# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import functools
import os
from logging import Logger as SacredLogger
from typing import Any, Callable, Dict, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from jumanji.env import Environment
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
from jumanji.wrappers import AutoResetWrapper
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from sacred import Experiment, observers, run, utils

from mava.evaluator import evaluator_setup
from mava.logger import logger_setup
from mava.types import ExperimentOutput, PPOTransition, RNNLearnerState
from mava.utils.logger_tools import config_copy, get_experiment_path, get_logger
from mava.utils.timing_utils import TimeIt
from mava.wrappers.jumanji import AgentIDWrapper, LogWrapper, RwareMultiAgentWrapper


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), hidden_size)


class Actor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: Tuple[chex.Array, chex.Array],
    ) -> Tuple[chex.Array, distrax.Categorical]:
        """Forward pass."""
        observation, done = observation_done

        policy_embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(observation.agents_view)
        policy_embedding = nn.relu(policy_embedding)

        policy_rnn_in = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_in)

        actor_output = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            policy_embedding
        )
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        return policy_hidden_state, pi


class Critic(nn.Module):
    """Critic Network."""

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: Tuple[chex.Array, chex.Array],
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        critic_embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(observation.agents_view)
        critic_embedding = nn.relu(critic_embedding)

        critic_rnn_in = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_in)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(critic_embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return critic_hidden_state, jnp.squeeze(critic, axis=-1)


def get_learner_fn(
    env: jumanji.Environment,
    apply_fns: Tuple[Callable, Callable],
    update_fns: Tuple[Callable, Callable],
    config: Dict,
) -> Callable:
    """Get the learner function."""

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - actor_params (FrozenDict): The current actor parameters.
                - critic_params (FrozenDict): The current critic parameters.
                - actor_opt_state (OptState): The current actor optimizer state.
                - critic_opt_state (OptState): The current critic optimizer state.
                - rng (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - dones (bool): Whether the last timestep was a terminal state.
                - policy_hstate (chex.Array): The current policy hidden state.
                - critic_hstate (chex.Array): The current critic hidden state.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, PPOTransition]:
            """Step the environment."""
            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                rng,
                env_state,
                last_timestep,
                last_done,
                policy_hidden_state,
                critic_hidden_state,
            ) = learner_state
            rng, policy_rng = jax.random.split(rng)

            # Add a batch dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], last_timestep.observation
            )
            ac_in = (
                batched_observation,
                last_done[:, 0][jnp.newaxis, :],
            )

            # Run the network.
            policy_hidden_state, actor_policy = actor_apply_fn(
                actor_params, policy_hidden_state, ac_in
            )
            critic_hidden_state, value = critic_apply_fn(critic_params, critic_hidden_state, ac_in)

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_rng)
            log_prob = actor_policy.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # log episode return and length
            done, reward = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, config["num_agents"]).reshape(config["num_envs"], -1),
                (timestep.last(), timestep.reward),
            )
            info = {
                "episode_return": env_state.episode_return_info,
                "episode_length": env_state.episode_length_info,
            }

            transition = PPOTransition(
                done, action, value, reward, log_prob, last_timestep.observation, info
            )
            learner_state = RNNLearnerState(
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                rng,
                env_state,
                timestep,
                done,
                policy_hidden_state,
                critic_hidden_state,
            )
            return learner_state, transition

        # INITIALISE RNN STATE
        init_policy_hstate, init_critic_hstate = (
            learner_state.policy_hstate,
            learner_state.critic_hstate,
        )

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config["rollout_length"]
        )

        # CALCULATE ADVANTAGE
        (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            rng,
            env_state,
            last_timestep,
            last_done,
            policy_hidden_state,
            critic_hidden_state,
        ) = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = jax.tree_util.tree_map(
            lambda x: x[jnp.newaxis, :], last_timestep.observation
        )
        ac_in = (
            batched_last_observation,
            last_done[:, 0][jnp.newaxis, :],
        )

        # Run the network.
        _, last_val = critic_apply_fn(critic_params, critic_hidden_state, ac_in)

        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

        def _calculate_gae(
            traj_batch: PPOTransition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            def _get_advantages(gae_and_next_value: Tuple, transition: PPOTransition) -> Tuple:
                """Calculate the GAE for a single transition."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["gamma"] * next_value * (1 - done) - value
                gae = delta + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                actor_params, critic_params, actor_opt_state, critic_opt_state = train_state
                (
                    init_policy_hstate,
                    init_critic_hstate,
                    traj_batch,
                    advantages,
                    targets,
                ) = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    _, actor_policy = actor_apply_fn(
                        actor_params,
                        init_policy_hstate.squeeze(0),
                        (
                            traj_batch.obs,
                            traj_batch.done[:, :, 0],
                        ),
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["clip_eps"],
                            1.0 + config["clip_eps"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = actor_policy.entropy().mean()

                    total_loss = loss_actor - config["ent_coef"] * entropy
                    return total_loss, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    critic_opt_state: OptState,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK
                    _, value = critic_apply_fn(
                        critic_params,
                        init_critic_hstate.squeeze(0),
                        (
                            traj_batch.obs,
                            traj_batch.done[:, :, 0],
                        ),
                    )

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config["clip_eps"], config["clip_eps"]
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config["vf_coef"] * value_loss
                    return total_loss, (value_loss)

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    actor_params, actor_opt_state, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss_info, critic_grads = critic_grad_fn(
                    critic_params, critic_opt_state, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(actor_grads, actor_opt_state)
                actor_new_params = optax.apply_updates(actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, critic_opt_state
                )
                critic_new_params = optax.apply_updates(critic_params, critic_updates)

                # PACK LOSS INFO
                total_loss = actor_loss_info[0] + critic_loss_info[0]
                value_loss = critic_loss_info[1]
                actor_loss = actor_loss_info[1][0]
                entropy = actor_loss_info[1][1]
                loss_info = (
                    total_loss,
                    (value_loss, actor_loss, entropy),
                )

                return (
                    actor_new_params,
                    critic_new_params,
                    actor_new_opt_state,
                    critic_new_opt_state,
                ), loss_info

            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                init_policy_hstate,
                init_critic_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, shuffle_rng = jax.random.split(rng)

            # SHUFFLE MINIBATCHES
            permutation = jax.random.permutation(shuffle_rng, config["num_envs"])
            batch = (init_policy_hstate, init_critic_hstate, traj_batch, advantages, targets)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            reshaped_batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (x.shape[0], config["num_minibatches"], -1, *x.shape[2:])),
                shuffled_batch,
            )
            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # UPDATE MINIBATCHES
            (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
            ), loss_info = jax.lax.scan(
                _update_minibatch,
                (actor_params, critic_params, actor_opt_state, critic_opt_state),
                minibatches,
            )

            update_state = (
                actor_params,
                critic_params,
                actor_opt_state,
                critic_opt_state,
                init_policy_hstate,
                init_critic_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, loss_info

        init_policy_hstate, init_critic_hstate = (
            init_policy_hstate[None, :],
            init_critic_hstate[None, :],
        )
        update_state = (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            init_policy_hstate,
            init_critic_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["ppo_epochs"]
        )

        (
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            _,
            _,
            traj_batch,
            advantages,
            targets,
            rng,
        ) = update_state
        learner_state = RNNLearnerState(
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            rng,
            env_state,
            last_timestep,
            last_done,
            policy_hidden_state,
            critic_hidden_state,
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> ExperimentOutput:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - actor_params (FrozenDict): The current actor parameters.
                - critic_params (FrozenDict): The current critic parameters.
                - actor_opt_state (OptState): The current actor optimizer state.
                - critic_opt_state (OptState): The current critic optimizer state.
                - rng (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - dones (bool): Whether the last timestep was a terminal state.
                - policy_hstate (chex.Array): The current policy hidden state.
                - critic_hstate (chex.Array): The current critic hidden state.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (metric, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config["num_updates_per_eval"]
        )
        total_loss, (value_loss, loss_actor, entropy) = loss_info
        return ExperimentOutput(
            learner_state=learner_state,
            episodes_info=metric,
            total_loss=total_loss,
            value_loss=value_loss,
            loss_actor=loss_actor,
            entropy=entropy,
        )

    return learner_fn


def learner_setup(
    env: Environment, rngs: chex.Array, config: Dict
) -> Tuple[Callable, Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""

    def get_learner_state(
        rngs: chex.Array,
        actor_network: Actor,
        critic_network: Critic,
        actor_optim: optax.GradientTransformation,
        critic_optim: optax.GradientTransformation,
    ) -> RNNLearnerState:
        """Get initial learner state."""
        # Get available TPU cores.
        n_devices = len(jax.devices())

        # PRNG keys.
        rng, rng_p = rngs

        # Initialise observation: Select only obs for a single agent.
        init_obs = env.observation_spec().generate_value()
        init_obs = jax.tree_util.tree_map(lambda x: x[0], init_obs)
        init_obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis, ...], config["num_envs"], axis=0),
            init_obs,
        )
        init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
        init_done = jnp.zeros((1, config["num_envs"]), dtype=bool)
        init_x = (init_obs, init_done)

        # Initialise hidden states.
        init_policy_hstate = ScannedRNN.initialize_carry((config["num_envs"]), 128)
        init_critic_hstate = ScannedRNN.initialize_carry((config["num_envs"]), 128)

        # initialise params and optimiser state.
        actor_params = actor_network.init(rng_p, init_policy_hstate, init_x)
        actor_opt_state = actor_optim.init(actor_params)
        critic_params = critic_network.init(rng_p, init_critic_hstate, init_x)
        critic_opt_state = critic_optim.init(critic_params)

        # Broadcast params and optimiser state to cores and batch.
        broadcast = lambda x: jnp.broadcast_to(
            x, (n_devices, config["update_batch_size"]) + x.shape
        )
        actor_params = jax.tree_map(broadcast, actor_params)
        actor_opt_state = jax.tree_map(broadcast, actor_opt_state)
        critic_params = jax.tree_map(broadcast, critic_params)
        critic_opt_state = jax.tree_map(broadcast, critic_opt_state)

        # Duplicate the hidden state for each agent.
        init_policy_hstate = jnp.expand_dims(init_policy_hstate, axis=1)
        init_policy_hstate = jnp.tile(init_policy_hstate, (1, config["num_agents"], 1))
        policy_hstates = jax.tree_map(broadcast, init_policy_hstate)

        init_critic_hstate = jnp.expand_dims(init_critic_hstate, axis=1)
        init_critic_hstate = jnp.tile(init_critic_hstate, (1, config["num_agents"], 1))
        critic_hstates = jax.tree_map(broadcast, init_critic_hstate)

        # Initialise environment states and timesteps.
        rng, *env_rngs = jax.random.split(
            rng, n_devices * config["update_batch_size"] * config["num_envs"] + 1
        )
        env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
            jnp.stack(env_rngs),
        )

        # Split rngs for each core.
        rng, *step_rngs = jax.random.split(rng, n_devices * config["update_batch_size"] + 1)
        # Add dimension to pmap over.
        reshape_step_rngs = lambda x: x.reshape(
            (n_devices, config["update_batch_size"]) + x.shape[1:]
        )
        step_rngs = reshape_step_rngs(jnp.stack(step_rngs))
        reshape_states = lambda x: x.reshape(
            (n_devices, config["update_batch_size"], config["num_envs"]) + x.shape[1:]
        )
        env_states = jax.tree_util.tree_map(reshape_states, env_states)
        timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

        # Initialise dones.
        dones = jnp.zeros(
            (
                n_devices,
                config["update_batch_size"],
                config["num_envs"],
                config["num_agents"],
            ),
            dtype=bool,
        )

        init_learner_state = RNNLearnerState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            key=step_rngs,
            env_state=env_states,
            timestep=timesteps,
            dones=dones,
            policy_hstate=policy_hstates,
            critic_hstate=critic_hstates,
        )
        return init_learner_state

    # Get number of actions and agents.
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    config["num_agents"] = num_agents

    # Define network and optimisers.
    actor_network = Actor(num_actions)
    critic_network = Critic()
    actor_optim = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["actor_lr"], eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["critic_lr"], eps=1e-5),
    )

    # Initialise learner state.
    init_learner_state = jax.vmap(
        get_learner_state, in_axes=(0, None, None, None, None), out_axes=1
    )(rngs, actor_network, critic_network, actor_optim, critic_optim)

    # Vmap network apply function over number of agents.
    vmapped_actor_network_apply_fn = jax.vmap(
        actor_network.apply, in_axes=(None, 1, (2, None)), out_axes=(1, 2)
    )
    # Vmap network apply function over number of agents.
    vmapped_critic_network_apply_fn = jax.vmap(
        critic_network.apply, in_axes=(None, 1, (2, None)), out_axes=(1, 2)
    )

    # Get network apply functions and optimiser updates.
    apply_fns = (vmapped_actor_network_apply_fn, vmapped_critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn_fn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(lambda x: jax.vmap(learn_fn, axis_name="train_seed")(x), axis_name="device")
    return learn, actor_network, init_learner_state


def run_experiment(_run: run.Run, _config: Dict, _log: SacredLogger) -> None:  # noqa: CCR001
    """Runs experiment."""
    # Logger setup
    config = config_copy(_config)
    log, stop_logger = logger_setup(_run, config, _log)

    # Create envs
    generator = RandomGenerator(**config["rware_scenario"]["task_config"])
    env = jumanji.make(config["env_name"], generator=generator)
    env = RwareMultiAgentWrapper(env)
    # Add agent id to observation.
    if config["add_agent_id"]:
        env = AgentIDWrapper(env)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    eval_env = jumanji.make(config["env_name"], generator=generator)
    eval_env = RwareMultiAgentWrapper(eval_env)
    if config["add_agent_id"]:
        eval_env = AgentIDWrapper(eval_env)

    # PRNG keys.
    if len(config["seeds"]) > 1:
        print(
            f"{Fore.RED}{Style.BRIGHT}",
            "Warning: Using multiple seeds may significantly increase experiment runtime.",
            f"{Style.RESET_ALL}",
        )
        if not config["split_json_by_seed"]:
            print(
                f"{Fore.RED}{Style.BRIGHT}",
                "Warning: All metrics will be saved in a single JSON file with seed number suffix.",
                f"{Style.RESET_ALL}",
            )
    seeds_array = jnp.array(config["seeds"])
    random_keys = jax.vmap(jax.random.PRNGKey)(seeds_array)
    rng, rng_e, rng_p = jax.vmap(jax.random.split, in_axes=(0, None), out_axes=1)(random_keys, 3)

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(env, (rng, rng_p), config)

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_rngs) = evaluator_setup(
        eval_env=eval_env,
        rng_e=rng_e,
        network=actor_network,
        params=learner_state.actor_params,
        config=config,
        use_recurrent_net=True,
        scanned_rnn=ScannedRNN,
    )

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config["num_updates_per_eval"] = config["num_updates"] // config["num_evaluation"]
    timesteps_per_training = (
        n_devices
        * config["num_updates_per_eval"]
        * config["rollout_length"]
        * config["update_batch_size"]
        * config["num_envs"]
    )

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.zeros(len(config["seeds"]))
    best_params = copy.deepcopy(trained_params)
    for i in range(config["num_evaluation"]):
        # Train.
        with TimeIt(
            tag=("COMPILATION" if i == 0 else "EXECUTION"),
            environment_steps=timesteps_per_training,
        ):
            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)

        # Log the results of the training.
        for seed_id, seed in enumerate(config["seeds"]):
            output = jax.tree_map(lambda x: x[:, seed_id], learner_output)  # noqa: B023
            log(
                metrics=output,
                t_env=timesteps_per_training * (i + 1),
                trainer_metric=True,
                seed=seed,
            )

        # Prepare for evaluation.
        trained_params = jax.tree_util.tree_map(
            lambda x: x[:, :, 0, ...], learner_output.learner_state.actor_params
        )
        split_and_reshape = lambda key: jnp.stack(jax.random.split(key, n_devices + 1)[1:]).reshape(
            n_devices, -1
        )
        eval_rngs = jax.vmap(split_and_reshape, out_axes=1)(rng_e)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_rngs)
        jax.block_until_ready(evaluator_output)

        for seed_id, seed in enumerate(config["seeds"]):
            output = jax.tree_map(lambda x: x[:, seed_id], evaluator_output)  # noqa: B023
            episode_return = log(metrics=output, t_env=timesteps_per_training * (i + 1), seed=seed)
            # Assess whether the present evaluator episode return outperforms
            # the best recorded return for the current seed.
            if config["absolute_metric"] and max_episode_return[seed_id] <= episode_return:
                # Update best params for only the current seed.
                best_params = jax.tree_util.tree_map(
                    lambda new, old: old.at[:, seed_id].set(new[:, seed_id]),  # noqa: B023
                    trained_params,
                    best_params,
                )
                max_episode_return = max_episode_return.at[seed_id].set(episode_return)

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute performance.
    if config["absolute_metric"]:
        split_and_reshape = lambda key: jnp.stack(jax.random.split(key, n_devices + 1)[1:]).reshape(
            n_devices, -1
        )
        eval_rngs = jax.vmap(split_and_reshape, out_axes=1)(rng_e)

        evaluator_output = absolute_metric_evaluator(best_params, eval_rngs)
        for seed_id, seed in enumerate(config["seeds"]):
            output = jax.tree_map(lambda x: x[:, seed_id], evaluator_output)  # noqa: B023
            log(
                metrics=output,
                t_env=timesteps_per_training * (i + 1),
                absolute_metric=True,
                seed=seed,
            )

    # Close logger in case of neptune.
    stop_logger()


@hydra.main(config_path="../configs", config_name="default.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""
    # Logger setup.
    logger = get_logger()
    ex = Experiment("mava", save_git_info=False)
    ex.logger = logger
    ex.captured_out_filter = utils.apply_backspaces_and_linefeeds

    # Set the base path for the experiment.
    cfg["system_name"] = "rec_ippo"
    exp_path = get_experiment_path(cfg, "sacred")
    file_obs_path = os.path.join(cfg["base_exp_path"], exp_path)
    ex.observers = [observers.FileStorageObserver.create(file_obs_path)]

    # Add configuration to the experiment.
    ex.add_config(OmegaConf.to_container(cfg, resolve=True))

    # Run experiment.
    ex.main(run_experiment)
    ex.run(config_updates={})

    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent IPPO experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    hydra_entry_point()