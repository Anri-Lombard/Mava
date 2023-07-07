# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

import functools
import json
import os
import pickle
import time
from functools import partial
from typing import Any, NamedTuple, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper
from jumanji.environments.routing.robot_warehouse.types import State
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper, Wrapper

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        state, timestep = self._env.reset(key)
        # timestep.extras = {}
        state = LogEnvState(state, 0, 0, 0, 0)
        return state, timestep

    def step(
        self,
        state: State,
        action: jnp.ndarray,
    ) -> Tuple[State, TimeStep]:

        env_state, timestep = self._env.step(state.env_state, action)

        new_episode_return = state.episode_returns + jnp.mean(timestep.reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - timestep.last()),
            episode_lengths=new_episode_length * (1 - timestep.last()),
            returned_episode_returns=state.returned_episode_returns
            * (1 - timestep.last())
            + new_episode_return * timestep.last(),
            returned_episode_lengths=state.returned_episode_lengths
            * (1 - timestep.last())
            + new_episode_length * timestep.last(),
        )
        # timestep.extras["returned_episode_returns"] = state.returned_episode_returns
        # timestep.extras["returned_episode_lengths"] = state.returned_episode_lengths
        # timestep.extras["returned_episode"] = timestep.last()
        return state, timestep


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, observation, action_mask):

        x = observation
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        logits = jnp.where(
            action_mask,
            actor_mean,
            jnp.finfo(jnp.float32).min,
        )
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: TimeStep
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = jumanji.make(config["ENV_NAME"])
    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK FOR SINGLE AGENT

        # Since agents action homogeneous, use action dim of first agent
        # network = ActorCritic(
        #     env.action_spec().num_values[0].tolist(), activation=config["ACTIVATION"]
        # )

        network = ActorCritic(5, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_obs = env.observation_spec().generate_value()

        init_obs_add = jnp.expand_dims(init_obs.agents_view, axis=0)
        init_mask_add = jnp.expand_dims(init_obs.action_mask, axis=0)

        network_params = network.init(_rng, init_obs_add, init_mask_add)

        # network_params = network.init(
        #     _rng, init_obs.agents_view, init_obs.action_mask
        # )
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        train_state = (jax.device_put_replicated(train_state, jax.local_devices()),)
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # env_state, timestep = env.reset(_rng)
        num_local_devices = jax.local_device_count()
        num_global_devices = jax.device_count()
        num_workers = num_global_devices // num_local_devices
        local_batch_size = config["NUM_ENVS"] // num_global_devices
        reset_keys = jax.random.split(rng, config["NUM_ENVS"]).reshape(
            (
                num_workers,
                num_local_devices,
                config["NUM_ENVS"] // num_local_devices,
                -1,
            )
        )
        """reset_keys = jnp.reshape(
            reset_rng, (num_local_devices, config["NUM_ENVS"]//num_local_devices, -1)
        )"""
        reset_keys_per_worker = reset_keys[jax.process_index()]
        # reset_keys_per_worker = reset_keys
        env_state, timestep = jax.pmap(
            jax.vmap(env.reset, in_axes=(0)), axis_name="devices"
        )(reset_keys_per_worker)
        # pi, values = network.apply(network_params, timestep.observation.agents_view, timestep.observation.action_mask)

        # TRAIN LOOP
        @functools.partial(jax.pmap, axis_name="devices")
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_timestep, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                pi, value = network.apply(
                    train_state.params,
                    last_timestep.observation.agents_view,
                    last_timestep.observation.action_mask,
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                env_state, next_timestep = jax.vmap(
                    env.step,
                    in_axes=(0, 0),
                )(env_state, action)

                transition = Transition(
                    jnp.repeat(next_timestep.last(), 4).reshape(
                        config["NUM_ENVS"] // num_local_devices, -1
                    ),  # TODO: duplicating for now. But this should be per agent.
                    action,
                    value,
                    jnp.repeat(next_timestep.reward, 4).reshape(
                        config["NUM_ENVS"] // num_local_devices, -1
                    ),  # TODO: duplicating for now. But this should be per agent.
                    log_prob,
                    last_timestep,
                    {
                        "returned_episode_returns": jnp.repeat(
                            env_state.returned_episode_returns, 4
                        ).reshape(config["NUM_ENVS"] // num_local_devices, -1),
                        "returned_episode_lengths": jnp.repeat(
                            env_state.returned_episode_lengths, 4
                        ).reshape(config["NUM_ENVS"] // num_local_devices, -1),
                        "returned_episode": jnp.repeat(next_timestep.last(), 4).reshape(
                            config["NUM_ENVS"] // num_local_devices, -1
                        ),
                    },
                )
                runner_state = (train_state, env_state, next_timestep, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_timestep, rng = runner_state
            _, last_val = network.apply(
                train_state.params,
                last_timestep.observation.agents_view,
                last_timestep.observation.action_mask,
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(
                            params,
                            traj_batch.obs.observation.agents_view,
                            traj_batch.obs.observation.action_mask,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    total_loss, grads = jax.lax.pmean(
                        (total_loss, grads), axis_name="devices"
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                batch_size = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] // num_local_devices
                )
                """assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]//num_local_devices
                ), "batch size must be equal to number of steps * number of envs"""
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            # metric = traj_batch.info
            metric = jax.tree_util.tree_map(
                lambda x: x[:, :, 0], traj_batch.info
            )  # only the team rewards
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_timestep, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, config["NUM_ENVS"]).reshape(
            num_local_devices, config["NUM_ENVS"] // num_local_devices, -1
        )
        _rng = _rng[:, :, 0]
        runner_state = (train_state[jax.process_index()], env_state, timestep, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


config = {
    "LR": 5e-3,
    "NUM_ENVS": 16,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ENV_NAME": "RobotWarehouse-v0",
    "ANNEAL_LR": True,
}

SEEDS = [42]
STORE_RESULTS = False

print("Starting run.")
rng = jax.random.PRNGKey(42)
print("Compiling")
compile_start_time = time.time()
# train_jit = jax.jit(chex.assert_max_traces(make_train(config), n=1))
train_jit = make_train(config)
# jax.block_until_ready(train_jit(rng))  # compile your code
print(f"Compile done and took {time.time() - compile_start_time} seconds.")

# Run and store multiple seeds.
for seed_num in SEEDS:
    print(f"Training starting for {seed_num}.")
    rng = jax.random.PRNGKey(seed_num)
    start_time = time.perf_counter()
    out = train_jit(rng)
    # jax.block_until_ready(out)
    total_time = time.perf_counter() - start_time
    print(f"TOTAL TIME TO RUN: {total_time}")

    out["metrics"]["run_time"] = total_time
    print(out["metrics"]["returned_episode_returns"].mean())

    store_dict = {}
    store_dict["returns"] = out["metrics"]["returned_episode_returns"].tolist()
    store_dict["run_time"] = total_time

    if STORE_RESULTS:
        with open(f"results_{config['NUM_ENVS']}/{seed_num}.json", "w") as f:
            json.dump(store_dict, f)