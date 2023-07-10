"""
An example following the Anakin podracer example found here:
https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing#scrollTo=myLN2J47oNGq
"""

import timeit
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: jnp.ndarray


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, observation):

        x = observation.agents_view

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

        masked_logits = jnp.where(
            observation.action_mask,
            actor_mean,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

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


def get_learner_fn(env, forward_pass, opt_update, config):
    def update_step_fn(params, opt_state, outer_rng, env_state, timestep):

        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            params, env_state, last_timestep, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = forward_pass(params, last_timestep.observation)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state, next_timestep = env.step(env_state, action)

            done, reward = jax.tree_map(
                lambda x: jnp.repeat(x, 4).reshape(-1),
                [next_timestep.last(), next_timestep.reward],
            )

            transition = Transition(
                done,
                action,
                value,
                reward,
                log_prob,
                last_timestep.observation,
                {},
            )
            runner_state = (params, env_state, next_timestep, rng)
            return runner_state, transition

        runner_state = (params, env_state, timestep, outer_rng)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        params, env_state, last_timestep, rng = runner_state
        _, last_val = forward_pass(params, last_timestep.observation)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                    pi, value = forward_pass(params, traj_batch.observation)
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
                total_loss, grads = grad_fn(params, traj_batch, advantages, targets)
                # pmean
                total_loss = jax.lax.pmean(total_loss, axis_name="i")
                grads = jax.lax.pmean(grads, axis_name="j")
                grads = jax.lax.pmean(grads, axis_name="i")
                updates, new_opt_state = opt_update(grads, opt_state)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), total_loss

            (params, opt_state), traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config['ROLLOUT_LENGTH']
            # batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            # assert (
            #     batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            # ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[1:]), batch
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
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )

            update_state = ((params, opt_state), traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = ((params, opt_state), traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        ((params, opt_state), traj_batch, advantages, targets, rng) = update_state

        return params, opt_state, rng, env_state, last_timestep

    def update_fn(params, opt_state, rng, env_state, timestep):
        """Compute a gradient update from a single trajectory."""
        rng, loss_rng = jax.random.split(rng)
        # grads, new_env_state = jax.grad(  # compute gradient on a single trajectory.
        #     loss_fn, has_aux=True)(params, loss_rng, env_state)
        # grads = lax.pmean(grads, axis_name='j')  # reduce mean across cores.
        # grads = lax.pmean(grads, axis_name='i')  # reduce mean across batch.
        # updates, new_opt_state = opt_update(grads, opt_state)  # transform grads.
        # new_params = optax.apply_updates(params, updates)  # update parameters.

        new_params, new_opt_state, rng, new_env_state, new_timestep = update_step_fn(
            params,
            opt_state,
            rng,
            env_state,
            timestep,
        )

        return new_params, new_opt_state, rng, new_env_state, new_timestep

    def learner_fn(params, opt_state, rngs, env_states, timesteps):
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            update_fn, axis_name="j"
        )  # vectorize across batch.

        def iterate_fn(_, val):  # repeat many times to avoid going back to Python.
            params, opt_state, rngs, env_states, timesteps = val
            # params = jax.lax.pmean(params, axis_name="j")  # reduce mean across batch.
            return batched_update_fn(params, opt_state, rngs, env_states, timesteps)

        return jax.lax.fori_loop(
            0,
            config["ITERATIONS"],
            iterate_fn,
            (params, opt_state, rngs, env_states, timesteps),
        )

    return learner_fn


def run_experiment(env, config):
    cores_count = len(jax.devices())
    num_actions = int(env.action_spec().num_values[0])
    network = ActorCritic(num_actions, activation=config["ACTIVATION"])
    optim = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_env, rng_params = jax.random.split(rng, 3)

    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: x[None, ...],
        init_obs,
    )

    params = network.init(rng_params, init_obs)
    opt_state = optim.init(params)

    # TODO: Complete this
    learn = get_learner_fn(
        env,
        network.apply,
        optim.update,
        config,
    )

    learn = jax.pmap(learn, axis_name="i")  # replicate over multiple cores.

    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
    opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch

    rng, *env_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)
    env_states, env_timesteps = jax.vmap(env.reset)(jnp.stack(env_rngs))  # init envs.
    rng, *step_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)

    reshape = lambda x: x.reshape((cores_count, config["BATCH_SIZE"]) + x.shape[1:])
    step_rngs = reshape(jnp.stack(step_rngs))  # add dimension to pmap over.
    env_states = jax.tree_util.tree_map(
        reshape,
        env_states,
    )  # add dimension to pmap over.
    env_timesteps = jax.tree_util.tree_map(
        reshape,
        env_timesteps,
    )
    # env_timesteps = reshape(env_timesteps)  # add dimension to pmap over.

    with TimeIt(tag="COMPILATION"):
        out = learn(params, opt_state, step_rngs, env_states, env_timesteps)  # compiles
        jax.block_until_ready(out)


    # Number of iterations
    timesteps_per_iteration = (cores_count*config["ROLLOUT_LENGTH"]* config["BATCH_SIZE"])
    config["ITERATIONS"] = config["TOTAL_TIMESTEPS"] // timesteps_per_iteration # Number of training updates 

    num_frames = config["TOTAL_TIMESTEPS"]

    with TimeIt(tag="EXECUTION", frames=num_frames):
        out = learn(  # runs compiled fn
            params, opt_state, step_rngs, env_states, env_timesteps, 
        )
        jax.block_until_ready(out)

if __name__ == "__main__":
    config = {
        "LR": 5e-3,
        "ENV_NAME": "RobotWarehouse-v0",
        "ACTIVATION": "relu",
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "BATCH_SIZE": 4, # Parallel updates / environmnents
        "ROLLOUT_LENGTH": 128, # Length of each rollout
        "TOTAL_TIMESTEPS": 204800,
        "SEED": 42,
    }

    run_experiment(jumanji.make(config["ENV_NAME"]), config)
