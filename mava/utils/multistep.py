from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp

# Borrowed from stoix (https://github.com/EdanToledo/Stoix/blob/main/stoix/utils/multistep.py)
# and adapted for MARL
def truncated_generalized_advantage_estimation(
    r_t: chex.Array,
    discount_t: chex.Array,
    lambda_: Union[chex.Array, chex.Scalar],
    values: chex.Array,
    stop_target_gradients: bool = True,
    time_major: bool = False,
    standardize_advantages: bool = False,
    truncation_flags: Optional[chex.Array] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ₊₁ + γₜ₊₁ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Note: This paper uses a different notation than the RLax standard
    convention that follows Sutton & Barto. We use rₜ₊₁ to denote the reward
    received after acting in state sₜ, while the PPO paper uses rₜ.

    Args:
        r_t: Sequence of rewards at times [1, k]
        discount_t: Sequence of discounts at times [1, k]
        lambda_: Mixing parameter; a scalar or sequence of lambda_t at times [1, k]
        values: Sequence of values under π at times [0, k]
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
        time_major: If True, the first dimension of the input tensors is the time
        dimension.
        standardize_advantages: If True, standardize the advantages.
        truncation_flags: Optional sequence of truncation flags at times [1, k].

    Returns:
        Multistep truncated generalized advantage estimation at times [0, k-1].
        The target values at times [0, k-1] are also returned.
    """

    if truncation_flags is None:
        truncation_flags = jnp.zeros_like(r_t)

    truncation_mask = 1.0 - truncation_flags

    # Swap axes to make time axis the first dimension
    if not time_major:
        batch_size = r_t.shape[0]
        r_t, discount_t, values, truncation_mask = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, values, truncation_mask)
        )
    else:
        batch_size = r_t.shape[1]

    chex.assert_type([r_t, values, discount_t, truncation_mask], float)

    lambda_ = jnp.ones_like(discount_t) * lambda_  # If scalar, make into vector.

    delta_t = r_t + discount_t * values[1:] - values[:-1]
    delta_t *= truncation_mask

    # Iterate backwards to calculate advantages.
    def _body(
        acc: chex.Array, xs: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        deltas, discounts, lambda_, trunc_mask = xs
        acc = deltas + discounts * lambda_ * trunc_mask * acc
        return acc, acc

    _, advantage_t = jax.lax.scan(
        _body,
        jnp.zeros_like(r_t[0]),  # (T, A)
        (delta_t, discount_t, lambda_, truncation_mask),
        reverse=True,
        unroll=16,
    )

    target_values = values[:-1] + advantage_t
    advantage_t *= truncation_mask

    if not time_major:
        # Swap axes back to original shape
        advantage_t, target_values = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (advantage_t, target_values)
        )

    if stop_target_gradients:
        advantage_t, target_values = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x), (advantage_t, target_values)
        )

    if standardize_advantages:
        advantage_t = jax.nn.standardize(advantage_t, axis=(0, 1))

    return advantage_t, target_values
