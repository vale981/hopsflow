"""Utilities for the energy flow calculation."""
import numpy as np
import scipy

def apply_operator(ψ: np.ndarray, op: np.ndarray) -> np.ndarray:
    """
    Applies the operator ``op`` to each element of the time series
    ψ of the dimensions ``(*, dim)`` where ``dim`` is the hilbert
    space dimension.
    """

    return np.array((op @ ψ.T).T)


def operator_expectation(ρ: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Calculates the expecation value of ``op`` as a time series.

    :param ρ: The state as time series. ``(time, dim-sys, dim-sys)``
    :param op: The operator.
    :returns: the expectation value
    """

    return np.einsum("ijk,kj", ρ, op).real


def mulitply_hierarchy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply each hierarchy member with a member of ``left`` for each time step.

    :param left: array of shape ``(hierarchy-width,)``
    :param right: array of shape ``(time-steps, hierarchy-width, system-dimension)``
    """

    return left[None, :, None] * right


def dot_with_hierarchy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    r"""Calculates $\sum_k \langle\mathrm{left} | \right^{(e_k)}$ for
    each time step.

    :param left: array of shape ``(time-steps, system-dimension, hierarchy-width,)``
    :param right: array of shape ``(time-steps, hierarchy-width, system-dimension)``
    """

    return np.sum(left[:, None, :] * right, axis=(1, 2))


def α_apprx(τ: np.ndarray, G: np.ndarray, W: np.ndarray) -> np.ndarray:
    r"""
    Calculate exponential expansion $\sum_i G_i \exp(W_i * τ)$ of the
    BCF along ``τ``.

    :param τ: the time
    :param G: pefactors
    :param W: exponents
    :returns: the exponential expansion evaluated at ``τ``
    """

    return np.sum(
        G[np.newaxis, :] * np.exp(-W[np.newaxis, :] * (τ[:, np.newaxis])), axis=1
    )


def integrate_array(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculates the antiderivative of the function sampled in ``arr``
    along ``t``.
    """

    return np.array(
      [0]
      + [
          scipy.integrate.simpson(arr[0:n], t[0:n])
          for n in range(1, len(t))
      ]
    )
