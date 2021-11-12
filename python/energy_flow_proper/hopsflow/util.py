"""Utilities for the energy flow calculation."""
import itertools
import functools
import multiprocessing
import numpy as np
import scipy
from typing import Iterator, Optional, Any, Tuple, Callable, Dict


def apply_operator(ψ: np.ndarray, op: np.ndarray) -> np.ndarray:
    """
    Applies the operator ``op`` to each element of the time series
    ψ of the dimensions ``(*, dim)`` where ``dim`` is the hilbert
    space dimension.
    """

    return np.array((op @ ψ.T).T)


def sandwhich_operator(
    ψ: np.ndarray, op: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """
    Applies the operator ``op`` to each element of the time
         series ψ of the dimensions ``(*, dim)`` where ``dim`` is the
         hilbert space dimension and sandwiches ``ψ`` onto it from the
         left.  If ``normalize`` is :py:`True` then the value will be
         divided by the squared norm.
    """

    exp_val = np.sum(ψ.conj() * apply_operator(ψ, op), axis=1)

    if normalize:
        exp_val /= np.sum(ψ.conj() * ψ, axis=1).real

    return exp_val


def operator_expectation(ρ: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Calculates the expecation value of ``op`` as a time series.

    :param ρ: The state as time series. ``(time, dim-sys, dim-sys)``
    :param op: The operator.
    :returns: the expectation value
    """

    return np.einsum("ijk,kj", ρ, op).real


def operator_expectation_ensemble(
    ψs: Iterator[np.ndarray], op: np.ndarray, N: Optional[int], normalize: bool = False
) -> np.ndarray:
    """Calculates the expecation value of ``op`` as a time series.

    :param ψs: A collection of stochastic trajectories.  Each
        element should have the shape  ``(time, dim-sys)``.
    :param op: The operator.
    :param N: Number of samples to take.

    :returns: the expectation value
    """

    return ensemble_mean(ψs, sandwhich_operator, N, const_args=(op, normalize))


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
        [0] + [scipy.integrate.simpson(arr[0:n], t[0:n]) for n in range(1, len(t))]
    )


###############################################################################
#                                Ensemble Mean                                #
###############################################################################

_ENSEMBLE_MEAN_ARGS: Tuple = tuple()
_ENSEMBLE_MEAN_KWARGS: Dict = dict()


def _ENSEMBLE_FUNC(_, *args, **kwargs):
    return _


def _ensemble_mean_call(arg) -> np.ndarray:
    global _ENSEMBLE_MEAN_ARGS
    global _ENSEMBLE_MEAN_KWARGS

    return _ENSEMBLE_FUNC(arg, *_ENSEMBLE_MEAN_ARGS, **_ENSEMBLE_MEAN_KWARGS)


def _ensemble_mean_init(func: Callable, args: Tuple, kwargs: Dict):
    global _ENSEMBLE_FUNC
    global _ENSEMBLE_MEAN_ARGS
    global _ENSEMBLE_MEAN_KWARGS

    _ENSEMBLE_FUNC = func
    _ENSEMBLE_MEAN_ARGS = args
    _ENSEMBLE_MEAN_KWARGS = kwargs


# TODO: Use paramspec
def ensemble_mean(
    arg_iter: Iterator[Any],
    function: Callable[..., np.ndarray],
    N: Optional[int] = None,
    const_args: Tuple = tuple(),
    const_kwargs: Dict = dict(),
    n_proc: Optional[int] = None,
):

    result = function(next(arg_iter), *const_args)

    if not n_proc:
        n_proc = multiprocessing.cpu_count()

    with multiprocessing.Pool(
        processes=n_proc,
        initializer=_ensemble_mean_init,
        initargs=(function, const_args, const_kwargs),
    ) as pool:
        result_iter = pool.imap_unordered(
            _ensemble_mean_call,
            itertools.islice(arg_iter, None, N - 1 if N else None),
            100,
        )

        n = 1
        for res in result_iter:
            result += res
            n += 1

    return result / n
