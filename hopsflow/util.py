"""Utilities for the energy flow calculation."""

import itertools
import multiprocessing
import numpy as np
from numpy.lib.npyio import savetxt
import scipy
import scipy.integrate
from typing import Iterator, Optional, Any, Callable, Union
from lmfit import minimize, Parameters
from numpy.polynomial import Polynomial
from tqdm import tqdm
from pathlib import Path

Aggregate = tuple[int, np.ndarray, np.ndarray]
EnsembleReturn = Union[Aggregate, list[Aggregate]]


def apply_operator(ψ: np.ndarray, op: np.ndarray) -> np.ndarray:
    """
    Applies the operator ``op`` to each element of the time series
    ψ of the dimensions ``(*, dim)`` where ``dim`` is the hilbert
    space dimension.
    """

    return np.array((op @ ψ.T).T)


def sandwhich_operator(
    ψ: np.ndarray,
    op: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Applies the operator ``op`` to each element of the time
         series ``ψ`` of the dimensions ``(*, dim)`` where ``dim`` is the
         hilbert space dimension and sandwiches ``ψ`` onto it from the
         left.  If ``normalize`` is :any:`True` then the value will be
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
    ψs: Iterator[np.ndarray],
    op: np.ndarray,
    N: Optional[int],
    normalize: bool = False,
    **kwargs,
) -> EnsembleReturn:
    """Calculates the expecation value of ``op`` as a time series.

    :param ψs: A collection of stochastic trajectories.  Each
        element should have the shape  ``(time, dim-sys)``.
    :param op: The operator.
    :param N: Number of samples to take.

    All the other kwargs are passed on to :any:`ensemble_mean`.

    :returns: the expectation value
    """

    return ensemble_mean(
        ψs, sandwhich_operator, N, const_args=(op, normalize), **kwargs
    )


def mulitply_hierarchy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply each hierarchy member with a member of ``left`` for each time step.

    :param left: array of shape ``(hierarchy-width,)``
    :param right: array of shape ``(time-steps, hierarchy-width, system-dimension)``
    """

    return left[None, :, None] * right


def dot_with_hierarchy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    r"""Calculates :math:`\sum_k \langle\mathrm{left} | \mathrm{right}^{(e_k)}\rangle` for
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

    return scipy.integrate.cumulative_trapezoid(arr, t, initial=0)


###############################################################################
#                                Ensemble Mean                                #
###############################################################################

_ENSEMBLE_MEAN_ARGS: tuple = tuple()
_ENSEMBLE_MEAN_KWARGS: dict = dict()


def _ensemble_mean_call(arg) -> np.ndarray:
    global _ENSEMBLE_MEAN_ARGS
    global _ENSEMBLE_MEAN_KWARGS

    return _ENSEMBLE_FUNC(arg, *_ENSEMBLE_MEAN_ARGS, **_ENSEMBLE_MEAN_KWARGS)


def _ensemble_mean_init(func: Callable, args: tuple, kwargs: dict):
    global _ENSEMBLE_FUNC
    global _ENSEMBLE_MEAN_ARGS
    global _ENSEMBLE_MEAN_KWARGS

    _ENSEMBLE_FUNC = func
    _ENSEMBLE_MEAN_ARGS = args
    _ENSEMBLE_MEAN_KWARGS = kwargs


# TODO: Use paramspec


class WelfordAggregator:
    __slots__ = ["n", "mean", "_m_2"]

    def __init__(self, first_value: np.ndarray):
        self.n = 1
        self.mean = first_value
        self._m_2 = np.zeros_like(first_value)

    def update(self, new_value: np.ndarray):
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self._m_2 += np.abs(delta) * np.abs(delta2)

    @property
    def sample_variance(self) -> np.ndarray:
        return self._m_2 / (self.n - 1)

    @property
    def ensemble_variance(self) -> np.ndarray:
        return self.sample_variance / self.n

    @property
    def ensemble_std(self) -> np.ndarray:
        return np.sqrt(self.ensemble_variance)


def ensemble_mean(
    arg_iter: Iterator[Any],
    function: Callable[..., np.ndarray],
    N: Optional[int] = None,
    const_args: tuple = tuple(),
    const_kwargs: dict = dict(),
    n_proc: Optional[int] = None,
    every: Optional[int] = None,
    save: Optional[str] = None,
) -> EnsembleReturn:

    results = []
    aggregate = WelfordAggregator(function(next(arg_iter), *const_args))

    if N == 1:
        results = [(1, aggregate.mean, np.zeros_like(aggregate.mean))]
        return results if every else results[0]

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
            10,
        )

        for res in tqdm(result_iter, total=(N - 1) if N else None):
            aggregate.update(res)

            if every is not None and (aggregate.n % every) == 0 or aggregate.n == N:
                results.append(
                    (aggregate.n, aggregate.mean.copy(), aggregate.ensemble_std.copy())
                )

    if not every:
        results = results[-1]

    if save:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            np.save(f, results)

    return results


def fit_α(
    α: Callable[[np.ndarray], np.ndarray],
    n: int,
    t_max: float,
    support_points: Union[int, np.ndarray] = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the BCF ``α`` to a sum of ``n`` exponentials up to
    ``t_max`` using a number of ``support_points``.
    """

    def residual(fit_params, x, data):
        resid = 0
        w = np.array([fit_params[f"w{i}"] for i in range(n)]) + 1j * np.array(
            [fit_params[f"wi{i}"] for i in range(n)]
        )
        g = np.array([fit_params[f"g{i}"] for i in range(n)]) + 1j * np.array(
            [fit_params[f"gi{i}"] for i in range(n)]
        )
        resid = data - α_apprx(x, g, w)

        return resid.view(float)

    fit_params = Parameters()
    for i in range(n):
        fit_params.add(f"g{i}", value=0.1)
        fit_params.add(f"gi{i}", value=0.1)
        fit_params.add(f"w{i}", value=0.1)
        fit_params.add(f"wi{i}", value=0.1)

    ts = support_points
    if isinstance(ts, int):
        ts = np.linspace(0, t_max, int(support_points))

    out = minimize(residual, fit_params, args=(ts, α(ts)))

    w = np.array([out.params[f"w{i}"] for i in range(n)]) + 1j * np.array(
        [out.params[f"wi{i}"] for i in range(n)]
    )
    g = np.array([out.params[f"g{i}"] for i in range(n)]) + 1j * np.array(
        [out.params[f"gi{i}"] for i in range(n)]
    )

    return w, g


def except_element(array: np.ndarray, index: int) -> np.ndarray:
    """Returns the ``array`` except the element with ``index``."""
    mask = np.array([i != index for i in range(array.size)])
    return array[mask]


def poly_real(p: Polynomial) -> Polynomial:
    """Return the real part of ``p``."""
    new = p.copy()
    new.coef = p.coef.real
    return new


def uni_to_gauss(x: np.ndarray):
    """Transforms ``x`` into ``len(x)/2`` normal distributed numbers."""
    n = len(x) // 2
    phi = x[:n] * 2 * np.pi
    r = np.sqrt(-np.log(x[n:]))
    return r * np.exp(1j * phi)
