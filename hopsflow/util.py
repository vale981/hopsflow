"""Utilities for the energy flow calculation."""

import itertools
import multiprocessing
import numpy as np
import scipy
import scipy.integrate
import scipy.optimize
from typing import Iterator, Optional, Any, Callable, Union
from lmfit import minimize, Parameters
from numpy.polynomial import Polynomial
from tqdm import tqdm
from pathlib import Path
import sys
import hashlib
import logging
import json
from functools import singledispatch, singledispatchmethod
from scipy.stats import NumericalInverseHermite
import copy

Aggregate = tuple[int, np.ndarray, np.ndarray]
EnsembleReturn = Union[Aggregate, list[Aggregate]]


# class EnsembleValue:
#     def __init__(self, value: Union[Aggregate, list[Aggregate]]):
#         self._value = value if isinstance(value, list) else [value]

#     @property
#     def final_aggregate(self):
#         return self._value[-1]

#     @property
#     def N(self):
#         return self.final_aggregate[0]

#     @property
#     def value(self):
#         return self.final_aggregate[1]

#     @property
#     def σ(self):
#         return self.final_aggregate[2]

#     @property
#     def Ns(self):
#         return [N for N, _, _ in self._value]

#     @property
#     def values(self):
#         return [val for _, val, _ in self._value]

#     @property
#     def σs(self):
#         return [σ for _, _, σ in self._value]

#     @property
#     def aggregate_iterator(self):
#         for agg in self._value:
#             yield agg

#     def __getitem__(self, index: int):
#         return self._value[index]

#     def __len__(self) -> int:
#         return len(self._value)

#     def insert(self, value: Aggregate):
#         where = len(self._value)
#         for i, (N, _, _) in enumerate(self._value):
#             if N > value[0]:
#                 where = i
#                 break

#         self._value.insert(where, value)

#     def insert_multi(self, values: list[Aggregate]):
#         for value in values:
#             self.insert(value)

#     @singledispatchmethod
#     def __add__(self, _):
#         return NotImplemented

#     @__add__.register
#     def _(self, value: Aggregate):
#         new = copy.deepcopy(self)
#         new.insert(value)
#         return new

#     @__add__.register
#     def _(self, value: list[Aggregate]):
#         new = copy.deepcopy(self)
#         new.insert_multi(value)
#         return new

#     @singledispatchmethod
#     def __mul__(self, _):
#         return NotImplemented

#     @__mul__.register
#     def _(self, scale: Union[float, int]):
#         return EnsembleValue(
#             [(N, val * scale, np.abs(σ * scale)) for N, val, σ in self._value]
#         )

#     def __subs__(self, other: "EnsembleValue") -> "EnsembleValue":
#         return self + (other * -1)


# @EnsembleValue.__add__.register
# def __add__(self, other: EnsembleValue) -> EnsembleValue:
#     if len(self) != len(other):
#         raise RuntimeError("Can only add values of equal length.")

#     left = self._value
#     right = other._value

#     out = []

#     for left_i, right_i in zip(left, right):
#         if left_i[0] != right_i[0]:
#             raise RuntimeError("Can only add equal sample counts.")

#         out.append(
#             (
#                 left_i[0],
#                 left_i[1] + right_i[1],
#                 np.sqrt(left_i[2] ** 2 + right_i[2] ** 2).real,
#             )
#         )

#     return EnsembleValue(out)


# @EnsembleValue.__mul__.register
# def _(self, other: EnsembleValue) -> EnsembleValue:
#     if len(self) != len(other):
#         raise RuntimeError("Can only multiply values of equal length.")

#     left = self._value
#     right = other._value

#     out = []

#     for left_i, right_i in zip(left, right):
#         if left_i[0] != right_i[0]:
#             raise RuntimeError("Can only multiply equal sample counts.")

#         out.append(
#             (
#                 left_i[0],
#                 left_i[1] * right_i[1],
#                 np.sqrt(
#                     (right_i[1] * left_i[2]) ** 2 + (left_i[1] * right_i[2]) ** 2
#                 ).real,
#             )
#         )

#     return EnsembleValue(out)


def ensemble_return_scale(left: float, right: EnsembleReturn) -> EnsembleReturn:
    """Scales ``right`` by ``left``."""

    single_return = False

    if not isinstance(right, list):
        single_return = True
        right = [right]

    result = [(N, left * val, np.abs(left * σ)) for N, val, σ in right]

    return result[0] if single_return else result


def ensemble_return_add(left: EnsembleReturn, right: EnsembleReturn) -> EnsembleReturn:
    """
    Adds the values of ``left`` and ``right``.  The standard
    deviations are calculated correctly by adding the variances.
    """

    single_return = False

    if not isinstance(left, list):
        assert not isinstance(
            right, list
        ), "Both ensemble returns have to be of the same shape"

        single_return = True
        left = [left]
        right = [right]

    assert isinstance(right, list)

    out = []

    for left_i, right_i in zip(left, right):
        if left_i[0] != right_i[0]:
            raise RuntimeError("Can only add equal sample counts.")

        out.append(
            (
                left_i[0],
                left_i[1] + right_i[1],
                np.sqrt(left_i[2] ** 2 + right_i[2] ** 2).real,
            )
        )

    return out[0] if single_return else out


class BCF:
    r"""A parameter object to hold information about a BCF.

    The BCFs will be expanded into a sum of exponentials like
    :math:`\alpha(\tau) = \sum_k G_k \cdot \exp(-W_k\cdot\tau)`.  You
    can either give the BCFs as parameter or the coefficients.  If
    you give the BCFs, the fit will be performed automatically.

    Calling this object will call the wrapped BCF function.

    :param resolution: the precision in the sampling for the fit,
        ``t_max/precision`` points will be used
    :param num_terms: the number of terms of the expansion of the BCF
        expansion
    """

    def __init__(
        self,
        t_max: float,
        function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        num_terms: Optional[int] = None,
        resolution: Optional[float] = None,
        factors: Optional[np.ndarray] = None,
        exponents: Optional[np.ndarray] = None,
    ):

        #: the maximum simulation time
        self.t_max = t_max

        if function is not None:
            #: the BCF as python function, will be set to the exponential
            #: expansion if the BCF coefficients are given.
            self.function = function

            if num_terms is None or resolution is None:
                raise ValueError(
                    "Either give the function, the number of terms and the resolution or the coefficients."
                )

            _exponents, _factors = fit_α(
                self.function,
                num_terms,
                self.t_max,
                int(self.t_max / resolution),
            )
            #: the factors in the BCF expansion
            self.factors = _factors

            #: the exponents in the BCF expansion
            self.exponents = _exponents

        else:
            if factors is None or exponents is None:
                raise ValueError(
                    "Either give the function and number of terms or the coefficients."
                )

            assert factors is not None
            assert exponents is not None
            self.factors = factors
            self.exponents = exponents

            if self.factors.size != self.exponents.size:
                raise ValueError(
                    "Factors and exponents have to have the same dimension."
                )

            self.function = self.approx

    def approx(self, t: np.ndarray) -> np.ndarray:
        """The BCF as exponential expansion."""
        return α_apprx(t, self.factors, self.exponents)

    def __call__(self, t: np.ndarray):
        return self.function(t)


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
    real: bool = False,
) -> np.ndarray:
    """
    Applies the operator ``op`` to each element of the time series
    ``ψ`` of the dimensions ``(*, dim)`` where ``dim`` is the hilbert
    space dimension and sandwiches ``ψ`` onto it from the left.  If
    ``normalize`` is :any:`True` then the value will be divided by the
    squared norm.  If ``real`` is :any:`True`, the real part is returned.
    """

    exp_val = np.sum(ψ.conj() * apply_operator(ψ, op), axis=1)

    if normalize:
        exp_val /= np.sum(ψ.conj() * ψ, axis=1).real

    if real:
        exp_val = np.real(exp_val)

    return exp_val


def operator_expectation(
    ρ: np.ndarray, op: np.ndarray, real: bool = False
) -> np.ndarray:
    """Calculates the expecation value of ``op`` as a time series.

    :param ρ: The state as time series. ``(time, dim-sys, dim-sys)``
    :param op: The operator.
    :param real: Whether to take the real part.
    :returns: the expectation value
    """

    expect = np.einsum("ijk,kj", ρ, op)
    if real:
        expect = np.real(expect)

    return expect


def operator_expectation_ensemble(
    ψs: Iterator[np.ndarray],
    op: np.ndarray,
    N: Optional[int],
    normalize: bool = False,
    real: bool = False,
    **kwargs,
) -> EnsembleReturn:
    """Calculates the expecation value of ``op`` as a time series.

    :param ψs: A collection of stochastic trajectories.  Each
        element should have the shape  ``(time, dim-sys)``.
    :param op: The operator.
    :param N: Number of samples to take.
    :param real: Whether to take the real part.

    All the other kwargs are passed on to :any:`ensemble_mean`.

    :returns: the expectation value
    """

    return ensemble_mean(
        ψs, sandwhich_operator, N, const_args=(op, normalize, real), **kwargs
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
        G[np.newaxis, :] * np.exp(-W[np.newaxis, :] * (τ[:, np.newaxis])),
        axis=1,
    )


def integrate_array(
    arr: np.ndarray, t: np.ndarray, err: Optional[np.ndarray]
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the antiderivative of the function sampled in ``arr``
    along ``t``.  Optionally the error ``err`` is being integrated alongside.
    """

    integral = scipy.integrate.cumulative_trapezoid(arr, t, initial=0)

    if err is not None:
        err_integral = np.sqrt(
            scipy.integrate.cumulative_trapezoid(err**2, t, initial=0)
        ).real

        return integral, err_integral

    return integral


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


class JSONEncoder(json.JSONEncoder):
    """
    A custom encoder to serialize objects occuring in
    :any:`ensemble_mean`.
    """

    @singledispatchmethod
    def default(self, obj: Any):
        if hasattr(obj, "__bfkey__"):
            return f"<{type(obj)} ({obj.__bfkey__()})>"

        return super().default(obj)

    @default.register
    def _(self, arr: np.ndarray):
        return {"type": "array", "value": arr.tolist()}

    @default.register
    def _(self, obj: complex):
        return {"type": "complex", "re": obj.real, "im": obj.imag}


def _object_hook(dct: dict[str, Any]):
    """A custom decoder for the types introduced in :any:`JSONEncoder`."""
    if "type" in dct:
        type = dct["type"]

        if type == "array":
            return np.array(dct["value"])

        if type == "complex":
            return dct["re"] + 1j * dct["im"]

    return dct


def ensemble_mean(
    arg_iter: Iterator[Any],
    function: Callable[..., np.ndarray],
    N: Optional[int] = None,
    const_args: tuple = tuple(),
    const_kwargs: dict = dict(),
    n_proc: Optional[int] = None,
    every: Optional[int] = None,
    save: Optional[str] = None,
    overwrite_cache: bool = False,
) -> EnsembleReturn:

    results = []
    aggregate = WelfordAggregator(function(next(arg_iter), *const_args))

    path = None
    json_meta_info = json.dumps(
        dict(
            N=N,
            every=every,
            const_args=const_args,
            const_kwargs=const_kwargs,
            function_name=function.__name__,
            first_iterator_value=aggregate.mean,
        ),
        cls=JSONEncoder,
        ensure_ascii=False,
        default=lambda obj: obj.__dict__
        if hasattr(obj, "__dict__")
        else "<not serializable>",
    ).encode("utf-8")

    if save:
        key = hashlib.sha256(json_meta_info).hexdigest()
        path = Path("results") / Path(
            f"{save}_{function.__name__}_{N}_{every}_{key}.npy"
        )

        if not overwrite_cache and path.exists():
            logging.warning(f"Loading cache from: {path}")
            return np.load(str(path), allow_pickle=True)

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
        )

        for res in tqdm(result_iter, total=(N - 1) if N else None):
            aggregate.update(res)

            if every is not None and (aggregate.n % every) == 0 or aggregate.n == N:
                results.append(
                    (aggregate.n, aggregate.mean.copy(), aggregate.ensemble_std.copy())
                )

    if not every:
        results = results[-1]

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing cache to: {path}")
        with path.open("wb") as f:
            np.save(f, np.array(results, dtype="object"))

        with path.with_suffix(".json").open("wb") as f:
            f.write(json_meta_info)

    return results


class BCFDist(scipy.stats.rv_continuous):
    """A distribution based on the absolute value of the BCF."""

    def __init__(self, α: Callable[[np.ndarray], np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self._α = α
        self._norm = scipy.integrate.quad(
            lambda t: np.abs(self._α(np.array([t]))), 0, np.inf
        )

    def _pdf(self, x: np.ndarray):
        return np.abs(self._α(x)) / self._norm


def fit_α(
    α: Callable[[np.ndarray], np.ndarray],
    n: int,
    t_max: float,
    support_points: int = 1000,
    with_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the BCF ``α`` to a sum of ``n`` exponentials up to ``t_max``
    using a number of ``support_points``.

    The fit result will be cached if ``with_cache`` is :any:`True`.
    """

    max_sol = scipy.optimize.minimize(
        lambda t: -np.abs(α(np.array([t])))[0], [0], bounds=((0, np.inf),)
    )
    max = -max_sol.fun[0]
    t_at_max = max_sol.x[0]

    t_tail = scipy.optimize.newton(
        lambda t: np.abs(α(np.array([t]))[0]) - max / 100,
        t_at_max + 0.1,
    )

    assert isinstance(t_tail, float), "Could not find tail time."

    # norm = scipy.integrate.quad(lambda t: np.abs(α(np.array([t]))), 0, t_tail)[0]

    # hit_prop = norm / (max * t_tail)

    # rng = np.random.default_rng(1)
    # ts = rng.random(size=int(support_points * 2 / 3 / hit_prop) + 1) * t_tail
    # ys = rng.random(size=len(ts)) * max
    # mask = ys < np.abs(α(ts))

    # ts = ts[mask]

    ts = np.linspace(0, t_tail, int(support_points * 2 / 3) + 1)
    ts = np.append(ts, np.linspace(t_tail, t_max, int(support_points * 1 / 3)))
    ys = α(ts)

    data_key = hash((np.array([ts, ys]).data.tobytes(), n))
    cache_path = Path(".cache") / "bcf_fit" / f"{data_key}.npy"

    logging.info(f"Looking up bcf fit at {cache_path}.")

    if with_cache and cache_path.exists():
        logging.info(f"Loading bcf fit from {cache_path}.")
        w, g = np.load(cache_path)
        return w, g

    # ts = np.linspace(0, t_max, support_points)

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
        fit_params.add(f"w{i}", value=0.1, min=0)
        fit_params.add(f"wi{i}", value=0.1)
        fit_params.add(f"g{i}", value=0.1)
        fit_params.add(f"gi{i}", value=0)

        # if i == n - 1:
        #     expr_im = "0"
        #     # expr_re = "0"
        #     for j in range(n - 1):
        #         expr_im += f"+gi{j}"
        #         # expr_re += f"+g{j}"

        #     fit_params.add(f"gi{i}", expr=f"-({expr_im})")
        # else:
        #     fit_params.add(f"gi{i}", value=0)

    out = minimize(residual, fit_params, args=(ts, ys), method="least_squares")

    w = np.array([out.params[f"w{i}"] for i in range(n)]) + 1j * np.array(
        [out.params[f"wi{i}"] for i in range(n)]
    )
    g = np.array([out.params[f"g{i}"] for i in range(n)]) + 1j * np.array(
        [out.params[f"gi{i}"] for i in range(n)]
    )

    if with_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, (w, g))

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


def expand_t(f):
    def wrapped(self, t):
        t = np.expand_dims(np.asarray(t), axis=0)
        return f(self, t)

    return wrapped
