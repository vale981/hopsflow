"""Utilities for the energy flow calculation."""

from __future__ import annotations
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
import shutil
import hashlib
import logging
import json
from functools import singledispatch, singledispatchmethod
from scipy.stats import NumericalInverseHermite
import scipy.interpolate
import copy
import ray
import numbers
from hops.util.dynamic_matrix import DynamicMatrix, ConstantMatrix
import opt_einsum as oe
import gc
import math
import time
import pickle
from hops.core.hierarchy_data import HIData

Aggregate = tuple[int, np.ndarray, np.ndarray]
EnsembleReturn = Union[Aggregate, list[Aggregate]]


class EnsembleValue:
    def __init__(
        self, value: Union[Aggregate, list[Aggregate], tuple[np.ndarray, np.ndarray]]
    ):
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], np.ndarray)
            and isinstance(value[1], np.ndarray)
        ):
            self._value: list[Aggregate] = [[0, *value]]  # type:ignore
        else:
            self._value: list[Aggregate] = (  # type:ignore
                value
                if (isinstance(value, list) or isinstance(value, np.ndarray))
                else [value]
            )

    @property
    def final_aggregate(self):
        return self._value[-1]

    @property
    def N(self):
        return self.final_aggregate[0]

    @property
    def value(self):
        return self.final_aggregate[1]

    @property
    def σ(self):
        return self.final_aggregate[2]

    @property
    def Ns(self):
        return [N for N, _, _ in self._value]

    @property
    def values(self):
        return [val for _, val, _ in self._value]

    @property
    def σs(self):
        return [σ for _, _, σ in self._value]

    @property
    def aggregate_iterator(self):
        for agg in self._value:
            yield agg

    @property
    def ensemble_value_iterator(self):
        for agg in self._value:
            yield EnsembleValue(agg)

    @property
    def mean(self):
        N, val, σ = self.final_aggregate

        return EnsembleValue(
            [(N, val.mean().copy(), np.sqrt((σ.copy() ** 2).sum() / val.size**2))]
        )

    @property
    def max(self):
        N, val, σ = self.final_aggregate
        max_index = np.argmax(val)

        return EnsembleValue([(N, val[max_index].copy(), σ[max_index].copy())])

    @property
    def min(self):
        N, val, σ = self.final_aggregate
        min_index = np.argmin(val)

        return EnsembleValue([(N, val[min_index].copy(), σ[min_index].copy())])

    def __getitem__(self, index):
        return EnsembleValue(self._value[index])

    def slice(self, slc: Union[np.ndarray, slice]):
        results = []
        for N, val, σ in self.aggregate_iterator:
            results.append((N, val[slc], σ[slc]))

        return EnsembleValue(results)

    def __len__(self) -> int:
        return len(self._value)

    def for_bath(self, bath: int):
        if self.num_baths == 1 and len(self.value.shape) in [0, 1]:
            return self

        return EnsembleValue([(N, val[bath], σ[bath]) for N, val, σ in self._value])

    @property
    def num_baths(self) -> int:
        shape = self.value.shape
        return self.value.shape[0] if len(shape) > 1 else 1

    def sum_baths(self) -> EnsembleValue:
        final = self.for_bath(0)
        for i in range(1, self.num_baths):
            final = final + self.for_bath(i)

        return final

    def insert(self, value: Aggregate):
        where = len(self._value)
        for i, (N, _, _) in enumerate(self._value):
            if N > value[0]:
                where = i
                break

        self._value.insert(where, value)

    def insert_multi(self, values: list[Aggregate]):
        for value in values:
            self.insert(value)

    def consistency(self, other: Union[EnsembleValue, np.ndarray]) -> float:
        diff = abs(
            self[-1] - (other[-1] if isinstance(other, self.__class__) else other)
        )

        diff_val = diff.for_bath(0).value
        return (
            (diff.value < diff.σ).sum()
            / (len(diff_val) if hasattr(diff_val, "__len__") else 1)
            * 100
        )

    def integrate(self, τ: np.ndarray) -> EnsembleValue:
        """Calculate the antiderivative along a 'time axis' ``τ``."""

        results = []
        for N, val, σ in self.aggregate_iterator:
            results.append((N, *integrate_array(val, τ, σ)))

        return EnsembleValue(results)

    def __abs__(self) -> "EnsembleValue":
        out = []

        for N, value, σ in self._value:
            out.append((N, abs(value), σ))

        return EnsembleValue(out)

    def __add__(
        self, other: Union["EnsembleValue", float, int, np.ndarray]
    ) -> EnsembleValue:
        if isinstance(other, EnsembleValue):
            if len(self) != len(other):
                raise RuntimeError("Can only add values of equal length.")

            if self.num_baths != other.num_baths:
                raise RuntimeError(
                    "Can only add values pertaining to an equal number of baths."
                )

            left = self._value
            right = other._value

            out = []

            for left_i, right_i in zip(left, right):
                if left_i[0] < right_i[0]:
                    samples = left_i[0]
                    σ = np.sqrt(
                        left_i[2] ** 2
                        + right_i[2] ** 2 * (right_i[0] - 1) / (left_i[0] - 1)
                    ).real
                else:
                    samples = right_i[0]
                    σ = np.sqrt(
                        left_i[2] ** 2 * (left_i[0] - 1) / (right_i[0] - 1)
                        + right_i[2] ** 2
                    ).real

                out.append(
                    (
                        samples,
                        left_i[1] + right_i[1],
                        σ,
                    )
                )

            return EnsembleValue(out)

        if isinstance(other, tuple):
            new = copy.deepcopy(self)
            new.insert(other)
            return new

        if isinstance(other, list) and isinstance(other[0], tuple):
            new = copy.deepcopy(self)
            new.insert_multi(other)
            return new

        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            out = []

            for N, value, σ in self.aggregate_iterator:
                out.append((N, value + other, σ))

            return EnsembleValue(out)

        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other: Union["EnsembleValue", float, int, np.ndarray]):
        if (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, np.ndarray)
        ):
            return EnsembleValue(
                [(N, val * other, np.abs(σ * other)) for N, val, σ in self._value]
            )

        if type(self) == type(other):
            if len(self) != len(other):
                raise RuntimeError("Can only multiply values of equal length.")

            left = self._value
            right = other._value

            out = []

            for left_i, right_i in zip(left, right):
                if left_i[0] != right_i[0]:
                    raise RuntimeError("Can only multiply equal sample counts.")

                out.append(
                    (
                        left_i[0],
                        left_i[1] * right_i[1],
                        np.sqrt(
                            (right_i[1] * left_i[2]) ** 2
                            + (left_i[1] * right_i[2]) ** 2
                        ).real,
                    )
                )

            return EnsembleValue(out)

        return NotImplemented

    __rmul__ = __mul__

    def __sub__(
        self, other: Union["EnsembleValue", float, int, np.ndarray]
    ) -> "EnsembleValue":
        if (
            type(self) == type(other)
            or isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, np.ndarray)
        ):
            return self + (-1 * other)

        return NotImplemented

    def __rsub__(self, other: Union["EnsembleValue", float, int]) -> "EnsembleValue":
        if (
            type(self) == type(other)
            or isinstance(other, float)
            or isinstance(other, int)
        ):
            return (self * -1) + other

        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"


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


def make_operator_expectation_task(
    op: Union[np.ndarray, DynamicMatrix],
    t: np.ndarray,
    normalize: bool = False,
    real: bool = False,
):
    """
    Returns a function that takes the first hierarchy states and
    returns the expectation value of the operator op.

    For the arguments see :any:`operator_expectation_ensemble`.
    """
    if isinstance(op, ConstantMatrix):
        op = op(0)

    if isinstance(op, DynamicMatrix):
        calc_sandwhich = oe.contract_expression(
            "ti,tij,tj->t",
            (len(t), op.shape[0]),
            op(t),
            (len(t), op.shape[0]),
            constants=[1],
        )
    else:
        calc_sandwhich = oe.contract_expression(
            "ti,ij,tj->t",
            (len(t), op.shape[0]),
            op,
            (len(t), op.shape[0]),
            constants=[1],
        )

    def op_exp_task(ψ: np.ndarray):
        sandwhiches: np.ndarray = calc_sandwhich(ψ.conj(), ψ)  # type: ignore

        if normalize:
            sandwhiches /= np.sum(ψ.conj() * ψ, axis=1).real

        if real:
            sandwhiches = sandwhiches.real

        return sandwhiches

    return op_exp_task


def operator_expectation_ensemble(
    ψs: Iterator[np.ndarray],
    op: Union[np.ndarray, DynamicMatrix],
    t: np.ndarray,
    normalize: bool = False,
    real: bool = False,
    **kwargs,
) -> EnsembleValue:
    """Calculates the expecation value of ``op`` as a time series.

    :param ψs: A collection of stochastic trajectories.  Each
        element should have the shape  ``(time, dim-sys)``.
    :param op: The operator.
    :param N: Number of samples to take.
    :param real: Whether to take the real part.

    All the other kwargs are passed on to :any:`ensemble_mean`.

    :returns: the expectation value
    """

    op_exp_task = make_operator_expectation_task(op, t, normalize, real)
    return ensemble_mean(ψs, op_exp_task, **kwargs)


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
    along ``t`` using spline interpolation.  Optionally the error
    ``err`` is being integrated alongside.
    """

    multiple_baths = len(arr.shape) > 1

    if not multiple_baths:
        arr = arr[None, ...]
        if err is not None:
            err = err[None, ...]

    splines = [scipy.interpolate.UnivariateSpline(t, y, s=0, k=5) for y in arr]
    integral = np.array([spline.antiderivative()(t) for spline in splines])
    if err is not None:
        dt = t[1:] - t[:-1]
        err_sum = [
            np.concatenate(([0], np.cumsum(((e[1:] ** 2 + e[:-1] ** 2) / 4) * dt**2)))
            for e in err
        ]
        err_integral = np.sqrt(err_sum).real

        if multiple_baths:
            return integral, err_integral
        return integral[0], err_integral[0]

    if multiple_baths:
        return integral

    return integral[0]


###############################################################################
#                                Ensemble Mean                                #
###############################################################################


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
        if self.n == 1:
            return np.zeros_like(self.mean)

        return self._m_2 / (self.n - 1)

    @property
    def ensemble_variance(self) -> np.ndarray:
        return self.sample_variance / self.n

    @property
    def ensemble_std(self) -> np.ndarray:
        return np.sqrt(self.ensemble_variance)

    @property
    def ensemble_value(self) -> EnsembleValue:
        return EnsembleValue([(self.n, self.mean, self.ensemble_std)])


class JSONEncoder(json.JSONEncoder):
    """
    A custom encoder to serialize objects occuring in
    :any:`ensemble_mean`.
    """

    @singledispatchmethod
    def default(self, obj: Any):
        if hasattr(obj, "__bfkey__"):
            return f"<{type(obj)} ({obj.__bfkey__()})>"

        return obj.__dict__ if hasattr(obj, "__dict__") else "<not serializable>"

    @default.register
    def _(self, arr: np.ndarray):
        return {"type": "array", "value": arr.tolist()}

    @default.register
    def _(self, integer: np.int64):
        return int(integer)

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


def _grouper(n: int, iterable: Iterator[Any]):
    """Groups the iteartor into tuples of at most length ``n``."""

    while True:
        chunk = tuple(itertools.islice(iterable, n))
        if not chunk:
            return
        yield chunk


@ray.remote
def _ensemble_remote_function(function, chunk: tuple, index: int):
    res = np.array([np.array(function(arg)) for arg in chunk])
    return res, index


def get_online_data_path(save: str):
    return Path("results") / Path(f"online_{save}.npy")


def load_online_cache(save: str):
    path = get_online_data_path(save)

    with path.open("rb") as agg_file:
        aggregate = pickle.load(agg_file)

    return aggregate.ensemble_value


def ensemble_mean_online(
    args: Any,
    save: str,
    function: Callable[..., np.ndarray],
) -> Optional[EnsembleValue]:
    path = get_online_data_path(save)

    if args is None:
        result = None
    else:
        result = function(args)

        if np.isnan(np.sum(result)):
            result = None

    if path.exists():
        with path.open("rb") as agg_file:
            aggregate = pickle.load(agg_file)
            if result is not None:
                aggregate.update(result)

    else:
        if result is None:
            raise RuntimeError("No cache and no result.")

        aggregate = WelfordAggregator(result)

    with path.open("wb") as agg_file:
        pickle.dump(aggregate, agg_file)

    return aggregate.ensemble_value


def ensemble_mean(
    arg_iter: Iterator[Any],
    function: Callable[..., np.ndarray],
    N: Optional[int] = None,
    every: Optional[int] = None,
    save: Optional[str] = None,
    overwrite_cache: bool = False,
    chunk_size: Optional[int] = None,
    in_flight: Optional[int] = None,
    gc_sleep: float = 0,
) -> EnsembleValue:
    results = []

    path = None
    json_meta_info = json.dumps(
        dict(
            N=N,
            every=every,
            function_name=function.__name__,
            first_iterator_value="<not serializable>",
        ),
        cls=JSONEncoder,
        ensure_ascii=False,
    ).encode("utf-8")

    if save:
        key = hashlib.sha256(json_meta_info).hexdigest()
        path = Path("results") / Path(
            f"{save}_{function.__name__}_{N}_{every}_{key}.npy"
        )

        if not overwrite_cache and path.exists():
            logging.debug(f"Loading cache from: {path}")
            results = np.load(str(path), allow_pickle=True)
            return EnsembleValue([tuple(res) for res in results])

    first_result = function(next(arg_iter))
    aggregate = WelfordAggregator(first_result)

    if N == 1:
        return EnsembleValue([(1, aggregate.mean, np.zeros_like(aggregate.mean))])

    if chunk_size is None:
        chunk_size = max(100000 // (first_result.size * first_result.itemsize), 1)
        logging.debug(f"Setting chunk size to {chunk_size}.")

    num_chunks = math.ceil((N - 1) / chunk_size) if N is not None else None
    chunk_iterator = iter(
        tqdm(
            zip(
                _grouper(
                    chunk_size,
                    itertools.islice(arg_iter, None, N - 1 if N else None),
                ),
                itertools.count(0),
            ),
            total=num_chunks,
            desc="Loading",
        )
    )

    finished = []
    processing_refs = []
    chunks = {}

    in_flight = in_flight or int(ray.available_resources().get("CPU", 0)) * 2
    function_on_store = ray.put(function)

    highest_index = 0

    while True:
        try:
            next_val = next(chunk_iterator)
        except StopIteration:
            next_val = None

        if len(processing_refs) > in_flight or not next_val:
            finished, processing_refs = ray.wait(
                processing_refs,
                num_returns=len(processing_refs) - in_flight
                if next_val is not None
                else len(processing_refs),
                fetch_local=True,
            )

        has_downloaded = len(finished) > 0
        for result in finished:
            res_chunk, idx = ray.get(result)
            # print(
            #     res_chunk[0].size * len(res_chunk) * res_chunk[0].itemsize / 1024 / 1024
            # )
            chunks[idx] = res_chunk

        finished = []
        if has_downloaded:
            while highest_index in chunks:
                next_chunk = chunks[highest_index]
                del chunks[highest_index]

                len_chunk = len(next_chunk) - 1
                for i, res in enumerate(next_chunk):
                    aggregate.update(res)
                    if (
                        every is not None
                        and (aggregate.n % every) == 0
                        or aggregate.n == N
                        or (not next_val and not chunks and i == len_chunk)
                    ):
                        results.append(
                            (
                                aggregate.n,
                                aggregate.mean.copy(),
                                aggregate.ensemble_std.copy(),
                            )
                        )

                highest_index += 1

            if gc_sleep and gc_sleep > 0:
                gc.collect()
                time.sleep(gc_sleep)  # wait for the ray store to catch on

        if next_val:
            chunk_ref = ray.put(next_val[0])
            processing_refs.append(
                _ensemble_remote_function.remote(
                    function_on_store, chunk_ref, next_val[1]
                )
            )
        else:
            break

    del (
        chunk_iterator,
        function_on_store,
        finished,
        chunk_ref,
        chunks,
        processing_refs,
    )

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing cache to: {path}")
        with path.open("wb") as f:
            np.save(f, np.array(results, dtype="object"))

        with path.with_suffix(".json").open("wb") as f:
            f.write(json_meta_info)

    del aggregate

    return EnsembleValue(results)


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

    max = -max_sol.fun
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


def operator_expectation_from_data(
    data: HIData,
    op: Union[np.ndarray, DynamicMatrix],
    **kwargs,
) -> EnsembleValue:
    """Calculates the expectation value of a system observable ``op``.

    The ``kwargs`` is passed on to
    :any:`util.operator_expectation_ensemble`.

    :param data: The data instance that contains the trajectories.
        Does not have to be opened yet.

    :returns: the expectation value of the observable ``op`` for each
              time step
    """

    with data as d:
        if "save" in kwargs:
            kwargs["save"] += "_" + data.get_hi_key_hash()

        return operator_expectation_ensemble(
            ψs=d.valid_sample_iterator(d.stoc_traj),
            op=op,
            t=d.get_time(),
            normalize=d.get_hi_key().HiP.nonlinear,
            **(dict(N=d.samples) | kwargs),
        )
