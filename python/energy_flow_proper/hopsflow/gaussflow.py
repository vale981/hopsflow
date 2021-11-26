"""Calculate the energy flow into the bath for a simple gaussian
  Quantum Brownian Motion model.

  This is done analytically for a BCF that is a sum of exponentials.
"""

import numpy as np
import itertools
from dataclasses import dataclass, field, InitVar
from typing import Callable, Union, Optional
import numpy.typing as npt
import lmfit
from . import util
import functools
from numpy.polynomial import Polynomial


@dataclass
class BCF:
    r"""A parameter object to hold information about a BCF.

    The BCFs will be expanded into a sum of exponentials like
    :math:`\alpha(\tau) = \sum_k G_k \cdot \exp(-W_k\cdot\tau)`.
    You can either give the BCFs as parameter or the coefficients.
    If you give the BCFs, the fit will be performed automatically.

    Calling this object will call the wrapped BCF function.

    :param resolution: the precision in the sampling for the fit, ``t_max/precision``
        points will be used
    :param num_terms: the number of terms of the expansion of the BCF expansion
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

            _exponents, _factors = util.fit_α(
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
            self.exponents = factors

            if self.factors.size != self.exponents.size:
                raise ValueError(
                    "Factors and exponents have to have the same dimension."
                )

            self.function = self.approx

    def approx(self, t: np.ndarray) -> np.ndarray:
        """The BCF as exponential expansion."""
        return util.α_apprx(t, self.factors, self.exponents)

    def __call__(self, t: np.ndarray):
        return self.function(t)


@dataclass
class SystemParams:
    r"""A parameter object to hold information about the physical
    system and the global parameters for the algorithm.

    :param Ω: the system hamiltonian energy scale, :math:`H_s =
              \Omega (p^2 + q^2)`
    :param η: the coupling strength (in essence a prefactor to the BCF)
    :param α: the BCF


    :attr t_max: the maximum simulation time, will be copied from :attr:`α`
    :attr W: the exponential factors in the BCF expansion
    :attr G: the pre-factors in the BCF expansion
    """

    Ω: float
    η: float
    α: BCF

    t_max: float = field(init=False)

    # BCF coefficients
    W: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)

    def __post_init__(self):
        self.t_max = self.α.t_max

        self.W = self.α.exponents
        self.G = self.α.factors * (self.η ** 2)


def construct_polynomials(sys: SystemParams) -> tuple[Polynomial, Polynomial]:
    r"""
    Construct the polynomials required to find the coefficients
    and exponents of the solution.

    :param sys: a parameter object with the system parameters
    :returns: a list of polynomials

        - :math:`f_0(z) = \prod_k (z-z_k) (z-z_k^{\ast})`
        - :math:`p(z) = p_1(z) + \sum_n q_n(z)` where
            - :math:`p_1(z) = (z^2 + \Omega^2)\prod_k (z-z_k)(z-z_k^{\ast})`
            - :math:`q_n(z)=\Omega f_n(z) \prod_{k\neq n}(z-z_k) (z-z_k^{\ast})`

    """
    # we begin by calculating all the constants we'll need
    φ = -np.angle(sys.G)
    G_abs = np.abs(sys.G)
    γ, δ = sys.W.real, sys.W.imag
    s, c = np.sin(φ), np.cos(φ)

    # the roots of -G ((z + γ)s + δc)
    roots_f = -(γ + δ * c / s)

    # the roots of the denominator of the laplace transform of α
    roots_z = -sys.W

    # now the polynomials
    f_0 = util.poly_real(
        Polynomial.fromroots(np.concatenate((roots_z, roots_z.conj())))
    )
    p_1 = Polynomial([sys.Ω ** 2, 0, 1]) * f_0

    q = [
        -G_c
        * sys.Ω
        * s_c
        * util.poly_real(
            Polynomial.fromroots(
                np.concatenate(
                    (
                        [root_f],
                        util.except_element(roots_z, i),
                        util.except_element(roots_z, i).conj(),
                    )
                )
            )
        )
        for root_f, G_c, γ_c, δ_c, s_c, c_c, i in zip(
            roots_f, G_abs, γ, δ, s, c, range(len(c))
        )
    ]

    p = p_1 + sum(q)

    return f_0, p


def calculate_coefficients(sys: SystemParams) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Calculates the coefficients required to construct the
    propagator matrix :math:`G`.

    :param sys: a parameter object with the system parameters
    :returns: the exponents and the residuals which play a role as pre-factors
    """

    f_0, p = construct_polynomials(sys)

    master_roots = p.roots()

    if np.unique(master_roots).size != master_roots.size:
        raise RuntimeError(
            """The roots of the polynomial are not unique.
You can try to alter the number of terms in the expansion of the BCF."""
        )

    resiquals = f_0(master_roots) / p.deriv()(master_roots)
    return master_roots, resiquals


def _expand_t(f):
    def wrapped(self, t):
        t = np.expand_dims(np.asarray(t), axis=0)
        return f(self, t)

    return wrapped


class Propagator:
    """The propagator matrix :math:`G` for the system.

    You can get inidividual matrix elements as functions of time
    through indexing this object.

    Calling it with a time argument returns the whole matrix.

    :param sys: a parameter object with the system parameters
    """

    def __init__(self, sys: SystemParams):
        self.params = sys
        self._roots, self._res = calculate_coefficients(sys)
        self._roots_exp = np.expand_dims(self._roots, axis=1)
        self._res_exp = np.expand_dims(self._res, axis=1)
        self._Ω = sys.Ω

        self._res_times_roots = self._res_exp * self._roots_exp
        self._res_times_roots_squared = self._res_times_roots * self._roots_exp
        self._elements = np.array([[self.el_11, self.el_12], [self.el_21, self.el_22]])
        pass

    @_expand_t
    def el_11(self, t):
        return (self._res_times_roots * np.exp(t * self._roots_exp)).real.sum(axis=0)

    @_expand_t
    def el_12(self, t):
        return self._Ω * (self._res_exp * np.exp(t * self._roots_exp)).real.sum(axis=0)

    @_expand_t
    def el_21(self, t):
        return (self._res_times_roots_squared * np.exp(t * self._roots_exp)).real.sum(
            axis=0
        ) / self._Ω

    def el_22(self, t):
        return self.el_11(t)

    def __getitem__(self, indices) -> Callable[[npt.ArrayLike], np.ndarray]:
        """Get an individual matrix elment as function."""
        return self._elements[indices]

    @_expand_t
    def __call__(self, t) -> np.ndarray:
        """Get the propagator as array of shape ``(time, 2, 2)``."""

        g_12 = self._res_exp * np.exp(t * self._roots_exp)
        diag = self._roots_exp * g_12
        g_21 = self._roots_exp * diag / self._Ω

        return (
            np.array([[diag, g_12 * self._Ω], [g_21, diag]])
            .real.sum(axis=2)
            .swapaxes(0, 2)
            .swapaxes(1, 2)
        )


@dataclass
class FlowParams:
    r"""
    A parameter object to hold the global parameters for the flow algorithm.

    :param system: the system parameters, see :any:`SystemParams`
    :param α_0_dot: the zero temperature BCF time derivative
    :param n: the excitation number of the initial state of the system :math:`|n\rangle`

    :attr L: the exponential factors in the BCF derivative
             expansion :math:`\dot{\alpha}_0=\sum_k P_k e^{-L_k \cdot t}`

    :attr P: the pre-factors in the BCF derivative expansion
    :attr q_s_0: the expectation value :math:`\langle q(0)^2\rangle`
    :attr p_s_0: the expectation value :math:`\langle q(0)^2\rangle`
    :attr qp_0: the expectation value :math:`\langle q(0)p(0)\rangle`
    :attr prop: the propagator matrix :math:`G(t)`, see :any:`Propagator`
    :attr B_coef: the pre-factors of the exponential sum of :math:`B=G_{12}=\sum_k B_k e^{-C_k \cdot t}`

                    - note the minus sign
    :attr C: the exponents of the exponential sum of ``B``
    :attr A_coef: the pre-factors of the exponential sum of :math:`A=G_{11}=\sum_k A_k e^{-C_k \cdot t}`
    """

    system: InitVar[SystemParams]
    α_0_dot: BCF
    n: int

    # BCF derivative coefficients
    L: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)

    # initial state
    q_s_0: float = field(init=False)
    p_s_0: float = field(init=False)
    qp_0: complex = field(init=False)

    # propagator
    prop: Propagator = field(init=False)
    B_coef: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    A_coef: np.ndarray = field(init=False)

    def __post_init__(self, system):
        self.L = self.α_0_dot.exponents
        self.P = self.α_0_dot.factors * (system.η ** 2)

        self.q_s_0 = 1 + 2 * self.n
        self.p_s_0 = 1 + 2 * self.n
        self.qp_0 = 1j

        self.C, self.B_coef = calculate_coefficients(system)
        self.A_coef = self.B_coef * self.C

        self.B_coef = system.Ω * self.B_coef
        self.C = -self.C  # mind the extra -

        self.prop = Propagator(system)


def get_flow(
    sys: SystemParams, flow: FlowParams
) -> Callable[[npt.ArrayLike], np.ndarray]:
    r"""Get the flow :math:`\frac{d}{dt}\langle H_B\rangle` as a function of time."""

    # TODO: Think about the too bloated ``FlowParams`` class and this ugly function
    A = flow.prop[0, 0]
    B = flow.prop[0, 1]
    Bk = flow.B_coef  #
    Ak = flow.A_coef  #
    Pk = flow.P  #
    Lk = flow.L  #
    Ck = flow.C  #
    Wk = sys.W  #
    Gk = sys.G  #
    Γ1 = (Bk[:, None] * Pk[None, :]) / (Lk[None, :] - Ck[:, None])
    Γ2 = (Bk[:, None] * Gk[None, :]) / (Ck[:, None] - Wk[None, :])
    Γ3 = (Bk[:, None] * Gk.conj()[None, :]) / (Ck[:, None] + Wk.conj()[None, :])
    ΓA = (Ak[:, None] * Pk[None, :]) / (Lk[None, :] - Ck[:, None])

    def A_conv(t):
        t = np.asarray(t)
        result = np.zeros_like(t, dtype="complex128")

        # print(np.sort(Ck.imag.flatten()))

        for (n, m) in itertools.product(range(len(Ak)), range(len(Pk))):
            result += ΓA[n, m] * (np.exp(-Ck[n] * t) - np.exp(-Lk[m] * t))

        return result

    def B_conv(t):
        t = np.asarray(t)
        result = np.zeros_like(t, dtype="complex128")
        for (n, m) in itertools.product(range(len(Bk)), range(len(Pk))):
            result += Γ1[n, m] * (np.exp(-Ck[n] * t) - np.exp(-Lk[m] * t))

        return result

    def flow_conv_free(t):
        a, b = A(t), B(t)
        ac, bc = A_conv(t), B_conv(t)
        # print(ac, bc)
        return (
            -1
            / 2
            * (
                flow.q_s_0 * a * ac
                + flow.p_s_0 * b * bc
                + flow.qp_0 * a * bc
                + flow.qp_0.conjugate() * b * ac
            ).imag
        )

    def conv_part(t):
        t = np.asarray(t)
        result = np.zeros_like(t, dtype="float")
        for (m, k, n, l) in itertools.product(
            range(len(Bk)), range(len(Pk)), range(len(Bk)), range(len(Gk))
        ):
            g_1_2 = (
                Γ1[m, k]
                * Γ2[n, l]
                * (
                    (1 - np.exp(-(Ck[m] + Wk[l]) * t)) / (Ck[m] + Wk[l])
                    + (np.exp(-(Ck[m] + Ck[n]) * t) - 1) / (Ck[m] + Ck[n])
                    + (np.exp(-(Lk[k] + Wk[l]) * t) - 1) / (Lk[k] + Wk[l])
                    + (1 - np.exp(-(Lk[k] + Ck[n]) * t)) / (Lk[k] + Ck[n])
                )
            )

            g_1_3 = (
                Γ1[m, k]
                * Γ3[n, l]
                * (
                    (1 - np.exp(-(Ck[m] + Ck[n]) * t)) / (Ck[m] + Ck[n])
                    - (1 - np.exp(-(Lk[k] + Ck[n]) * t)) / (Lk[k] + Ck[n])
                    - (
                        np.exp(-(Ck[n] + Wk[l].conj() * t))
                        - np.exp(-(Ck[m] + Ck[n]) * t)
                    )
                    / (Ck[m] - Wk[l].conj())
                    + (
                        np.exp(-(Ck[n] + Wk[l].conj() * t))
                        - np.exp(-(Lk[k] + Ck[n]) * t)
                    )
                    / (Lk[k] - Wk[l].conj())
                )
            )

            result += -1 / 2 * (g_1_2.imag + g_1_3.imag)

        return result

    def final_flow(t: npt.ArrayLike) -> np.ndarray:
        return flow_conv_free(t) + conv_part(t)

    return final_flow
