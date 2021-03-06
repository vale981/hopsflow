"""Calculate the energy flow into the bath for a simple gaussian
Quantum Brownian Motion model with two coupled HOs coupled to a bath each.

This is done analytically for a BCF that is a sum of exponentials.
"""

from dataclasses import dataclass, field
from . import util
from .util import BCF
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import Polynomial
from typing import Union, Optional
from itertools import product
from collections.abc import Iterator, Callable
import scipy.optimize


@dataclass
class SystemParams:
    r"""A parameter object to hold information about the physical
    system and the global parameters for the algorithm.

    :math:`H_s = \frac{Ω}{4} (p^2 + (1+γ^2/Ω^2)q^2) + \frac{Λ}{4}
    (p^2 + (1+γ^2/Ω^2)r^2)`

    :param Ω: The system hamiltonian energy scales for the first HO.

    :param η: The coupling strengths (in essence a prefactor to the
              BCF).

    :param α_0: The zero temperature BCFs.
    :param γ: The coupling strength between the HOs.

    :attr t_max:
    """

    Ω: float
    Λ: float
    η: list[float]
    γ: float
    α_0: list[BCF]

    t_max: float = field(init=False)
    """The maximum simulation time, will be copied from :any:`α_0`.
    """

    W: list[np.ndarray] = field(init=False)
    """
    The exponential factors in the BCF expansion.
    """

    G: list[np.ndarray] = field(init=False)
    """
    The pre-factors in the BCF expansion.
    """

    root_tol: float = 1e-7
    """
    The relative tolerance of the roots.

    The roots are being calculated with the numpy polynomial module
    and then refined with newtons method.  This parameter is passed as
    ``rtol`` to :any:`scipy.optimize.newton`.
    """

    def __post_init__(self):
        self.t_max = self.α_0[0].t_max

        assert self.α_0[0].t_max == self.α_0[1].t_max

        self.W = [α.exponents.copy() for α in self.α_0]
        self.G = [α.factors.copy() * scale for α, scale in zip(self.α_0, self.η)]


def bcf_polynomials(
    G: NDArray[np.complex128], W: NDArray[np.complex128]
) -> tuple[Polynomial, Polynomial]:
    """Construct the polynomials related to the BCF laplace transform.
    The first return value is the denominator and the second one is
    the numerator.

    :param W: The exponential factors in the BCF expansion.
    :param G: The pre-factors in the BCF expansion.
    """
    # we begin by calculating all the constants we'll need
    φ = -np.angle(G)
    G_abs = np.abs(G)
    γ, δ = np.real(W), np.imag(W)
    s, c = np.sin(φ), np.cos(φ)
    roots_z = -W

    f_0 = util.poly_real(
        Polynomial.fromroots(np.concatenate((roots_z, roots_z.conj())))
    )

    bcf_numerator: Polynomial = sum(
        -G_c
        * Polynomial((δ_c * c_c + γ_c * s_c, s_c))
        * (
            util.poly_real(
                Polynomial.fromroots(
                    np.concatenate(
                        (
                            util.except_element(roots_z, i),
                            util.except_element(roots_z, i).conj(),
                        )
                    )
                )
            )
            if len(roots_z) > 1
            else 1  # corner case
        )
        for G_c, γ_c, δ_c, s_c, c_c, i in zip(G_abs, γ, δ, s, c, range(len(c)))
    )  # type: ignore

    return f_0, bcf_numerator


def construct_polynomials(
    sys: SystemParams,
) -> tuple[Polynomial, Polynomial, Polynomial, Polynomial, Polynomial]:
    r"""
    Construct the polynomials required to find the coefficients and
    exponents of the solution.

    :param sys: A parameter object with the system parameters.

    :returns: The numerators, denominators of the BCF laplace
              transforms and the denominator of the propagator
              matrix.
    """

    Ω = sys.Ω
    Λ = sys.Λ
    γ = sys.γ
    p_0 = Polynomial(
        [
            γ * Λ**2 * Ω + γ * Λ * Ω**2 + Λ**2 * Ω**2,
            0,
            γ * Λ + Λ**2 + γ * Ω + Ω**2,
            0,
            1,
        ]
    )

    p_a = Polynomial([γ * Λ * Ω + Λ**2 * Ω, 0, Ω])
    p_b = Polynomial([γ * Λ * Ω + Λ * Ω**2, 0, Λ])
    p_ab = Polynomial([Λ * Ω])

    f_0_a, gn_a = bcf_polynomials(sys.G[0], sys.W[0])
    f_0_b, gn_b = bcf_polynomials(sys.G[1], sys.W[1])

    p = (
        p_0 * f_0_a * f_0_b
        + p_a * f_0_b * gn_a
        + p_b * f_0_a * gn_b
        + gn_a * gn_b * p_ab
    )

    return f_0_a, f_0_b, gn_a, gn_b, p


def calculate_coefficients(sys: SystemParams) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Calculates the coefficients required to construct the propagator
    matrix :math:`G`.

    :param sys: a parameter object with the system parameters
    :returns: The exponents and the residuals which play a role as
              pre-factors in the shape ``(num sum terms, unique
              matrix element count)``.
    """

    f_0_a, f_0_b, gn_a, gn_b, p = construct_polynomials(sys)

    master_roots = p.roots()

    if np.unique(master_roots).size != master_roots.size:
        raise RuntimeError(
            """The roots of the polynomial are not unique.
You can try to alter the number of terms in the expansion of the BCF."""
        )

    # improving the accuracy of the roots seems to be essential
    p_prime = p.deriv()
    p_prime_2 = p_prime.deriv()
    for i, root in enumerate(master_roots):
        res = scipy.optimize.newton(
            p,
            root,
            p_prime,
            rtol=sys.root_tol,
            fprime2=p_prime_2,
            maxiter=100000,  # we can allow that, there aren't that many roots
        )
        master_roots[i] = res

    Λ = sys.Λ * np.ones_like(master_roots)
    Ω = sys.Ω * np.ones_like(master_roots)
    γ = sys.γ * np.ones_like(master_roots)

    a = gn_a(master_roots) / f_0_a(master_roots)
    b = gn_b(master_roots) / f_0_b(master_roots)

    matrix_elements = np.array(
        [
            master_roots**3 + master_roots * Λ * (b + γ + Λ),
            (master_roots**2 + Λ * (b + γ + Λ)) * Ω,
            master_roots * γ * Ω,
            γ * Λ * Ω,
            -(a * (master_roots**2 + Λ * (b + γ + Λ)))
            - master_roots**2 * (γ + Ω)
            - Λ * (Λ * Ω + b * (γ + Ω) + γ * (Λ + Ω)),
            master_roots**2 * γ,
            master_roots * γ * Λ,
            master_roots**3 + master_roots * Ω * (a + γ + Ω),
            Λ * (master_roots**2 + Ω * (a + γ + Ω)),
            -(master_roots**2 * (γ + Λ))
            - b * (master_roots**2 + Ω * (a + γ + Ω))
            - Ω * (a * (γ + Λ) + Λ * Ω + γ * (Λ + Ω)),
        ]
    )

    residuals = (f_0_a(master_roots) * f_0_b(master_roots) / p.deriv()(master_roots))[
        None, :
    ] * matrix_elements

    return master_roots, residuals


class Propagator:
    """The propagator matrix :math:`G` for the system.

    You can get inidividual matrix elements as functions of time
    through indexing this object.

    Calling it with a time argument returns the whole matrix as
    array of shape ``(time, 4, 4)``.

    :param sys: a parameter object with the system parameters
    """

    def __init__(self, sys: SystemParams):
        self.params = sys
        self._roots, self._res = calculate_coefficients(sys)

        mat = self._res
        self._residual_matrix = np.array(
            [
                [mat[0], mat[1], mat[2], mat[3]],
                [mat[4], mat[0], mat[5], mat[6]],
                [mat[6], mat[3], mat[7], mat[8]],
                [mat[5], mat[2], mat[9], mat[7]],
            ]
        )

    def propagator(self, t: Union[NDArray[np.float64], float]) -> np.ndarray:
        """
        Get the propagator as array of shape ``(time, 4, 4)`` or
        ``(4, 4)`` if ``t`` is a float.
        """
        t = np.asarray(t)
        was_float = False
        if t.shape == tuple():
            t = np.array([t])
            was_float = True

        res = np.sum(
            self._residual_matrix[None, :, :, :]
            * np.exp(self._roots[None, :] * t[:, None])[:, None, None, :],
            axis=3,
        ).real

        return res[0] if was_float else res

    def __call__(self, *args, **kwargs):
        return self.propagator(*args, **kwargs)

    def inv(self, t: float) -> np.ndarray:
        """Get the inverse of the propagator matrix at time ``t``."""

        return np.linalg.inv(self.__call__(t))


def iterate_ragged(*ranges: int) -> Iterator[tuple[int, ...]]:
    return product(*(range(r) for r in ranges))


class CorrelationMatrix(Propagator):
    def __init__(
        self,
        sys: SystemParams,
        initial_corr: np.ndarray,
        αs: tuple[Optional[BCF], Optional[BCF]] = (None, None),
    ):
        super().__init__(sys)

        self.G = self._residual_matrix.copy()
        self.G_e = -self._roots.copy()

        α = []
        αe = []

        for (i, bcf), scale in zip(enumerate(αs), self.params.η):
            if bcf is not None:
                α.append(bcf.factors.copy())
                αe.append(bcf.exponents.copy())
                α[i] *= scale

            else:
                α.append(self.params.G[i].copy())
                αe.append(self.params.W[i].copy())

        self.αc = np.array(list(map(np.conj, α)), dtype=object)
        self.αce = np.array(list(map(np.conj, αe)), dtype=object)
        self.α = np.array(α, dtype=object)
        self.αe = np.array(αe, dtype=object)
        self.initial_corr = initial_corr.copy()

    def __call__(
        self,
        t: np.ndarray,
        s: Optional[np.ndarray] = None,
    ) -> NDArray[np.complex128]:
        r"""Caluclate the corellation matrix :math:`\langle x_i(t)
        x_j(s)\rangle` of the system.

        :param t: The first time point.
        :param s: The second timepoint.
        :param αs: The (finite temparature) bath correlation
            functions.  If they're :any:`None` they're chosen to be
            the zero temperature BCF.
        :param initial_corr: The initial correlation matrix
            :math:`\langle x_i x_j\rangle`.
        """

        G = self.G
        G_e = self.G_e
        αc = self.αc
        αc_e = self.αce
        α = self.α
        α_e = self.αe
        initial_corr = self.initial_corr

        if s is not None and (s > t).any():
            raise ValueError("`s` must be smaller than or equal to `t`")

        Gt = self.propagator(t)
        Gs = self.propagator(s) if s is not None else Gt

        result = np.einsum("tik,tjl,kl->tij", Gt, Gs, initial_corr)

        if not s:
            for l in range(len(α)):
                for i, j, m, n, g in iterate_ragged(
                    G.shape[0],
                    G.shape[0],
                    G_e.shape[0],
                    G_e.shape[0],
                    len(α[l]),
                ):
                    # straight from mathematica
                    result[:, i, j] += (
                        (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
                        / ((G_e[m] + G_e[n]) * (G_e[n] + α_e[l][g]))
                        - (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
                        * np.exp(-t * (G_e[n] + α_e[l][g]))
                        / ((G_e[m] - α_e[l][g]) * (G_e[n] + α_e[l][g]))
                        + (
                            -(
                                (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
                                / ((G_e[m] + G_e[n]) * (G_e[n] + α_e[l][g]))
                            )
                            + (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
                            / ((G_e[m] - α_e[l][g]) * (G_e[n] + α_e[l][g]))
                            + (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
                            / ((G_e[m] + G_e[n]) * (G_e[n] - αc_e[l][g]))
                        )
                        * np.exp(-t * (G_e[m] + G_e[n]))
                        - (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
                        / ((G_e[m] + G_e[n]) * (G_e[n] - αc_e[l][g]))
                        + (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
                        / ((G_e[n] - αc_e[l][g]) * (G_e[m] + αc_e[l][g]))
                        - (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
                        * np.exp(-t * (G_e[m] + αc_e[l][g]))
                        / ((G_e[n] - αc_e[l][g]) * (G_e[m] + αc_e[l][g]))
                    )

        # else:
        #     for l in range(len(α)):
        #         for i, j, m, n, g in iterate_ragged(
        #             G.shape[0],
        #             G.shape[0],
        #             G_e.shape[0],
        #             G_e.shape[0],
        #             len(α[l]),
        #         ):

        #             result[:, i, j] += (
        #                 -(
        #                     (
        #                         np.exp(-(s * G_e[n]) - t * α_e[l][g])
        #                         * G[i, 1 + 2 * l, m]
        #                         * G[j, 1 + 2 * l, n]
        #                         * α[l][g]
        #                     )
        #                     / ((G_e[m] - α_e[l][g]) * (G_e[n] + α_e[l][g]))
        #                 )
        #                 + (
        #                     np.exp(s * α_e[l][g] - t * α_e[l][g])
        #                     * G[i, 1 + 2 * l, m]
        #                     * G[j, 1 + 2 * l, n]
        #                     * α[l][g]
        #                 )
        #                 / ((G_e[m] - α_e[l][g]) * (G_e[n] + α_e[l][g]))
        #                 + np.exp(-(t * G_e[m]) - s * G_e[n])
        #                 * (
        #                     -(
        #                         (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
        #                         / ((G_e[m] + G_e[n]) * (G_e[n] + α_e[l][g]))
        #                     )
        #                     + (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
        #                     / ((G_e[m] - α_e[l][g]) * (G_e[n] + α_e[l][g]))
        #                     + (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
        #                     / ((G_e[m] + G_e[n]) * (G_e[n] - αc_e[l][g]))
        #                 )
        #                 - (
        #                     np.exp(-(t * G_e[m]) - s * αc_e[l][g])
        #                     * G[i, 1 + 2 * l, m]
        #                     * G[j, 1 + 2 * l, n]
        #                     * αc[l][g]
        #                 )
        #                 / ((G_e[n] - αc_e[l][g]) * (G_e[m] + αc_e[l][g]))
        #                 + np.exp(s * G_e[m] - t * G_e[m])
        #                 * (
        #                     (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
        #                     / ((G_e[m] + G_e[n]) * (G_e[n] + α_e[l][g]))
        #                     - (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * α[l][g])
        #                     / ((G_e[m] - α_e[l][g]) * (G_e[n] + α_e[l][g]))
        #                     - (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
        #                     / ((G_e[m] + G_e[n]) * (G_e[n] - αc_e[l][g]))
        #                     + (G[i, 1 + 2 * l, m] * G[j, 1 + 2 * l, n] * αc[l][g])
        #                     / ((G_e[n] - αc_e[l][g]) * (G_e[m] + αc_e[l][g]))
        #                 )
        #             )

        return result

    def system_energy(self, t: np.ndarray) -> np.ndarray:
        correlation_matrix = self.__call__(t)
        corr = np.real(np.diagonal(correlation_matrix, axis1=1, axis2=2))

        Ω = self.params.Ω
        Λ = self.params.Λ
        γ = self.params.γ

        return (
            1
            / 4
            * (
                (Ω + γ) * corr[:, 0]  # type: ignore
                + Ω * corr[:, 1]  # type: ignore
                + (Λ + γ) * corr[:, 2]  # type: ignore
                + Λ * corr[:, 3]  # type: ignore
            )
        ) - 1 / 2 * γ * np.real(correlation_matrix[:, 0, 2])

    def Q1(
        self, t: np.ndarray, u: int, exp: Callable[[np.ndarray], np.ndarray] = np.exp
    ) -> NDArray[np.complex128]:
        G = self.G
        G_e = self.G_e

        ic = self.initial_corr
        α0d_e = self.params.W[u].copy()
        α0d = -self.params.G[u] * self.params.W[u]

        result = np.zeros_like(t, dtype=np.complex128)
        for j, h, k, l, m in iterate_ragged(
            G.shape[1], G.shape[2], G.shape[1], G.shape[2], len(α0d_e)
        ):
            result += (
                (
                    (exp(t * G_e[h]) - exp(t * α0d_e[m]))
                    * G[2 * u, j, h]
                    * G[2 * u, k, l]
                    * ic[k, j]
                    * α0d[m]
                )
                * exp(-t * (G_e[h] + G_e[l] + α0d_e[m]))
                / ((G_e[h] - α0d_e[m]))
            )

        return result

    def Q2(
        self, t: np.ndarray, u: int, exp: Callable[[np.ndarray], np.ndarray] = np.exp
    ) -> NDArray[np.complex128]:
        G = self.G
        G_e = self.G_e

        αc = self.αc
        αc_e = self.αce
        α = self.α
        α_e = self.αe
        α0d_e = self.params.W[u]
        α0d = -self.params.G[u] * self.params.W[u]

        result = np.zeros_like(t, dtype=np.complex128)
        for l, r, m, n in iterate_ragged(len(α_e), len(α0d), G.shape[2], G.shape[2]):
            for g in range(len(α_e[l])):
                result += (
                    (G[2 * u, 1 + 2 * l, m] * G[2 * u, 1 + 2 * l, n] * α0d[r] * α[l][g])
                    / ((G_e[m] + G_e[n]) * (G_e[m] + α0d_e[r]) * (G_e[n] + α_e[l][g]))
                    - (
                        G[2 * u, 1 + 2 * l, m]
                        * G[2 * u, 1 + 2 * l, n]
                        * α0d[r]
                        * α[l][g]
                    )
                    / (
                        (G_e[m] + α0d_e[r])
                        * (G_e[m] - α_e[l][g])
                        * (G_e[n] + α_e[l][g])
                    )
                    - (
                        G[2 * u, 1 + 2 * l, m]
                        * G[2 * u, 1 + 2 * l, n]
                        * α0d[r]
                        * α[l][g]
                    )
                    * exp(-t * (G_e[n] + α_e[l][g]))
                    / (
                        (-G_e[n] + α0d_e[r])
                        * (G_e[m] - α_e[l][g])
                        * (G_e[n] + α_e[l][g])
                    )
                    + (
                        G[2 * u, 1 + 2 * l, m]
                        * G[2 * u, 1 + 2 * l, n]
                        * α0d[r]
                        * α[l][g]
                    )
                    / (
                        (G_e[m] - α_e[l][g])
                        * (G_e[n] + α_e[l][g])
                        * (α0d_e[r] + α_e[l][g])
                    )
                    + (
                        (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * α[l][g]
                        )
                        / (
                            (-G_e[n] + α0d_e[r])
                            * (G_e[m] - α_e[l][g])
                            * (G_e[n] + α_e[l][g])
                        )
                        - (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * α[l][g]
                        )
                        / (
                            (G_e[m] - α_e[l][g])
                            * (G_e[n] + α_e[l][g])
                            * (α0d_e[r] + α_e[l][g])
                        )
                    )
                    * exp(-t * (α0d_e[r] + α_e[l][g]))
                    + (
                        -(
                            (
                                G[2 * u, 1 + 2 * l, m]
                                * G[2 * u, 1 + 2 * l, n]
                                * α0d[r]
                                * α[l][g]
                            )
                            / (
                                (G_e[m] + G_e[n])
                                * (-G_e[n] + α0d_e[r])
                                * (G_e[n] + α_e[l][g])
                            )
                        )
                        + (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * α[l][g]
                        )
                        / (
                            (-G_e[n] + α0d_e[r])
                            * (G_e[m] - α_e[l][g])
                            * (G_e[n] + α_e[l][g])
                        )
                        + (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * αc[l][g]
                        )
                        / (
                            (G_e[m] + G_e[n])
                            * (-G_e[n] + α0d_e[r])
                            * (G_e[n] - αc_e[l][g])
                        )
                    )
                    * exp(-t * (G_e[m] + G_e[n]))
                    - (
                        G[2 * u, 1 + 2 * l, m]
                        * G[2 * u, 1 + 2 * l, n]
                        * α0d[r]
                        * αc[l][g]
                    )
                    / ((G_e[m] + G_e[n]) * (G_e[m] + α0d_e[r]) * (G_e[n] - αc_e[l][g]))
                    + (
                        G[2 * u, 1 + 2 * l, m]
                        * G[2 * u, 1 + 2 * l, n]
                        * α0d[r]
                        * αc[l][g]
                    )
                    / (
                        (G_e[m] + α0d_e[r])
                        * (G_e[n] - αc_e[l][g])
                        * (G_e[m] + αc_e[l][g])
                    )
                    - (
                        G[2 * u, 1 + 2 * l, m]
                        * G[2 * u, 1 + 2 * l, n]
                        * α0d[r]
                        * αc[l][g]
                    )
                    * exp(-t * (G_e[m] + αc_e[l][g]))
                    / (
                        (G_e[n] - αc_e[l][g])
                        * (α0d_e[r] - αc_e[l][g])
                        * (G_e[m] + αc_e[l][g])
                    )
                    + (
                        -(
                            (
                                G[2 * u, 1 + 2 * l, m]
                                * G[2 * u, 1 + 2 * l, n]
                                * α0d[r]
                                * α[l][g]
                            )
                            / (
                                (G_e[m] + G_e[n])
                                * (G_e[m] + α0d_e[r])
                                * (G_e[n] + α_e[l][g])
                            )
                        )
                        + (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * α[l][g]
                        )
                        / (
                            (G_e[m] + G_e[n])
                            * (-G_e[n] + α0d_e[r])
                            * (G_e[n] + α_e[l][g])
                        )
                        + (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * α[l][g]
                        )
                        / (
                            (G_e[m] + α0d_e[r])
                            * (G_e[m] - α_e[l][g])
                            * (G_e[n] + α_e[l][g])
                        )
                        - (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * α[l][g]
                        )
                        / (
                            (-G_e[n] + α0d_e[r])
                            * (G_e[m] - α_e[l][g])
                            * (G_e[n] + α_e[l][g])
                        )
                        + (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * αc[l][g]
                        )
                        / (
                            (G_e[m] + G_e[n])
                            * (G_e[m] + α0d_e[r])
                            * (G_e[n] - αc_e[l][g])
                        )
                        - (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * αc[l][g]
                        )
                        / (
                            (G_e[m] + G_e[n])
                            * (-G_e[n] + α0d_e[r])
                            * (G_e[n] - αc_e[l][g])
                        )
                        - (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * αc[l][g]
                        )
                        / (
                            (G_e[m] + α0d_e[r])
                            * (G_e[n] - αc_e[l][g])
                            * (G_e[m] + αc_e[l][g])
                        )
                        + (
                            G[2 * u, 1 + 2 * l, m]
                            * G[2 * u, 1 + 2 * l, n]
                            * α0d[r]
                            * αc[l][g]
                        )
                        / (
                            (G_e[n] - αc_e[l][g])
                            * (α0d_e[r] - αc_e[l][g])
                            * (G_e[m] + αc_e[l][g])
                        )
                    )
                    * exp(-t * (G_e[m] + α0d_e[r]))
                )

        return result

    def Q3(
        self, t: np.ndarray, u: int, exp: Callable[[np.ndarray], np.ndarray] = np.exp
    ) -> NDArray[np.complex128]:
        G = self.G
        G_e = self.G_e

        αd = np.array([-self.αe[i] * self.α[i] for i in range(2)], dtype=object)
        αd_e = np.array([self.αe[i] for i in range(2)], dtype=object)
        α0d_e = np.array([self.params.W[i] for i in range(2)], dtype=object)
        α0d = np.array(
            [-self.params.G[i] * self.params.W[i] for i in range(2)],
            dtype=object,
        )

        result = np.zeros_like(t, dtype=np.complex128)
        for i in range(len(G_e)):
            for j in range(len(αd[u])):
                result += -(
                    (G[2 * u, 1 + 2 * u, i] * αd[u][j]) / (-G_e[i] - αd_e[u][j])
                ) + (
                    exp(-t * (G_e[i] + αd_e[u][j])) * G[2 * u, 1 + 2 * u, i] * αd[u][j]
                ) / (
                    -G_e[i] - αd_e[u][j]
                )

        for i in range(len(G_e)):
            for l in range(len(α0d[u])):
                result += (G[2 * u, 1 + 2 * u, i] * α0d[u][l]) / (
                    -G_e[i] - α0d_e[u][l]
                ) - (
                    exp(-t * (G_e[i] + α0d_e[u][l]))
                    * G[2 * u, 1 + 2 * u, i]
                    * α0d[u][l]
                ) / (
                    -G_e[i] - α0d_e[u][l]
                )

        return result

    def flow(self, t: np.ndarray, u: int, steady: bool = False) -> NDArray[np.float64]:
        exp = np.zeros_like if steady else np.exp
        return (
            1
            / 2
            * (
                np.real(self.Q3(t, u, exp))
                - np.imag(self.Q1(t, u, exp) + self.Q2(t, u, exp))
            )
        )


def initial_correlation_pure_osci(n: int, m: int):
    r"""
    Initial correlation matrix :math:`\langle x_i[] x_j\rangle` for
    two HO in product eigenstates with labels ``n`` and ``m``.
    """
    corr = np.diag(
        np.array([1 + 2 * n, 1 + 2 * n, 1 + 2 * m, 1 + 2 * m], dtype=np.complex128)
    )

    corr[0, 1] = 1j
    corr[1, 0] = -1j
    corr[2, 3] = 1j
    corr[3, 2] = -1j

    return corr
