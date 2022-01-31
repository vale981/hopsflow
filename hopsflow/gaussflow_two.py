"""Calculate the energy flow into the bath for a simple gaussian
Quantum Brownian Motion model with two coupled HOs coupled to a bath each.

This is done analytically for a BCF that is a sum of exponentials.
"""

from dataclasses import dataclass, field
import util
from .util import BCF, expand_t
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import Polynomial


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

    def __post_init__(self):
        self.t_max = self.α_0[0].t_max

        assert self.α_0[0].t_max == self.α_0[1].t_max

        self.W = [α.exponents for α in self.α_0]
        self.G = [α.factors * scale for α, scale in zip(self.α_0, self.η)]


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
            γ * Λ ** 2 * Ω + γ * Λ * Ω ** 2 + Λ ** 2 * Ω ** 2,
            0,
            γ * Λ + Λ ** 2 + γ * Ω + Ω ** 2,
            0,
            1,
        ]
    )

    p_a = Polynomial([γ * Λ * Ω + Λ ** 2 * Ω, 0, Ω])
    p_b = Polynomial([γ * Λ * Ω + Λ * Ω ** 2, 0, Λ])
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

    Λ = sys.Λ * np.ones_like(master_roots)
    Ω = sys.Ω * np.ones_like(master_roots)
    γ = sys.γ * np.ones_like(master_roots)

    a = gn_a(master_roots) / f_0_a(master_roots)
    b = gn_b(master_roots) / f_0_b(master_roots)

    matrix_elements = np.array(
        [
            master_roots ** 3 + master_roots * Λ * (b + γ + Λ),
            (master_roots ** 2 + Λ * (b + γ + Λ)) * Ω,
            master_roots * γ * Ω,
            γ * Λ * Ω,
            -(a * (master_roots ** 2 + Λ * (b + γ + Λ)))
            - master_roots ** 2 * (γ + Ω)
            - Λ * (Λ * Ω + b * (γ + Ω) + γ * (Λ + Ω)),
            master_roots ** 2 * γ,
            master_roots * γ * Λ,
            master_roots ** 3 + master_roots * Ω * (a + γ + Ω),
            Λ * (master_roots ** 2 + Ω * (a + γ + Ω)),
            -(master_roots ** 2 * (γ + Λ))
            - b * (master_roots ** 2 + Ω * (a + γ + Ω))
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

    @expand_t
    def __call__(self, t: NDArray[np.float64]) -> np.ndarray:
        """Get the propagator as array of shape ``(time, 4, 4)``."""

        mat = self._res
        residual_matrix = np.array(
            [
                [mat[0], mat[1], mat[2], mat[3]],
                [mat[4], mat[0], mat[5], mat[6]],
                [mat[6], mat[3], mat[7], mat[8]],
                [mat[5], mat[2], mat[9], mat[7]],
            ]
        )

        return (
            np.sum(
                residual_matrix * np.exp(self._roots * t)[None, None, :], axis=2
            ).real
            * 2
        )

    def inv(self, t) -> np.ndarray:
        return np.linalg.inv(self.__call__(t))
