import numpy as np
cimport numpy as np
from itertools import product
def iterate_ragged(*ranges: int):
    return product(*(range(r) for r in ranges))

cdef Q2(np.ndarray t,
        int u,
        np.ndarray G,
        np.ndarray G_e,
        np.ndarray alphac,
        np.ndarray alphac_e,
        np.ndarray alpha,
        np.ndarray alpha_e,
        np.ndarray alpha0d,
        np.ndarray alpha0d_e):
        cdef np.ndarray result = np.zeros_like(t, dtype=np.complex128)

        for l, m, n, r, g in iterate_ragged(
            len(alpha_e), G.shape[2], G.shape[2], len(alpha0d_e), len(alpha0d_e)
        ):
            result += (
                G[2 * u, l, m]
                * G[2 * u, l, n]
                * alpha0d[r]
                * (
                    (
                        alpha[l, g]
                        * (
                            -(
                                (
                                    (G_e[m] + G_e[n])
                                    * (G_e[n] + alpha_e[l, g])
                                    * (G_e[m] + alpha0d_e[r])
                                )
                                / np.exp(t * (alpha_e[l, g] + alpha0d_e[r]))
                            )
                            + (
                                (G_e[m] + G_e[n])
                                * (G_e[n] + alpha_e[l, g])
                                * (alpha_e[l, g] + alpha0d_e[r])
                            )
                            / np.exp(t * (G_e[m] + alpha0d_e[r]))
                            + (
                                (G_e[m] + G_e[n])
                                * (G_e[m] + alpha0d_e[r])
                                * (alpha_e[l, g] + alpha0d_e[r])
                            )
                            / np.exp(t * (G_e[n] + alpha_e[l, g]))
                            - (
                                (G_e[n] + alpha_e[l, g])
                                * (G_e[m] + alpha0d_e[r])
                                * (alpha_e[l, g] + alpha0d_e[r])
                            )
                            / np.exp(t * (G_e[m] + G_e[n]))
                            + (G_e[m] - alpha_e[l, g])
                            * (G_e[n] - alpha0d_e[r])
                            * (G_e[m] + G_e[n] + alpha_e[l, g] + alpha0d_e[r])
                        )
                    )
                    / (
                        (G_e[m] + G_e[n])
                        * (G_e[m] - alpha_e[l, g])
                        * (G_e[n] + alpha_e[l, g])
                        * (G_e[n] - alpha0d_e[r])
                        * (G_e[m] + alpha0d_e[r])
                        * (alpha_e[l, g] + alpha0d_e[r])
                    )
                    + alphac[l, g]
                    * (
                        -(
                            1
                            / (
                                np.exp(t * (G_e[m] + G_e[n]))
                                * (G_e[m] + G_e[n])
                                * (G_e[n] - alpha0d_e[r])
                                * (G_e[n] - alphac_e[l, g])
                            )
                        )
                        + 1
                        / (
                            np.exp(t * (G_e[m] + alpha0d_e[r]))
                            * (G_e[n] - alpha0d_e[r])
                            * (G_e[m] + alpha0d_e[r])
                            * (alpha0d_e[r] - alphac_e[l, g])
                        )
                        + 1
                        / (
                            (G_e[m] + G_e[n])
                            * (G_e[m] + alpha0d_e[r])
                            * (G_e[m] + alphac_e[l, g])
                        )
                        + 1
                        / (
                            np.exp(t * (G_e[m] + alphac_e[l, g]))
                            * (G_e[n] - alphac_e[l, g])
                            * (G_e[m] + alphac_e[l, g])
                            * (-alpha0d_e[r] + alphac_e[l, g])
                        )
                    )
                )
            )
