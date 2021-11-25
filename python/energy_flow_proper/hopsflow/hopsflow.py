"""Calculate the reservoir energy change from HOPS data.

The functions in this module mostly take parameter objects which hold
relevant information and compute commonly used values ahead of time.

The parameter objects are passed separately but may depend on each other.
"""

import numpy as np
import scipy.misc
from . import util
from typing import Optional, Tuple, Iterator, Union
from stocproc import StocProc


###############################################################################
#                          Interface/Parameter Object#
###############################################################################


class SystemParams:
    """A parameter object to hold information about the physical
    system and global HOPS parameters.

    :param L: the coupling operator as system matrix
    :param G: the coupling factors in the exponential expansion of the BCF
    :param W: the exponents in the exponential expansion of the BCF
    :param bcf_scale: BCF scale factor
    :param nonlinear: whether the trajectory was obtained through the nonlinear HOPS
    :param g: individual scaling factors for the hierarchy states
    :attr dim: the dimensionality of the system
    """

    __slots__ = [
        "L",
        "G",
        "W",
        "bcf_scale",
        "g",
        "nonlinear",
        "dim",
    ]

    def __init__(
        self,
        L: np.ndarray,
        G: np.ndarray,
        W: np.ndarray,
        bcf_scale: float = 1,
        nonlinear: bool = False,
        g: Optional[np.ndarray] = None,
    ):
        """Class initializer. Computes the most useful attributes
        ahead of time."""

        self.L = L
        self.G = G
        self.W = W
        self.bcf_scale = bcf_scale
        self.g = g
        self.nonlinear = nonlinear
        self.dim = L.shape[0]


class HOPSRun:
    """A parameter object to hold data commonly used by the energy
    flow calculations.

    :param ψ_0: the HOPS trajectory ``(time, dim-state)``
    :param ψ_1: the first HOPS hierarchy states ``(time, hierarchy-width * dim-state)``
                 - will be reshaped to ``(time, hierarchy-width, dim_state)``
                 - will automatically be rescaled if scaling factors ``g`` are given

    :attr norm_squared: the squared norm of the state, may be ``None`` for linear HOPS
    :attr ψ_coup: ψ_0 with the coupling operator applied
    :attr hierarch_width: the width of the hierarchy (number of exponential terms)
    :attr t_steps: the number of timesteps
    :attr nonlinear: whether the trajectory was obtained through the nonlinear HOPS
    """

    __slots__ = [
        "ψ_0",
        "ψ_1",
        "norm_squared",
        "ψ_coup",
        "hierarchy_width",
        "t_steps",
        "nonlinear",
    ]

    def __init__(self, ψ_0: np.ndarray, ψ_1: np.ndarray, params: SystemParams):
        """Class initializer. Computes the most useful attributes
        ahead of time."""

        self.ψ_0 = ψ_0
        self.nonlinear = params.nonlinear
        self.norm_squared = np.sum(self.ψ_0.conj() * self.ψ_0, axis=1).real
        self.ψ_coup = util.apply_operator(self.ψ_0, params.L)
        self.t_steps = self.ψ_0.shape[0]
        self.hierarchy_width = ψ_1.shape[1] // params.dim

        ψ_1_rs = ψ_1.reshape(self.t_steps, self.hierarchy_width, params.dim)

        if params.g:
            ψ_1_rs /= params.g[None, :, None]

        self.ψ_1 = ψ_1_rs

    def normalize_maybe(self, array: np.ndarray):
        """Normalize the ``array`` if nonlinear HOPS is being used."""
        if self.nonlinear:
            return array / self.norm_squared

        return array


class ThermalParams:
    """Aparameter object to hold information abouth the thermal
    stochastic process.

    :param ξ: the thermal stochastic process
    :param τ: the time points of the trajectory
    :param num_deriv: whether to calculate the derivative of the
        process numerically or use it directly from the
        :class:`StocProc`.  The latter alternative is strongly preferred
        if available.
    :param dx: step size for the calculation of the derivative of ξ,
        only relevant if numerical differentiation is used
    :param order: order the calculation of the derivative of ξ, must
        be odd, see :any:`scipy.misc.derivative`, only relevant if
        numerical differentiation is used
    """

    __slots__ = ["ξ", "τ", "dx", "order", "num_deriv"]

    def __init__(
        self,
        ξ: StocProc,
        τ: np.ndarray,
        num_deriv: bool = False,
        dx: float = 1e-3,
        order: int = 3,
    ):
        """Class initializer. Computes the most useful attributes
        ahead of time.

        The process ξ is intialized with ξ_coeff and its derivative is
        being calculated.
        """

        self.ξ = ξ
        self.τ = τ
        self.dx = 1e-3
        self.order = order
        self.num_deriv = num_deriv


class ThermalRunParams:
    """A parameter object to hold information abouth the thermal
    stochastic process.

    :param therm_params: information abouth the thermal stochastic
        process
    :param ξ_coeff: the coefficients of the realization of the thermal
        stochastic process

    :attr ξ_dot: the process derivative evaluated at τ :attr
        ξ_values: the process evaluated at τ
    """

    __slots__ = ["ξ_coeff", "ξ_dot", "ξ_values"]

    def __init__(self, therm_params: ThermalParams, ξ_coeff: np.ndarray):
        """Class initializer.  Computes the most useful attributes
        ahead of time.

        The process ξ is intialized with ξ_coeff and its derivative is
        being calculated.
        """

        self.ξ_coeff = ξ_coeff

        therm_params.ξ.new_process(self.ξ_coeff)
        self.ξ_values = therm_params.ξ(therm_params.τ)
        self.ξ_dot: np.ndarray = (
            scipy.misc.derivative(
                therm_params.ξ,
                therm_params.τ,
                dx=therm_params.dx,
                order=therm_params.order,
            )
            if therm_params.num_deriv
            else therm_params.ξ.dot(therm_params.τ)
        )


###############################################################################
#                              Single Trajectory                              #
###############################################################################


def flow_trajectory_coupling(
    run: HOPSRun,
    params: SystemParams,
) -> np.ndarray:
    r"""Calculates the $\langle L^\dagger \dot{B} + c.c.$ part of the
    energy flow for a single trajectory.

    :param run: a parameter object for the current trajectory, see :any:`HOPSRun`
    :param therma_run: a parameter object for the current thermal process, see :any:`HOPSRun`
    :param params: a parameter object for the system, see :any:`SystemParams`

    :returns: the value of the flow for each time step
    """

    # here we apply the prefactors to each hierarchy state
    ψ_hops = util.mulitply_hierarchy(
        (1j * params.W * params.G * params.bcf_scale), run.ψ_1
    )

    flow = 2 * util.dot_with_hierarchy(run.ψ_coup.conj(), ψ_hops).real

    # optionally normalize
    return run.normalize_maybe(flow)


def flow_trajectory_therm(run: HOPSRun, therm_run: ThermalRunParams) -> np.ndarray:
    r"""Calculates the $\langle L^\dagger \dot{\xi} + c.c.$ part of the
    energy flow for a single trajectory.

    :param run: a parameter object, see :any:`HOPSRun`
    :param therm_run: a parameter object, see :any:`ThermalRunParams`
    :returns: the value of the flow for each time step
    """

    flow = (
        2
        * (
            run.normalize_maybe(np.sum(run.ψ_coup.conj() * run.ψ_0, axis=1))
            * therm_run.ξ_dot
        ).real
    )
    return flow


def flow_trajectory(
    run: HOPSRun, params: SystemParams, therm_run: Optional[ThermalRunParams]
) -> np.ndarray:
    r"""Calculates the total energy flow for a trajectory.

    :param run: a parameter object, see :any:`HOPSRun`
    :param therm: a parameter object, see :any:`ThermalParams`
    :param params: a parameter object for the system, see :any:`SystemParams`
    :returns: the value of the flow for each time step
    """

    flow = flow_trajectory_coupling(run, params)
    if therm_run:
        flow += flow_trajectory_therm(run, therm_run)

    return flow


def interaction_energy_coupling(run: HOPSRun, params: SystemParams) -> np.ndarray:
    """Calculates the coupling part of the interaction energy
    expectation value for a trajectory.

    :param run: a parameter object, see :any:`HOPSRun`
    :param params: a parameter object for the system, see
        :any:`SystemParams`

    :returns: the value of the coupling interaction energy for each time step
    """

    ψ_hops = util.mulitply_hierarchy(-1j * params.G * params.bcf_scale, run.ψ_1)
    energy = util.dot_with_hierarchy(run.ψ_coup.conj(), ψ_hops).real * 2
    return run.normalize_maybe(energy)


def interaction_energy_therm(run: HOPSRun, therm_run: ThermalRunParams) -> np.ndarray:
    r"""Calculates the thermal part of the interaction energy.

    :param run: a parameter object, see :any:`HOPSRun`
    :param therm_run: a parameter object, see :any:`ThermalParams`
    :returns: the value of the thermal interaction for each time step
    """

    energy = (
        run.normalize_maybe((run.ψ_coup.conj() * run.ψ_0).sum(axis=1))
        * therm_run.ξ_values
    ).real * 2

    return energy


###############################################################################
#                                   Ensemble                                  #
###############################################################################


def _heat_flow_ensemble_body(
    ψs: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: SystemParams,
    thermal: ThermalParams,
):
    ψ_0, ψ_1 = ψs[0:2]

    with_therm = len(ψs) == 3
    if with_therm:
        ys = ψs[-1]

    run = HOPSRun(ψ_0, ψ_1, params)
    flow = flow_trajectory_coupling(run, params)

    if with_therm:
        therm_run = ThermalRunParams(thermal, ys)
        flow += flow_trajectory_therm(run, therm_run)

    return flow


def heat_flow_ensemble(
    ψ_0s: Iterator[np.ndarray],
    ψ_1s: Iterator[np.ndarray],
    params: SystemParams,
    N: Optional[int],
    therm_args: Optional[Tuple[Iterator[np.ndarray], ThermalParams]] = None,
) -> np.ndarray:
    """Calculates the heat flow for an ensemble of trajectories.

    :param ψ_0s: array of trajectories ``(N, time-steps, dim-state)``
    :param ψ_0s: array of the first HOPS hierarchy states ``(N, time,
        hierarchy-width * dim-state)``
    :param params: a parameter object for the system, see
        :any:`SystemParams`
    :param therm_args: the realization parameters and the parameter
        object, see :any:`ThermalParams`

    :returns: the value of the flow for each time step
    """

    return util.ensemble_mean(
        iter(zip(ψ_0s, ψ_1s, therm_args[0])) if therm_args else iter(zip(ψ_0s, ψ_1s)),
        _heat_flow_ensemble_body,
        N,
        (params, therm_args[1] if therm_args else None),
    )


def _interaction_energy_ensemble_body(
    ψs: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: SystemParams,
    thermal: ThermalParams,
):
    ψ_0, ψ_1 = ψs[0:2]

    with_therm = len(ψs) == 3
    if with_therm:
        ys = ψs[-1]

    run = HOPSRun(ψ_0, ψ_1, params)
    energy = interaction_energy_coupling(run, params)

    if with_therm:
        therm_run = ThermalRunParams(thermal, ys)
        energy += interaction_energy_therm(run, therm_run)

    return energy


def interaction_energy_ensemble(
    ψ_0s: Iterator[np.ndarray],
    ψ_1s: Iterator[np.ndarray],
    params: SystemParams,
    N: Optional[int],
    therm_args: Optional[Tuple[Iterator[np.ndarray], ThermalParams]] = None,
) -> np.ndarray:
    """Calculates the heat flow for an ensemble of trajectories.

    :param ψ_0s: array of trajectories ``(N, time-steps, dim-state)``
    :param ψ_0s: array of the first HOPS hierarchy states ``(N, time,
        hierarchy-width * dim-state)``
    :param params: a parameter object for the system, see
        :any:`SystemParams`
    :param therm_args: the realization parameters and the parameter
        object, see :any:`ThermalParams`

    :returns: the value of the flow for each time step
    """

    return util.ensemble_mean(
        iter(zip(ψ_0s, ψ_1s, therm_args[0])) if therm_args else iter(zip(ψ_0s, ψ_1s)),
        _interaction_energy_ensemble_body,
        N,
        (params, therm_args[1] if therm_args else None),
    )
