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
import itertools
import ray
from hops.util.dynamic_matrix import DynamicMatrix, DynamicMatrixList
from numpy.typing import NDArray
import opt_einsum as oe
from hops.core.hierarchy_data import HIData
from hops.core.hierarchy_parameters import HIParams
from typing import Callable
import copy

###############################################################################
#                          Interface/Parameter Object                         #
###############################################################################


class SystemParams:
    """A parameter object to hold information about the physical
    system and global HOPS parameters.

    :param L: The coupling operators as system matrices.
    :param G: The coupling factors in the exponential expansion of the
              BCF.
    :param W: The exponents in the exponential expansion of the BCF.
    :param t: The time points of the evaluation.
    :param bcf_scale: The BCF scale factors.
    :param nonlinear: Whether the trajectory was obtained through the
        nonlinear HOPS.
    :param fock_hops: Whether the now fock hops hierarchy
        normalization is used.
    """

    __slots__ = [
        "L",
        "G",
        "W",
        "t",
        "bcf_scale",
        "nonlinear",
        "coupling_interaction_prefactors",
        "coupling_flow_prefactors",
        "dim",
        "apply_L",
        "apply_L_dot",
    ]

    def __init__(
        self,
        L: list[DynamicMatrix],
        G: list[np.ndarray],
        W: list[np.ndarray],
        t: NDArray[np.float64],
        bcf_scale: Optional[list[float]] = None,
        nonlinear: bool = False,
        fock_hops: bool = True,
    ):
        self.t = t

        self.G = [g * scale for g, scale in zip(G, bcf_scale)] if bcf_scale else G
        self.W = W

        self.coupling_interaction_prefactors = [
            (np.sqrt(G) if fock_hops else -1j * G) for G in self.G
        ]

        self.coupling_flow_prefactors = [
            fac * -W for fac, W in zip(self.coupling_interaction_prefactors, self.W)
        ]

        self.nonlinear = nonlinear

        #: the dimensionality of the system
        self.dim = L[0].shape[0]

        """
        A fast einstein sum to apply the ``L`` operators to the zeroth
        hierarchy state for each time step.
        """
        self.apply_L = oe.contract_expression(
            "ntij,tj->nti",
            (DynamicMatrixList(L))(self.t),
            (len(t), self.dim),
            constants=[0],
        )

        """
        Same as :any:`apply_L` but for the time derivatives of the
        ``L``.
        """
        self.apply_L_dot = None

        try:
            self.apply_L_dot = oe.contract_expression(
                "ntij,tj->nti",
                (DynamicMatrixList(L).derivative())(self.t),
                (len(t), self.dim),
                constants=[0],
            )
        except NotImplementedError:
            self.apply_L_dot = None

    @classmethod
    def from_hi_params(cls, params: HIParams, **kwargs) -> "SystemParams":
        """Construct a :any:`SystemParams` object from the HOPS
        configuration.


        The ``kwargs`` are forwarded to the constructor.

        :param params: The :any:`hops.core.hierarchy_parameters`
            object that holds the HOPS config.
        """
        return cls(
            L=params.SysP.L,
            G=params.SysP.g,
            W=params.SysP.w,
            t=params.IntP.t,
            nonlinear=params.HiP.nonlinear,
            fock_hops=True,
            **kwargs,  # no bcf scale, as this has been built in already
        )


class HOPSRun:
    """A parameter object to hold data commonly used by the energy
    flow calculations.

    :param ψ_0: The HOPS trajectory ``(time, dim-state)``.
    :param ψ_1: The first HOPS hierarchy states ``(time,
        hierarchy-width * dim-state)``.

        It will be reshaped to a list containing arrays of the
        shape``(time, hierarchy-width, dim_state)`` (one for each
        bath).

    :param power: whether to calculate the interaction power (replaces
        the coupling operators with their time derivatives)
    """

    __slots__ = [
        "ψ_0",
        "ψ_1_n",
        "norm_squared",
        "ψ_coups",
        "t_steps",
        "nonlinear",
        "power",
    ]

    def __init__(
        self, ψ_0: np.ndarray, ψ_1: np.ndarray, params: SystemParams, power=False
    ):
        """Class initializer. Computes the most useful attributes
        ahead of time."""

        self.ψ_0 = ψ_0
        self.nonlinear = params.nonlinear
        self.norm_squared = (
            np.sum(self.ψ_0.conj() * self.ψ_0, axis=1).real
            if params.nonlinear
            else None
        )
        """
        The squared norm of the state, may be ``None`` for linear HOPS.
        """

        if power:
            if params.apply_L_dot is None:
                raise NotImplementedError

            ψ_coups = params.apply_L_dot(self.ψ_0)

        else:
            ψ_coups = params.apply_L(self.ψ_0)

        self.ψ_coups = ψ_coups
        """
        ψ_0 with the coupling operator (or its derivative) applied for each bath.
        """

        self.power = power
        """Wether the ``L`` operators have been replaced with their derivatives."""

        self.t_steps = self.ψ_0.shape[0]
        """The number of timesteps."""

        hierarchy_width = ψ_1.shape[1] // params.dim
        ψ_1_rs = ψ_1.reshape(self.t_steps, hierarchy_width, params.dim)

        offsets = np.cumsum([0] + [len(g) for g in params.G])
        self.ψ_1_n = [
            ψ_1_rs[:, offsets[i] : offsets[i + 1], :] for i in range(len(offsets) - 1)
        ]
        """The first hierarchy states for each bath."""

    def normalize_maybe(self, array: np.ndarray):
        """Normalize the ``array`` if nonlinear HOPS is being used."""
        if self.nonlinear:
            return array / self.norm_squared

        return array


class ThermalParams:
    """Aparameter object to hold information abouth the thermal
    stochastic process.

    :param ξ: the thermal stochastic processes
    :param τ: the time points of the trajectory
    :param rand_skip: how many random numbers to skip when
        parametrizing the stochastic process
    :param num_deriv: whether to calculate the derivative of the
        process numerically or use it directly from the
        :class:`StocProc`.  The latter alternative is strongly
        preferred if available.
    :param dx: step size for the calculation of the derivative of ξ,
        only relevant if numerical differentiation is used
    :param order: order the calculation of the derivative of ξ, must
        be odd, see :any:`scipy.misc.derivative`, only relevant if
        numerical differentiation is used
    """

    __slots__ = ["ξs", "τ", "dx", "order", "num_deriv", "rand_skip"]

    def __init__(
        self,
        ξs: list[Optional[StocProc]],
        τ: np.ndarray,
        rand_skip: int = 0,
        num_deriv: bool = False,
        dx: float = 1e-3,
        order: int = 3,
    ):
        self.ξs = ξs
        self.τ = τ
        self.dx = dx
        self.order = order
        self.num_deriv = num_deriv
        self.rand_skip = rand_skip

    @classmethod
    def from_hi_params(cls, params: HIParams, **kwargs) -> "SystemParams":
        """Construct a :any:`ThermalParams` object from the HOPS
        configuration.


        The ``kwargs`` are forwarded to the constructor.

        :param params: The :any:`hops.core.hierarchy_parameters`
            object that holds the HOPS config.
        """

        ξs = [copy.deepcopy(proc) for proc in params.EtaTherm]

        for i, proc in enumerate(ξs):
            if proc is not None:
                proc.calc_deriv = True
                proc.set_scale(params.SysP.bcf_scale[i])

        return cls(
            ξs=ξs,
            τ=params.IntP.t,
            rand_skip=params.HiP.rand_skip,
            **kwargs,  # no bcf scale, as this has been built in already
        )


class ThermalRunParams:
    """A parameter object to hold information abouth the thermal
    stochastic process.

    :param therm_params: information abouth the thermal stochastic
        process

    :param seed: the seed used to generate the random numbers for the
        process realization

    :attr ξ_dots: the process derivatives evaluated at τ :attr

    :attr ξ_values: the processes evaluated at τ

    :attr ξ_coeff: the coefficients of the realization of the thermal
        stochastic processes
    """

    __slots__ = ["ξ_coeff", "ξ_dots", "ξ_values"]

    def __init__(self, therm_params: ThermalParams, seed: int):
        """Class initializer.  Computes the most useful attributes
        ahead of time.

        The processes ξs are intialized with ξ_coeff and their
        derivative is being calculated.
        """

        np.random.seed(seed)
        np.random.rand(therm_params.rand_skip)

        self.ξ_coeff = []
        self.ξ_values = []
        self.ξ_dots = []

        for ξ in therm_params.ξs:
            if ξ:
                coeff = util.uni_to_gauss(np.random.rand((ξ.get_num_y() or 0) * 2))
                self.ξ_coeff.append(coeff)
                ξ.new_process(coeff)
                self.ξ_values.append(ξ(therm_params.τ))

                self.ξ_dots.append(
                    scipy.misc.derivative(
                        ξ,
                        therm_params.τ,
                        dx=therm_params.dx,
                        order=therm_params.order,
                    )
                    if therm_params.num_deriv
                    else ξ.dot(therm_params.τ)
                )
            else:
                self.ξ_coeff.append(0)
                self.ξ_values.append(0)
                self.ξ_dots.append(None)


###############################################################################
#                              Single Trajectory                              #
###############################################################################


def flow_trajectory_coupling(
    run: HOPSRun,
    params: SystemParams,
) -> np.ndarray:
    r"""Calculates the :math:`\langle L^\dagger \dot{B}\rangle + c.c.` part of the
    energy flow for a single trajectory and for each bath.

    :param run: a parameter object for the current trajectory, see :any:`HOPSRun`
    :param params: a parameter object for the system, see :any:`SystemParams`

    :returns: the value of the flow for each time step
    """

    # here we apply the prefactors to each hierarchy state
    ψ_hopses = [
        util.mulitply_hierarchy(f, ψ_1)
        for f, ψ_1 in zip(params.coupling_flow_prefactors, run.ψ_1_n)
    ]

    flows = np.array(
        [
            2 * util.dot_with_hierarchy(ψ_coup.conj(), ψ_hops).real
            for ψ_coup, ψ_hops in zip(run.ψ_coups, ψ_hopses)
        ]
    )  # fromiter crashed...

    # optionally normalize
    return run.normalize_maybe(flows)


def flow_trajectory_therm(run: HOPSRun, therm_run: ThermalRunParams) -> np.ndarray:
    r"""Calculates the :math:`\langle L^\dagger \dot{\xi} + c.c.` part of the
    energy flow for a single trajectory.

    :param run: a parameter object, see :any:`HOPSRun`
    :param therm_run: a parameter object, see :any:`ThermalRunParams`
    :returns: the value of the flow for each time step
    """

    flows = np.array(
        [
            2
            * (
                run.normalize_maybe(np.sum(ψ_coup.conj() * run.ψ_0, axis=1)) * ξ_dot
                if ξ_dot is not None
                else np.zeros(
                    ψ_coup.shape[0]
                )  # note: this is already scaled by stocproc
            ).real
            for ψ_coup, ξ_dot in zip(run.ψ_coups, therm_run.ξ_dots)
        ]
    )

    return flows


def flow_trajectory(
    run: HOPSRun, params: SystemParams, therm_run: Optional[ThermalRunParams] = None
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
    r"""Calculates the coupling part of the interaction energy
    expectation value for a trajectory.

    :param run: a parameter object for the current trajectory, see :any:`HOPSRun`
    :param params: a parameter object for the system, see :any:`SystemParams`

    :returns: the value of the interaction energy (or power) for each time step
    """

    # here we apply the prefactors to each hierarchy state
    ψ_hopses = [
        util.mulitply_hierarchy(f, ψ_1)
        for f, ψ_1 in zip(params.coupling_interaction_prefactors, run.ψ_1_n)
    ]

    flows = np.array(
        [
            2 * util.dot_with_hierarchy(ψ_coup.conj(), ψ_hops).real
            for ψ_coup, ψ_hops in zip(run.ψ_coups, ψ_hopses)
        ]
    )  # fromiter crashed...

    # optionally normalize
    return run.normalize_maybe(flows)


def interaction_energy_therm(
    run: HOPSRun, therm_run: ThermalRunParams, power: bool = False
) -> np.ndarray:
    r"""Calculates the thermal part of the interaction energy.

    :param run: a parameter object, see :any:`HOPSRun`
    :param params: a parameter object for the system, see :any:`SystemParams`
    :param therm_run: a parameter object, see :any:`ThermalParams`

    :returns: the value of the thermal interaction (or power) for each time step
    """

    energies = np.array(
        [
            2
            * (
                run.normalize_maybe(np.sum(ψ_coup.conj() * run.ψ_0, axis=1)) * ξ
                if ξ is not None
                else np.zeros(
                    ψ_coup.shape[0]
                )  # note: this is already scaled by stocproc
            ).real
            for ψ_coup, ξ in zip(run.ψ_coups, therm_run.ξ_values)
        ]
    )

    return energies


###############################################################################
#                                   Ensemble                                  #
###############################################################################


def heat_flow_ensemble(
    ψ_0s: Iterator[np.ndarray],
    ψ_1s: Iterator[np.ndarray],
    params: SystemParams,
    therm_args: Optional[Tuple[Iterator[np.ndarray], ThermalParams]] = None,
    only_therm: bool = False,
    **kwargs,
) -> util.EnsembleValue:
    """Calculates the heat flow for an ensemble of trajectories.

    :param ψ_0s: array of trajectories ``(N, time-steps, dim-state)``
    :param ψ_0s: array of the first HOPS hierarchy states ``(N, time,
        hierarchy-width * dim-state)``
    :param params: a parameter object for the system, see
        :any:`SystemParams`
    :param therm_args: the realization parameters and the parameter
        object, see :any:`ThermalParams`
    :param only_therm: whether to only calculate the thermal part of the flow

    The rest of the ``kwargs`` is passed on to :any:`util.ensemble_mean`.

    :returns: the value of the flow for each time step
    """

    flow_worker = make_heat_flow_worker(
        params, therm_args[1] if therm_args else None, only_therm
    )

    return util.ensemble_mean(
        iter(zip(ψ_0s, ψ_1s, therm_args[0]))
        if therm_args
        else iter(zip(ψ_0s, ψ_1s, itertools.repeat(0))),
        flow_worker,
        **kwargs,
    )


def make_heat_flow_worker(
    params: SystemParams,
    thermal: Optional[ThermalParams] = None,
    only_therm: bool = False,
) -> Callable[..., np.ndarray]:
    """Constructs a worker function that takes ``(first hierarchy
    state, aux states, thermal process seed)`` and returns the flow.

    :param params: a parameter object for the system, see
        :any:`SystemParams`
    :param therm: the parameter object for the nonzero temperature,
        see :any:`ThermalParams`
    :param only_therm: whether to only calculate the thermal part of
        the flow
    """

    if thermal is None and only_therm:
        raise ValueError("Can't calculate only thermal part if therm_args are None.")

    def flow_worker(ψs: tuple[np.ndarray, np.ndarray, int]):
        ψ_0, ψ_1, seed = ψs

        run = HOPSRun(ψ_0, ψ_1, params)  # type: ignore
        flow = (
            flow_trajectory_coupling(run, params)  # type: ignore
            if not only_therm
            else np.zeros((len(params.G), ψ_0.shape[0]))
        )

        if thermal is not None:
            therm_run = ThermalRunParams(thermal, seed)  # type: ignore
            flow += flow_trajectory_therm(run, therm_run)

        return flow

    return flow_worker


def heat_flow_from_data(
    data: HIData,
    *args,
    thermal_params: Optional[ThermalParams] = None,
    **kwargs,
) -> util.EnsembleValue:
    """Calculates the heat flow for an ensemble of trajectories.

    The rest of the ``args`` and ``kwargs`` is passed on to
    :any:`util.heat_flow_ensemble`.

    :param data: The data instance that contains the trajectories.
        Does not have to be opened yet.

    :returns: the value of the flow for each time step
    """

    with data as d:
        if "save" in kwargs:
            kwargs["save"] += "_" + data.get_hi_key_hash()

        if thermal_params is not None:
            kwargs["therm_args"] = (
                d.valid_sample_iterator(d.rng_seed),
                thermal_params,
            )

        return heat_flow_ensemble(
            d.valid_sample_iterator(d.stoc_traj),
            d.valid_sample_iterator(d.aux_states),
            *args,
            **(dict(N=data.samples) | kwargs),
        )


def make_interaction_worker(
    params: SystemParams,
    thermal: Optional[ThermalParams] = None,
    power: bool = False,
):
    """
    :param params: a parameter object for the system, see
        :any:`SystemParams`
    :param thermal: the thermal object, see :any:`ThermalParams`
    :param power: whether to calculate the interaction power
    """

    def interaction_energy_task(
        ψs: Tuple[np.ndarray, np.ndarray, int],
    ) -> np.ndarray:
        ψ_0, ψ_1, seeds = ψs
        run = HOPSRun(ψ_0, ψ_1, params, power)  # type: ignore
        energy = interaction_energy_coupling(run, params)  # type: ignore

        if thermal is not None:
            therm_run = ThermalRunParams(thermal, seeds)  # type: ignore
            energy += interaction_energy_therm(run, therm_run)

        return energy

    return interaction_energy_task


def interaction_energy_ensemble(
    ψ_0s: Iterator[np.ndarray],
    ψ_1s: Iterator[np.ndarray],
    params: SystemParams,
    therm_args: Optional[Tuple[Iterator[int], ThermalParams]] = None,
    power: bool = False,
    **kwargs,
) -> util.EnsembleValue:
    """Calculates the heat flow for an ensemble of trajectories.

    :param ψ_0s: array of trajectories ``(N, time-steps, dim-state)``
    :param ψ_0s: array of the first HOPS hierarchy states ``(N, time,
        hierarchy-width * dim-state)``
    :param params: a parameter object for the system, see
        :any:`SystemParams`
    :param therm_args: the realization parameters and the parameter
        object, see :any:`ThermalParams`
    :param power: whether to calculate the interaction power


    The ``**kwargs`` are passed to :any:`hopsflow.util.ensemble_mean`.
    :returns: the value of the interaction energy (or power) for each time step
    """

    thermal = therm_args[1] if therm_args else None

    interaction_energy_task = make_interaction_worker(params, thermal, power)
    return util.ensemble_mean(
        iter(zip(ψ_0s, ψ_1s, therm_args[0]))
        if therm_args
        else iter(zip(ψ_0s, ψ_1s, itertools.repeat(0))),
        interaction_energy_task,
        **kwargs,
    )


def energy_change_from_interaction_power(
    τ: np.ndarray,
    *args,
    **kwargs,
) -> util.EnsembleValue:
    """Calculates the energy change due to the explicit time
    dependence of the interaction hamiltonian.

    For the arguments see :any:`interaction_energy_ensemble`.

    :param τ: The time points of the simulations.

    :returns: The value of the total energy change due to the time
              dependence of the interaction for each time step.
    """

    kwargs = kwargs | dict(power=True)
    power = interaction_energy_ensemble(*args, **kwargs)

    return power.integrate(τ)


def bath_energy_from_flow(
    τ: np.ndarray,
    *args,
    **kwargs,
) -> util.EnsembleValue:
    """Calculates the bath energy by integrating the flow.
    For the arguments see :any:`heat_flow_ensemble`.

    :param τ: The time points of the simulations.
    :returns: The value of the bath energy for each time step.
    """

    flow = heat_flow_ensemble(*args, **kwargs)

    return -1 * flow.integrate(τ)
