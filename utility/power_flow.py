"""
Tensor-based fixed-point power-flow solver for IEEE 33-bus DESS coordination.

This module implements a constant-power distribution power-flow solver using
a reduced admittance-matrix formulation.

Main tensor interface:
    run_pf(active_power, reactive_power)

where active_power and reactive_power are non-slack bus power vectors with
shape:

    (batch_size, num_buses - 1)

Environment wrapper:
    run_power_flow(load_kw, load_kvar, pv_kw, dess_power_kw)

where all inputs are full bus-level vectors with shape:

    (num_buses,)

Sign convention:
    load_kw > 0        : active-power consumption
    pv_kw > 0          : active-power generation
    dess_power_kw > 0  : DESS discharging / active-power injection
    dess_power_kw < 0  : DESS charging / active-power absorption

Therefore:

    P_net = load_kw - pv_kw - dess_power_kw

The solver returns bus voltages, branch currents, branch power-flow
quantities, voltage violations, line-current violations, and convergence
status. These quantities are used by the MARL environment to enforce
network feasibility.
"""

from time import perf_counter

import numpy as np
import pandas as pd


class PowerFlowSolver:
    """
    Fixed-point distribution power-flow solver.
    """

    def __init__(self, grid, config):
        self.grid = grid
        self.config = config

        self.num_buses = int(config["network"]["num_buses"])
        self.slack_bus = int(config["network"]["slack_bus"])

        self.s_base_kva = float(config["network"]["s_base_kva"])
        self.v_base_kv = float(config["network"]["v_base_kv"])

        self.z_base_ohm = (self.v_base_kv ** 2 * 1000.0) / self.s_base_kva
        self.i_base_a = self.s_base_kva / (np.sqrt(3.0) * self.v_base_kv)

        self.v_ref = float(config["network"]["v_ref"])
        self.v_min = float(config["network"]["v_min"])
        self.v_max = float(config["network"]["v_max"])

        self.tolerance = float(config["network"]["power_flow_tolerance"])
        self.max_iterations = int(config["network"]["power_flow_max_iterations"])

        self.return_branch_flows = bool(
            config["network"].get("return_branch_flows", True)
        )
        self.return_line_currents = bool(
            config["network"].get("return_line_currents", True)
        )

        self.bus_file = config["paths"]["gridtensor_bus_data"]
        self.line_file = config["paths"]["gridtensor_line_data"]

        self.bus_info = pd.read_csv(self.bus_file)
        self.line_info = pd.read_csv(self.line_file)

        self._validate_input_files()
        self._prepare_in_service_lines()
        self._build_ybus()
        self._precompute_tensor_matrices()

    # ============================================================
    # Validation
    # ============================================================

    def _validate_input_files(self):
        """
        Validate solver-specific GridTensor-format network files.

        Expected bus columns:
            NODES, Tb

        Expected line columns:
            FROM, TO, R, X, B, STATUS, TAP

        The solver-specific files use one-based bus numbering.
        """

        required_bus_cols = ["NODES", "Tb"]
        required_line_cols = ["FROM", "TO", "R", "X", "B", "STATUS", "TAP"]

        for col in required_bus_cols:
            if col not in self.bus_info.columns:
                raise ValueError("Missing bus-data column: {}".format(col))

        for col in required_line_cols:
            if col not in self.line_info.columns:
                raise ValueError("Missing line-data column: {}".format(col))

        if len(self.bus_info) != self.num_buses:
            raise ValueError(
                "Expected {} buses, found {}.".format(
                    self.num_buses,
                    len(self.bus_info),
                )
            )

        slack_nodes = self.bus_info[self.bus_info["Tb"] == 1]["NODES"].tolist()

        if len(slack_nodes) != 1:
            raise ValueError("Exactly one slack bus is required.")

        expected_slack = self.slack_bus + 1
        actual_slack = int(slack_nodes[0])

        if actual_slack != expected_slack:
            raise ValueError(
                "Slack-bus mismatch. Config expects {}, file gives {}.".format(
                    expected_slack,
                    actual_slack,
                )
            )

        for line_idx, row in self.line_info.iterrows():
            from_bus = int(row["FROM"])
            to_bus = int(row["TO"])

            if from_bus < 1 or from_bus > self.num_buses:
                raise ValueError(
                    "Invalid FROM bus index in line {}: {}".format(
                        line_idx,
                        from_bus,
                    )
                )

            if to_bus < 1 or to_bus > self.num_buses:
                raise ValueError(
                    "Invalid TO bus index in line {}: {}".format(
                        line_idx,
                        to_bus,
                    )
                )

            if from_bus == to_bus:
                raise ValueError(
                    "Line {} has identical FROM and TO buses.".format(line_idx)
                )

            if float(row["R"]) < 0.0:
                raise ValueError(
                    "Line resistance cannot be negative in line {}.".format(line_idx)
                )

            if float(row["X"]) < 0.0:
                raise ValueError(
                    "Line reactance cannot be negative in line {}.".format(line_idx)
                )

            if float(row["TAP"]) < 0.0:
                raise ValueError(
                    "Transformer tap ratio cannot be negative in line {}.".format(
                        line_idx
                    )
                )

            if float(row["STATUS"]) not in [0.0, 1.0]:
                raise ValueError(
                    "Line STATUS must be either 0 or 1 in line {}.".format(line_idx)
                )

    def _prepare_in_service_lines(self):
        """
        Store in-service line data used by the solver.
        """

        self.in_service_line_info = self.line_info[
            self.line_info["STATUS"].astype(float) == 1.0
        ].reset_index(drop=True)

        if len(self.in_service_line_info) == 0:
            raise ValueError("At least one in-service line is required.")

    # ============================================================
    # Y-bus construction
    # ============================================================

    def _build_ybus(self):
        """
        Build full and reduced admittance matrices.
        """

        n = self.num_buses
        ybus = np.zeros((n, n), dtype=np.complex128)

        for _, row in self.in_service_line_info.iterrows():
            from_bus = int(row["FROM"]) - 1
            to_bus = int(row["TO"]) - 1

            r_ohm = float(row["R"])
            x_ohm = float(row["X"])
            b_value = float(row["B"])
            tap = float(row["TAP"])

            if tap == 0.0:
                tap = 1.0

            z_pu = (r_ohm + 1j * x_ohm) / self.z_base_ohm
            y_series = 1.0 / z_pu

            b_pu = b_value * self.z_base_ohm

            y_tt = y_series + 1j * b_pu / 2.0
            y_ff = y_tt / (tap ** 2)
            y_ft = -y_series / tap
            y_tf = -y_series / tap

            ybus[from_bus, from_bus] += y_ff
            ybus[to_bus, to_bus] += y_tt
            ybus[from_bus, to_bus] += y_ft
            ybus[to_bus, from_bus] += y_tf

        self.ybus = ybus

        self.non_slack_buses = np.array(
            [bus for bus in range(self.num_buses) if bus != self.slack_bus],
            dtype=int,
        )

        self.ydd = ybus[np.ix_(self.non_slack_buses, self.non_slack_buses)]
        self.yds = ybus[np.ix_(self.non_slack_buses, [self.slack_bus])]

    def _precompute_tensor_matrices(self):
        """
        Precompute fixed-point update matrices.

        Fixed-point update:

            V^{k+1} = K @ conj(S / V^k) + L

        where:
            K = -Ydd^{-1}
            L = -Ydd^{-1} @ Yds @ Vs
        """

        self.K = -np.linalg.inv(self.ydd)

        # Slack voltage is fixed at 1∠0 p.u.
        self.L = -(np.linalg.inv(self.ydd) @ self.yds).reshape(-1, 1)

    # ============================================================
    # Tensor power-flow interface
    # ============================================================

    def run_pf(
        self,
        active_power,
        reactive_power=None,
        flat_start=True,
        start_value=None,
        tolerance=None,
    ):
        """
        Run batched fixed-point power flow.

        Parameters
        ----------
        active_power : np.ndarray
            Net active demand excluding the slack bus, in kW.

        reactive_power : np.ndarray or None
            Net reactive demand excluding the slack bus, in kvar.

        flat_start : bool
            If True, initialize all non-slack voltages at 1∠0 p.u.

        start_value : np.ndarray or None
            Optional initial complex voltage array.

        tolerance : float or None
            Solver convergence tolerance.

        Returns
        -------
        dict
            Power-flow solution dictionary.
        """

        start_time = perf_counter()

        p = self._prepare_power_tensor(active_power, "active_power")

        if reactive_power is None:
            q = np.zeros_like(p)
        else:
            q = self._prepare_power_tensor(reactive_power, "reactive_power")

        if p.shape != q.shape:
            raise ValueError("active_power and reactive_power must have the same shape.")

        batch_size = p.shape[0]
        n_non_slack = self.num_buses - 1

        if p.shape[1] != n_non_slack:
            raise ValueError(
                "Power tensors must have {} columns, got {}.".format(
                    n_non_slack,
                    p.shape[1],
                )
            )

        tol = self.tolerance if tolerance is None else float(tolerance)

        s_pu = (p + 1j * q) / self.s_base_kva

        if flat_start:
            v0 = np.ones((batch_size, n_non_slack), dtype=np.complex128)
        elif start_value is not None:
            v0 = np.asarray(start_value, dtype=np.complex128)

            if v0.shape != (batch_size, n_non_slack):
                raise ValueError(
                    "start_value must have shape {}, got {}.".format(
                        (batch_size, n_non_slack),
                        v0.shape,
                    )
                )
        else:
            v0 = np.ones((batch_size, n_non_slack), dtype=np.complex128)

        v, iterations, converged = self._power_flow_tensor_constant_power(
            K=self.K,
            L=self.L,
            S=s_pu,
            v0=v0,
            iterations=self.max_iterations,
            tolerance=tol,
        )

        end_time = perf_counter()

        return {
            "v": v,
            "convergence": converged,
            "iterations": iterations,
            "time_algorithm": end_time - start_time,
        }

    def _power_flow_tensor_constant_power(
        self,
        K,
        L,
        S,
        v0,
        iterations,
        tolerance,
    ):
        """
        Batched constant-power fixed-point solver.
        """

        iteration = 0
        error = np.inf
        converged = False

        s_t = S.T
        v_t = v0.T

        while iteration < iterations and error >= tolerance:
            v_safe = np.where(np.abs(v_t) < 1e-9, 1e-9 + 0.0j, v_t)

            current_injection = np.conj(s_t / v_safe)
            v_next_t = K @ current_injection + L

            error = np.max(np.abs(v_next_t - v_t))

            v_t = v_next_t
            iteration += 1

            if error < tolerance:
                converged = True
                break

        return v_t.T, iteration, converged

    # ============================================================
    # Environment wrapper
    # ============================================================

    def run_power_flow(
        self,
        load_kw=None,
        load_kvar=None,
        pv_kw=None,
        dess_power_kw=None,
    ):
        """
        Run one power-flow calculation using full bus-level vectors.

        Parameters
        ----------
        load_kw : np.ndarray or None
            Bus-level active load vector.

        load_kvar : np.ndarray or None
            Bus-level reactive load vector.

        pv_kw : np.ndarray or None
            Bus-level photovoltaic generation vector.

        dess_power_kw : np.ndarray or None
            Bus-level DESS active-power vector.

        Returns
        -------
        dict
            Power-flow result dictionary.
        """

        load_kw = self._prepare_bus_vector(
            value=load_kw,
            default=self.grid.get_base_load_kw(),
            name="load_kw",
        )

        load_kvar = self._prepare_bus_vector(
            value=load_kvar,
            default=self.grid.get_base_load_kvar(),
            name="load_kvar",
        )

        pv_kw = self._prepare_bus_vector(
            value=pv_kw,
            default=self.grid.get_base_pv_kw(),
            name="pv_kw",
        )

        dess_power_kw = self._prepare_bus_vector(
            value=dess_power_kw,
            default=np.zeros(self.num_buses, dtype=np.float64),
            name="dess_power_kw",
        )

        net_load_kw = load_kw - pv_kw - dess_power_kw
        net_load_kvar = load_kvar

        p_non_slack = net_load_kw[self.non_slack_buses]
        q_non_slack = net_load_kvar[self.non_slack_buses]

        solution = self.run_pf(
            active_power=p_non_slack.reshape(1, -1),
            reactive_power=q_non_slack.reshape(1, -1),
            flat_start=True,
        )

        v_non_slack = solution["v"][0]

        voltage_complex = np.ones(self.num_buses, dtype=np.complex128)
        voltage_complex[self.slack_bus] = 1.0 + 0.0j
        voltage_complex[self.non_slack_buses] = v_non_slack

        voltage_pu = np.abs(voltage_complex).astype(np.float64)

        slack_power = self._compute_slack_power(voltage_complex)

        branch_results = self._compute_branch_quantities(voltage_complex)

        voltage_violation = self._compute_voltage_violation(voltage_pu)
        line_current_violation = self._compute_line_current_violation(
            branch_results["line_current_pu"]
        )

        max_voltage_violation = float(np.max(voltage_violation))
        max_line_current_violation = float(np.max(line_current_violation))

        feasible = bool(
            solution["convergence"]
            and max_voltage_violation <= 1e-9
            and max_line_current_violation <= 1e-9
        )

        return {
            "voltage_pu": voltage_pu.astype(np.float32),
            "voltage_complex": voltage_complex,
            "grid_import_kw": slack_power["grid_import_kw"],
            "grid_import_kvar": slack_power["grid_import_kvar"],
            "slack_power_pu": slack_power["slack_power_pu"],
            "line_current_pu": branch_results["line_current_pu"].astype(np.float32),
            "line_current_a": branch_results["line_current_a"].astype(np.float32),
            "line_active_power_kw": branch_results["line_active_power_kw"].astype(
                np.float32
            ),
            "line_reactive_power_kvar": branch_results[
                "line_reactive_power_kvar"
            ].astype(np.float32),
            "voltage_violation": voltage_violation.astype(np.float32),
            "line_current_violation": line_current_violation.astype(np.float32),
            "max_voltage_violation": max_voltage_violation,
            "max_line_current_violation": max_line_current_violation,
            "feasible": feasible,
            "converged": bool(solution["convergence"]),
            "iterations": int(solution["iterations"]),
            "time_algorithm": float(solution["time_algorithm"]),
        }

    # ============================================================
    # Result calculations
    # ============================================================

    def _compute_slack_power(self, voltage_complex):
        """
        Compute slack-bus complex power injection.

        Positive grid_import_kw means the upstream grid supplies active power
        to the distribution network.
        """

        current_injection = self.ybus @ voltage_complex

        s_slack_pu = voltage_complex[self.slack_bus] * np.conj(
            current_injection[self.slack_bus]
        )

        grid_import_kw = float(np.real(s_slack_pu) * self.s_base_kva)
        grid_import_kvar = float(np.imag(s_slack_pu) * self.s_base_kva)

        return {
            "grid_import_kw": grid_import_kw,
            "grid_import_kvar": grid_import_kvar,
            "slack_power_pu": s_slack_pu,
        }

    def _compute_branch_quantities(self, voltage_complex):
        """
        Compute line currents and branch complex powers.

        Returns
        -------
        dict
            Branch quantities for each in-service physical line.
        """

        line_current_pu = []
        line_current_a = []
        line_active_power_kw = []
        line_reactive_power_kvar = []

        for _, row in self.in_service_line_info.iterrows():
            from_bus = int(row["FROM"]) - 1
            to_bus = int(row["TO"]) - 1

            r_ohm = float(row["R"])
            x_ohm = float(row["X"])
            b_value = float(row["B"])
            tap = float(row["TAP"])

            if tap == 0.0:
                tap = 1.0

            z_pu = (r_ohm + 1j * x_ohm) / self.z_base_ohm
            y_series = 1.0 / z_pu
            b_pu = b_value * self.z_base_ohm

            v_from = voltage_complex[from_bus]
            v_to = voltage_complex[to_bus]

            current_from_to = ((v_from / tap) - v_to) * y_series
            current_from_to += 1j * b_pu / 2.0 * (v_from / tap)

            s_from_to_pu = v_from * np.conj(current_from_to)

            line_current_pu.append(abs(current_from_to))
            line_current_a.append(abs(current_from_to) * self.i_base_a)
            line_active_power_kw.append(np.real(s_from_to_pu) * self.s_base_kva)
            line_reactive_power_kvar.append(np.imag(s_from_to_pu) * self.s_base_kva)

        return {
            "line_current_pu": np.asarray(line_current_pu, dtype=np.float64),
            "line_current_a": np.asarray(line_current_a, dtype=np.float64),
            "line_active_power_kw": np.asarray(line_active_power_kw, dtype=np.float64),
            "line_reactive_power_kvar": np.asarray(
                line_reactive_power_kvar,
                dtype=np.float64,
            ),
        }

    def _compute_voltage_violation(self, voltage_pu):
        """
        Compute voltage-limit violation at each bus.
        """

        voltage_pu = np.asarray(voltage_pu, dtype=np.float64)

        below_limit = np.maximum(0.0, self.v_min - voltage_pu)
        above_limit = np.maximum(0.0, voltage_pu - self.v_max)

        return below_limit + above_limit

    def _compute_line_current_violation(self, line_current_pu):
        """
        Compute line-current limit violation for each in-service line.
        """

        line_current_pu = np.asarray(line_current_pu, dtype=np.float64)

        try:
            limits = self.grid.get_line_current_limits_pu()
        except ValueError:
            limits = None

        if limits is None:
            return np.zeros_like(line_current_pu, dtype=np.float64)

        limits = np.asarray(limits, dtype=np.float64)

        if limits.shape != line_current_pu.shape:
            raise ValueError(
                "Line-current limit shape {} does not match current shape {}.".format(
                    limits.shape,
                    line_current_pu.shape,
                )
            )

        return np.maximum(0.0, line_current_pu - limits)

    # ============================================================
    # Baseline line-current calibration
    # ============================================================

    def calibrate_line_current_limits_from_baseline(
        self,
        time_series,
        indices,
        rho=None,
    ):
        """
        Calibrate line-current limits from no-control baseline operation.

        Parameters
        ----------
        time_series : dict
            Processed time-series dictionary from data_loader.py.

        indices : list[int]
            Row indices used for baseline calibration.

        rho : float or None
            Security margin. If None, the value is read from config.

        Returns
        -------
        np.ndarray
            Calibrated line-current limits in per-unit.
        """

        if rho is None:
            rho = float(
                self.config["network"].get("line_current_security_margin_rho", 1.20)
            )

        if rho <= 1.0:
            raise ValueError("rho must be greater than 1.0.")

        if len(indices) == 0:
            raise ValueError("Baseline calibration requires at least one index.")

        from utility.data_loader import select_time_series_row

        max_currents = None

        for idx in indices:
            row = select_time_series_row(time_series, idx)

            result = self.run_power_flow(
                load_kw=row["load_kw"],
                load_kvar=self.grid.get_base_load_kvar(),
                pv_kw=row["pv_kw"],
                dess_power_kw=np.zeros(self.num_buses, dtype=np.float64),
            )

            currents = result["line_current_pu"].astype(np.float64)

            if max_currents is None:
                max_currents = currents.copy()
            else:
                max_currents = np.maximum(max_currents, currents)

        limits = rho * max_currents

        self.grid.set_line_current_limits_pu(limits)

        return limits

    # ============================================================
    # Shape utilities
    # ============================================================

    def _prepare_power_tensor(self, value, name):
        """
        Convert active/reactive power input into a 2D tensor.
        """

        array = np.asarray(value, dtype=np.float64)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                "{} must be a 1D or 2D array, got shape {}.".format(
                    name,
                    array.shape,
                )
            )

        if not np.all(np.isfinite(array)):
            raise ValueError("{} contains non-finite values.".format(name))

        return array

    def _prepare_bus_vector(self, value, default, name):
        """
        Convert full bus-level input into a validated vector.
        """

        if value is None:
            array = np.asarray(default, dtype=np.float64)
        else:
            array = np.asarray(value, dtype=np.float64)

        array = array.reshape(-1)

        if array.shape != (self.num_buses,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    self.num_buses,
                    array.shape,
                )
            )

        if not np.all(np.isfinite(array)):
            raise ValueError("{} contains non-finite values.".format(name))

        return array


def build_power_flow_solver(grid, config):
    """
    Build a PowerFlowSolver object.

    Parameters
    ----------
    grid : DistributionGrid
        Static grid representation.

    config : dict
        Configuration dictionary.

    Returns
    -------
    PowerFlowSolver
        Power-flow solver instance.
    """

    return PowerFlowSolver(grid=grid, config=config)