"""
Grid representation utilities for the IEEE 33-bus graph-aware cooperative
multi-agent reinforcement learning framework for distributed energy storage
system coordination.

This module stores and validates the static distribution-network structure,
including:

    - bus data,
    - line data,
    - DESS placement,
    - adjacency matrix,
    - directed edge index,
    - edge attributes,
    - DESS indicator vector,
    - optional line-current limit data.

This module does not solve power flow. Power-flow calculations are handled
by utility/power_flow.py.
"""

import numpy as np


class DistributionGrid:
    """
    Static distribution-grid representation.

    The grid is represented as an undirected physical network for electrical
    connectivity and as a directed graph for graph neural network inputs.
    For each physical line i--j, two directed edges are created:

        i -> j
        j -> i

    All bus indices are assumed to use zero-based indexing.
    """

    def __init__(self, bus_df, line_df, dess_buses, config):
        self.bus_df = bus_df.copy()
        self.line_df = line_df.copy()
        self.dess_buses = list(dess_buses)
        self.config = config

        self.num_buses = int(config["network"]["num_buses"])
        self.slack_bus = int(config["network"]["slack_bus"])

        self.s_base_kva = float(config["network"]["s_base_kva"])
        self.v_base_kv = float(config["network"]["v_base_kv"])

        self.v_ref = float(config["network"]["v_ref"])
        self.v_min = float(config["network"]["v_min"])
        self.v_max = float(config["network"]["v_max"])

        self.line_current_limit_mode = config["network"].get(
            "line_current_limit_mode",
            "baseline_margin",
        )
        self.global_line_current_limit_pu = float(
            config["network"].get("global_line_current_limit_pu", np.inf)
        )
        self.use_line_specific_current_limits = bool(
            config["network"].get("use_line_specific_current_limits", True)
        )

        self._validate()
        self._prepare_line_status()

        self.in_service_line_df = self._get_in_service_lines()

        self.adjacency_matrix = self._build_adjacency_matrix()
        self.edge_index = self._build_edge_index()
        self.edge_attr = self._build_edge_attr()

        self.dess_indicator = self._build_dess_indicator()

        self.line_current_limits_pu = self._initialize_line_current_limits()

    # ============================================================
    # Validation
    # ============================================================

    def _validate(self):
        """
        Validate static grid data.
        """

        if len(self.bus_df) != self.num_buses:
            raise ValueError(
                "Expected {} buses, found {}.".format(
                    self.num_buses,
                    len(self.bus_df),
                )
            )

        if self.slack_bus < 0 or self.slack_bus >= self.num_buses:
            raise ValueError("Invalid slack bus index.")

        required_bus_cols = ["bus", "load_kw", "load_kvar", "pv_kw"]
        for col in required_bus_cols:
            if col not in self.bus_df.columns:
                raise ValueError("Missing bus column: {}".format(col))

        required_line_cols = ["from_bus", "to_bus", "r_ohm", "x_ohm"]
        for col in required_line_cols:
            if col not in self.line_df.columns:
                raise ValueError("Missing line column: {}".format(col))

        bus_indices = set(self.bus_df["bus"].astype(int).tolist())
        expected_bus_indices = set(range(self.num_buses))

        if bus_indices != expected_bus_indices:
            raise ValueError(
                "Bus indices must be exactly 0 to {}. Found: {}".format(
                    self.num_buses - 1,
                    sorted(bus_indices),
                )
            )

        if len(set(self.dess_buses)) != len(self.dess_buses):
            raise ValueError("DESS bus list contains duplicate entries.")

        for bus in self.dess_buses:
            bus = int(bus)

            if bus < 0 or bus >= self.num_buses:
                raise ValueError("Invalid DESS bus index: {}".format(bus))

            if bus == self.slack_bus:
                raise ValueError("DESS cannot be placed at the slack bus.")

        for line_idx, row in self.line_df.iterrows():
            i = int(row["from_bus"])
            j = int(row["to_bus"])

            if i < 0 or i >= self.num_buses:
                raise ValueError(
                    "Invalid from_bus index in line {}: {}".format(line_idx, i)
                )

            if j < 0 or j >= self.num_buses:
                raise ValueError(
                    "Invalid to_bus index in line {}: {}".format(line_idx, j)
                )

            if i == j:
                raise ValueError(
                    "Line {} has identical from_bus and to_bus.".format(line_idx)
                )

            if float(row["r_ohm"]) < 0.0:
                raise ValueError(
                    "Line resistance cannot be negative in line {}.".format(line_idx)
                )

            if float(row["x_ohm"]) < 0.0:
                raise ValueError(
                    "Line reactance cannot be negative in line {}.".format(line_idx)
                )

    def _prepare_line_status(self):
        """
        Ensure that the line dataframe contains an in-service status column.
        """

        if "status" not in self.line_df.columns:
            self.line_df["status"] = 1.0

        self.line_df["status"] = self.line_df["status"].astype(float)

        invalid_status = ~self.line_df["status"].isin([0.0, 1.0])

        if invalid_status.any():
            bad_indices = self.line_df.index[invalid_status].tolist()
            raise ValueError(
                "Line status must be either 0 or 1. Invalid rows: {}".format(
                    bad_indices
                )
            )

    def _get_in_service_lines(self):
        """
        Return only in-service physical lines.
        """

        return self.line_df[self.line_df["status"] == 1.0].reset_index(drop=True)

    # ============================================================
    # Graph construction
    # ============================================================

    def _build_adjacency_matrix(self):
        """
        Build an undirected adjacency matrix from in-service line data.

        Returns
        -------
        np.ndarray
            Adjacency matrix with shape (num_buses, num_buses).
        """

        adjacency = np.zeros((self.num_buses, self.num_buses), dtype=np.float32)

        for _, row in self.in_service_line_df.iterrows():
            i = int(row["from_bus"])
            j = int(row["to_bus"])

            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

        return adjacency

    def _build_edge_index(self):
        """
        Build directed edge index from in-service physical lines.

        Returns
        -------
        np.ndarray
            Directed edge index with shape (2, 2 * num_in_service_lines).
        """

        edges = []

        for _, row in self.in_service_line_df.iterrows():
            i = int(row["from_bus"])
            j = int(row["to_bus"])

            edges.append([i, j])
            edges.append([j, i])

        if not edges:
            return np.empty((2, 0), dtype=np.int64)

        return np.array(edges, dtype=np.int64).T

    def _build_edge_attr(self):
        """
        Build directed edge attributes.

        The default edge feature vector is:

            [r_ohm, x_ohm]

        Since the graph uses directed edges, each physical line contributes
        two identical edge-attribute rows.

        Returns
        -------
        np.ndarray
            Edge-attribute matrix with shape (2 * num_in_service_lines, 2).
        """

        attrs = []

        for _, row in self.in_service_line_df.iterrows():
            r_ohm = float(row["r_ohm"])
            x_ohm = float(row["x_ohm"])

            attrs.append([r_ohm, x_ohm])
            attrs.append([r_ohm, x_ohm])

        if not attrs:
            return np.empty((0, 2), dtype=np.float32)

        return np.array(attrs, dtype=np.float32)

    def _build_dess_indicator(self):
        """
        Build binary DESS indicator vector.

        Returns
        -------
        np.ndarray
            Binary vector with shape (num_buses,).
        """

        indicator = np.zeros(self.num_buses, dtype=np.float32)
        indicator[self.dess_buses] = 1.0

        return indicator

    # ============================================================
    # Line-current limits
    # ============================================================

    def _initialize_line_current_limits(self):
        """
        Initialize line-current limits.

        If physical or per-unit current limits are available in line_data.csv,
        they are converted or loaded here. If no ratings are available and the
        selected mode is baseline_margin, limits are initialized as None and
        should later be assigned from a baseline power-flow scan.

        Returns
        -------
        np.ndarray or None
            Line-current limits for physical in-service lines.
        """

        num_lines = self.get_num_lines()

        if num_lines == 0:
            return np.array([], dtype=np.float64)

        if "current_limit_pu" in self.in_service_line_df.columns:
            limits = self.in_service_line_df["current_limit_pu"].to_numpy(
                dtype=np.float64
            )

            if np.any(limits <= 0.0):
                raise ValueError("current_limit_pu values must be positive.")

            return limits

        if "max_current_a" in self.in_service_line_df.columns:
            current_a = self.in_service_line_df["max_current_a"].to_numpy(
                dtype=np.float64
            )
            return self._physical_current_to_pu(current_a)

        if "max_i_a" in self.in_service_line_df.columns:
            current_a = self.in_service_line_df["max_i_a"].to_numpy(dtype=np.float64)
            return self._physical_current_to_pu(current_a)

        if "max_i_ka" in self.in_service_line_df.columns:
            current_a = (
                self.in_service_line_df["max_i_ka"].to_numpy(dtype=np.float64)
                * 1000.0
            )
            return self._physical_current_to_pu(current_a)

        if self.line_current_limit_mode == "global_pu":
            return np.ones(num_lines, dtype=np.float64) * self.global_line_current_limit_pu

        if self.line_current_limit_mode == "baseline_margin":
            return None

        return None

    def _physical_current_to_pu(self, current_a):
        """
        Convert physical three-phase line-current limits from amperes to per-unit.

        Base current:
            I_base = S_base / (sqrt(3) * V_base)

        where S_base is in kVA and V_base is in kV, giving I_base in amperes.

        Parameters
        ----------
        current_a : np.ndarray
            Physical current limits in amperes.

        Returns
        -------
        np.ndarray
            Current limits in per-unit.
        """

        current_a = np.asarray(current_a, dtype=np.float64)

        if np.any(current_a <= 0.0):
            raise ValueError("Physical current limits must be positive.")

        i_base_a = self.get_base_current_a()

        return current_a / i_base_a

    def set_line_current_limits_pu(self, line_current_limits_pu):
        """
        Assign line-current limits in per-unit.

        This method is used when line-current limits are calibrated from
        no-control baseline operation.

        Parameters
        ----------
        line_current_limits_pu : array-like
            Per-unit current limits for each in-service physical line.
        """

        limits = np.asarray(line_current_limits_pu, dtype=np.float64).reshape(-1)

        if limits.shape[0] != self.get_num_lines():
            raise ValueError(
                "Expected {} line-current limits, got {}.".format(
                    self.get_num_lines(),
                    limits.shape[0],
                )
            )

        if np.any(limits <= 0.0):
            raise ValueError("Line-current limits must be positive.")

        self.line_current_limits_pu = limits

    def get_line_current_limits_pu(self):
        """
        Return line-current limits in per-unit.

        Returns
        -------
        np.ndarray
            Per-unit line-current limits.

        Raises
        ------
        ValueError
            If limits have not yet been assigned.
        """

        if self.line_current_limits_pu is None:
            raise ValueError(
                "Line-current limits have not been assigned. "
                "Run a baseline-current calibration or provide ratings."
            )

        return self.line_current_limits_pu.copy()

    # ============================================================
    # Base quantities
    # ============================================================

    def get_base_current_a(self):
        """
        Return the three-phase base current in amperes.

        I_base = S_base / (sqrt(3) * V_base)

        With S_base in kVA and V_base in kV, the result is in amperes.
        """

        return self.s_base_kva / (np.sqrt(3.0) * self.v_base_kv)

    def get_z_base_ohm(self):
        """
        Return the base impedance in ohms.

        Z_base = V_base^2 * 1000 / S_base
        """

        return (self.v_base_kv ** 2 * 1000.0) / self.s_base_kva

    # ============================================================
    # Static profile accessors
    # ============================================================

    def get_base_load_kw(self):
        """
        Return base active load at each bus.

        Returns
        -------
        np.ndarray
            Active load vector with shape (num_buses,).
        """

        return self.bus_df["load_kw"].to_numpy(dtype=np.float32)

    def get_base_load_kvar(self):
        """
        Return base reactive load at each bus.

        Returns
        -------
        np.ndarray
            Reactive load vector with shape (num_buses,).
        """

        return self.bus_df["load_kvar"].to_numpy(dtype=np.float32)

    def get_base_pv_kw(self):
        """
        Return base photovoltaic generation at each bus.

        Returns
        -------
        np.ndarray
            Photovoltaic generation vector with shape (num_buses,).
        """

        return self.bus_df["pv_kw"].to_numpy(dtype=np.float32)

    # ============================================================
    # DESS accessors
    # ============================================================

    def get_dess_indicator(self):
        """
        Return binary DESS indicator for all buses.

        Returns
        -------
        np.ndarray
            Binary indicator vector with shape (num_buses,).
        """

        return self.dess_indicator.copy()

    def get_dess_buses(self):
        """
        Return DESS bus indices.

        Returns
        -------
        list[int]
            DESS bus list.
        """

        return list(self.dess_buses)

    def is_dess_bus(self, bus):
        """
        Check whether a bus contains a DESS.

        Parameters
        ----------
        bus : int
            Bus index.

        Returns
        -------
        bool
            True if the bus contains a DESS.
        """

        return int(bus) in self.dess_buses

    # ============================================================
    # Line accessors
    # ============================================================

    def get_num_lines(self):
        """
        Return the number of in-service physical lines.

        Returns
        -------
        int
            Number of in-service physical lines.
        """

        return len(self.in_service_line_df)

    def get_line_endpoints(self):
        """
        Return in-service physical line endpoints.

        Returns
        -------
        np.ndarray
            Array with shape (num_lines, 2), where each row is [from_bus, to_bus].
        """

        if self.get_num_lines() == 0:
            return np.empty((0, 2), dtype=np.int64)

        return self.in_service_line_df[["from_bus", "to_bus"]].to_numpy(
            dtype=np.int64
        )

    def get_line_impedances_ohm(self):
        """
        Return in-service physical line impedances.

        Returns
        -------
        np.ndarray
            Array with shape (num_lines, 2), where each row is [r_ohm, x_ohm].
        """

        if self.get_num_lines() == 0:
            return np.empty((0, 2), dtype=np.float64)

        return self.in_service_line_df[["r_ohm", "x_ohm"]].to_numpy(
            dtype=np.float64
        )

    # ============================================================
    # Summary
    # ============================================================

    def get_summary(self):
        """
        Return a compact grid summary.

        Returns
        -------
        dict
            Grid metadata and aggregate quantities.
        """

        line_limits_available = self.line_current_limits_pu is not None

        return {
            "num_buses": self.num_buses,
            "num_lines": self.get_num_lines(),
            "slack_bus": self.slack_bus,
            "dess_buses": self.get_dess_buses(),
            "total_load_kw": float(self.bus_df["load_kw"].sum()),
            "total_load_kvar": float(self.bus_df["load_kvar"].sum()),
            "total_pv_kw": float(self.bus_df["pv_kw"].sum()),
            "v_base_kv": self.v_base_kv,
            "s_base_kva": self.s_base_kva,
            "i_base_a": float(self.get_base_current_a()),
            "z_base_ohm": float(self.get_z_base_ohm()),
            "v_min": self.v_min,
            "v_max": self.v_max,
            "line_current_limit_mode": self.line_current_limit_mode,
            "line_current_limits_available": line_limits_available,
        }


def build_grid_from_static_data(static_data, config):
    """
    Build a DistributionGrid object from loaded static data.

    Parameters
    ----------
    static_data : dict
        Output of the static data loader.

    config : dict
        Configuration dictionary.

    Returns
    -------
    DistributionGrid
        Static grid representation.
    """

    return DistributionGrid(
        bus_df=static_data["bus_data"],
        line_df=static_data["line_data"],
        dess_buses=static_data["dess_buses"],
        config=config,
    )