"""
Graph construction utilities for the IEEE 33-bus graph-aware cooperative
multi-agent reinforcement learning framework for distributed energy storage
system coordination.

This module builds graph-structured inputs for the GNN encoder.

The node feature matrix is constructed according to the feature list defined
in the configuration file. The current default feature vector is:

    [time, price, load_kw, pv_kw, soc_masked, voltage_pu, dess_indicator]

IMPORTANT:
    Normalization here affects only graph observations used by the neural
    network. It does not change the physical values used by power flow,
    battery dynamics, rewards, or feasibility checks.
"""

import numpy as np

from utility.normalization import ObservationNormalizer


class GraphBuilder:
    """
    Builder for graph-structured GNN inputs.
    """

    def __init__(self, grid, config):
        self.grid = grid
        self.config = config

        self.num_buses = int(config["network"]["num_buses"])
        self.node_features = list(config["graph"]["node_features"])
        self.node_feature_dim = int(config["graph"]["node_feature_dim"])

        self.edge_features = list(config["graph"].get("edge_features", []))
        self.edge_feature_dim = int(config["graph"]["edge_feature_dim"])

        self.v_ref = float(config["network"]["v_ref"])

        self.voltage_feature_mode = config["graph"].get(
            "voltage_feature_mode",
            "all_buses",
        )
        self.soc_feature_mode = config["graph"].get(
            "soc_feature_mode",
            "dess_only_masked",
        )

        self.dess_buses = grid.get_dess_buses()
        self.dess_indicator = grid.get_dess_indicator().astype(np.float32)

        self.normalize_observations = bool(
            config.get("normalization", {}).get("normalize_observations", True)
        )

        self.normalizer = self._build_normalizer()

        self._validate_config()

    # ============================================================
    # Normalization
    # ============================================================

    def _build_normalizer(self):
        """
        Build observation normalizer from config.

        If selected values are None in config, infer safe defaults from
        static grid data.
        """

        norm_cfg = self.config.get("normalization", {})

        total_base_load_kw = max(float(np.sum(self.grid.get_base_load_kw())), 1.0)
        total_base_pv_kw = max(float(np.sum(self.grid.get_base_pv_kw())), 1.0)

        max_time_index = norm_cfg.get(
            "max_time_index",
            float(self.config["time"]["episode_length"] - 1),
        )

        max_price = norm_cfg.get("max_price", 300.0)

        max_load_kw = norm_cfg.get("max_load_kw", None)
        if max_load_kw is None:
            max_load_kw = total_base_load_kw

        max_pv_kw = norm_cfg.get("max_pv_kw", None)
        if max_pv_kw is None:
            max_pv_kw = total_base_pv_kw

        max_grid_import_kw = norm_cfg.get("max_grid_import_kw", None)
        if max_grid_import_kw is None:
            max_grid_import_kw = total_base_load_kw

        max_voltage_deviation = norm_cfg.get(
            "max_voltage_deviation",
            float(self.num_buses) * 0.10,
        )

        max_line_current_pu = norm_cfg.get("max_line_current_pu", 5.0)

        return ObservationNormalizer(
            max_time_index=float(max_time_index),
            max_price=float(max_price),
            max_load_kw=max(float(max_load_kw), 1.0),
            max_pv_kw=max(float(max_pv_kw), 1.0),
            max_grid_import_kw=max(float(max_grid_import_kw), 1.0),
            max_voltage_deviation=max(float(max_voltage_deviation), 1e-6),
            max_line_current_pu=max(float(max_line_current_pu), 1e-6),
        )

    # ============================================================
    # Validation
    # ============================================================

    def _validate_config(self):
        """
        Validate graph-related configuration settings.
        """

        if len(self.node_features) != self.node_feature_dim:
            raise ValueError(
                "node_feature_dim must match the number of node_features. "
                "Expected {}, got {}.".format(
                    len(self.node_features),
                    self.node_feature_dim,
                )
            )

        supported_node_features = {
            "time",
            "price",
            "load_kw",
            "pv_kw",
            "soc",
            "soc_masked",
            "voltage_pu",
            "voltage_masked",
            "dess_indicator",
        }

        unsupported = [
            feature
            for feature in self.node_features
            if feature not in supported_node_features
        ]

        if unsupported:
            raise ValueError(
                "Unsupported node feature(s): {}. Supported features are: {}".format(
                    unsupported,
                    sorted(supported_node_features),
                )
            )

        supported_voltage_modes = {"all_buses", "dess_only_masked"}

        if self.voltage_feature_mode not in supported_voltage_modes:
            raise ValueError(
                "Unsupported voltage_feature_mode '{}'. Supported modes are: {}.".format(
                    self.voltage_feature_mode,
                    sorted(supported_voltage_modes),
                )
            )

        supported_soc_modes = {"dess_only_masked"}

        if self.soc_feature_mode not in supported_soc_modes:
            raise ValueError(
                "Unsupported soc_feature_mode '{}'. Supported modes are: {}.".format(
                    self.soc_feature_mode,
                    sorted(supported_soc_modes),
                )
            )

    # ============================================================
    # Node feature construction
    # ============================================================

    def build_node_features(
        self,
        time_index,
        price,
        load_kw=None,
        pv_kw=None,
        soc_by_bus=None,
        voltage_pu=None,
    ):
        """
        Build node feature matrix X_t.

        Returns
        -------
        np.ndarray
            Node feature matrix with shape (num_buses, node_feature_dim).
        """

        feature_data = self._build_feature_dictionary(
            time_index=time_index,
            price=price,
            load_kw=load_kw,
            pv_kw=pv_kw,
            soc_by_bus=soc_by_bus,
            voltage_pu=voltage_pu,
        )

        feature_columns = []

        for feature_name in self.node_features:
            feature_columns.append(feature_data[feature_name])

        x = np.stack(feature_columns, axis=1).astype(np.float32)

        expected_shape = (self.num_buses, self.node_feature_dim)

        if x.shape != expected_shape:
            raise ValueError(
                "Expected node feature shape {}, got {}.".format(
                    expected_shape,
                    x.shape,
                )
            )

        if not np.all(np.isfinite(x)):
            raise ValueError("Graph node feature matrix contains non-finite values.")

        return x

    def _build_feature_dictionary(
        self,
        time_index,
        price,
        load_kw=None,
        pv_kw=None,
        soc_by_bus=None,
        voltage_pu=None,
    ):
        """
        Build all supported node-feature vectors.

        Returns
        -------
        dict
            Mapping from feature name to bus-level feature vector.
        """

        load_kw = self._prepare_vector(
            value=load_kw,
            default=self.grid.get_base_load_kw(),
            name="load_kw",
        )

        pv_kw = self._prepare_vector(
            value=pv_kw,
            default=self.grid.get_base_pv_kw(),
            name="pv_kw",
        )

        soc_by_bus = self._prepare_vector(
            value=soc_by_bus,
            default=np.zeros(self.num_buses, dtype=np.float32),
            name="soc_by_bus",
        )

        voltage_pu = self._prepare_vector(
            value=voltage_pu,
            default=np.ones(self.num_buses, dtype=np.float32) * self.v_ref,
            name="voltage_pu",
        )

        if self.normalize_observations:
            time_value = self.normalizer.normalize_time_index(float(time_index))
            price_value = self.normalizer.normalize_price(float(price))
            load_feature = self.normalizer.normalize_load_kw(load_kw).astype(np.float32)
            pv_feature = self.normalizer.normalize_pv_kw(pv_kw).astype(np.float32)
            soc_feature = self.normalizer.normalize_soc(soc_by_bus).astype(np.float32)
            voltage_feature_raw = self.normalizer.normalize_voltage_pu(voltage_pu).astype(
                np.float32
            )
        else:
            time_value = float(time_index)
            price_value = float(price)
            load_feature = load_kw.astype(np.float32)
            pv_feature = pv_kw.astype(np.float32)
            soc_feature = soc_by_bus.astype(np.float32)
            voltage_feature_raw = voltage_pu.astype(np.float32)

        time_vec = np.ones(self.num_buses, dtype=np.float32) * float(time_value)
        price_vec = np.ones(self.num_buses, dtype=np.float32) * float(price_value)

        soc_masked = self._mask_to_dess_buses(soc_feature)

        if self.voltage_feature_mode == "all_buses":
            voltage_feature = voltage_feature_raw
        else:
            voltage_feature = self._mask_to_dess_buses(voltage_feature_raw)

        voltage_masked = self._mask_to_dess_buses(voltage_feature_raw)

        return {
            "time": time_vec.astype(np.float32),
            "price": price_vec.astype(np.float32),
            "load_kw": load_feature.astype(np.float32),
            "pv_kw": pv_feature.astype(np.float32),
            "soc": soc_feature.astype(np.float32),
            "soc_masked": soc_masked.astype(np.float32),
            "voltage_pu": voltage_feature.astype(np.float32),
            "voltage_masked": voltage_masked.astype(np.float32),
            "dess_indicator": self.dess_indicator.astype(np.float32),
        }

    def _mask_to_dess_buses(self, vector):
        """
        Retain values only at DESS buses and set all other buses to zero.
        """

        vector = self._prepare_vector(
            value=vector,
            default=np.zeros(self.num_buses, dtype=np.float32),
            name="vector",
        )

        masked = np.zeros(self.num_buses, dtype=np.float32)
        masked[self.dess_buses] = vector[self.dess_buses]

        return masked

    # ============================================================
    # Graph construction
    # ============================================================

    def build_graph(
        self,
        time_index,
        price,
        load_kw=None,
        pv_kw=None,
        soc_by_bus=None,
        voltage_pu=None,
    ):
        """
        Build the full graph input dictionary.
        """

        x = self.build_node_features(
            time_index=time_index,
            price=price,
            load_kw=load_kw,
            pv_kw=pv_kw,
            soc_by_bus=soc_by_bus,
            voltage_pu=voltage_pu,
        )

        return {
            "x": x.astype(np.float32),
            "edge_index": self.grid.edge_index.astype(np.int64),
            "edge_attr": self.grid.edge_attr.astype(np.float32),
            "adjacency": self.grid.adjacency_matrix.astype(np.float32),
            "dess_buses": np.asarray(self.dess_buses, dtype=np.int64),
            "dess_indicator": self.dess_indicator.astype(np.float32),
        }

    # ============================================================
    # DESS SOC utility
    # ============================================================

    def build_soc_vector(self, dess_soc_values):
        """
        Convert DESS SOC values into a full bus-level SOC vector.
        """

        dess_soc_values = np.asarray(dess_soc_values, dtype=np.float32).reshape(-1)

        if len(dess_soc_values) != len(self.dess_buses):
            raise ValueError(
                "Expected {} DESS SOC values, got {}.".format(
                    len(self.dess_buses),
                    len(dess_soc_values),
                )
            )

        soc_by_bus = np.zeros(self.num_buses, dtype=np.float32)

        for idx, bus in enumerate(self.dess_buses):
            soc_by_bus[int(bus)] = float(dess_soc_values[idx])

        return soc_by_bus

    # ============================================================
    # Validation utility
    # ============================================================

    def _prepare_vector(self, value, default, name):
        """
        Validate or create a bus-level vector.
        """

        if value is None:
            arr = np.asarray(default, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)

        arr = arr.reshape(-1)

        if arr.shape != (self.num_buses,):
            raise ValueError(
                "{} must have shape ({},), got {}.".format(
                    name,
                    self.num_buses,
                    arr.shape,
                )
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError("{} contains non-finite values.".format(name))

        return arr

    # ============================================================
    # Metadata
    # ============================================================

    def get_feature_names(self):
        """
        Return node feature names in column order.
        """

        return list(self.node_features)

    def get_summary(self):
        """
        Return graph-builder metadata.
        """

        norm_cfg = self.config.get("normalization", {})

        return {
            "num_buses": self.num_buses,
            "node_feature_dim": self.node_feature_dim,
            "node_features": self.get_feature_names(),
            "edge_feature_dim": self.edge_feature_dim,
            "edge_features": list(self.edge_features),
            "num_physical_lines": self.grid.get_num_lines(),
            "num_directed_edges": int(self.grid.edge_index.shape[1]),
            "dess_buses": list(self.dess_buses),
            "voltage_feature_mode": self.voltage_feature_mode,
            "soc_feature_mode": self.soc_feature_mode,
            "normalize_observations": self.normalize_observations,
            "normalization": {
                "max_time_index": norm_cfg.get("max_time_index", None),
                "max_price": norm_cfg.get("max_price", None),
                "max_load_kw": norm_cfg.get("max_load_kw", None),
                "max_pv_kw": norm_cfg.get("max_pv_kw", None),
                "max_grid_import_kw": norm_cfg.get("max_grid_import_kw", None),
                "max_voltage_deviation": norm_cfg.get("max_voltage_deviation", None),
                "max_line_current_pu": norm_cfg.get("max_line_current_pu", None),
            },
        }


def build_graph_builder(grid, config):
    """
    Build a GraphBuilder object.
    """

    return GraphBuilder(grid=grid, config=config)