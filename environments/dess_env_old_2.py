"""
Distributed energy storage coordination environment for graph-aware
cooperative multi-agent reinforcement learning.

This environment combines:

    - DESS battery dynamics,
    - static distribution-network data from the provided config,
    - SMART-DS / NYISO node-level 15-minute time-series data,
    - tensor-based fixed-point power-flow evaluation,
    - graph construction for GNN-based policies,
    - normalized neural-network observations,
    - hard feasibility handling for operational constraints,
    - cooperative reward calculation.

Action convention:
    action > 0  : DESS discharge / active-power injection
    action < 0  : DESS charge / active-power absorption
    action = 0  : idle

Important design choice:
    This environment does NOT run the GNN encoder. It only returns:

        x, edge_index, edge_attr, agent_obs

    The GNN encoder should be used later inside the model/training stack:

        H_t = GNNEncoder(x, edge_index, edge_attr)
        h_i(t) = H_t[dess_bus_i]
        graph_aware_agent_obs_i = concat(agent_obs_i, h_i(t))

All accepted environment transitions are evaluated through the power-flow
model. Under hard feasibility handling, voltage and line-current limits are
enforced before the transition is accepted.

Reward convention:
    r_i(t) = w_grid  * R_grid(t)
           + w_local * R_local,i(t)
           + w_kpi   * R_KPI(t)

where w_grid, w_local, and w_kpi are defined in the provided config.
"""

import copy

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from environments.battery import Battery, BatteryConfig
from utility.data_loader import (
    load_all_data,
    select_time_series_row,
)
from utility.grid import build_grid_from_static_data
from utility.graph_builder import build_graph_builder
from utility.power_flow import build_power_flow_solver
from utility.normalization import ObservationNormalizer


class DESSEnv(gym.Env):
    """
    Graph-ready cooperative MARL environment for DESS coordination.

    The environment returns graph-structured observations but does not apply
    any neural network internally. This keeps simulation, feasibility handling,
    and learning-model code cleanly separated.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None, mode="train", seed=None):
        super(DESSEnv, self).__init__()

        if config is None:
            raise ValueError(
                "DESSEnv requires a configuration dictionary. "
                "Pass IEEE33_CONFIG, IEEE69_CONFIG, or another compatible config."
            )

        self.config = copy.deepcopy(config)
        self.mode = mode

        if self.mode not in ["train", "val", "test", "all"]:
            raise ValueError("mode must be one of: train, val, test, all.")

        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------
        # Core network and time settings
        # ------------------------------------------------------------
        self.num_buses = int(self.config["network"]["num_buses"])
        self.slack_bus = int(self.config["network"]["slack_bus"])

        self.v_ref = float(self.config["network"]["v_ref"])
        self.v_min = float(self.config["network"]["v_min"])
        self.v_max = float(self.config["network"]["v_max"])

        self.grid_export_limit_kw = float(
            self.config["network"].get("grid_export_limit_kw", 0.0)
        )

        self.episode_length = int(self.config["time"]["episode_length"])
        self.delta_t_hours = float(self.config["time"]["delta_t_hours"])

        # ------------------------------------------------------------
        # Reward and constraint settings
        # ------------------------------------------------------------
        self.reward_cfg = self.config["reward"]
        self.constraint_cfg = self.config["constraints"]

        self.constraint_handling = self.constraint_cfg.get(
            "handling",
            "hard_feasibility",
        )
        self.hard_constraint_action = self.constraint_cfg.get(
            "hard_constraint_action",
            "clip_or_correct",
        )
        self.max_correction_attempts = int(
            self.constraint_cfg.get("max_correction_attempts", 20)
        )
        self.action_correction_factor = float(
            self.constraint_cfg.get("action_correction_factor", 0.90)
        )

        # ------------------------------------------------------------
        # Data scaling
        # ------------------------------------------------------------
        data_scaling = self.config.get("data_scaling", {})
        self.load_scaling_factor = float(data_scaling.get("load_scaling_factor", 1.0))
        self.pv_scaling_factor = float(data_scaling.get("pv_scaling_factor", 1.0))
        self.price_scaling_factor = float(data_scaling.get("price_scaling_factor", 1.0))

        self.normalize_observations = bool(
            self.config.get("normalization", {}).get("normalize_observations", True)
        )

        # ------------------------------------------------------------
        # Data, grid, graph builder, and power flow solver
        # ------------------------------------------------------------
        all_data = load_all_data(
            self.config,
            require_time_series=True,
        )

        self.static_data = all_data["static"]
        self.time_series = all_data["processed_time_series"]
        self.splits = all_data["splits"]

        self.grid = build_grid_from_static_data(
            self.static_data,
            self.config,
        )

        self.normalizer = self._build_normalizer()

        self.graph_builder = build_graph_builder(
            self.grid,
            self.config,
        )

        self.power_flow = build_power_flow_solver(
            self.grid,
            self.config,
        )

        self.dess_buses = list(self.grid.get_dess_buses())
        self.num_agents = len(self.dess_buses)

        self._initialize_line_current_limits()
        self._build_batteries()

        self.episode_start_indices = self._build_episode_start_indices()

        if len(self.episode_start_indices) == 0:
            raise ValueError(
                "No valid episode start indices were found for mode '{}'.".format(
                    self.mode
                )
            )

        # ------------------------------------------------------------
        # Action and observation spaces
        # ------------------------------------------------------------
        self.action_space = spaces.Box(
            low=float(self.config["action"]["low"]),
            high=float(self.config["action"]["high"]),
            shape=(self.num_agents,),
            dtype=np.float32,
        )

        # Base local observation only. GNN embeddings are NOT included here.
        # Later, in the model stack:
        #     graph_aware_obs_i = concat(agent_obs_i, h_i)
        self.agent_obs_dim = 11

        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.num_buses,
                        int(self.config["graph"]["node_feature_dim"]),
                    ),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=self.num_buses - 1,
                    shape=self.grid.edge_index.shape,
                    dtype=np.int64,
                ),
                "edge_attr": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.grid.edge_attr.shape,
                    dtype=np.float32,
                ),
                "agent_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, self.agent_obs_dim),
                    dtype=np.float32,
                ),
            }
        )

        self.reset(seed=seed)

    # ============================================================
    # Normalization utilities
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
            float(self.episode_length - 1),
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
    # Initialization utilities
    # ============================================================

    def _initialize_line_current_limits(self):
        """
        Initialize or calibrate line-current limits.
        """

        mode = self.config["network"].get(
            "line_current_limit_mode",
            "baseline_margin",
        )

        if self.grid.line_current_limits_pu is not None:
            return

        if mode != "baseline_margin":
            return

        candidate_indices = self.splits["train_indices"]

        if len(candidate_indices) == 0:
            raise ValueError(
                "Cannot calibrate line-current limits because the training split "
                "contains no indices."
            )

        rho = float(
            self.config["network"].get("line_current_security_margin_rho", 1.20)
        )

        self._calibrate_line_current_limits_from_scaled_baseline(
            indices=candidate_indices,
            rho=rho,
        )

    def _calibrate_line_current_limits_from_scaled_baseline(self, indices, rho):
        """
        Calibrate line-current limits from no-control baseline operation.

        I_line,max = rho * max_t I_line,baseline(t)
        """

        if rho <= 1.0:
            raise ValueError("rho must be greater than 1.0.")

        max_currents = None

        for idx in indices:
            load_kw, load_kvar, pv_kw, _, _ = self._get_profiles(idx)

            result = self.power_flow.run_power_flow(
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
                dess_power_kw=np.zeros(self.num_buses, dtype=np.float64),
            )

            currents = result["line_current_pu"].astype(np.float64)

            if max_currents is None:
                max_currents = currents.copy()
            else:
                max_currents = np.maximum(max_currents, currents)

        limits = rho * max_currents
        self.grid.set_line_current_limits_pu(limits)

    def _build_batteries(self):
        """
        Create one DESS battery model for each DESS bus.
        """

        dess_cfg = self.config["dess"]

        battery_config = BatteryConfig(
            capacity_kwh=dess_cfg["capacity_kwh"],
            max_charge_kw=dess_cfg["max_charge_kw"],
            max_discharge_kw=dess_cfg["max_discharge_kw"],
            efficiency=dess_cfg.get("efficiency", 0.95),
            charge_efficiency=dess_cfg.get("charge_efficiency", 0.95),
            discharge_efficiency=dess_cfg.get("discharge_efficiency", 0.95),
            soc_min=dess_cfg["soc_min"],
            soc_max=dess_cfg["soc_max"],
            soc_init=dess_cfg["soc_init"],
        )

        self.batteries = [
            Battery(battery_config, delta_t_hours=self.delta_t_hours)
            for _ in range(self.num_agents)
        ]

    # ============================================================
    # Episode indexing
    # ============================================================

    def _build_episode_start_indices(self):
        """
        Build valid episode-start indices for the selected data split.

        For 15-minute data, an episode starts exactly at 00:00.
        """

        df = self.time_series["df"]
        datetime_col = self.time_series["datetime_col"]

        if self.mode == "train":
            candidate_indices = self.splits["train_indices"]
        elif self.mode == "val":
            candidate_indices = self.splits["val_indices"]
        elif self.mode == "test":
            candidate_indices = self.splits["test_indices"]
        else:
            candidate_indices = list(range(len(df)))

        candidate_set = set(candidate_indices)
        starts = []

        for idx in candidate_indices:
            if idx + self.episode_length - 1 >= len(df):
                continue

            dt = df.loc[idx, datetime_col]

            if dt.hour != 0 or dt.minute != 0:
                continue

            episode_indices = range(idx, idx + self.episode_length)

            if all(j in candidate_set for j in episode_indices):
                starts.append(idx)

        return starts

    # ============================================================
    # Reset
    # ============================================================

    def reset(self, seed=None, options=None):
        """
        Reset the environment at the beginning of an episode.
        """

        del options

        try:
            super().reset(seed=seed)
        except TypeError:
            pass

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_start_index = int(self.rng.choice(self.episode_start_indices))
        self.current_step = 0
        self.current_index = self.episode_start_index

        for battery in self.batteries:
            battery.reset()

        self.previous_actions = np.zeros(self.num_agents, dtype=np.float32)
        self.last_dess_power_kw = np.zeros(self.num_buses, dtype=np.float64)

        self.previous_kpis = None
        self.current_kpis = None

        load_kw, load_kvar, pv_kw, price, date_time = self._get_profiles(
            self.current_index
        )

        pv_used_kw, curtailment_by_bus, curtailment_kw = self._apply_curtailment(
            load_kw=load_kw,
            pv_kw=pv_kw,
            dess_power_kw=self.last_dess_power_kw,
        )

        pf_result = self.power_flow.run_power_flow(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_used_kw,
            dess_power_kw=self.last_dess_power_kw,
        )

        self.last_pf_result = pf_result
        self.last_curtailment_by_bus = curtailment_by_bus.copy()
        self.last_curtailment_kw = float(curtailment_kw)

        self.current_kpis = self._compute_kpis(
            pf_result=pf_result,
            price=price,
            curtailment_kw=curtailment_kw,
        )
        self.previous_kpis = dict(self.current_kpis)
        self.last_date_time = date_time

        obs = self._build_observation(
            load_kw=load_kw,
            pv_kw=pv_used_kw,
            price=price,
            voltage_pu=pf_result["voltage_pu"],
        )

        info = self._build_info()
        return obs, info

    # ============================================================
    # Observation construction
    # ============================================================

    def _build_observation(self, load_kw, pv_kw, price, voltage_pu):
        """
        Build graph-level and agent-level observations.

        Returned fields:
            x          : node feature matrix, shape [num_buses, node_feature_dim]
            edge_index : directed graph connectivity, shape [2, num_edges]
            edge_attr  : edge features, shape [num_edges, edge_feature_dim]
            agent_obs  : local per-DESS observations, shape [num_agents, 11]

        The GNN embedding is not included in agent_obs. It should be computed
        later by models/gnn_encoder.py.
        """

        soc_values = np.array(
            [battery.get_soc() for battery in self.batteries],
            dtype=np.float32,
        )

        soc_by_bus = self.graph_builder.build_soc_vector(soc_values)

        graph = self.graph_builder.build_graph(
            time_index=self.current_step,
            price=price,
            load_kw=load_kw,
            pv_kw=pv_kw,
            soc_by_bus=soc_by_bus,
            voltage_pu=voltage_pu,
        )

        agent_obs = self._build_agent_observation(
            load_kw=load_kw,
            pv_kw=pv_kw,
            price=price,
            voltage_pu=voltage_pu,
            soc_values=soc_values,
        )

        return {
            "x": graph["x"].astype(np.float32),
            "edge_index": graph["edge_index"].astype(np.int64),
            "edge_attr": graph["edge_attr"].astype(np.float32),
            "agent_obs": agent_obs.astype(np.float32),
        }

    def _build_agent_observation(self, load_kw, pv_kw, price, voltage_pu, soc_values):
        """
        Build local per-agent observations.

        Current local observation vector for each DESS agent is:

            [
                time,
                price,
                soc_i,
                voltage_i,
                load_i,
                pv_i,
                previous_action_i,
                grid_import,
                voltage_deviation,
                curtailment,
                grid_stress,
            ]

        GNN embeddings are intentionally not computed here.
        """

        agent_obs = np.zeros((self.num_agents, self.agent_obs_dim), dtype=np.float32)

        grid_import = float(self.current_kpis["grid_import_kw"])
        voltage_deviation = float(self.current_kpis["voltage_deviation"])
        curtailment = float(self.current_kpis["curtailment_kw"])
        grid_stress = float(self.current_kpis["grid_stress"])

        if self.normalize_observations:
            time_value = float(self.normalizer.normalize_time_index(self.current_step))
            price_value = float(self.normalizer.normalize_price(price))
            grid_import_value = float(
                self.normalizer.normalize_grid_import_kw(grid_import)
            )
            voltage_deviation_value = float(
                self.normalizer.normalize_voltage_deviation(voltage_deviation)
            )
            curtailment_value = float(self.normalizer.normalize_pv_kw(curtailment))
            grid_stress_value = float(np.clip(grid_stress, 0.0, 1.0))
        else:
            time_value = float(self.current_step)
            price_value = float(price)
            grid_import_value = grid_import
            voltage_deviation_value = voltage_deviation
            curtailment_value = curtailment
            grid_stress_value = grid_stress

        for agent_idx, bus in enumerate(self.dess_buses):
            if self.normalize_observations:
                load_value = float(self.normalizer.normalize_load_kw(load_kw[bus]))
                pv_value = float(self.normalizer.normalize_pv_kw(pv_kw[bus]))
                voltage_value = float(
                    self.normalizer.normalize_voltage_pu(voltage_pu[bus])
                )
                soc_value = float(self.normalizer.normalize_soc(soc_values[agent_idx]))
            else:
                load_value = float(load_kw[bus])
                pv_value = float(pv_kw[bus])
                voltage_value = float(voltage_pu[bus])
                soc_value = float(soc_values[agent_idx])

            agent_obs[agent_idx, :] = np.array(
                [
                    time_value,
                    price_value,
                    soc_value,
                    voltage_value,
                    load_value,
                    pv_value,
                    float(self.previous_actions[agent_idx]),
                    grid_import_value,
                    voltage_deviation_value,
                    curtailment_value,
                    grid_stress_value,
                ],
                dtype=np.float32,
            )

        if not np.all(np.isfinite(agent_obs)):
            raise ValueError("Agent observation contains non-finite values.")

        return agent_obs

    # ============================================================
    # Environment transition
    # ============================================================

    def step(self, action):
        """
        Apply DESS actions and advance the environment by one step.
        """

        action = self._prepare_action(action)

        load_kw, load_kvar, pv_kw, price, date_time = self._get_profiles(
            self.current_index
        )

        soc_before = np.array(
            [battery.get_soc() for battery in self.batteries],
            dtype=np.float32,
        )

        candidate = self._evaluate_candidate_action(
            action=action,
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_kw,
        )

        infeasible_action = False

        if self.constraint_handling == "hard_feasibility":
            if not candidate["feasible"]:
                infeasible_action = True
                candidate = self._handle_infeasible_action(
                    original_action=action,
                    load_kw=load_kw,
                    load_kvar=load_kvar,
                    pv_kw=pv_kw,
                )
        else:
            candidate["accepted_action"] = action.copy()

        # Commit the accepted battery states.
        self._restore_battery_states(candidate["battery_states_after"])

        accepted_action = candidate["accepted_action"]
        dess_power_kw = candidate["dess_power_kw"]
        battery_outputs = candidate["battery_outputs"]
        pv_used_kw = candidate["pv_used_kw"]
        curtailment_by_bus = candidate["curtailment_by_bus"]
        curtailment_kw = candidate["curtailment_kw"]
        pf_result = candidate["pf_result"]

        self.previous_kpis = dict(self.current_kpis)

        self.last_pf_result = pf_result
        self.last_dess_power_kw = dess_power_kw.copy()
        self.last_curtailment_by_bus = curtailment_by_bus.copy()
        self.last_curtailment_kw = float(curtailment_kw)

        self.current_kpis = self._compute_kpis(
            pf_result=pf_result,
            price=price,
            curtailment_kw=curtailment_kw,
        )

        soc_after = np.array(
            [battery.get_soc() for battery in self.batteries],
            dtype=np.float32,
        )

        rewards = self._calculate_rewards(
            price=price,
            dess_power_kw=dess_power_kw,
            soc_before=soc_before,
            soc_after=soc_after,
            infeasible_action=infeasible_action,
        )

        self.previous_actions = accepted_action.copy()
        self.last_date_time = date_time

        terminated = False
        truncated = self.current_step >= self.episode_length - 1

        info = self._build_info()
        info["battery_outputs"] = battery_outputs
        info["dess_power_kw"] = dess_power_kw.copy()
        info["curtailment_by_bus_kw"] = curtailment_by_bus.copy()
        info["requested_action"] = action.copy()
        info["accepted_action"] = accepted_action.copy()
        info["infeasible_action"] = bool(infeasible_action)
        info["price"] = float(price)
        info["date_time"] = date_time
        info["dess_buses"] = list(self.dess_buses)

        # If episode ends here, return the current post-action observation.
        if truncated:
            next_obs = self._build_observation(
                load_kw=load_kw,
                pv_kw=pv_used_kw,
                price=price,
                voltage_pu=pf_result["voltage_pu"],
            )
            return next_obs, rewards, terminated, truncated, info

        # Advance time.
        self.current_step += 1
        self.current_index += 1

        next_load_kw, next_load_kvar, next_pv_kw, next_price, next_date_time = (
            self._get_profiles(self.current_index)
        )

        next_pv_used_kw, next_curtailment_by_bus, next_curtailment_kw = (
            self._apply_curtailment(
                load_kw=next_load_kw,
                pv_kw=next_pv_kw,
                dess_power_kw=self.last_dess_power_kw,
            )
        )

        next_pf_result = self.power_flow.run_power_flow(
            load_kw=next_load_kw,
            load_kvar=next_load_kvar,
            pv_kw=next_pv_used_kw,
            dess_power_kw=self.last_dess_power_kw,
        )

        self.last_pf_result = next_pf_result
        self.last_curtailment_by_bus = next_curtailment_by_bus.copy()
        self.last_curtailment_kw = float(next_curtailment_kw)

        self.current_kpis = self._compute_kpis(
            pf_result=next_pf_result,
            price=next_price,
            curtailment_kw=next_curtailment_kw,
        )

        self.last_date_time = next_date_time

        next_obs = self._build_observation(
            load_kw=next_load_kw,
            pv_kw=next_pv_used_kw,
            price=next_price,
            voltage_pu=next_pf_result["voltage_pu"],
        )

        return next_obs, rewards, terminated, truncated, info

    # ============================================================
    # Candidate action evaluation
    # ============================================================

    def _prepare_action(self, action):
        """
        Validate and clip the normalized action vector.
        """

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (self.num_agents,):
            raise ValueError(
                "Action must have shape ({},), got {}.".format(
                    self.num_agents,
                    action.shape,
                )
            )

        return np.clip(
            action,
            float(self.config["action"]["low"]),
            float(self.config["action"]["high"]),
        ).astype(np.float32)

    def _evaluate_candidate_action(self, action, load_kw, load_kvar, pv_kw):
        """
        Evaluate a candidate DESS action without permanently committing it.
        """

        battery_states_before = self._snapshot_battery_states()

        dess_power_kw = np.zeros(self.num_buses, dtype=np.float64)
        battery_outputs = []

        for agent_idx, bus in enumerate(self.dess_buses):
            battery_result = self.batteries[agent_idx].step(action[agent_idx])
            dess_power_kw[bus] = battery_result["power_kw"]
            battery_outputs.append(battery_result)

        battery_states_after = self._snapshot_battery_states()

        pv_used_kw, curtailment_by_bus, curtailment_kw = self._apply_curtailment(
            load_kw=load_kw,
            pv_kw=pv_kw,
            dess_power_kw=dess_power_kw,
        )

        pf_result = self.power_flow.run_power_flow(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_used_kw,
            dess_power_kw=dess_power_kw,
        )

        feasible = self._is_feasible_pf_result(pf_result)

        # Restore because this method only evaluates the candidate.
        self._restore_battery_states(battery_states_before)

        return {
            "accepted_action": action.copy(),
            "dess_power_kw": dess_power_kw,
            "battery_outputs": battery_outputs,
            "battery_states_after": battery_states_after,
            "pv_used_kw": pv_used_kw,
            "curtailment_by_bus": curtailment_by_bus,
            "curtailment_kw": float(curtailment_kw),
            "pf_result": pf_result,
            "feasible": bool(feasible),
        }

    def _handle_infeasible_action(self, original_action, load_kw, load_kvar, pv_kw):
        """
        Handle an infeasible action according to the configured rule.
        """

        if self.hard_constraint_action == "reject_action":
            zero_action = np.zeros(self.num_agents, dtype=np.float32)
            candidate = self._evaluate_candidate_action(
                action=zero_action,
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
            )
            candidate["accepted_action"] = zero_action.copy()
            return candidate

        if self.hard_constraint_action == "clip_or_correct":
            corrected_action = original_action.copy()

            for _ in range(self.max_correction_attempts):
                corrected_action = corrected_action * self.action_correction_factor

                candidate = self._evaluate_candidate_action(
                    action=corrected_action,
                    load_kw=load_kw,
                    load_kvar=load_kvar,
                    pv_kw=pv_kw,
                )

                if candidate["feasible"]:
                    candidate["accepted_action"] = corrected_action.copy()
                    return candidate

            zero_action = np.zeros(self.num_agents, dtype=np.float32)
            candidate = self._evaluate_candidate_action(
                action=zero_action,
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
            )
            candidate["accepted_action"] = zero_action.copy()
            return candidate

        if self.hard_constraint_action == "terminate_episode":
            zero_action = np.zeros(self.num_agents, dtype=np.float32)
            candidate = self._evaluate_candidate_action(
                action=zero_action,
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
            )
            candidate["accepted_action"] = zero_action.copy()
            return candidate

        raise ValueError(
            "Unsupported hard_constraint_action: {}".format(
                self.hard_constraint_action
            )
        )

    def _is_feasible_pf_result(self, pf_result):
        """
        Check whether a power-flow result satisfies enabled feasibility limits.
        """

        if not bool(pf_result["converged"]):
            return False

        if self.constraint_cfg.get("enforce_voltage_limits", True):
            if float(pf_result["max_voltage_violation"]) > 1e-9:
                return False

        if self.constraint_cfg.get("enforce_line_current_limits", True):
            if float(pf_result["max_line_current_violation"]) > 1e-9:
                return False

        return True

    # ============================================================
    # Battery-state snapshots
    # ============================================================

    def _snapshot_battery_states(self):
        """
        Save current battery states.
        """

        return [
            {
                "soc": float(battery.soc),
                "last_power_kw": float(battery.last_power_kw),
                "last_energy_kwh": float(battery.last_energy_kwh),
                "last_requested_power_kw": float(
                    getattr(battery, "last_requested_power_kw", 0.0)
                ),
                "last_action": float(getattr(battery, "last_action", 0.0)),
                "last_was_limited": bool(
                    getattr(battery, "last_was_limited", False)
                ),
            }
            for battery in self.batteries
        ]

    def _restore_battery_states(self, states):
        """
        Restore battery states.
        """

        for battery, state in zip(self.batteries, states):
            battery.soc = float(state["soc"])
            battery.last_power_kw = float(state["last_power_kw"])
            battery.last_energy_kwh = float(state["last_energy_kwh"])
            battery.last_requested_power_kw = float(state["last_requested_power_kw"])
            battery.last_action = float(state["last_action"])
            battery.last_was_limited = bool(state["last_was_limited"])

    # ============================================================
    # Time-series profiles
    # ============================================================

    def _get_profiles(self, row_index):
        """
        Load one scaled node-level time-series row.
        """

        row = select_time_series_row(self.time_series, row_index)

        load_kw = row["load_kw"].astype(np.float64) * self.load_scaling_factor
        pv_kw = row["pv_kw"].astype(np.float64) * self.pv_scaling_factor
        price = float(row["price"]) * self.price_scaling_factor
        date_time = row["date_time"]

        base_load_kw = self.grid.get_base_load_kw().astype(np.float64)
        base_load_kvar = self.grid.get_base_load_kvar().astype(np.float64)

        total_base_load = float(np.sum(base_load_kw))

        if total_base_load <= 0.0:
            raise ValueError("Total base active load must be positive.")

        load_scale = float(np.sum(load_kw)) / total_base_load
        load_kvar = base_load_kvar * load_scale

        return load_kw, load_kvar, pv_kw, price, date_time

    # ============================================================
    # Curtailment calculation
    # ============================================================

    def _apply_curtailment(self, load_kw, pv_kw, dess_power_kw):
        """
        Apply feeder-level renewable curtailment using the configured export limit.
        """

        total_load_kw = float(np.sum(load_kw))
        total_pv_kw = float(np.sum(pv_kw))

        total_dess_discharge_kw = float(np.sum(np.maximum(0.0, dess_power_kw)))
        total_dess_charge_kw = float(np.sum(np.maximum(0.0, -dess_power_kw)))

        surplus_kw = (
            total_pv_kw
            + total_dess_discharge_kw
            - total_load_kw
            - total_dess_charge_kw
            - self.grid_export_limit_kw
        )

        curtailment_kw = max(0.0, surplus_kw)
        curtailment_kw = min(curtailment_kw, total_pv_kw)

        if curtailment_kw <= 0.0 or total_pv_kw <= 0.0:
            return pv_kw.copy(), np.zeros_like(pv_kw), 0.0

        curtailment_fraction = curtailment_kw / total_pv_kw
        curtailment_by_bus = pv_kw * curtailment_fraction
        pv_used_kw = np.maximum(0.0, pv_kw - curtailment_by_bus)

        return pv_used_kw, curtailment_by_bus, float(np.sum(curtailment_by_bus))

    # ============================================================
    # KPI and reward calculation
    # ============================================================

    def _compute_kpis(self, pf_result, price, curtailment_kw):
        """
        Compute grid-level performance indicators.
        """

        voltage_pu = pf_result["voltage_pu"]

        voltage_deviation = float(np.sum(np.abs(voltage_pu - self.v_ref)))
        voltage_violation_max = float(pf_result["max_voltage_violation"])
        line_current_violation = float(pf_result["max_line_current_violation"])

        grid_stress = voltage_violation_max + line_current_violation

        return {
            "grid_import_kw": float(pf_result["grid_import_kw"]),
            "grid_import_kvar": float(pf_result["grid_import_kvar"]),
            "voltage_deviation": voltage_deviation,
            "voltage_violation_max": voltage_violation_max,
            "curtailment_kw": float(curtailment_kw),
            "line_current_violation": line_current_violation,
            "grid_stress": grid_stress,
            "price": float(price),
        }

    def _calculate_rewards(
        self,
        price,
        dess_power_kw,
        soc_before,
        soc_after,
        infeasible_action=False,
    ):
        """
        Compute per-agent cooperative rewards.

        Research_Summer-consistent structure:

            r_i(t) = w_grid  * R_grid(t)
                   + w_local * R_local,i(t)
                   + w_kpi   * R_KPI(t)
        """

        cfg = self.reward_cfg

        if not bool(self.last_pf_result["converged"]):
            penalty = float(cfg.get("nonconvergence_penalty", -10000.0))
            return np.ones(self.num_agents, dtype=np.float32) * penalty

        w_grid = float(cfg.get("w_grid", 0.60))
        w_local = float(cfg.get("w_local", 0.15))
        w_kpi = float(cfg.get("w_kpi", 0.25))

        grid_import_kw = float(self.current_kpis["grid_import_kw"])
        voltage_deviation = float(self.current_kpis["voltage_deviation"])
        curtailment_kw = float(self.current_kpis["curtailment_kw"])

        grid_import_mwh = max(0.0, grid_import_kw) * self.delta_t_hours / 1000.0
        curtailment_mwh = curtailment_kw * self.delta_t_hours / 1000.0

        energy_cost = float(price) * grid_import_mwh

        grid_reward = -(
            float(cfg.get("phi_cost", 1.0)) * energy_cost
            + float(cfg.get("phi_voltage_deviation", 1.0)) * voltage_deviation
            + float(cfg.get("lambda_curtailment", 1.0)) * curtailment_mwh
        )

        kpi_reward = 0.0
        if bool(cfg.get("use_kpi_reward", True)):
            kpi_reward = self._calculate_kpi_reward()

        rewards = np.zeros(self.num_agents, dtype=np.float32)

        for agent_idx, bus in enumerate(self.dess_buses):
            power_abs_kw = abs(float(dess_power_kw[bus]))
            power_abs_mwh = power_abs_kw * self.delta_t_hours / 1000.0
            delta_soc = float(soc_after[agent_idx] - soc_before[agent_idx])

            local_reward = -(
                float(cfg.get("phi_cycling", 0.05)) * power_abs_mwh
                + float(cfg.get("phi_soc_reserve", 0.05)) * (delta_soc ** 2)
            )

            rewards[agent_idx] = (
                w_grid * grid_reward
                + w_local * local_reward
                + w_kpi * kpi_reward
            )

        if infeasible_action:
            rewards += float(cfg.get("infeasible_action_penalty", -50.0))

        return rewards.astype(np.float32)

    def _calculate_kpi_reward(self):
        """
        Compute KPI-improvement reward using consecutive grid indicators.
        """

        if self.previous_kpis is None:
            return 0.0

        cfg = self.reward_cfg

        current = self.current_kpis
        previous = self.previous_kpis

        delta_import = (
            previous["grid_import_kw"] - current["grid_import_kw"]
        ) * self.delta_t_hours / 1000.0

        delta_curtailment = (
            previous["curtailment_kw"] - current["curtailment_kw"]
        ) * self.delta_t_hours / 1000.0

        delta_vdev = previous["voltage_deviation"] - current["voltage_deviation"]
        delta_stress = previous["grid_stress"] - current["grid_stress"]

        kpi_reward = (
            float(cfg.get("beta_import", 1.0)) * delta_import
            + float(cfg.get("beta_voltage_dev", 1.0)) * delta_vdev
            + float(cfg.get("beta_curtailment", 1.0)) * delta_curtailment
            + float(cfg.get("beta_grid_stress", 1.0)) * delta_stress
        )

        return float(kpi_reward)

    # ============================================================
    # Diagnostics
    # ============================================================

    def _build_info(self):
        """
        Build diagnostic information dictionary.
        """

        min_voltage = float(np.min(self.last_pf_result["voltage_pu"]))
        max_voltage = float(np.max(self.last_pf_result["voltage_pu"]))

        return {
            "mode": self.mode,
            "episode_start_index": self.episode_start_index,
            "current_step": self.current_step,
            "current_index": self.current_index,
            "date_time": getattr(self, "last_date_time", None),
            "dess_buses": list(self.dess_buses),
            "num_agents": int(self.num_agents),
            "agent_obs_dim": int(self.agent_obs_dim),
            "kpis": dict(self.current_kpis) if self.current_kpis is not None else None,
            "feasible": bool(self.last_pf_result["feasible"]),
            "converged": bool(self.last_pf_result["converged"]),
            "iterations": int(self.last_pf_result["iterations"]),
            "min_voltage": min_voltage,
            "max_voltage": max_voltage,
            "min_voltage_pu": min_voltage,
            "max_voltage_pu": max_voltage,
            "max_voltage_violation": float(
                self.last_pf_result["max_voltage_violation"]
            ),
            "max_line_current_pu": float(
                np.max(self.last_pf_result["line_current_pu"])
            ),
            "max_line_current_violation": float(
                self.last_pf_result["max_line_current_violation"]
            ),
            "curtailment_kw": float(self.current_kpis["curtailment_kw"]),
            "load_scaling_factor": self.load_scaling_factor,
            "pv_scaling_factor": self.pv_scaling_factor,
            "price_scaling_factor": self.price_scaling_factor,
            "normalize_observations": self.normalize_observations,
        }

    def render(self):
        """
        Print compact environment diagnostics.
        """

        info = self._build_info()
        kpis = info["kpis"]

        print(
            "mode={}, step={}, date_time={}, feasible={}, minV={:.4f}, "
            "maxV={:.4f}, grid_import={:.2f} kW, curtailment={:.2f} kW, "
            "max_line_current={:.4f} pu".format(
                info["mode"],
                info["current_step"],
                info["date_time"],
                info["feasible"],
                info["min_voltage_pu"],
                info["max_voltage_pu"],
                kpis["grid_import_kw"],
                kpis["curtailment_kw"],
                info["max_line_current_pu"],
            )
        )