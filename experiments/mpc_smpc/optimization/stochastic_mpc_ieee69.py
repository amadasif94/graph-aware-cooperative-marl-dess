# ============================================================
# stochastic_mpc_ieee69.py
# ============================================================

from __future__ import annotations

import copy
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv
from experiments.mpc_smpc.optimization import baseline_config_ieee69 as cfg


_PF_ENV_CACHE: Dict[str, DESSEnv] = {}


def _normalize_topology_case(topology_case: str = "TP1") -> str:
    if hasattr(cfg, "normalize_topology_case"):
        return cfg.normalize_topology_case(topology_case)
    return str(topology_case or "TP1").upper()


def get_pf_env(
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
) -> DESSEnv:
    global _PF_ENV_CACHE

    topology_case = _normalize_topology_case(topology_case)

    if topology_case not in _PF_ENV_CACHE:
        config = copy.deepcopy(
            topology_config if topology_config is not None else IEEE69_CONFIG
        )

        _PF_ENV_CACHE[topology_case] = DESSEnv(
            config=config,
            mode="test",
            seed=42,
        )

    return _PF_ENV_CACHE[topology_case]


def clear_pf_env_cache() -> None:
    global _PF_ENV_CACHE
    _PF_ENV_CACHE = {}


def apply_export_limited_curtailment(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    dess_power_kw_full: np.ndarray,
) -> tuple[np.ndarray, float]:

    load_kw = np.asarray(load_kw, dtype=np.float64).reshape(cfg.NUM_BUSES)
    pv_kw = np.asarray(pv_kw, dtype=np.float64).reshape(cfg.NUM_BUSES)
    dess_power_kw_full = np.asarray(
        dess_power_kw_full,
        dtype=np.float64,
    ).reshape(cfg.NUM_BUSES)

    total_load = float(np.sum(load_kw))
    total_pv = float(np.sum(pv_kw))

    total_dis = float(np.sum(np.maximum(dess_power_kw_full, 0.0)))
    total_ch = float(np.sum(np.maximum(-dess_power_kw_full, 0.0)))

    surplus = (
        total_pv
        + total_dis
        - total_load
        - total_ch
        - cfg.GRID_EXPORT_LIMIT_KW
    )

    curtailment_kw = min(total_pv, max(0.0, surplus))

    if total_pv <= 1e-9 or curtailment_kw <= 0.0:
        return pv_kw.copy(), 0.0

    pv_used_kw = pv_kw - (pv_kw / total_pv) * curtailment_kw
    pv_used_kw = np.maximum(pv_used_kw, 0.0)

    return pv_used_kw, float(curtailment_kw)


def build_full_dess_power_from_beta(
    beta_ch_kw: np.ndarray,
    beta_dis_kw: np.ndarray,
) -> np.ndarray:

    beta_ch_kw = np.asarray(beta_ch_kw, dtype=np.float64).reshape(cfg.NUM_DESS)
    beta_dis_kw = np.asarray(beta_dis_kw, dtype=np.float64).reshape(cfg.NUM_DESS)

    full = np.zeros(cfg.NUM_BUSES, dtype=np.float64)

    for i, bus in enumerate(cfg.DESS_BUSES):
        full[int(bus)] = beta_dis_kw[i] - beta_ch_kw[i]

    return full


def check_first_action_network_feasible(
    load_S69_kw: np.ndarray,
    pv_S69_kw: np.ndarray,
    beta_ch_kw: np.ndarray,
    beta_dis_kw: np.ndarray,
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
) -> Dict[str, Any]:

    env = get_pf_env(
        topology_config=topology_config,
        topology_case=topology_case,
    )

    load_S69_kw = np.asarray(load_S69_kw, dtype=np.float64)
    pv_S69_kw = np.asarray(pv_S69_kw, dtype=np.float64)

    if load_S69_kw.ndim != 2 or load_S69_kw.shape[1] != cfg.NUM_BUSES:
        raise ValueError(
            f"load_S69_kw must have shape [S, {cfg.NUM_BUSES}], "
            f"got {load_S69_kw.shape}"
        )

    if pv_S69_kw.ndim != 2 or pv_S69_kw.shape[1] != cfg.NUM_BUSES:
        raise ValueError(
            f"pv_S69_kw must have shape [S, {cfg.NUM_BUSES}], "
            f"got {pv_S69_kw.shape}"
        )

    if load_S69_kw.shape != pv_S69_kw.shape:
        raise ValueError(
            f"load_S69_kw shape {load_S69_kw.shape} != "
            f"pv_S69_kw shape {pv_S69_kw.shape}"
        )

    dess_power_full = build_full_dess_power_from_beta(
        beta_ch_kw=beta_ch_kw,
        beta_dis_kw=beta_dis_kw,
    )

    feasible_all = True
    worst_min_voltage = np.inf
    worst_max_voltage = -np.inf
    worst_line_current = -np.inf
    max_voltage_violation = 0.0
    max_line_violation = 0.0

    for s in range(load_S69_kw.shape[0]):
        load_kw = load_S69_kw[s]
        pv_kw = pv_S69_kw[s]

        pv_used_kw, _ = apply_export_limited_curtailment(
            load_kw=load_kw,
            pv_kw=pv_kw,
            dess_power_kw_full=dess_power_full,
        )

        load_kvar = 0.30 * load_kw

        pf = env.power_flow.run_power_flow(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_used_kw,
            dess_power_kw=dess_power_full,
        )

        voltage = np.asarray(pf["voltage_pu"], dtype=np.float64)
        current = np.asarray(pf["line_current_pu"], dtype=np.float64)

        min_v = float(np.min(voltage))
        max_v = float(np.max(voltage))
        max_i = float(np.max(current))

        v_viol = max(
            0.0,
            float(env.v_min - min_v),
            float(max_v - env.v_max),
        )

        if env.grid.line_current_limits_pu is not None:
            limits = np.asarray(env.grid.line_current_limits_pu, dtype=np.float64)
            i_viol = float(np.max(np.maximum(0.0, current - limits)))
        else:
            i_viol = 0.0

        feasible = (
            bool(pf.get("converged", True))
            and v_viol <= 1e-8
            and i_viol <= 1e-8
        )

        feasible_all = feasible_all and feasible

        worst_min_voltage = min(worst_min_voltage, min_v)
        worst_max_voltage = max(worst_max_voltage, max_v)
        worst_line_current = max(worst_line_current, max_i)
        max_voltage_violation = max(max_voltage_violation, v_viol)
        max_line_violation = max(max_line_violation, i_viol)

    return {
        "topology_case": _normalize_topology_case(topology_case),
        "network_feasible": bool(feasible_all),
        "worst_min_voltage_pu": float(worst_min_voltage),
        "worst_max_voltage_pu": float(worst_max_voltage),
        "worst_line_current_pu": float(worst_line_current),
        "max_voltage_violation": float(max_voltage_violation),
        "max_line_current_violation": float(max_line_violation),
    }


def correct_first_action_by_power_flow(
    load_S69_kw: np.ndarray,
    pv_S69_kw: np.ndarray,
    beta_ch_kw: np.ndarray,
    beta_dis_kw: np.ndarray,
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
) -> Dict[str, Any]:

    beta_ch = np.asarray(beta_ch_kw, dtype=np.float64).copy()
    beta_dis = np.asarray(beta_dis_kw, dtype=np.float64).copy()

    original_ch = beta_ch.copy()
    original_dis = beta_dis.copy()

    for attempt in range(cfg.MAX_CORRECTION_ATTEMPTS + 1):
        check = check_first_action_network_feasible(
            load_S69_kw=load_S69_kw,
            pv_S69_kw=pv_S69_kw,
            beta_ch_kw=beta_ch,
            beta_dis_kw=beta_dis,
            topology_config=topology_config,
            topology_case=topology_case,
        )

        if check["network_feasible"]:
            return {
                **check,
                "corrected": bool(attempt > 0),
                "correction_attempts": int(attempt),
                "beta_ch_kw_corrected": beta_ch,
                "beta_dis_kw_corrected": beta_dis,
                "beta_ch_kw_original": original_ch,
                "beta_dis_kw_original": original_dis,
            }

        beta_ch *= cfg.ACTION_CORRECTION_FACTOR
        beta_dis *= cfg.ACTION_CORRECTION_FACTOR

    zero = np.zeros(cfg.NUM_DESS, dtype=np.float64)

    check = check_first_action_network_feasible(
        load_S69_kw=load_S69_kw,
        pv_S69_kw=pv_S69_kw,
        beta_ch_kw=zero,
        beta_dis_kw=zero,
        topology_config=topology_config,
        topology_case=topology_case,
    )

    return {
        **check,
        "corrected": True,
        "correction_attempts": int(cfg.MAX_CORRECTION_ATTEMPTS + 1),
        "beta_ch_kw_corrected": zero,
        "beta_dis_kw_corrected": zero,
        "beta_ch_kw_original": original_ch,
        "beta_dis_kw_original": original_dis,
    }


def build_stochastic_mpc_problem(
    S: int,
    T: int,
    num_dess: int = cfg.NUM_DESS,
    dt_hours: float = cfg.DT_HOURS,
    soc_min: float = cfg.SOC_MIN,
    soc_max: float = cfg.SOC_MAX,
    eta_ch: float = cfg.ETA_CH,
    eta_dis: float = cfg.ETA_DIS,
    p_ch_max_kw: float = cfg.P_CH_MAX_KW,
    p_dis_max_kw: float = cfg.P_DIS_MAX_KW,
    grid_export_limit_kw: float = cfg.GRID_EXPORT_LIMIT_KW,
    u_unmet_max_kw: float = cfg.U_UNMET_MAX_KW,
    u_curt_max_kw: float = cfg.U_CURT_MAX_KW,
) -> Dict[str, Any]:

    return {
        "meta": {
            "S": int(S),
            "T": int(T),
            "num_dess": int(num_dess),
            "dt_hours": float(dt_hours),
            "soc_min": float(soc_min),
            "soc_max": float(soc_max),
            "eta_ch": float(eta_ch),
            "eta_dis": float(eta_dis),
            "capacity_kwh": float(cfg.BATTERY_CAPACITY_KWH),
            "p_ch_max_kw": float(p_ch_max_kw),
            "p_dis_max_kw": float(p_dis_max_kw),
            "grid_export_limit_kw": float(grid_export_limit_kw),
            "u_unmet_max_kw": float(u_unmet_max_kw),
            "u_curt_max_kw": float(u_curt_max_kw),
        }
    }


def solve_stochastic_mpc(
    bundle: Dict[str, Any],
    aggregate_netload_ST_kw: Optional[np.ndarray],
    price_T: np.ndarray,
    soc0: np.ndarray,
    load_ST69_kw: Optional[np.ndarray] = None,
    pv_ST69_kw: Optional[np.ndarray] = None,
    load_ST33_kw: Optional[np.ndarray] = None,
    pv_ST33_kw: Optional[np.ndarray] = None,
    enforce_network_first_action: bool = True,
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
    solver_name: str = cfg.SOLVER_NAME,
    verbose: bool = cfg.VERBOSE,
    threads: int = cfg.THREADS,
    mip_gap: float = cfg.MIP_GAP,
    time_limit: float = cfg.TIME_LIMIT,
    warm_start: bool = cfg.WARM_START,
) -> Dict[str, Any]:

    del warm_start

    topology_case = _normalize_topology_case(topology_case)

    if load_ST69_kw is None and load_ST33_kw is not None:
        load_ST69_kw = load_ST33_kw

    if pv_ST69_kw is None and pv_ST33_kw is not None:
        pv_ST69_kw = pv_ST33_kw

    if solver_name.upper() != "GUROBI":
        raise ValueError(f"Unsupported solver_name: {solver_name}")

    meta = bundle["meta"]

    S = int(meta["S"])
    T = int(meta["T"])
    N = int(meta["num_dess"])

    dt = float(meta["dt_hours"])
    soc_min = float(meta["soc_min"])
    soc_max = float(meta["soc_max"])
    eta_ch = float(meta["eta_ch"])
    eta_dis = float(meta["eta_dis"])
    capacity_kwh = float(meta["capacity_kwh"])

    p_ch_max_kw = float(meta["p_ch_max_kw"])
    p_dis_max_kw = float(meta["p_dis_max_kw"])
    grid_export_limit_kw = float(meta["grid_export_limit_kw"])
    u_unmet_max_kw = float(meta["u_unmet_max_kw"])
    u_curt_max_kw = float(meta["u_curt_max_kw"])

    price_T = np.asarray(price_T, dtype=np.float64).reshape(-1)
    soc0 = np.asarray(soc0, dtype=np.float64).reshape(-1)

    if load_ST69_kw is not None and pv_ST69_kw is not None:
        load_ST69_kw = np.asarray(load_ST69_kw, dtype=np.float64)
        pv_ST69_kw = np.asarray(pv_ST69_kw, dtype=np.float64)

        if load_ST69_kw.shape != (S, T, cfg.NUM_BUSES):
            raise ValueError(
                f"load_ST69_kw shape {load_ST69_kw.shape} "
                f"!= {(S, T, cfg.NUM_BUSES)}"
            )

        if pv_ST69_kw.shape != (S, T, cfg.NUM_BUSES):
            raise ValueError(
                f"pv_ST69_kw shape {pv_ST69_kw.shape} "
                f"!= {(S, T, cfg.NUM_BUSES)}"
            )

        aggregate_netload_ST_kw = np.sum(load_ST69_kw - pv_ST69_kw, axis=2)

    else:
        aggregate_netload_ST_kw = np.asarray(
            aggregate_netload_ST_kw,
            dtype=np.float64,
        )

    if aggregate_netload_ST_kw.shape != (S, T):
        raise ValueError(
            f"aggregate_netload_ST_kw shape {aggregate_netload_ST_kw.shape} "
            f"!= {(S, T)}"
        )

    if len(price_T) != T:
        raise ValueError(f"price_T length {len(price_T)} != T={T}")

    if len(soc0) != N:
        raise ValueError(f"soc0 length {len(soc0)} != num_dess={N}")

    solve_start = time.time()

    try:
        m = gp.Model(f"ieee69_network_aware_smpc_{topology_case}")

        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.Threads = int(threads)
        m.Params.MIPGap = float(mip_gap)
        m.Params.TimeLimit = float(time_limit)

        P_ch = m.addVars(S, T, N, lb=0.0, ub=p_ch_max_kw, vtype=GRB.CONTINUOUS, name="P_ch")
        P_dis = m.addVars(S, T, N, lb=0.0, ub=p_dis_max_kw, vtype=GRB.CONTINUOUS, name="P_dis")
        SoC = m.addVars(S, T + 1, N, lb=soc_min, ub=soc_max, vtype=GRB.CONTINUOUS, name="SoC")

        z_batt = m.addVars(S, T, N, vtype=GRB.BINARY, name="z_batt")

        P_grid_import = m.addVars(S, T, lb=0.0, vtype=GRB.CONTINUOUS, name="P_grid_import")
        P_grid_export = m.addVars(S, T, lb=0.0, ub=grid_export_limit_kw, vtype=GRB.CONTINUOUS, name="P_grid_export")
        z_grid = m.addVars(S, T, vtype=GRB.BINARY, name="z_grid")

        U_unmet = m.addVars(S, T, lb=0.0, ub=u_unmet_max_kw, vtype=GRB.CONTINUOUS, name="U_unmet")
        U_curt = m.addVars(S, T, lb=0.0, ub=u_curt_max_kw, vtype=GRB.CONTINUOUS, name="U_curt")

        beta_ch = m.addVars(N, lb=0.0, ub=p_ch_max_kw, vtype=GRB.CONTINUOUS, name="beta_ch")
        beta_dis = m.addVars(N, lb=0.0, ub=p_dis_max_kw, vtype=GRB.CONTINUOUS, name="beta_dis")
        z_batt_0 = m.addVars(N, vtype=GRB.BINARY, name="z_batt_0")

        for s in range(S):
            for i in range(N):
                m.addConstr(SoC[s, 0, i] == float(soc0[i]))

        for s in range(S):
            for t in range(T):
                for i in range(N):
                    m.addConstr(P_ch[s, t, i] <= p_ch_max_kw * z_batt[s, t, i])
                    m.addConstr(P_dis[s, t, i] <= p_dis_max_kw * (1 - z_batt[s, t, i]))

        for s in range(S):
            for i in range(N):
                m.addConstr(P_ch[s, 0, i] == beta_ch[i])
                m.addConstr(P_dis[s, 0, i] == beta_dis[i])
                m.addConstr(z_batt[s, 0, i] == z_batt_0[i])

        for s in range(S):
            for t in range(T):
                for i in range(N):
                    m.addConstr(
                        SoC[s, t + 1, i]
                        == SoC[s, t, i]
                        + eta_ch * P_ch[s, t, i] * dt / capacity_kwh
                        - P_dis[s, t, i] * dt / (eta_dis * capacity_kwh)
                    )

        import_big_m_kw = max(
            1.0,
            float(np.max(aggregate_netload_ST_kw))
            + N * p_ch_max_kw
            + u_unmet_max_kw
            + 1000.0,
        )

        for s in range(S):
            for t in range(T):
                m.addConstr(P_grid_import[s, t] <= import_big_m_kw * z_grid[s, t])
                m.addConstr(P_grid_export[s, t] <= grid_export_limit_kw * (1 - z_grid[s, t]))

        for s in range(S):
            for t in range(T):
                total_ch = gp.quicksum(P_ch[s, t, i] for i in range(N))
                total_dis = gp.quicksum(P_dis[s, t, i] for i in range(N))

                m.addConstr(
                    P_grid_import[s, t]
                    - P_grid_export[s, t]
                    + total_dis
                    - total_ch
                    + U_unmet[s, t]
                    - U_curt[s, t]
                    == float(aggregate_netload_ST_kw[s, t])
                )

        grid_cost = gp.quicksum(
            float(price_T[t]) * (P_grid_import[s, t] / 1000.0)
            for s in range(S)
            for t in range(T)
        )

        battery_throughput_cost = gp.quicksum(
            cfg.W_CYC * ((P_ch[s, t, i] + P_dis[s, t, i]) / 1000.0)
            for s in range(S)
            for t in range(T)
            for i in range(N)
        )

        unmet_cost = gp.quicksum(
            cfg.C_UNMET * (U_unmet[s, t] / 1000.0)
            for s in range(S)
            for t in range(T)
        )

        curtailment_cost = gp.quicksum(
            cfg.C_CURT * (U_curt[s, t] / 1000.0)
            for s in range(S)
            for t in range(T)
        )

        objective = (
            cfg.W_ENERGY * grid_cost
            + battery_throughput_cost
            + unmet_cost
            + curtailment_cost
        )

        m.setObjective((dt / S) * objective, GRB.MINIMIZE)
        m.optimize()

    except Exception as e:
        print("\nIEEE69 SMPC solve crashed.")
        print("=" * 70)
        print("Exception type    :", type(e).__name__)
        print("Exception message :", str(e))
        traceback.print_exc()
        raise

    solve_time_sec = time.time() - solve_start

    status = int(m.Status)
    sol_count = int(m.SolCount)

    if status == GRB.OPTIMAL:
        status_str = "optimal"
    elif status == GRB.SUBOPTIMAL:
        status_str = "suboptimal"
    elif status == GRB.TIME_LIMIT and sol_count > 0:
        status_str = "time_limit_feasible"
    elif sol_count > 0:
        status_str = f"feasible_status_{status}"
    else:
        raise RuntimeError(f"Gurobi solve failed: status={status}, SolCount={sol_count}")

    P_ch_val = np.zeros((S, T, N), dtype=np.float64)
    P_dis_val = np.zeros((S, T, N), dtype=np.float64)
    SoC_val = np.zeros((S, T + 1, N), dtype=np.float64)

    for s in range(S):
        for t in range(T):
            for i in range(N):
                P_ch_val[s, t, i] = P_ch[s, t, i].X
                P_dis_val[s, t, i] = P_dis[s, t, i].X

        for t in range(T + 1):
            for i in range(N):
                SoC_val[s, t, i] = SoC[s, t, i].X

    P_grid_import_val = np.array(
        [[P_grid_import[s, t].X for t in range(T)] for s in range(S)],
        dtype=np.float64,
    )

    P_grid_export_val = np.array(
        [[P_grid_export[s, t].X for t in range(T)] for s in range(S)],
        dtype=np.float64,
    )

    U_unmet_val = np.array(
        [[U_unmet[s, t].X for t in range(T)] for s in range(S)],
        dtype=np.float64,
    )

    U_curt_val = np.array(
        [[U_curt[s, t].X for t in range(T)] for s in range(S)],
        dtype=np.float64,
    )

    beta_ch_val = np.array([beta_ch[i].X for i in range(N)], dtype=np.float64)
    beta_dis_val = np.array([beta_dis[i].X for i in range(N)], dtype=np.float64)

    network_check = {}

    if enforce_network_first_action and load_ST69_kw is not None and pv_ST69_kw is not None:
        network_check = correct_first_action_by_power_flow(
            load_S69_kw=load_ST69_kw[:, 0, :],
            pv_S69_kw=pv_ST69_kw[:, 0, :],
            beta_ch_kw=beta_ch_val,
            beta_dis_kw=beta_dis_val,
            topology_config=topology_config,
            topology_case=topology_case,
        )

        beta_ch_val = network_check["beta_ch_kw_corrected"]
        beta_dis_val = network_check["beta_dis_kw_corrected"]

    action = np.zeros(N, dtype=np.float64)

    for i in range(N):
        if beta_dis_val[i] > beta_ch_val[i]:
            action[i] = beta_dis_val[i] / p_dis_max_kw
        elif beta_ch_val[i] > beta_dis_val[i]:
            action[i] = -beta_ch_val[i] / p_ch_max_kw
        else:
            action[i] = 0.0

    action = np.clip(action, -1.0, 1.0)

    return {
        "topology_case": topology_case,
        "status": status_str,
        "objective_value": float(m.ObjVal),
        "solve_time_sec": float(solve_time_sec),

        "beta_ch_kw": beta_ch_val,
        "beta_dis_kw": beta_dis_val,
        "action": action,

        "P_ch_kw": P_ch_val,
        "P_dis_kw": P_dis_val,
        "SoC": SoC_val,

        "P_grid_import_kw": P_grid_import_val,
        "P_grid_export_kw": P_grid_export_val,
        "U_unmet_kw": U_unmet_val,
        "U_curt_kw": U_curt_val,

        "network_check": network_check,
    }


def solve_first_action(
    load_ST69_kw: np.ndarray,
    pv_ST69_kw: np.ndarray,
    price_T: np.ndarray,
    soc0: np.ndarray,
    n_scenarios: int,
    horizon_t: int,
    topology_config: Optional[dict] = None,
    topology_case: str = "TP1",
) -> Dict[str, Any]:

    bundle = build_stochastic_mpc_problem(
        S=int(n_scenarios),
        T=int(horizon_t),
    )

    return solve_stochastic_mpc(
        bundle=bundle,
        aggregate_netload_ST_kw=None,
        load_ST69_kw=load_ST69_kw,
        pv_ST69_kw=pv_ST69_kw,
        price_T=price_T,
        soc0=soc0,
        enforce_network_first_action=True,
        topology_config=topology_config,
        topology_case=topology_case,
    )