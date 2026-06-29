"""
Configuration file for the IEEE 33-bus graph-aware cooperative MARL
framework for distributed energy storage system (DESS) coordination.

This configuration uses SMART-DS based 15-minute load/PV profiles and
NYISO day-ahead LBMP price data processed into:

    data/time_series/processed_15min_smartds.csv

All bus indices in the implementation use zero-based indexing:
    buses 0 to 32

DESS placement:
    one-based reference buses: [12, 16, 25, 30, 33]
    zero-based code buses:     [11, 15, 24, 29, 32]

The environment uses:
    - 15-minute decision intervals,
    - one-day episodes with 96 time steps,
    - active-power DESS actions,
    - GridTensor-based power-flow evaluation,
    - normalized neural-network observations,
    - hard feasibility handling for voltage and line-current limits.

Important reward/constraint convention:
    - Minor one-step correction is logged but not penalized.
    - Major correction / fallback-to-zero is penalized through infeasible_action.
"""

from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
NETWORK_DIR = DATA_DIR / "network" / "ieee33"
TIME_SERIES_DIR = DATA_DIR / "time_series"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

BUS_DATA_PATH = NETWORK_DIR / "bus_data.csv"
LINE_DATA_PATH = NETWORK_DIR / "line_data.csv"
DESS_BUSES_PATH = NETWORK_DIR / "dess_buses.csv"

GRIDTENSOR_BUS_DATA_PATH = NETWORK_DIR / "gridtensor_bus.csv"
GRIDTENSOR_LINE_DATA_PATH = NETWORK_DIR / "gridtensor_line.csv"

PROCESSED_TIME_SERIES_PATH = TIME_SERIES_DIR / "processed_15min_smartds.csv"
GRAPH_SEQUENCE_PATH = PROCESSED_DIR / "ieee33_graph_sequence.npz"


# ============================================================
# Network settings
# ============================================================

CASE_NAME = "ieee33"

NUM_BUSES = 33
SLACK_BUS = 0

S_BASE_KVA = 1000.0
V_BASE_KV = 12.66

V_REF = 1.0
V_MIN = 0.95
V_MAX = 1.05


# ============================================================
# Data scaling
# ============================================================

LOAD_SCALING_FACTOR = 1.0
PV_SCALING_FACTOR = 1.0
PRICE_SCALING_FACTOR = 1.0


# ============================================================
# Time settings
# ============================================================

TIME_STEP_MINUTES = 15
DELTA_T_HOURS = TIME_STEP_MINUTES / 60.0
EPISODE_LENGTH = int(24 * 60 / TIME_STEP_MINUTES)


# ============================================================
# Observation normalization settings
# ============================================================

NORMALIZE_OBSERVATIONS = True

NORMALIZATION_MAX_TIME_INDEX = EPISODE_LENGTH - 1
NORMALIZATION_MAX_PRICE = 300.0

NORMALIZATION_MAX_LOAD_KW = None
NORMALIZATION_MAX_PV_KW = None
NORMALIZATION_MAX_GRID_IMPORT_KW = None

NORMALIZATION_MAX_VOLTAGE_DEVIATION = NUM_BUSES * 0.10
NORMALIZATION_MAX_LINE_CURRENT_PU = 5.0


# ============================================================
# DESS settings
# ============================================================

DESS_BUSES = [11, 15, 24, 29, 32]
NUM_DESS = len(DESS_BUSES)

BATTERY_CAPACITY_KWH = 500.0
BATTERY_MAX_CHARGE_KW = 75.0
BATTERY_MAX_DISCHARGE_KW = 75.0

BATTERY_CHARGE_EFFICIENCY = 0.95
BATTERY_DISCHARGE_EFFICIENCY = 0.95

SOC_MIN = 0.10
SOC_MAX = 0.90
SOC_INIT = 0.50


# ============================================================
# Power-flow settings
# ============================================================

POWER_FLOW_METHOD = "gridtensor"
POWER_FLOW_TOLERANCE = 1e-6
POWER_FLOW_MAX_ITERATIONS = 100

POWER_FLOW_NUMBA = True
POWER_FLOW_GPU = False

RETURN_BRANCH_FLOWS = True
RETURN_LINE_CURRENTS = True


# ============================================================
# Network operating limits
# ============================================================

GRID_EXPORT_LIMIT_KW = 500.0

LINE_CURRENT_LIMIT_MODE = "baseline_margin"
LINE_CURRENT_SECURITY_MARGIN_RHO = 2.0

GLOBAL_LINE_CURRENT_LIMIT_PU = 5.0
USE_LINE_SPECIFIC_CURRENT_LIMITS = True


# ============================================================
# Graph representation settings
# ============================================================

NODE_FEATURES = [
    "time",
    "price",
    "load_kw",
    "pv_kw",
    "soc_masked",
    "voltage_pu",
    "dess_indicator",
]

NODE_FEATURE_DIM = len(NODE_FEATURES)

EDGE_FEATURES = ["r_ohm", "x_ohm"]
EDGE_FEATURE_DIM = len(EDGE_FEATURES)

USE_EDGE_ATTR = True

VOLTAGE_FEATURE_MODE = "all_buses"
SOC_FEATURE_MODE = "dess_only_masked"


# ============================================================
# Action settings
# ============================================================

ACTION_LOW = -1.0
ACTION_HIGH = 1.0
ACTION_DIM_PER_AGENT = 1

ACTION_SCALING = "separate_charge_discharge_limits"


# ============================================================
# Constraint handling settings
# ============================================================

CONSTRAINT_HANDLING = "hard_feasibility"

ENFORCE_SOC_LIMITS = True
ENFORCE_DESS_POWER_LIMITS = True
ENFORCE_VOLTAGE_LIMITS = True
ENFORCE_LINE_CURRENT_LIMITS = True

HARD_CONSTRAINT_ACTION = "clip_or_correct"

MAX_CORRECTION_ATTEMPTS = 20
ACTION_CORRECTION_FACTOR = 0.80

# New corrected-feasibility logging/penalty behavior.
# A one-step correction, e.g. action ±1.0 -> ±0.75, is treated as minor
# and is logged but not penalized.
MINOR_CORRECTION_MAX_ITERATIONS = 2
PENALIZE_MINOR_CORRECTIONS = False


# ============================================================
# Reward settings
# ============================================================
# Research_Summer-consistent reward structure:
#
# r_i(t) =
#     W_GRID  * R_grid(t)
#   + W_LOCAL * R_local,i(t)
#   + W_KPI   * R_KPI,i(t)
#
# where:
#   W_GRID  = 0.60
#   W_LOCAL = 0.15
#   W_KPI   = 0.25
#
# The PHI_* values below are internal weights used inside each component.

W_GRID = 0.65
W_LOCAL = 0.15
W_KPI = 0.20

PHI_COST = 2.0
PHI_VOLTAGE_DEVIATION = 2.0
PHI_CYCLING = 0.15
PHI_SOC_RESERVE = 0.15

USE_KPI_REWARD = True

BETA_IMPORT = 1.0
BETA_VOLTAGE_DEV = 1.0
BETA_CURTAILMENT = 1.0
BETA_GRID_STRESS = 1.0

LAMBDA_CURTAILMENT = 1.0

INFEASIBLE_ACTION_PENALTY = -20.0
NONCONVERGENCE_PENALTY = -10000.0


# ============================================================
# Train / validation / test split
# ============================================================

TRAIN_START = "2018-01-01"
TRAIN_END = "2018-09-30"

VAL_START = "2018-10-01"
VAL_END = "2018-11-30"

TEST_START = "2018-12-01"
TEST_END = "2018-12-31"


# ============================================================
# Main configuration dictionary
# ============================================================

IEEE33_CONFIG = {
    "case_name": CASE_NAME,

    "paths": {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "network_dir": NETWORK_DIR,
        "time_series_dir": TIME_SERIES_DIR,
        "processed_dir": PROCESSED_DIR,
        "results_dir": RESULTS_DIR,

        "bus_data": BUS_DATA_PATH,
        "line_data": LINE_DATA_PATH,
        "dess_buses": DESS_BUSES_PATH,

        "gridtensor_bus_data": GRIDTENSOR_BUS_DATA_PATH,
        "gridtensor_line_data": GRIDTENSOR_LINE_DATA_PATH,

        "processed_time_series": PROCESSED_TIME_SERIES_PATH,
        "graph_sequence": GRAPH_SEQUENCE_PATH,
    },

    "network": {
        "num_buses": NUM_BUSES,
        "slack_bus": SLACK_BUS,
        "s_base_kva": S_BASE_KVA,
        "v_base_kv": V_BASE_KV,
        "v_ref": V_REF,
        "v_min": V_MIN,
        "v_max": V_MAX,

        "grid_export_limit_kw": GRID_EXPORT_LIMIT_KW,

        "line_current_limit_mode": LINE_CURRENT_LIMIT_MODE,
        "line_current_security_margin_rho": LINE_CURRENT_SECURITY_MARGIN_RHO,
        "global_line_current_limit_pu": GLOBAL_LINE_CURRENT_LIMIT_PU,
        "use_line_specific_current_limits": USE_LINE_SPECIFIC_CURRENT_LIMITS,

        "power_flow_method": POWER_FLOW_METHOD,
        "power_flow_tolerance": POWER_FLOW_TOLERANCE,
        "power_flow_max_iterations": POWER_FLOW_MAX_ITERATIONS,
        "power_flow_numba": POWER_FLOW_NUMBA,
        "power_flow_gpu": POWER_FLOW_GPU,
        "return_branch_flows": RETURN_BRANCH_FLOWS,
        "return_line_currents": RETURN_LINE_CURRENTS,
    },

    "data_scaling": {
        "load_scaling_factor": LOAD_SCALING_FACTOR,
        "pv_scaling_factor": PV_SCALING_FACTOR,
        "price_scaling_factor": PRICE_SCALING_FACTOR,
    },

    "normalization": {
        "normalize_observations": NORMALIZE_OBSERVATIONS,
        "max_time_index": NORMALIZATION_MAX_TIME_INDEX,
        "max_price": NORMALIZATION_MAX_PRICE,
        "max_load_kw": NORMALIZATION_MAX_LOAD_KW,
        "max_pv_kw": NORMALIZATION_MAX_PV_KW,
        "max_grid_import_kw": NORMALIZATION_MAX_GRID_IMPORT_KW,
        "max_voltage_deviation": NORMALIZATION_MAX_VOLTAGE_DEVIATION,
        "max_line_current_pu": NORMALIZATION_MAX_LINE_CURRENT_PU,
    },

    "dess": {
        "dess_buses": DESS_BUSES,
        "num_dess": NUM_DESS,
        "capacity_kwh": BATTERY_CAPACITY_KWH,
        "max_charge_kw": BATTERY_MAX_CHARGE_KW,
        "max_discharge_kw": BATTERY_MAX_DISCHARGE_KW,
        "charge_efficiency": BATTERY_CHARGE_EFFICIENCY,
        "discharge_efficiency": BATTERY_DISCHARGE_EFFICIENCY,
        "efficiency": BATTERY_CHARGE_EFFICIENCY,
        "soc_min": SOC_MIN,
        "soc_max": SOC_MAX,
        "soc_init": SOC_INIT,
    },

    "time": {
        "time_step_minutes": TIME_STEP_MINUTES,
        "delta_t_hours": DELTA_T_HOURS,
        "episode_length": EPISODE_LENGTH,
    },

    "graph": {
        "node_features": NODE_FEATURES,
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_features": EDGE_FEATURES,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "use_edge_attr": USE_EDGE_ATTR,
        "voltage_feature_mode": VOLTAGE_FEATURE_MODE,
        "soc_feature_mode": SOC_FEATURE_MODE,
    },

    "action": {
        "low": ACTION_LOW,
        "high": ACTION_HIGH,
        "dim_per_agent": ACTION_DIM_PER_AGENT,
        "scaling": ACTION_SCALING,
    },

    "constraints": {
        "handling": CONSTRAINT_HANDLING,

        "enforce_soc_limits": ENFORCE_SOC_LIMITS,
        "enforce_dess_power_limits": ENFORCE_DESS_POWER_LIMITS,
        "enforce_voltage_limits": ENFORCE_VOLTAGE_LIMITS,
        "enforce_line_current_limits": ENFORCE_LINE_CURRENT_LIMITS,

        "hard_constraint_action": HARD_CONSTRAINT_ACTION,

        "max_correction_attempts": MAX_CORRECTION_ATTEMPTS,
        "action_correction_factor": ACTION_CORRECTION_FACTOR,

        "minor_correction_max_iterations": MINOR_CORRECTION_MAX_ITERATIONS,
        "penalize_minor_corrections": PENALIZE_MINOR_CORRECTIONS,
    },

    "reward": {
        # Component-combination weights
        "w_grid": W_GRID,
        "w_local": W_LOCAL,
        "w_kpi": W_KPI,

        # Internal grid/local component weights
        "phi_cost": PHI_COST,
        "phi_voltage_deviation": PHI_VOLTAGE_DEVIATION,
        "phi_cycling": PHI_CYCLING,
        "phi_soc_reserve": PHI_SOC_RESERVE,

        # KPI reward settings
        "use_kpi_reward": USE_KPI_REWARD,

        "beta_import": BETA_IMPORT,
        "beta_voltage_dev": BETA_VOLTAGE_DEV,
        "beta_curtailment": BETA_CURTAILMENT,
        "beta_grid_stress": BETA_GRID_STRESS,

        "lambda_curtailment": LAMBDA_CURTAILMENT,

        # Penalties
        "infeasible_action_penalty": INFEASIBLE_ACTION_PENALTY,
        "nonconvergence_penalty": NONCONVERGENCE_PENALTY,
    },

    "splits": {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "val_start": VAL_START,
        "val_end": VAL_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
    },
}
