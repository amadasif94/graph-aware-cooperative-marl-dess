# ============================================================
# Optimization baseline configuration for IEEE33 MPC/SMPC
# ============================================================

from pathlib import Path

# ============================================================
# SYSTEM
# ============================================================

SYSTEM_NAME = "ieee33"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = PROJECT_ROOT / "results" / "mpc_smpc"

FORECAST_DIR = RESULTS_DIR / "forecasts"
SCENARIO_DIR = RESULTS_DIR / "scenarios"
OPT_RESULTS_DIR = RESULTS_DIR / "optimization"

MPC_RESULTS_DIR = OPT_RESULTS_DIR / "mpc"
SMPC_RESULTS_DIR = OPT_RESULTS_DIR / "smpc"
UNCONTROLLED_RESULTS_DIR = OPT_RESULTS_DIR / "uncontrolled"

for _p in [
    OPT_RESULTS_DIR,
    MPC_RESULTS_DIR,
    SMPC_RESULTS_DIR,
    UNCONTROLLED_RESULTS_DIR,
]:
    _p.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

LOAD_SCENARIO_FILE = SCENARIO_DIR / "load_scenarios_december_33bus.npz"
PV_SCENARIO_FILE = SCENARIO_DIR / "pv_scenarios_december_33bus.npz"

FORECAST_WIDE_FILE = FORECAST_DIR / "forecast_33bus_load_pv_sep_dec_wide.csv"
DECEMBER_FORECAST_FILE = FORECAST_DIR / "december_forecast_33bus_load_pv.csv"

SARIMA_MANIFEST_FILE = SCENARIO_DIR / "sarima_scenario_manifest.json"


# ============================================================
# RUN TOGGLES
# ============================================================

RUN_SMPC = True
RUN_DETERMINISTIC_MPC = True
RUN_UNCONTROLLED = False


# ============================================================
# TOPOLOGY SETTINGS
# ============================================================

DEFAULT_TOPOLOGY_CASE = "TP1"

TOPOLOGY_CASES = [
    "TP1",
    "TP2",
    "TP3",
    "TP4",
    "TP5",
    "TP6",
    "TP7",
]


# ============================================================
# IEEE33 / TIME SETTINGS
# ============================================================

NUM_BUSES = 33

# zero-based DESS buses: one-based [12, 16, 25, 30, 33]
DESS_BUSES = [11, 15, 24, 29, 32]
NUM_DESS = len(DESS_BUSES)

TIME_STEP_MINUTES = 15
DT_HOURS = 0.25

EPISODE_LENGTH = 96
HORIZON_T = 96

# For debugging, reduce this. For full December use None.
MAX_STEPS = None


# ============================================================
# SCENARIO SETTINGS
# ============================================================

N_SCEN_TOTAL = 100
SCENARIO_GRID = [100]

ALPHA_GRID = [1.0]


# ============================================================
# DESS / BATTERY SETTINGS
# ============================================================

BATTERY_CAPACITY_KWH = 500.0

P_CH_MAX_KW = 75.0
P_DIS_MAX_KW = 75.0

SOC_MIN = 0.10
SOC_MAX = 0.90
SOC0 = 0.50

ETA_CH = 0.95
ETA_DIS = 0.95


# ============================================================
# GRID / NETWORK SETTINGS
# ============================================================

GRID_EXPORT_LIMIT_KW = 500.0

V_REF = 1.0
V_MIN = 0.95
V_MAX = 1.05

ENFORCE_VOLTAGE_LIMITS = True
ENFORCE_LINE_CURRENT_LIMITS = True
ENFORCE_SOC_LIMITS = True
ENFORCE_POWER_LIMITS = True


# ============================================================
# POWER-FLOW ACTION CORRECTION SETTINGS
# ============================================================

MAX_CORRECTION_ATTEMPTS = 20
ACTION_CORRECTION_FACTOR = 0.80


# ============================================================
# OBJECTIVE / PENALTY SETTINGS
# ============================================================

W_ENERGY = 1.0
W_CYC = 1.0
W_VOLTAGE = 1.0
W_CURTAILMENT = 1.0

C_UNMET = 1000.0
C_CURT = 500.0

U_UNMET_MAX_KW = 10000.0
U_CURT_MAX_KW = 10000.0

SELL_PRICE_FACTOR = 1.0


# ============================================================
# SOLVER SETTINGS
# ============================================================

SOLVER_NAME = "GUROBI"

THREADS = 12
MIP_GAP = 1e-3
TIME_LIMIT = 120
WARM_START = True
VERBOSE = False


# ============================================================
# OUTPUT FILE HELPERS
# ============================================================

UNCONTROLLED_STEP_CSV = UNCONTROLLED_RESULTS_DIR / "step_metrics_uncontrolled.csv"
UNCONTROLLED_SUMMARY_CSV = UNCONTROLLED_RESULTS_DIR / "summary_uncontrolled.csv"

MPC_STEP_CSV = MPC_RESULTS_DIR / f"step_metrics_mpc_h{HORIZON_T}.csv"
MPC_SUMMARY_CSV = MPC_RESULTS_DIR / f"summary_mpc_h{HORIZON_T}.csv"

SMPC_STEP_CSV_TEMPLATE = str(
    SMPC_RESULTS_DIR / "step_metrics_smpc_h{horizon}_S{scenarios}_alpha_{alpha}.csv"
)

SMPC_SUMMARY_CSV_TEMPLATE = str(
    SMPC_RESULTS_DIR / "summary_smpc_h{horizon}_S{scenarios}_alpha_{alpha}.csv"
)


# ============================================================
# UTILITY
# ============================================================

def alpha_tag(alpha):
    return str(alpha).replace(".", "p")


def normalize_topology_case(topology_case):
    if topology_case is None:
        topology_case = DEFAULT_TOPOLOGY_CASE

    topology_case = str(topology_case).upper()

    if topology_case not in TOPOLOGY_CASES:
        raise ValueError(
            f"Unknown topology case: {topology_case}. "
            f"Available cases: {TOPOLOGY_CASES}"
        )

    return topology_case


def get_topology_output_dir(base_dir, topology_case):
    topology_case = normalize_topology_case(topology_case)
    out_dir = Path(base_dir) / topology_case
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_mpc_output_paths(horizon_t, topology_case=DEFAULT_TOPOLOGY_CASE):
    out_dir = get_topology_output_dir(MPC_RESULTS_DIR, topology_case)

    step_path = out_dir / f"step_metrics_mpc_{topology_case}_h{horizon_t}.csv"
    summary_path = out_dir / f"summary_mpc_{topology_case}_h{horizon_t}.csv"

    return step_path, summary_path


def get_smpc_output_paths(
    horizon_t,
    n_scen,
    alpha,
    topology_case=DEFAULT_TOPOLOGY_CASE,
):
    topology_case = normalize_topology_case(topology_case)
    tag = alpha_tag(alpha)

    out_dir = get_topology_output_dir(SMPC_RESULTS_DIR, topology_case)

    step_path = out_dir / (
        f"step_metrics_smpc_{topology_case}_h{horizon_t}_S{n_scen}_alpha_{tag}.csv"
    )

    summary_path = out_dir / (
        f"summary_smpc_{topology_case}_h{horizon_t}_S{n_scen}_alpha_{tag}.csv"
    )

    return step_path, summary_path


def print_config_summary():
    print("=" * 72)
    print("IEEE33 MPC/SMPC OPTIMIZATION BASELINE CONFIG")
    print("=" * 72)
    print(f"PROJECT_ROOT              : {PROJECT_ROOT}")
    print(f"RESULTS_DIR               : {RESULTS_DIR}")
    print(f"LOAD_SCENARIO_FILE        : {LOAD_SCENARIO_FILE}")
    print(f"PV_SCENARIO_FILE          : {PV_SCENARIO_FILE}")
    print(f"FORECAST_WIDE_FILE        : {FORECAST_WIDE_FILE}")
    print(f"DEFAULT_TOPOLOGY_CASE     : {DEFAULT_TOPOLOGY_CASE}")
    print(f"TOPOLOGY_CASES            : {TOPOLOGY_CASES}")
    print(f"NUM_BUSES                 : {NUM_BUSES}")
    print(f"DESS_BUSES                : {DESS_BUSES}")
    print(f"NUM_DESS                  : {NUM_DESS}")
    print(f"DT_HOURS                  : {DT_HOURS}")
    print(f"HORIZON_T                 : {HORIZON_T}")
    print(f"N_SCEN_TOTAL              : {N_SCEN_TOTAL}")
    print(f"SCENARIO_GRID             : {SCENARIO_GRID}")
    print(f"BATTERY_CAPACITY_KWH      : {BATTERY_CAPACITY_KWH}")
    print(f"P_CH_MAX_KW               : {P_CH_MAX_KW}")
    print(f"P_DIS_MAX_KW              : {P_DIS_MAX_KW}")
    print(f"SOC_MIN/MAX/INIT          : {SOC_MIN}, {SOC_MAX}, {SOC0}")
    print(f"ETA_CH/DIS                : {ETA_CH}, {ETA_DIS}")
    print(f"GRID_EXPORT_LIMIT_KW      : {GRID_EXPORT_LIMIT_KW}")
    print(f"V_MIN/MAX                 : {V_MIN}, {V_MAX}")
    print(f"MAX_CORRECTION_ATTEMPTS   : {MAX_CORRECTION_ATTEMPTS}")
    print(f"ACTION_CORRECTION_FACTOR  : {ACTION_CORRECTION_FACTOR}")
    print(f"SOLVER_NAME               : {SOLVER_NAME}")
    print("=" * 72)