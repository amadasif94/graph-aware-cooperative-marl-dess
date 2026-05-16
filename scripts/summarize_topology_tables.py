from pathlib import Path
import pandas as pd

BASE = Path("results/topology_generalization/csv")
OUT_DIR = Path("results/topology_generalization/summary_tables")

OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gcn", "mlp", "tagconv", "gat"]

RUN_NAME_MAP = {
    "gcn": [
        "tp_full_eval_gcn_run0_best",
        "tp_full_eval_gcn_run1_best",
        "tp_full_eval_gcn_best",
    ],
    "gat": [
        "tp_full_eval_gat_run0_best",
        "tp_full_eval_gat_best",
        "tp_full_eval_gat_run2_best",
    ],
    "mlp": [
        "tp_full_eval_mlp_run0_best",
        "tp_full_eval_mlp_run1_best",
        "tp_full_eval_mlp_best",
    ],
    "tagconv": [
        "tp_full_eval_tagconv_best",
        "tp_full_eval_tagconv_run1_best",
        "tp_full_eval_tagconv_run2_best",
    ],
}

METRICS = {
    "mean_total_reward_mean": "Avg Reward",
    "mean_energy_cost": "Energy Cost",
    "mean_voltage_deviation": "Voltage Dev.",
    "mean_throughput_mwh": "Throughput",
    "worst_min_voltage_pu": "Worst Min Voltage",
    "mean_feasible_rate": "Feasible Rate",
}

def learned_policy_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "policy" not in df.columns:
        return df

    df["policy"] = (
        df["policy"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df[
        ~df["policy"].isin(["zero", "random"])
    ].copy()

def mean_std_text(mean, std, decimals=2):
    if pd.isna(std):
        return f"{mean:.{decimals}f}"

    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"

all_rows = []

# ============================================================
# Read all topology evaluation runs
# ============================================================

for model, run_names in RUN_NAME_MAP.items():

    for run_idx, run_name in enumerate(run_names):

        agg_path = BASE / run_name / "aggregate_summary.csv"

        if not agg_path.exists():
            print(f"WARNING: missing {agg_path}")
            continue

        df = pd.read_csv(agg_path)
        df = learned_policy_only(df)

        if df.empty:
            print(f"WARNING: no learned-policy rows in {agg_path}")
            continue

        df["model"] = model
        df["seed_run"] = f"run_{run_idx}"
        df["source_folder"] = run_name

        all_rows.append(df)

if not all_rows:
    raise RuntimeError("No topology generalization results found.")

all_df = pd.concat(all_rows, ignore_index=True)

# Save raw merged results
all_df.to_csv(
    OUT_DIR / "all_topology_generalization_runs.csv",
    index=False
)

# ============================================================
# Create one summary table per model
# ============================================================

for model in MODELS:

    model_df = all_df[
        all_df["model"] == model
    ].copy()

    if model_df.empty:
        print(f"WARNING: no data for {model}")
        continue

    table_rows = []

    for tp in sorted(model_df["topology_case"].unique()):

        tp_df = model_df[
            model_df["topology_case"] == tp
        ]

        row = {
            "Topology": tp
        }

        for metric, label in METRICS.items():

            if metric not in tp_df.columns:
                continue

            mean_val = tp_df[metric].mean()
            std_val = tp_df[metric].std(ddof=1)

            decimals = 4 if metric in [
                "mean_voltage_deviation",
                "worst_min_voltage_pu",
                "mean_feasible_rate",
            ] else 2

            row[label] = mean_std_text(
                mean_val,
                std_val,
                decimals
            )

        table_rows.append(row)

    table = pd.DataFrame(table_rows)

    csv_path = (
        OUT_DIR /
        f"{model}_topology_table_mean_std.csv"
    )

    table.to_csv(csv_path, index=False)

    print("\n===================================================")
    print(f"{model.upper()} TOPOLOGY TABLE")
    print("===================================================")
    print(table.to_string(index=False))

print("\n===================================================")
print("Saved CSV tables to:")
print(OUT_DIR)
print("===================================================\n")