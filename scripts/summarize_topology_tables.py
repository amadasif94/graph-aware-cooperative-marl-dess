from pathlib import Path
import pandas as pd

BASE = Path("results/topology_generalization/csv")
OUT_DIR = Path("results/topology_generalization/summary_tables_ieee33")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gcn", "mlp", "tagconv", "gat"]

RUN_NAME_MAP = {
    "gat": [
        "gat_ieee33_tp_full_eval_run0_best",
        "gat_ieee33_tp_full_eval_run1_best",
        "gat_ieee33_tp_full_eval_run2_best",
    ],
    "gcn": [
        "gcn_ieee33_tp_full_eval_run0_best",
        "gcn_ieee33_tp_full_eval_run1_best",
        "gcn_ieee33_tp_full_eval_run2_best",
    ],
    "mlp": [
        "mlp_ieee33_tp_full_eval_run0_best",
        "mlp_ieee33_tp_full_eval_run1_best",
        "mlp_ieee33_tp_full_eval_run2_best",
    ],
    "tagconv": [
        "tagconv_ieee33_tp_full_eval_run0_best",
        "tagconv_ieee33_tp_full_eval_run1_best",
        "tagconv_ieee33_tp_full_eval_run2_best",
    ],
}

METRICS = {
    "mean_total_reward_mean": "Avg Reward",
    "mean_energy_cost": "Energy Cost",
    "mean_grid_import_mwh": "Grid Import",
    "mean_curtailment_mwh": "Curtailment",
    "mean_voltage_deviation": "Voltage Dev.",
    "mean_throughput_mwh": "Throughput",
    "worst_min_voltage_pu": "Worst Min Voltage",
    "worst_max_voltage_pu": "Worst Max Voltage",
    "worst_line_current_pu": "Worst Line Current",
    "mean_feasible_rate": "Feasible Rate",
    "mean_converged_rate": "Converged Rate",
}

def learned_policy_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "policy" not in df.columns:
        return df

    df["policy"] = df["policy"].astype(str).str.lower().str.strip()
    return df[~df["policy"].isin(["zero", "random"])].copy()

def mean_std_text(mean, std, decimals=2):
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"

all_rows = []

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

all_df.to_csv(
    OUT_DIR / "all_ieee33_topology_generalization_runs.csv",
    index=False,
)

for model in MODELS:
    model_df = all_df[all_df["model"] == model].copy()

    if model_df.empty:
        print(f"WARNING: no data for {model}")
        continue

    table_rows = []

    for tp in sorted(model_df["topology_case"].unique()):
        tp_df = model_df[model_df["topology_case"] == tp]

        row = {"Topology": tp}

        for metric, label in METRICS.items():
            if metric not in tp_df.columns:
                continue

            mean_val = tp_df[metric].mean()
            std_val = tp_df[metric].std(ddof=1)

            decimals = 4 if metric in [
                "mean_voltage_deviation",
                "worst_min_voltage_pu",
                "worst_max_voltage_pu",
                "worst_line_current_pu",
                "mean_feasible_rate",
                "mean_converged_rate",
            ] else 2

            row[label] = mean_std_text(mean_val, std_val, decimals)

        table_rows.append(row)

    table = pd.DataFrame(table_rows)

    csv_path = OUT_DIR / f"{model}_ieee33_topology_table_mean_std.csv"
    table.to_csv(csv_path, index=False)

    print("\n===================================================")
    print(f"{model.upper()} IEEE33 TOPOLOGY TABLE")
    print("===================================================")
    print(table.to_string(index=False))

print("\n===================================================")
print("Saved CSV tables to:")
print(OUT_DIR)
print("===================================================")