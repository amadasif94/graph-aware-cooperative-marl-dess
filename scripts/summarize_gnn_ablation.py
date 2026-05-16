from pathlib import Path
import pandas as pd

BASE = Path("results/topology_generalization/csv")
OUT_DIR = Path("results/topology_generalization/summary_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_NAME_MAP = {
    "GCN": [
        "tp_full_eval_gcn_run0_best",
        "tp_full_eval_gcn_run1_best",
        "tp_full_eval_gcn_best",
    ],
    "MLP": [
        "tp_full_eval_mlp_run0_best",
        "tp_full_eval_mlp_run1_best",
        "tp_full_eval_mlp_best",
    ],
    "TAGConv": [
        "tp_full_eval_tagconv_best",
        "tp_full_eval_tagconv_run1_best",
        "tp_full_eval_tagconv_run2_best",
    ],
    "GAT": [
        "tp_full_eval_gat_run0_best",
        "tp_full_eval_gat_best",
        "tp_full_eval_gat_run2_best",
    ],
}

METRICS = {
    "mean_total_reward_mean": "Avg Reward",
    "mean_energy_cost": "Energy Cost",
    "mean_voltage_deviation": "Voltage Dev.",
    "mean_throughput_mwh": "Throughput",
    "worst_line_current_pu": "Worst Line Current",
    "mean_feasible_rate": "Feasible Rate",
}

def keep_learned_policy(df):
    df = df.copy()
    if "policy" in df.columns:
        df["policy"] = df["policy"].astype(str).str.lower().str.strip()
        df = df[~df["policy"].isin(["zero", "random"])].copy()
    return df

def fmt(mean, std, decimals):
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"

rows = []

for model, run_names in RUN_NAME_MAP.items():
    for run_idx, run_name in enumerate(run_names):
        path = BASE / run_name / "aggregate_summary.csv"

        if not path.exists():
            print(f"WARNING: missing {path}")
            continue

        df = pd.read_csv(path)
        df = keep_learned_policy(df)

        if df.empty:
            print(f"WARNING: no learned-policy rows in {path}")
            continue

        df["model"] = model
        df["seed_run"] = f"run_{run_idx}"
        df["source_folder"] = run_name
        rows.append(df)

if not rows:
    raise RuntimeError("No topology-generalization aggregate summaries found.")

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(OUT_DIR / "gnn_ablation_all_runs_tp_results.csv", index=False)

summary_rows = []

for model in RUN_NAME_MAP.keys():
    model_df = all_df[all_df["model"] == model].copy()

    row = {"Model": model}

    for metric, label in METRICS.items():
        if metric not in model_df.columns:
            continue

        mean_val = model_df[metric].mean()
        std_val = model_df[metric].std(ddof=1)

        decimals = 4 if metric in [
            "mean_voltage_deviation",
            "worst_line_current_pu",
            "mean_feasible_rate",
        ] else 2

        row[label] = fmt(mean_val, std_val, decimals)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Sort by Avg Reward numerically, highest/least negative first
def extract_mean(x):
    return float(str(x).split("+/-")[0].strip())

summary_df["_reward_mean"] = summary_df["Avg Reward"].apply(extract_mean)
summary_df = summary_df.sort_values("_reward_mean", ascending=False).drop(columns="_reward_mean")

out_csv = OUT_DIR / "gnn_ablation_summary_mean_std.csv"
summary_df.to_csv(out_csv, index=False)

print("\n===================================================")
print("GNN ABLATION SUMMARY")
print("Averaged across TP1-TP7 and run_0/run_1/run_2")
print("===================================================\n")
print(summary_df.to_string(index=False))

print("\nSaved:")
print(out_csv)
print(OUT_DIR / "gnn_ablation_all_runs_tp_results.csv")
print("===================================================\n")