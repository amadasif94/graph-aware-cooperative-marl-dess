from pathlib import Path
import pandas as pd


BASE = Path("results/csv/eval_ieee69")
OUT_DIR = BASE / "summary"

OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for model_dir in BASE.iterdir():
    if not model_dir.is_dir():
        continue

    if model_dir.name == "summary":
        continue

    model = model_dir.name.lower().strip()

    for run_dir in model_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        for step_dir in run_dir.glob("step_*"):
            if not step_dir.is_dir():
                continue

            agg_csv = step_dir / "aggregate_summary.csv"

            if not agg_csv.exists():
                continue

            try:
                step = int(step_dir.name.replace("step_", ""))
            except ValueError:
                continue

            df = pd.read_csv(agg_csv)

            if df.empty:
                continue

            if "policy" in df.columns:
                df["policy"] = (
                    df["policy"]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                )
                df = df[~df["policy"].isin(["zero", "random"])].copy()

            if df.empty:
                continue

            row = df.iloc[0].to_dict()
            row["model"] = model
            row["run"] = run_name
            row["checkpoint_step"] = step
            row["result_dir"] = str(step_dir)

            rows.append(row)

all_df = pd.DataFrame(rows)

if all_df.empty:
    raise RuntimeError(
        "No IEEE69 evaluation results found under results/csv/eval_ieee69."
    )

all_df = all_df.sort_values(
    ["model", "run", "checkpoint_step"]
).reset_index(drop=True)

safe_df = all_df.copy()

if "mean_feasible_rate" in safe_df.columns:
    safe_df = safe_df[safe_df["mean_feasible_rate"] >= 1.0]

if "mean_converged_rate" in safe_df.columns:
    safe_df = safe_df[safe_df["mean_converged_rate"] >= 1.0]

if "worst_voltage_violation" in safe_df.columns:
    safe_df = safe_df[safe_df["worst_voltage_violation"] <= 0.0]

if "worst_line_current_violation" in safe_df.columns:
    safe_df = safe_df[safe_df["worst_line_current_violation"] <= 0.0]

if safe_df.empty:
    print("WARNING: No checkpoints passed the IEEE69 safety filter.")
    print("Using all checkpoints instead.")
    safe_df = all_df.copy()

all_csv = OUT_DIR / "all_eval_results.csv"
safe_csv = OUT_DIR / "safe_eval_results.csv"

all_df.to_csv(all_csv, index=False)
safe_df.to_csv(safe_csv, index=False)

best_per_model_run = (
    safe_df
    .sort_values("mean_total_reward_mean", ascending=False)
    .groupby(["model", "run"], as_index=False)
    .head(1)
    .sort_values(["model", "run"])
    .reset_index(drop=True)
)

best_per_model_run_csv = OUT_DIR / "best_checkpoint_per_model_run.csv"
best_per_model_run.to_csv(best_per_model_run_csv, index=False)

best_per_model = (
    safe_df
    .sort_values("mean_total_reward_mean", ascending=False)
    .groupby("model", as_index=False)
    .head(1)
    .sort_values("model")
    .reset_index(drop=True)
)

best_per_model_csv = OUT_DIR / "best_checkpoint_per_model.csv"
best_per_model.to_csv(best_per_model_csv, index=False)

best_overall = (
    safe_df
    .sort_values("mean_total_reward_mean", ascending=False)
    .head(1)
    .reset_index(drop=True)
)

best_overall_csv = OUT_DIR / "best_overall_checkpoint.csv"
best_overall.to_csv(best_overall_csv, index=False)

summary_cols = [
    "policy",
    "model",
    "run",
    "checkpoint_step",
    "mean_total_reward_mean",
    "mean_energy_cost",
    "mean_grid_import_mwh",
    "mean_curtailment_mwh",
    "mean_throughput_mwh",
    "mean_voltage_deviation",
    "mean_feasible_rate",
    "mean_converged_rate",
    "result_dir",
]

summary_cols = [c for c in summary_cols if c in safe_df.columns]

print("\n===================================================")
print("IEEE69 BEST CHECKPOINT PER MODEL/RUN")
print("===================================================\n")
print(best_per_model_run[summary_cols].to_string(index=False))

print("\n===================================================")
print("IEEE69 BEST CHECKPOINT PER MODEL")
print("===================================================\n")
print(best_per_model[summary_cols].to_string(index=False))

print("\n===================================================")
print("IEEE69 BEST OVERALL CHECKPOINT")
print("===================================================\n")
print(best_overall[summary_cols].to_string(index=False))

print("\n===================================================")
print("Saved files:")
print(all_csv)
print(safe_csv)
print(best_per_model_run_csv)
print(best_per_model_csv)
print(best_overall_csv)
print("===================================================\n")