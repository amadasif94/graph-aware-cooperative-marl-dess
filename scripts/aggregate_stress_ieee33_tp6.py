#!/usr/bin/env python3
"""
Aggregate IEEE 33-bus TP6 load-stress results across three training runs.

Expected result folders:
    results/topology_generalization/csv/
        mlp_ieee33_tp6_load1p3_run0/
        mlp_ieee33_tp6_load1p3_run1/
        mlp_ieee33_tp6_load1p3_run2/
        gcn_ieee33_tp6_load1p_run0/
        ...
        tagconv_ieee33_tp6_load1p3_run2/

Each folder must contain:
    aggregate_summary.csv

Outputs:
    results/topology_generalization/csv/
        ieee33_tp6_load1p3_aggregate/
            all_runs.csv
            aggregate_across_runs.csv
            paper_summary.csv
            paper_summary_formatted.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("results/topology_generalization/csv")

OUTPUT_DIR = BASE_DIR / "ieee33_tp6_load1p3_aggregate"

MODELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "tagconv": "TAGConv",
}

MODEL_ORDER = ["MLP", "GCN", "GAT", "TAGConv"]

EXPECTED_RUNS = {0, 1, 2}

FOLDER_PATTERN = re.compile(
    r"^(mlp|gcn|gat|tagconv)_ieee33_tp6_load1p3_run([0-2])$"
)

# Policy name expected in each model's aggregate_summary.csv
EXPECTED_POLICIES = {
    "mlp": {"mlp_maddpg", "mlp", "maddpg"},
    "gcn": {"maddpg", "gcn_maddpg", "gcn"},
    "gat": {"maddpg", "gat_maddpg", "gat"},
    "tagconv": {"maddpg", "tagconv_maddpg", "tagconv"},
}

# Metrics to aggregate across the three independently trained runs.
METRICS = [
    "mean_total_reward_mean",
    "mean_total_reward_team",
    "mean_energy_cost",
    "mean_grid_import_mwh",
    "mean_curtailment_mwh",
    "mean_throughput_mwh",
    "mean_voltage_deviation",
    "worst_min_voltage_pu",
    "worst_max_voltage_pu",
    "worst_line_current_pu",
    "worst_voltage_violation",
    "worst_line_current_violation",
    "mean_infeasible_requested_count",
    "mean_feasible_rate",
    "mean_converged_rate",
]


def find_result_folders() -> list[tuple[str, str, int, Path]]:
    """Find all model/run stress-test result folders."""
    found: list[tuple[str, str, int, Path]] = []

    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {BASE_DIR.resolve()}"
        )

    for folder in sorted(BASE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        match = FOLDER_PATTERN.match(folder.name)
        if match is None:
            continue

        model_key = match.group(1)
        run_id = int(match.group(2))
        display_name = MODELS[model_key]

        found.append((model_key, display_name, run_id, folder))

    return found


def validate_folder_set(
    folders: list[tuple[str, str, int, Path]],
) -> None:
    """Verify that all four models contain runs 0, 1, and 2."""
    runs_by_model: dict[str, set[int]] = {
        model_key: set() for model_key in MODELS
    }

    for model_key, _, run_id, _ in folders:
        runs_by_model[model_key].add(run_id)

    errors = []

    for model_key, expected_display_name in MODELS.items():
        actual_runs = runs_by_model[model_key]
        missing_runs = EXPECTED_RUNS - actual_runs
        extra_runs = actual_runs - EXPECTED_RUNS

        if missing_runs:
            errors.append(
                f"{expected_display_name}: missing runs "
                f"{sorted(missing_runs)}"
            )

        if extra_runs:
            errors.append(
                f"{expected_display_name}: unexpected runs "
                f"{sorted(extra_runs)}"
            )

    if errors:
        raise RuntimeError(
            "The complete set of stress-test results was not found:\n  - "
            + "\n  - ".join(errors)
        )


def select_controller_row(
    df: pd.DataFrame,
    model_key: str,
    csv_path: Path,
) -> pd.DataFrame:
    """
    Select the TP6 learned-controller row.

    This removes zero-action and random-policy rows if they are present.
    """
    required_columns = {"topology_case", "policy"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    filtered = df.copy()

    filtered["topology_case"] = (
        filtered["topology_case"].astype(str).str.upper()
    )
    filtered["policy_normalized"] = (
        filtered["policy"].astype(str).str.lower().str.strip()
    )

    filtered = filtered[filtered["topology_case"] == "TP6"]

    expected_policies = EXPECTED_POLICIES[model_key]
    filtered = filtered[
        filtered["policy_normalized"].isin(expected_policies)
    ]

    if len(filtered) != 1:
        available = (
            df[["topology_case", "policy"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )

        raise ValueError(
            f"Expected exactly one TP6 learned-policy row in {csv_path}, "
            f"but found {len(filtered)}.\n"
            f"Available topology/policy combinations: {available}"
        )

    return filtered.drop(columns=["policy_normalized"])


def load_all_runs(
    folders: list[tuple[str, str, int, Path]],
) -> pd.DataFrame:
    """Read and combine the aggregate summary from every run."""
    rows = []

    for model_key, model_name, run_id, folder in folders:
        csv_path = folder / "aggregate_summary.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing aggregate summary: {csv_path}"
            )

        df = pd.read_csv(csv_path)
        selected = select_controller_row(
            df=df,
            model_key=model_key,
            csv_path=csv_path,
        ).copy()

        selected.insert(0, "model", model_name)
        selected.insert(1, "run", run_id)
        selected["source_folder"] = folder.name

        rows.append(selected)

    combined = pd.concat(rows, ignore_index=True)

    combined["model"] = pd.Categorical(
        combined["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(
        ["model", "run"]
    ).reset_index(drop=True)

    return combined


def validate_run_contents(all_runs: pd.DataFrame) -> None:
    """Perform basic consistency checks before aggregation."""
    errors = []

    for model in MODEL_ORDER:
        model_df = all_runs[all_runs["model"] == model]

        if len(model_df) != 3:
            errors.append(
                f"{model}: expected 3 rows but found {len(model_df)}"
            )

        actual_runs = set(model_df["run"].astype(int))
        if actual_runs != EXPECTED_RUNS:
            errors.append(
                f"{model}: expected runs {sorted(EXPECTED_RUNS)}, "
                f"found {sorted(actual_runs)}"
            )

        if "episodes" in model_df.columns:
            episode_counts = model_df["episodes"].dropna().unique()

            if len(episode_counts) != 1:
                errors.append(
                    f"{model}: inconsistent episode counts "
                    f"{episode_counts.tolist()}"
                )

            if len(episode_counts) == 1 and int(episode_counts[0]) != 31:
                errors.append(
                    f"{model}: expected 31 episodes per run, "
                    f"found {episode_counts[0]}"
                )

    if errors:
        raise RuntimeError(
            "Result validation failed:\n  - "
            + "\n  - ".join(errors)
        )


def aggregate_across_runs(
    all_runs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and sample standard deviation across run 0, 1, and 2.

    The standard deviation here represents variability across independently
    trained policies, matching the paper's mean ± standard-deviation format.
    """
    available_metrics = [
        metric for metric in METRICS if metric in all_runs.columns
    ]

    missing_metrics = [
        metric for metric in METRICS if metric not in all_runs.columns
    ]

    if missing_metrics:
        print(
            "Warning: the following optional metrics were not found and "
            "will be omitted:"
        )
        for metric in missing_metrics:
            print(f"  - {metric}")

    for metric in available_metrics:
        all_runs[metric] = pd.to_numeric(
            all_runs[metric],
            errors="raise",
        )

    records = []

    for model in MODEL_ORDER:
        model_df = all_runs[all_runs["model"] == model]

        record = {
            "model": model,
            "n_runs": len(model_df),
        }

        if "episodes" in model_df.columns:
            record["episodes_per_run"] = int(
                model_df["episodes"].iloc[0]
            )

        for metric in available_metrics:
            record[f"{metric}_across_runs_mean"] = model_df[
                metric
            ].mean()

            # pandas std() uses sample standard deviation with ddof=1.
            record[f"{metric}_across_runs_std"] = model_df[
                metric
            ].std(ddof=1)

        records.append(record)

    return pd.DataFrame(records)


def build_paper_summary(
    aggregate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create compact numerical and formatted paper-oriented summaries."""
    metric_map = {
        "reward": "mean_total_reward_mean",
        "cost_usd": "mean_energy_cost",
        "grid_import_mwh": "mean_grid_import_mwh",
        "curtailment_mwh": "mean_curtailment_mwh",
        "throughput_mwh": "mean_throughput_mwh",
        "voltage_deviation_pu": "mean_voltage_deviation",
        "worst_min_voltage_pu": "worst_min_voltage_pu",
        "worst_max_voltage_pu": "worst_max_voltage_pu",
        "worst_line_current_pu": "worst_line_current_pu",
        "infeasible_requested_count": (
            "mean_infeasible_requested_count"
        ),
        "feasible_rate": "mean_feasible_rate",
        "converged_rate": "mean_converged_rate",
    }

    numeric_records = []
    formatted_records = []

    for _, row in aggregate_df.iterrows():
        numeric = {
            "model": row["model"],
            "n_runs": int(row["n_runs"]),
        }

        if "episodes_per_run" in aggregate_df.columns:
            numeric["episodes_per_run"] = int(
                row["episodes_per_run"]
            )

        formatted = {
            "model": row["model"],
        }

        for output_name, source_metric in metric_map.items():
            mean_col = f"{source_metric}_across_runs_mean"
            std_col = f"{source_metric}_across_runs_std"

            if mean_col not in aggregate_df.columns:
                continue

            mean_value = float(row[mean_col])
            std_value = float(row[std_col])

            numeric[f"{output_name}_mean"] = mean_value
            numeric[f"{output_name}_std"] = std_value

            if output_name in {
                "reward",
                "cost_usd",
                "infeasible_requested_count",
            }:
                decimals = 2
            elif output_name in {
                "grid_import_mwh",
                "curtailment_mwh",
                "throughput_mwh",
            }:
                decimals = 4
            else:
                decimals = 4

            formatted[output_name] = (
                f"{mean_value:.{decimals}f} ± "
                f"{std_value:.{decimals}f}"
            )

        numeric_records.append(numeric)
        formatted_records.append(formatted)

    numeric_df = pd.DataFrame(numeric_records)
    formatted_df = pd.DataFrame(formatted_records)

    return numeric_df, formatted_df


def main() -> int:
    print("=" * 78)
    print("IEEE33 TP6 1.3x LOAD-STRESS AGGREGATION")
    print("=" * 78)
    print(f"Searching under: {BASE_DIR.resolve()}")

    folders = find_result_folders()
    validate_folder_set(folders)

    print(f"Found {len(folders)} result folders:")

    for _, model_name, run_id, folder in folders:
        print(f"  {model_name:8s} run_{run_id}: {folder}")

    all_runs = load_all_runs(folders)
    validate_run_contents(all_runs)

    aggregate_df = aggregate_across_runs(all_runs)

    paper_numeric_df, paper_formatted_df = build_paper_summary(
        aggregate_df
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs_path = OUTPUT_DIR / "all_runs.csv"
    aggregate_path = OUTPUT_DIR / "aggregate_across_runs.csv"
    paper_path = OUTPUT_DIR / "paper_summary.csv"
    formatted_path = OUTPUT_DIR / "paper_summary_formatted.csv"

    all_runs.to_csv(all_runs_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    paper_numeric_df.to_csv(paper_path, index=False)
    paper_formatted_df.to_csv(formatted_path, index=False)

    print()
    print("=" * 78)
    print("COMPACT STRESS-TEST SUMMARY")
    print("=" * 78)

    display_columns = [
        column
        for column in [
            "model",
            "reward",
            "cost_usd",
            "voltage_deviation_pu",
            "feasible_rate",
        ]
        if column in paper_formatted_df.columns
    ]

    print(
        paper_formatted_df[display_columns].to_string(index=False)
    )

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)
    print(f"Individual run rows:       {all_runs_path}")
    print(f"Complete aggregation:      {aggregate_path}")
    print(f"Numerical paper summary:   {paper_path}")
    print(f"Formatted mean ± std:      {formatted_path}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print()
        print("=" * 78)
        print("AGGREGATION FAILED")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)