#!/usr/bin/env python3
"""
Build paired episode-level cost data and Wilcoxon significance tables for the
IEEE 33-bus and IEEE 69-bus topology-generalization experiments.

Expected project structure
--------------------------
results/topology_generalization/
    csv/
        mlp_ieee33_tp_full_eval_run0_best/episode_summary.csv
        gcn_ieee33_tp_full_eval_run0_best/episode_summary.csv
        ...
    csv_ieee69/
        mlp_ieee69_tp_full_eval_run0_best/episode_summary.csv
        gcn_ieee69_tp_full_eval_run0_best/episode_summary.csv
        ...

What this script does
---------------------
1. Finds every episode_summary.csv under csv/ and csv_ieee69/.
2. Infers architecture (MLP, GCN, GAT, TAGConv) and seed/run from folder names.
3. Checks that episode IDs and start indices align across runs/controllers.
4. Averages the three independent runs for each architecture, topology, and day.
5. Pairs each GNN with MLP on the same:
       feeder + topology_case + episode_id + start_index
6. Runs paired Wilcoxon signed-rank tests on daily total_energy_cost.
7. Reports win rate, mean/median differences, percentage improvement,
   matched-pairs rank-biserial effect size, and Holm-adjusted p-values.
8. Writes clean CSV files that can be sent directly to a collaborator.

Usage
-----
From the repository root:

    python scripts/build_paired_wilcoxon_data.py

Or with explicit paths:

    python scripts/build_paired_wilcoxon_data.py \
        --project_root /network/rit/lab/dahlin_lab/amad/graph_marl_dess \
        --out_dir results/statistical_analysis

Dependencies
------------
    pandas, numpy, scipy
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ARCHITECTURES = ("mlp", "gcn", "gat", "tagconv")
GNN_ARCHITECTURES = ("gcn", "gat", "tagconv")

REQUIRED_COLUMNS = {
    "topology_case",
    "episode_id",
    "start_index",
    "total_energy_cost",
}

# Additional episode-level metrics retained when present.
OPTIONAL_METRICS = [
    "total_reward_mean",
    "total_reward_team",
    "total_grid_import_mwh",
    "total_curtailment_mwh",
    "total_throughput_mwh",
    "mean_voltage_deviation",
    "max_voltage_deviation",
    "min_voltage_pu",
    "max_voltage_pu",
    "max_line_current_pu",
    "max_voltage_violation",
    "max_line_current_violation",
    "infeasible_requested_count",
    "feasible_rate",
    "converged_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired episode data and Wilcoxon cost tests."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("."),
        help="Root of graph_marl_dess repository. Default: current directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<project_root>/results/statistical_analysis"
        ),
    )
    parser.add_argument(
        "--expected_runs",
        type=int,
        default=3,
        help="Expected number of independent runs per architecture. Default: 3.",
    )
    parser.add_argument(
        "--expected_episodes",
        type=int,
        default=31,
        help="Expected episodes per topology and run. Default: 31.",
    )
    return parser.parse_args()


def infer_architecture(folder_name: str) -> str:
    name = folder_name.lower()
    for arch in ARCHITECTURES:
        if re.search(rf"(^|_){re.escape(arch)}(_|$)", name):
            return arch
    raise ValueError(
        f"Could not infer architecture from folder name: {folder_name}"
    )


def infer_run(folder_name: str) -> int:
    match = re.search(r"(?:^|_)run[_-]?(\d+)(?:_|$)", folder_name.lower())
    if not match:
        raise ValueError(f"Could not infer run number from: {folder_name}")
    return int(match.group(1))


def discover_files(project_root: Path) -> list[tuple[str, Path]]:
    base = project_root / "results" / "topology_generalization"
    roots = [
        ("IEEE33", base / "csv"),
        ("IEEE69", base / "csv_ieee69"),
    ]

    found: list[tuple[str, Path]] = []
    for feeder, root in roots:
        if not root.exists():
            print(f"WARNING: directory not found: {root}", file=sys.stderr)
            continue
        for path in sorted(root.glob("*/episode_summary.csv")):
            found.append((feeder, path))

    if not found:
        raise FileNotFoundError(
            "No episode_summary.csv files were found under:\n"
            f"  {base / 'csv'}\n"
            f"  {base / 'csv_ieee69'}"
        )
    return found


def load_all_episode_files(
    files: Iterable[tuple[str, Path]]
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []

    for feeder, path in files:
        folder = path.parent.name
        architecture = infer_architecture(folder)
        run = infer_run(folder)

        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        # Normalize key fields.
        df["topology_case"] = df["topology_case"].astype(str).str.upper()
        df["episode_id"] = pd.to_numeric(df["episode_id"], errors="raise").astype(int)
        df["start_index"] = pd.to_numeric(df["start_index"], errors="raise").astype(int)
        df["total_energy_cost"] = pd.to_numeric(
            df["total_energy_cost"], errors="raise"
        )

        # Keep only the trained controller if the file also contains
        # zero-action or random-policy baselines.
        if "policy" in df.columns:
            policy_norm = df["policy"].astype(str).str.strip().str.lower()
            learned_aliases = {
                "maddpg",
                "mlp",
                "mlp_maddpg",
                "gcn",
                "gat",
                "tagconv",
            }
            learned_mask = policy_norm.isin(learned_aliases)

            if learned_mask.any():
                df = df.loc[learned_mask].copy()
            else:
                raise ValueError(
                    f"{path} has a policy column, but no recognized learned "
                    f"controller rows. Found: {sorted(policy_norm.unique())}"
                )

        key_cols = ["topology_case", "episode_id", "start_index"]

        # Some evaluation files may contain repeated copies caused by
        # appending results across repeated executions. Exact duplicate rows
        # are safe to remove. Conflicting duplicates remain an error.
        exact_before = len(df)
        df = df.drop_duplicates().copy()
        exact_duplicates_removed = exact_before - len(df)

        duplicate_mask = df.duplicated(key_cols, keep=False)
        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count:
            conflict_preview = (
                df.loc[
                    duplicate_mask,
                    key_cols + ["total_energy_cost"],
                ]
                .sort_values(key_cols)
                .head(20)
                .to_string(index=False)
            )
            raise ValueError(
                f"{path} contains conflicting duplicate episode keys after "
                f"policy filtering and exact-row deduplication.\n"
                f"Preview:\n{conflict_preview}\n"
                "This usually means the same output file contains results "
                "from multiple repeated evaluations. Use a clean run folder "
                "or separate the repeated evaluations before testing."
            )

        keep_cols = [
            "topology_case",
            "episode_id",
            "start_index",
            "total_energy_cost",
        ] + [c for c in OPTIONAL_METRICS if c in df.columns]

        part = df[keep_cols].copy()
        part.insert(0, "run", run)
        part.insert(0, "architecture", architecture)
        part.insert(0, "feeder", feeder)
        part["source_folder"] = folder
        part["source_file"] = str(path)

        frames.append(part)

        for topology, group in part.groupby("topology_case", sort=True):
            quality_rows.append(
                {
                    "feeder": feeder,
                    "architecture": architecture,
                    "run": run,
                    "topology_case": topology,
                    "episodes": int(len(group)),
                    "unique_episode_ids": int(group["episode_id"].nunique()),
                    "unique_start_indices": int(group["start_index"].nunique()),
                    "duplicate_keys": duplicate_count,
                    "exact_duplicate_rows_removed": int(exact_duplicates_removed),
                    "source_folder": folder,
                    "status": "loaded",
                }
            )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(
        ["feeder", "architecture", "run", "topology_case", "episode_id"]
    ).reset_index(drop=True)

    return all_df, quality_rows


def validate_design(
    all_df: pd.DataFrame,
    expected_runs: int,
    expected_episodes: int,
    quality_rows: list[dict],
) -> pd.DataFrame:
    """
    Add design checks. Warnings are recorded rather than silently ignored.
    """
    report = pd.DataFrame(quality_rows)

    grouped = (
        all_df.groupby(["feeder", "architecture", "topology_case"])
        .agg(
            n_runs=("run", "nunique"),
            total_rows=("episode_id", "size"),
            n_episode_ids=("episode_id", "nunique"),
            n_start_indices=("start_index", "nunique"),
        )
        .reset_index()
    )

    grouped["expected_runs"] = expected_runs
    grouped["expected_episodes_per_run"] = expected_episodes
    grouped["run_count_ok"] = grouped["n_runs"].eq(expected_runs)
    grouped["episode_count_ok"] = grouped["n_episode_ids"].eq(expected_episodes)
    grouped["row_count_ok"] = grouped["total_rows"].eq(
        expected_runs * expected_episodes
    )
    grouped["status"] = np.where(
        grouped[["run_count_ok", "episode_count_ok", "row_count_ok"]].all(axis=1),
        "OK",
        "CHECK",
    )

    # Check whether all controllers use the exact same episode/start-index keys.
    alignment_rows: list[dict] = []
    for (feeder, topology), block in all_df.groupby(
        ["feeder", "topology_case"], sort=True
    ):
        keys_by_arch: dict[str, set[tuple[int, int]]] = {}
        for arch, arch_df in block.groupby("architecture"):
            keys_by_arch[arch] = set(
                zip(
                    arch_df["episode_id"].astype(int),
                    arch_df["start_index"].astype(int),
                )
            )

        mlp_keys = keys_by_arch.get("mlp", set())
        for arch in GNN_ARCHITECTURES:
            gnn_keys = keys_by_arch.get(arch, set())
            alignment_rows.append(
                {
                    "feeder": feeder,
                    "topology_case": topology,
                    "comparison": f"{arch}_vs_mlp",
                    "mlp_unique_keys": len(mlp_keys),
                    "gnn_unique_keys": len(gnn_keys),
                    "common_keys": len(mlp_keys & gnn_keys),
                    "mlp_only_keys": len(mlp_keys - gnn_keys),
                    "gnn_only_keys": len(gnn_keys - mlp_keys),
                    "aligned": mlp_keys == gnn_keys and len(mlp_keys) > 0,
                }
            )

    alignment = pd.DataFrame(alignment_rows)
    return report, grouped, alignment


def average_runs_per_episode(all_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in ["total_energy_cost"] + OPTIONAL_METRICS
        if c in all_df.columns
    ]

    averaged = (
        all_df.groupby(
            [
                "feeder",
                "architecture",
                "topology_case",
                "episode_id",
                "start_index",
            ],
            as_index=False,
        )
        .agg(
            **{f"{c}_seed_mean": (c, "mean") for c in metric_cols},
            **{f"{c}_seed_std": (c, "std") for c in metric_cols},
            n_runs=("run", "nunique"),
        )
    )

    return averaged.sort_values(
        ["feeder", "architecture", "topology_case", "episode_id"]
    ).reset_index(drop=True)


def matched_rank_biserial(differences: np.ndarray) -> float:
    """
    Matched-pairs rank-biserial correlation.

    Differences are defined as GNN cost - MLP cost.
    Negative values therefore favor the GNN.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = rankdata(np.abs(d), method="average")
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg

    return 0.0 if denom == 0 else (w_pos - w_neg) / denom


def safe_wilcoxon(
    differences: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) == 0:
        return np.nan, np.nan

    if np.allclose(d, 0.0):
        return 0.0, 1.0

    result = wilcoxon(
        d,
        alternative=alternative,
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """
    Holm family-wise error correction.
    """
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(p))[0]

    if len(valid_idx) == 0:
        return pd.Series(out, index=pvalues.index)

    valid_p = p[valid_idx]
    order = np.argsort(valid_p)
    m = len(valid_p)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank_position, sorted_position in enumerate(order):
        raw = valid_p[sorted_position]
        adjusted = (m - rank_position) * raw
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank_position] = min(running_max, 1.0)

    for rank_position, sorted_position in enumerate(order):
        out[valid_idx[sorted_position]] = adjusted_sorted[rank_position]

    return pd.Series(out, index=pvalues.index)


def build_pairs_and_tests(
    averaged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cost_col = "total_energy_cost_seed_mean"
    key_cols = ["feeder", "topology_case", "episode_id", "start_index"]

    cost_long = averaged[
        key_cols + ["architecture", cost_col, "n_runs"]
    ].copy()

    wide = cost_long.pivot_table(
        index=key_cols,
        columns="architecture",
        values=cost_col,
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    pair_rows: list[pd.DataFrame] = []
    test_rows: list[dict] = []

    for (feeder, topology), group in wide.groupby(
        ["feeder", "topology_case"], sort=True
    ):
        if "mlp" not in group.columns:
            continue

        for arch in GNN_ARCHITECTURES:
            if arch not in group.columns:
                continue

            pair = group[
                ["feeder", "topology_case", "episode_id", "start_index", "mlp", arch]
            ].dropna().copy()

            if pair.empty:
                continue

            pair = pair.rename(
                columns={
                    "mlp": "mlp_cost",
                    arch: "gnn_cost",
                }
            )
            pair["gnn_architecture"] = arch
            pair["difference_gnn_minus_mlp"] = (
                pair["gnn_cost"] - pair["mlp_cost"]
            )
            pair["improvement_percent"] = np.where(
                pair["mlp_cost"].abs() > 1e-12,
                100.0
                * (pair["mlp_cost"] - pair["gnn_cost"])
                / pair["mlp_cost"],
                np.nan,
            )
            pair["winner"] = np.select(
                [
                    pair["gnn_cost"] < pair["mlp_cost"],
                    pair["gnn_cost"] > pair["mlp_cost"],
                ],
                [arch, "mlp"],
                default="tie",
            )
            pair_rows.append(pair)

            diff = pair["difference_gnn_minus_mlp"].to_numpy(dtype=float)
            stat_two, p_two = safe_wilcoxon(diff, alternative="two-sided")
            stat_less, p_less = safe_wilcoxon(diff, alternative="less")

            wins = int((diff < 0).sum())
            losses = int((diff > 0).sum())
            ties = int((diff == 0).sum())

            mean_mlp = float(pair["mlp_cost"].mean())
            mean_gnn = float(pair["gnn_cost"].mean())
            mean_diff = float(np.mean(diff))
            median_diff = float(np.median(diff))
            mean_improvement_pct = (
                float(100.0 * (mean_mlp - mean_gnn) / mean_mlp)
                if abs(mean_mlp) > 1e-12
                else np.nan
            )

            test_rows.append(
                {
                    "feeder": feeder,
                    "topology_case": topology,
                    "gnn_architecture": arch,
                    "comparison": f"{arch.upper()} vs MLP",
                    "paired_episodes": int(len(pair)),
                    "mlp_mean_cost": mean_mlp,
                    "gnn_mean_cost": mean_gnn,
                    "mean_difference_gnn_minus_mlp": mean_diff,
                    "median_difference_gnn_minus_mlp": median_diff,
                    "mean_cost_improvement_percent": mean_improvement_pct,
                    "gnn_wins": wins,
                    "mlp_wins": losses,
                    "ties": ties,
                    "gnn_win_rate_excluding_ties": (
                        wins / (wins + losses) if (wins + losses) else np.nan
                    ),
                    "wilcoxon_statistic_two_sided": stat_two,
                    "p_value_two_sided": p_two,
                    "wilcoxon_statistic_gnn_lower": stat_less,
                    "p_value_gnn_lower_one_sided": p_less,
                    "rank_biserial_gnn_minus_mlp": matched_rank_biserial(diff),
                }
            )

    paired_long = (
        pd.concat(pair_rows, ignore_index=True)
        if pair_rows
        else pd.DataFrame()
    )
    tests = pd.DataFrame(test_rows)

    if not tests.empty:
        # Correct the three architecture comparisons within each
        # feeder/topology family.
        # transform() preserves the original one-level row index across
        # pandas versions, avoiding MultiIndex/reset_index incompatibilities.
        tests["p_holm_two_sided"] = (
            tests.groupby(["feeder", "topology_case"])[
                "p_value_two_sided"
            ]
            .transform(holm_adjust)
        )
        tests["p_holm_gnn_lower_one_sided"] = (
            tests.groupby(["feeder", "topology_case"])[
                "p_value_gnn_lower_one_sided"
            ]
            .transform(holm_adjust)
        )
        tests["significant_two_sided_0p05"] = (
            tests["p_holm_two_sided"] < 0.05
        )
        tests["significant_gnn_lower_0p05"] = (
            tests["p_holm_gnn_lower_one_sided"] < 0.05
        )

    return wide, paired_long, tests


def write_readme(out_dir: Path) -> None:
    text = """PAIRED WILCOXON ANALYSIS OUTPUTS
================================

1. all_episode_results_long.csv
   Raw episode-level data from every architecture and every independent run.

2. seed_averaged_episode_results.csv
   Three-run mean and standard deviation for each architecture, topology,
   episode_id, and start_index.

3. paired_costs_wide.csv
   One row per test day, with architecture costs in separate columns.

4. paired_daily_cost_differences.csv
   Direct GNN-vs-MLP matched pairs. The column
   difference_gnn_minus_mlp is negative when the GNN is cheaper.

5. wilcoxon_cost_results.csv
   Per-feeder, per-topology statistical results:
   - two-sided paired Wilcoxon p-value
   - one-sided p-value for GNN cost < MLP cost
   - Holm-adjusted p-values across GCN/GAT/TAGConv
   - GNN win counts and win rates
   - mean and median paired differences
   - percentage cost improvement
   - matched-pairs rank-biserial effect size

6. design_summary.csv
   Checks expected runs and episode counts.

7. pairing_alignment_check.csv
   Confirms that MLP and each GNN share identical
   (episode_id, start_index) pairing keys.

8. loaded_file_report.csv
   Inventory of every loaded episode_summary.csv.

Recommended paper reporting
---------------------------
Use the two-sided Holm-adjusted p-value as the conservative primary result.
The directional one-sided result may be included only if the hypothesis
"GNN cost is lower than MLP cost" was specified before examining the data.

Interpretation
--------------
difference_gnn_minus_mlp < 0  : GNN is cheaper
rank_biserial < 0             : effect favors GNN
"""
    (out_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else project_root / "results" / "statistical_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(project_root)
    print(f"Found {len(files)} episode_summary.csv files.")

    all_df, quality_rows = load_all_episode_files(files)
    loaded_report, design_summary, alignment = validate_design(
        all_df=all_df,
        expected_runs=args.expected_runs,
        expected_episodes=args.expected_episodes,
        quality_rows=quality_rows,
    )

    averaged = average_runs_per_episode(all_df)
    paired_wide, paired_long, tests = build_pairs_and_tests(averaged)

    outputs = {
        "all_episode_results_long.csv": all_df,
        "seed_averaged_episode_results.csv": averaged,
        "paired_costs_wide.csv": paired_wide,
        "paired_daily_cost_differences.csv": paired_long,
        "wilcoxon_cost_results.csv": tests,
        "design_summary.csv": design_summary,
        "pairing_alignment_check.csv": alignment,
        "loaded_file_report.csv": loaded_report,
    }

    for filename, df in outputs.items():
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved: {path}  ({len(df)} rows)")

    write_readme(out_dir)

    print("\nValidation summary")
    print("------------------")
    if not design_summary.empty:
        print(design_summary["status"].value_counts(dropna=False).to_string())
    if not alignment.empty:
        print(
            f"Aligned comparisons: {int(alignment['aligned'].sum())}/"
            f"{len(alignment)}"
        )

    if not tests.empty:
        print("\nWilcoxon results preview")
        print("------------------------")
        preview_cols = [
            "feeder",
            "topology_case",
            "comparison",
            "paired_episodes",
            "mean_cost_improvement_percent",
            "gnn_wins",
            "mlp_wins",
            "p_value_two_sided",
            "p_holm_two_sided",
        ]
        print(tests[preview_cols].to_string(index=False))

    print(f"\nAll outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
