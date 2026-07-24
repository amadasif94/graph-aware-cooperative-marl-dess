"""
Paired Wilcoxon signed-rank analysis for the IEEE 33-bus TP6 stress test.

Statistical unit:
    Test episode/day.

For each architecture, load factor, and episode, results are first averaged
across the three independent training seeds. Each graph architecture is then
paired with the MLP using the same episode_id.

For each metric and load factor:
    GCN vs. MLP
    GAT vs. MLP
    TAGConv vs. MLP

The three p-values are corrected jointly using the Holm procedure.

Input:
    results/topology_generalization/csv/stress_combined.csv

Output:
    results/topology_generalization/csv/stress_wilcoxon_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "topology_generalization"
    / "csv"
)

INPUT_FILE = DATA_DIR / "stress_combined.csv"
OUTPUT_FILE = DATA_DIR / "stress_wilcoxon_results.csv"


# =============================================================================
# Analysis configuration
# =============================================================================

ALPHA = 0.05

BASELINE_MODEL = "mlp"

GRAPH_MODELS = [
    "gcn",
    "gat",
    "tagconv",
]

METRICS = {
    "cost": {
        "column": "total_energy_cost",
        "better": "lower",
    },
    "feasibility": {
        "column": "feasible_rate",
        "better": "higher",
    },
    "infeasible_requests": {
        "column": "infeasible_requested_count",
        "better": "lower",
    },
}

REQUIRED_COLUMNS = [
    "model",
    "load_factor",
    "seed",
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]


# =============================================================================
# Helper functions
# =============================================================================

def validate_input_file(file_path: Path) -> None:
    """Confirm that the expected input CSV exists."""

    if not file_path.exists():
        raise FileNotFoundError(
            "\nCombined stress-test file was not found.\n"
            f"Expected path:\n{file_path.resolve()}"
        )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Confirm that the input contains all required columns."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nThe input file is missing required columns.\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def holm_adjust(
    p_values: list[float],
) -> np.ndarray:
    """
    Apply Holm's step-down family-wise error correction.

    For m ordered raw p-values:
        adjusted p_(i) = max_{j <= i} [(m-j+1) p_(j)]

    Adjusted values are constrained to [0, 1].
    """

    p_array = np.asarray(
        p_values,
        dtype=float,
    )

    number_of_tests = len(p_array)

    if number_of_tests == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_array)
    ordered_p = p_array[order]

    adjusted_ordered = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, raw_p in enumerate(ordered_p):
        multiplier = number_of_tests - rank
        candidate = multiplier * raw_p

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted_ordered[rank] = min(
            running_maximum,
            1.0,
        )

    adjusted = np.empty(
        number_of_tests,
        dtype=float,
    )

    adjusted[order] = adjusted_ordered

    return adjusted


def run_paired_wilcoxon(
    baseline_values: np.ndarray,
    architecture_values: np.ndarray,
) -> tuple[float, float]:
    """
    Run a two-sided paired Wilcoxon signed-rank test.

    Returns:
        statistic, raw_p_value
    """

    differences = (
        architecture_values
        - baseline_values
    )

    if np.allclose(
        differences,
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        return 0.0, 1.0

    result = wilcoxon(
        architecture_values,
        baseline_values,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def count_outcomes(
    baseline_values: np.ndarray,
    architecture_values: np.ndarray,
    better_direction: str,
) -> tuple[int, int, int]:
    """
    Count architecture wins, ties, and losses against the MLP.

    A win means:
        lower value when lower is better;
        higher value when higher is better.
    """

    tolerance = 1e-12

    differences = (
        architecture_values
        - baseline_values
    )

    ties = int(
        np.sum(
            np.isclose(
                differences,
                0.0,
                rtol=0.0,
                atol=tolerance,
            )
        )
    )

    if better_direction == "lower":
        wins = int(
            np.sum(
                differences < -tolerance
            )
        )

        losses = int(
            np.sum(
                differences > tolerance
            )
        )

    elif better_direction == "higher":
        wins = int(
            np.sum(
                differences > tolerance
            )
        )

        losses = int(
            np.sum(
                differences < -tolerance
            )
        )

    else:
        raise ValueError(
            f"Unsupported better direction: {better_direction}"
        )

    return wins, ties, losses


def effect_difference(
    baseline_values: np.ndarray,
    architecture_values: np.ndarray,
) -> np.ndarray:
    """
    Return architecture minus MLP.

    Interpretation:
        Cost:
            negative values mean the graph model is cheaper.

        Feasibility:
            positive values mean the graph model is more feasible.

        Infeasible requests:
            negative values mean fewer infeasible requested actions.
    """

    return (
        architecture_values
        - baseline_values
    )


# =============================================================================
# Load and validate data
# =============================================================================

validate_input_file(
    INPUT_FILE
)

df = pd.read_csv(
    INPUT_FILE
)

validate_columns(
    df,
    REQUIRED_COLUMNS,
)


# =============================================================================
# Normalize data types
# =============================================================================

df["model"] = (
    df["model"]
    .astype(str)
    .str.strip()
    .str.lower()
)

numeric_columns = [
    "load_factor",
    "seed",
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    )


# =============================================================================
# Dataset checks
# =============================================================================

expected_models = {
    BASELINE_MODEL,
    *GRAPH_MODELS,
}

found_models = set(
    df["model"].unique()
)

missing_models = sorted(
    expected_models
    - found_models
)

if missing_models:
    raise ValueError(
        "\nExpected models are missing from the dataset:\n"
        f"{missing_models}"
    )

unexpected_models = sorted(
    found_models
    - expected_models
)

if unexpected_models:
    raise ValueError(
        "\nUnexpected models were found in the dataset:\n"
        f"{unexpected_models}"
    )

duplicate_mask = df.duplicated(
    subset=[
        "model",
        "load_factor",
        "seed",
        "episode_id",
    ],
    keep=False,
)

if duplicate_mask.any():
    duplicates = df.loc[
        duplicate_mask,
        [
            "model",
            "load_factor",
            "seed",
            "episode_id",
        ],
    ]

    raise ValueError(
        "\nDuplicate model/load/seed/episode rows were found:\n"
        f"{duplicates.to_string(index=False)}"
    )

if not np.allclose(
    df["converged_rate"].to_numpy(dtype=float),
    1.0,
):
    print(
        "Warning: converged_rate is not equal to 1.0 "
        "for every input row."
    )


# =============================================================================
# Average across training seeds for each test episode
# =============================================================================

episode_means = (
    df.groupby(
        [
            "model",
            "load_factor",
            "episode_id",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        total_energy_cost=(
            "total_energy_cost",
            "mean",
        ),
        feasible_rate=(
            "feasible_rate",
            "mean",
        ),
        infeasible_requested_count=(
            "infeasible_requested_count",
            "mean",
        ),
    )
)

load_factors = sorted(
    episode_means["load_factor"].unique()
)

print(
    "=" * 100
)

print(
    "PAIRED WILCOXON ANALYSIS"
)

print(
    "=" * 100
)

print(
    f"Input file: {INPUT_FILE.resolve()}"
)

print(
    "Statistical unit: episode after averaging across training seeds"
)

print(
    "Test: two-sided paired Wilcoxon signed-rank test"
)

print(
    "Multiple-testing correction: Holm correction across the "
    "three graph architectures within each metric and load factor"
)

print(
    f"Significance threshold: alpha = {ALPHA}"
)


# =============================================================================
# Run paired comparisons
# =============================================================================

all_results: list[dict] = []

for metric_name, metric_config in METRICS.items():
    metric_column = metric_config["column"]
    better_direction = metric_config["better"]

    for load_factor in load_factors:
        family_results: list[dict] = []

        baseline = (
            episode_means[
                episode_means["model"].eq(
                    BASELINE_MODEL
                )
                & np.isclose(
                    episode_means["load_factor"],
                    load_factor,
                )
            ][
                [
                    "episode_id",
                    metric_column,
                ]
            ]
            .rename(
                columns={
                    metric_column: "baseline_value",
                }
            )
        )

        if baseline.empty:
            raise ValueError(
                "\nNo MLP baseline observations were found.\n"
                f"Load factor: {load_factor}\n"
                f"Metric: {metric_name}"
            )

        for architecture in GRAPH_MODELS:
            graph_data = (
                episode_means[
                    episode_means["model"].eq(
                        architecture
                    )
                    & np.isclose(
                        episode_means["load_factor"],
                        load_factor,
                    )
                ][
                    [
                        "episode_id",
                        metric_column,
                    ]
                ]
                .rename(
                    columns={
                        metric_column: "architecture_value",
                    }
                )
            )

            paired = baseline.merge(
                graph_data,
                on="episode_id",
                how="inner",
                validate="one_to_one",
            ).sort_values(
                "episode_id"
            )

            expected_baseline_episodes = set(
                baseline["episode_id"]
            )

            expected_architecture_episodes = set(
                graph_data["episode_id"]
            )

            if (
                expected_baseline_episodes
                != expected_architecture_episodes
            ):
                missing_from_architecture = sorted(
                    expected_baseline_episodes
                    - expected_architecture_episodes
                )

                missing_from_baseline = sorted(
                    expected_architecture_episodes
                    - expected_baseline_episodes
                )

                raise ValueError(
                    "\nEpisode pairing mismatch.\n"
                    f"Metric: {metric_name}\n"
                    f"Load factor: {load_factor}\n"
                    f"Architecture: {architecture}\n"
                    f"Missing from architecture: "
                    f"{missing_from_architecture}\n"
                    f"Missing from MLP: "
                    f"{missing_from_baseline}"
                )

            baseline_values = paired[
                "baseline_value"
            ].to_numpy(dtype=float)

            architecture_values = paired[
                "architecture_value"
            ].to_numpy(dtype=float)

            statistic, raw_p_value = run_paired_wilcoxon(
                baseline_values=baseline_values,
                architecture_values=architecture_values,
            )

            differences = effect_difference(
                baseline_values=baseline_values,
                architecture_values=architecture_values,
            )

            wins, ties, losses = count_outcomes(
                baseline_values=baseline_values,
                architecture_values=architecture_values,
                better_direction=better_direction,
            )

            baseline_mean = float(
                np.mean(baseline_values)
            )

            architecture_mean = float(
                np.mean(architecture_values)
            )

            mean_difference = float(
                np.mean(differences)
            )

            median_difference = float(
                np.median(differences)
            )

            if metric_name == "cost":
                percent_difference = float(
                    100.0
                    * (
                        architecture_mean
                        - baseline_mean
                    )
                    / baseline_mean
                )

                percent_improvement = float(
                    100.0
                    * (
                        baseline_mean
                        - architecture_mean
                    )
                    / baseline_mean
                )

            else:
                percent_difference = np.nan
                percent_improvement = np.nan

            family_results.append(
                {
                    "metric": metric_name,
                    "arch": architecture,
                    "baseline": BASELINE_MODEL,
                    "load_factor": float(load_factor),
                    "n_pairs": int(len(paired)),
                    "wilcoxon_statistic": statistic,
                    "p_raw": raw_p_value,
                    "baseline_mean": baseline_mean,
                    "arch_mean": architecture_mean,
                    "mean_difference_arch_minus_mlp": mean_difference,
                    "median_difference_arch_minus_mlp": median_difference,
                    "percent_difference_arch_vs_mlp": percent_difference,
                    "percent_improvement_vs_mlp": percent_improvement,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "better_direction": better_direction,
                }
            )

        # Holm correction across GCN, GAT, and TAGConv for this
        # metric/load-factor family.
        raw_p_values = [
            row["p_raw"]
            for row in family_results
        ]

        corrected_p_values = holm_adjust(
            raw_p_values
        )

        for row, corrected_p_value in zip(
            family_results,
            corrected_p_values,
        ):
            row["p_holm"] = float(
                corrected_p_value
            )

            row["sig"] = bool(
                corrected_p_value < ALPHA
            )

            all_results.append(
                row
            )


# =============================================================================
# Construct and save output
# =============================================================================

results = pd.DataFrame(
    all_results
)

output_columns = [
    "metric",
    "arch",
    "baseline",
    "load_factor",
    "n_pairs",
    "wilcoxon_statistic",
    "p_raw",
    "p_holm",
    "sig",
    "baseline_mean",
    "arch_mean",
    "mean_difference_arch_minus_mlp",
    "median_difference_arch_minus_mlp",
    "percent_difference_arch_vs_mlp",
    "percent_improvement_vs_mlp",
    "wins",
    "ties",
    "losses",
    "better_direction",
]

results = (
    results[
        output_columns
    ]
    .sort_values(
        [
            "metric",
            "load_factor",
            "arch",
        ]
    )
    .reset_index(drop=True)
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# Print readable summaries
# =============================================================================

for metric_name in METRICS:
    print(
        "\n"
        + "=" * 100
    )

    print(
        metric_name.upper()
    )

    print(
        "=" * 100
    )

    metric_results = results[
        results["metric"].eq(
            metric_name
        )
    ].copy()

    display_columns = [
        "load_factor",
        "arch",
        "n_pairs",
        "baseline_mean",
        "arch_mean",
        "mean_difference_arch_minus_mlp",
        "wins",
        "ties",
        "losses",
        "p_raw",
        "p_holm",
        "sig",
    ]

    if metric_name == "cost":
        display_columns.insert(
            6,
            "percent_improvement_vs_mlp",
        )

    print(
        metric_results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.10g}",
        )
    )


print(
    "\n"
    + "=" * 100
)

print(
    "ANALYSIS COMPLETE"
)

print(
    "=" * 100
)

print(
    f"Rows written: {len(results)}"
)

print(
    f"Output file:\n{OUTPUT_FILE.resolve()}"
)

print(
    "\nExpected comparisons:"
)

print(
    f"  {len(METRICS)} metrics"
    f" × {len(load_factors)} load factors"
    f" × {len(GRAPH_MODELS)} architectures"
    f" = {len(METRICS) * len(load_factors) * len(GRAPH_MODELS)} rows"
)