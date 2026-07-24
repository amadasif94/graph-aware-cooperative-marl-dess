"""
Create the final IEEE 33-bus TP6 stress-test figure and LaTeX table
from episode-level evaluation data.

The figure contains:
    (a) operational feasibility,
    (b) percentage cost reduction relative to MLP,
    (c) requested-infeasible actions per episode.

The script also generates a LaTeX table body containing mean ± standard
deviation across the three training seeds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "topology_generalization"
    / "csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "topology_generalization"
    / "figures"
)

EPISODE_FILE = DATA_DIR / "stress_combined.csv"

WILCOXON_FILE = DATA_DIR / "stress_wilcoxon_results.csv"

OUTPUT_PDF = FIGURE_DIR / "stress_divergence_final.pdf"
OUTPUT_PNG = FIGURE_DIR / "stress_divergence_final.png"
OUTPUT_TABLE = DATA_DIR / "stress_table.tex"


# =============================================================================
# Plot configuration
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.6,
    }
)

MODELS = [
    "mlp",
    "gcn",
    "gat",
    "tagconv",
]

COLORS = {
    "mlp": "#7F7F7F",
    "gcn": "#2E75B6",
    "gat": "#8064A2",
    "tagconv": "#C0504D",
}

MARKERS = {
    "mlp": "s",
    "gcn": "o",
    "gat": "^",
    "tagconv": "D",
}

LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "tagconv": "TAGConv",
}


# =============================================================================
# Expected columns
# =============================================================================

EPISODE_REQUIRED_COLUMNS = [
    "model",
    "load_factor",
    "seed",
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]

STAT_REQUIRED_COLUMNS = [
    "metric",
    "arch",
    "load_factor",
    "sig",
]


# =============================================================================
# Validation helpers
# =============================================================================

def validate_file_exists(
    file_path: Path,
    description: str,
) -> None:
    """Raise a clear error when an expected input file is missing."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"\n{description} was not found.\n"
            f"Expected path:\n{file_path.resolve()}"
        )


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    file_path: Path,
) -> None:
    """Confirm that all required columns are available."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nRequired columns are missing.\n"
            f"File:\n{file_path.resolve()}\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_boolean_column(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common Boolean encodings to True/False.

    Supported examples:
        True, False
        1, 0
        "true", "false"
        "yes", "no"
        "*", ""
    """

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "significant",
        "sig",
        "*",
    }

    return normalized.isin(true_values)


def get_significance_flag(
    results: pd.DataFrame,
    metric: str,
    architecture: str,
    load_factor: float,
) -> bool:
    """
    Return the corrected significance flag for a model-load comparison.

    The comparison is assumed to be architecture versus MLP.
    """

    selected = results[
        results["metric"].eq(metric)
        & results["arch"].eq(architecture)
        & np.isclose(
            results["load_factor"].astype(float),
            float(load_factor),
        )
    ]

    if selected.empty:
        return False

    return bool(selected["sig"].any())


# =============================================================================
# Load and validate data
# =============================================================================

validate_file_exists(
    EPISODE_FILE,
    "Combined episode-level stress-test file",
)

validate_file_exists(
    WILCOXON_FILE,
    "Wilcoxon statistical-results file",
)

df = pd.read_csv(EPISODE_FILE)
res = pd.read_csv(WILCOXON_FILE)

validate_columns(
    df,
    EPISODE_REQUIRED_COLUMNS,
    EPISODE_FILE,
)

validate_columns(
    res,
    STAT_REQUIRED_COLUMNS,
    WILCOXON_FILE,
)


# =============================================================================
# Normalize input data
# =============================================================================

df["model"] = (
    df["model"]
    .astype(str)
    .str.strip()
    .str.lower()
)

res["arch"] = (
    res["arch"]
    .astype(str)
    .str.strip()
    .str.lower()
)

res["metric"] = (
    res["metric"]
    .astype(str)
    .str.strip()
    .str.lower()
)

numeric_episode_columns = [
    "load_factor",
    "seed",
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]

for column in numeric_episode_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    )

res["load_factor"] = pd.to_numeric(
    res["load_factor"],
    errors="raise",
)

res["sig"] = normalize_boolean_column(
    res["sig"]
)


# =============================================================================
# Dataset integrity checks
# =============================================================================

unexpected_models = sorted(
    set(df["model"].unique())
    - set(MODELS)
)

if unexpected_models:
    raise ValueError(
        "\nUnexpected model labels were found in the episode data:\n"
        f"{unexpected_models}"
    )

missing_models = sorted(
    set(MODELS)
    - set(df["model"].unique())
)

if missing_models:
    raise ValueError(
        "\nExpected model labels are missing from the episode data:\n"
        f"{missing_models}"
    )

duplicate_keys = df.duplicated(
    subset=[
        "model",
        "load_factor",
        "seed",
        "episode_id",
    ],
    keep=False,
)

if duplicate_keys.any():
    duplicate_rows = df.loc[
        duplicate_keys,
        [
            "model",
            "load_factor",
            "seed",
            "episode_id",
        ],
    ]

    raise ValueError(
        "\nDuplicate model/load/seed/episode rows were found:\n"
        f"{duplicate_rows.to_string(index=False)}"
    )

if not np.allclose(
    df["converged_rate"].to_numpy(dtype=float),
    1.0,
):
    print(
        "Warning: some episodes have converged_rate != 1.0."
    )


# =============================================================================
# Aggregate first within seed, then across seeds
# =============================================================================

seed_means = (
    df.groupby(
        [
            "model",
            "load_factor",
            "seed",
        ],
        observed=True,
    )
    .agg(
        cost=(
            "total_energy_cost",
            "mean",
        ),
        feas=(
            "feasible_rate",
            "mean",
        ),
        infreq=(
            "infeasible_requested_count",
            "mean",
        ),
    )
    .reset_index()
)

summ = (
    seed_means.groupby(
        [
            "model",
            "load_factor",
        ],
        observed=True,
    )
    .agg(
        cost_m=(
            "cost",
            "mean",
        ),
        cost_s=(
            "cost",
            "std",
        ),
        feas_m=(
            "feas",
            "mean",
        ),
        feas_s=(
            "feas",
            "std",
        ),
        inf_m=(
            "infreq",
            "mean",
        ),
        inf_s=(
            "infreq",
            "std",
        ),
    )
    .reset_index()
)

summary_nan_columns = [
    "cost_s",
    "feas_s",
    "inf_s",
]

if summ[summary_nan_columns].isna().any().any():
    raise ValueError(
        "\nStandard deviations could not be computed. "
        "Each model/load-factor combination must contain multiple seeds."
    )

load_factors = np.array(
    sorted(df["load_factor"].unique()),
    dtype=float,
)


# =============================================================================
# Verify every model contains every load factor
# =============================================================================

for model in MODELS:
    model_loads = np.array(
        sorted(
            summ.loc[
                summ["model"].eq(model),
                "load_factor",
            ].unique()
        ),
        dtype=float,
    )

    if not np.array_equal(
        model_loads,
        load_factors,
    ):
        raise ValueError(
            "\nLoad-factor mismatch.\n"
            f"Model: {model}\n"
            f"Expected: {load_factors.tolist()}\n"
            f"Found: {model_loads.tolist()}"
        )


# =============================================================================
# Create figure
# =============================================================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fig, (
    ax1,
    ax2,
    ax3,
) = plt.subplots(
    1,
    3,
    figsize=(7.16, 2.45),
)


# -----------------------------------------------------------------------------
# Panel (a): operational feasibility
# -----------------------------------------------------------------------------

for model in MODELS:
    model_summary = (
        summ[
            summ["model"].eq(model)
        ]
        .sort_values("load_factor")
    )

    x = model_summary["load_factor"].to_numpy(
        dtype=float
    )

    mean_values = model_summary["feas_m"].to_numpy(
        dtype=float
    )

    std_values = model_summary["feas_s"].to_numpy(
        dtype=float
    )

    ax1.plot(
        x,
        mean_values,
        color=COLORS[model],
        marker=MARKERS[model],
        markersize=3.5,
        linewidth=1.3,
        label=LABELS[model],
    )

    ax1.fill_between(
        x,
        mean_values - std_values,
        mean_values + std_values,
        color=COLORS[model],
        alpha=0.13,
        linewidth=0,
    )

ax1.set_xlabel(
    r"Load scaling $\lambda$"
)

ax1.set_ylabel(
    "Feasibility rate"
)

ax1.set_xticks(
    load_factors
)

ax1.grid(
    alpha=0.3,
    linewidth=0.4,
)

ax1.legend(
    frameon=False,
    loc="lower left",
    handlelength=1.4,
)

ax1.set_title(
    "(a) Operational feasibility"
)


# -----------------------------------------------------------------------------
# Panel (b): percentage cost reduction relative to MLP
# -----------------------------------------------------------------------------

mlp_summary = (
    summ[
        summ["model"].eq("mlp")
    ]
    .sort_values("load_factor")
)

mlp_cost = mlp_summary[
    "cost_m"
].to_numpy(dtype=float)

for model in [
    "gcn",
    "gat",
    "tagconv",
]:
    model_summary = (
        summ[
            summ["model"].eq(model)
        ]
        .sort_values("load_factor")
    )

    x = model_summary["load_factor"].to_numpy(
        dtype=float
    )

    model_cost = model_summary[
        "cost_m"
    ].to_numpy(dtype=float)

    cost_reduction = (
        100.0
        * (
            mlp_cost
            - model_cost
        )
        / mlp_cost
    )

    ax2.plot(
        x,
        cost_reduction,
        color=COLORS[model],
        marker=MARKERS[model],
        markersize=3.5,
        linewidth=1.3,
        label=LABELS[model],
    )

    for load_factor, reduction in zip(
        x,
        cost_reduction,
    ):
        significant = get_significance_flag(
            results=res,
            metric="cost",
            architecture=model,
            load_factor=load_factor,
        )

        if significant:
            ax2.annotate(
                "*",
                (
                    load_factor,
                    reduction,
                ),
                textcoords="offset points",
                xytext=(0, 2.2),
                horizontalalignment="center",
                fontsize=8,
                color=COLORS[model],
            )

ax2.axhline(
    0,
    color="#999999",
    linewidth=0.7,
    linestyle=":",
)

ax2.set_xlabel(
    r"Load scaling $\lambda$"
)

ax2.set_ylabel(
    "Cost reduction vs. MLP (%)"
)

ax2.set_xticks(
    load_factors
)

current_bottom, current_top = ax2.get_ylim()

ax2.set_ylim(
    bottom=min(-0.1, current_bottom),
    top=current_top,
)

ax2.grid(
    alpha=0.3,
    linewidth=0.4,
)

ax2.legend(
    frameon=False,
    loc="upper left",
    handlelength=1.4,
)

ax2.set_title(
    "(b) Economic advantage"
)


# -----------------------------------------------------------------------------
# Panel (c): requested-infeasible actions
# -----------------------------------------------------------------------------

for model in MODELS:
    model_summary = (
        summ[
            summ["model"].eq(model)
        ]
        .sort_values("load_factor")
    )

    x = model_summary["load_factor"].to_numpy(
        dtype=float
    )

    mean_values = model_summary["inf_m"].to_numpy(
        dtype=float
    )

    std_values = model_summary["inf_s"].to_numpy(
        dtype=float
    )

    ax3.plot(
        x,
        mean_values,
        color=COLORS[model],
        marker=MARKERS[model],
        markersize=3.5,
        linewidth=1.3,
        label=LABELS[model],
    )

    ax3.fill_between(
        x,
        mean_values - std_values,
        mean_values + std_values,
        color=COLORS[model],
        alpha=0.13,
        linewidth=0,
    )

ax3.set_xlabel(
    r"Load scaling $\lambda$"
)

ax3.set_ylabel(
    "Infeasible requested actions/day"
)

ax3.set_xticks(
    load_factors
)

ax3.grid(
    alpha=0.3,
    linewidth=0.4,
)

ax3.set_title(
    "(c) Requested-action infeasibility"
)


# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------

fig.tight_layout(
    pad=0.4,
    w_pad=0.9,
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(fig)

print(
    f"\nFigure written to:\n"
    f"  {OUTPUT_PDF.resolve()}\n"
    f"  {OUTPUT_PNG.resolve()}"
)


# =============================================================================
# Generate LaTeX table body
# =============================================================================

def format_mean_std(
    mean: float,
    std: float,
    decimal_places: int = 2,
    bold: bool = False,
) -> str:
    """
    Format a mean ± standard-deviation value for an IEEE-style table.
    """

    mean_text = f"{mean:.{decimal_places}f}"
    std_text = f"{std:.{decimal_places}f}"

    if bold:
        return (
            f"$\\mathbf{{{mean_text}}}"
            f"{{\\scriptstyle\\pm{std_text}}}$"
        )

    return (
        f"${mean_text}"
        f"{{\\scriptstyle\\pm{std_text}}}$"
    )


table_lines = []

for load_factor in load_factors:
    load_summary = summ[
        np.isclose(
            summ["load_factor"],
            load_factor,
        )
    ].copy()

    best_cost = load_summary[
        "cost_m"
    ].min()

    best_feasibility = load_summary[
        "feas_m"
    ].max()

    best_infeasible_requests = load_summary[
        "inf_m"
    ].min()

    for row_index, model in enumerate(MODELS):
        selected_row = load_summary[
            load_summary["model"].eq(model)
        ]

        if len(selected_row) != 1:
            raise ValueError(
                "\nExpected exactly one summary row.\n"
                f"Model: {model}\n"
                f"Load factor: {load_factor}\n"
                f"Rows found: {len(selected_row)}"
            )

        row = selected_row.iloc[0]

        if model == "mlp":
            significant_cost = False
            significant_feasibility = False
        else:
            significant_cost = get_significance_flag(
                results=res,
                metric="cost",
                architecture=model,
                load_factor=load_factor,
            )

            significant_feasibility = get_significance_flag(
                results=res,
                metric="feasibility",
                architecture=model,
                load_factor=load_factor,
            )

        cost_marker = (
            "$^{\\dagger}$"
            if significant_cost
            else ""
        )

        feasibility_marker = (
            "$^{\\dagger}$"
            if significant_feasibility
            else ""
        )

        topology_cell = (
            f"\\multirow{{4}}{{*}}{{{load_factor:.1f}}}"
            if row_index == 0
            else ""
        )

        cost_is_best = np.isclose(
            row["cost_m"],
            best_cost,
        )

        feasibility_is_best = np.isclose(
            row["feas_m"],
            best_feasibility,
        )

        infeasible_is_best = np.isclose(
            row["inf_m"],
            best_infeasible_requests,
        )

        table_lines.append(
            f"{topology_cell} & "
            f"{LABELS[model]} & "
            f"{format_mean_std(row['cost_m'], row['cost_s'], 2, cost_is_best)}"
            f"{cost_marker} & "
            f"{format_mean_std(row['feas_m'], row['feas_s'], 4, feasibility_is_best)}"
            f"{feasibility_marker} & "
            f"{format_mean_std(row['inf_m'], row['inf_s'], 1, infeasible_is_best)} "
            f"\\\\"
        )

    table_lines.append(
        "\\hline"
    )


OUTPUT_TABLE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_TABLE.write_text(
    "\n".join(table_lines),
    encoding="utf-8",
)

print(
    f"\nLaTeX table body written to:\n"
    f"  {OUTPUT_TABLE.resolve()}"
)

print(
    "\nCompleted successfully."
)