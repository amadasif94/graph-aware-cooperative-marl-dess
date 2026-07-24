from pathlib import Path

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("results/topology_generalization/csv")
OUTPUT_FILE = BASE_DIR / "stress_combined.csv"

MODELS = [
    "mlp",
    "gcn",
    "gat",
    "tagconv",
]

SEEDS = [0, 1, 2]

STRESS_LOAD_TOKENS = {
    1.1: "1p1",
    1.2: "1p2",
    1.3: "1p3",
    1.4: "1p4",
}

SOURCE_COLUMNS = [
    "topology_case",
    "policy",
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]

ANALYSIS_COLUMNS = [
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]

OUTPUT_COLUMNS = [
    "model",
    "load_factor",
    "seed",
    "episode_id",
    "total_energy_cost",
    "feasible_rate",
    "infeasible_requested_count",
    "converged_rate",
]

EXPECTED_EPISODES_PER_RUN = 31
EXPECTED_EPISODE_IDS = set(range(EXPECTED_EPISODES_PER_RUN))

EXPECTED_LOAD_FACTORS = {
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
}

EXPECTED_NUMBER_OF_FILES = (
    len(MODELS)
    * len(EXPECTED_LOAD_FACTORS)
    * len(SEEDS)
)

EXPECTED_ROWS_PER_MODEL_LOAD = (
    len(SEEDS)
    * EXPECTED_EPISODES_PER_RUN
)

EXPECTED_TOTAL_ROWS = (
    len(MODELS)
    * len(EXPECTED_LOAD_FACTORS)
    * len(SEEDS)
    * EXPECTED_EPISODES_PER_RUN
)


# =============================================================================
# Helper functions
# =============================================================================

def normalize_text(series: pd.Series) -> pd.Series:
    """
    Normalize string values for reliable comparisons.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def validate_required_columns(
    df: pd.DataFrame,
    file_path: Path,
) -> None:
    """
    Validate that all required source columns are present.
    """

    missing_columns = [
        column
        for column in SOURCE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nRequired columns are missing.\n"
            f"File:\n{file_path.resolve()}\n\n"
            f"Missing columns:\n{missing_columns}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )


def convert_analysis_columns_to_numeric(
    df: pd.DataFrame,
    file_path: Path,
) -> pd.DataFrame:
    """
    Convert analysis columns to numeric values.
    """

    converted = df.copy()

    for column in ANALYSIS_COLUMNS:
        try:
            converted[column] = pd.to_numeric(
                converted[column],
                errors="raise",
            )
        except Exception as error:
            raise ValueError(
                "\nA required analysis column contains a non-numeric value.\n"
                f"File:\n{file_path.resolve()}\n"
                f"Column: {column}\n"
                f"Original error: {error}"
            ) from error

    converted["episode_id"] = (
        converted["episode_id"]
        .astype(int)
    )

    return converted


def select_learned_policy_rows(
    source_df: pd.DataFrame,
    file_path: Path,
    model: str,
    load_factor: float,
    seed: int,
) -> pd.DataFrame:
    """
    Select TP6 rows belonging only to the learned controller.

    The nominal full-evaluation files include:
      - learned MADDPG policy
      - random policy
      - zero-action policy

    Only the learned policy is retained.
    """

    topology_labels = (
        source_df["topology_case"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    selected = source_df[
        topology_labels.eq("TP6")
    ].copy()

    if selected.empty:
        available_topologies = sorted(
            topology_labels
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "\nNo TP6 rows were found.\n"
            f"File:\n{file_path.resolve()}\n\n"
            f"Available topology labels:\n{available_topologies}"
        )

    normalized_policy = normalize_text(
        selected["policy"]
    )

    # Explicitly remove diagnostic baselines.
    selected = selected[
        ~normalized_policy.isin(
            {
                "random",
                "zero",
            }
        )
    ].copy()

    if selected.empty:
        available_policies = sorted(
            source_df["policy"]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        raise ValueError(
            "\nNo learned-policy TP6 rows remained after excluding "
            "the random and zero policies.\n"
            f"File:\n{file_path.resolve()}\n\n"
            f"Available policies:\n{available_policies}"
        )

    remaining_policies = sorted(
        selected["policy"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(remaining_policies) != 1:
        raise ValueError(
            "\nMore than one learned-policy label remained.\n"
            f"File:\n{file_path.resolve()}\n"
            f"Model: {model}\n"
            f"Load factor: {load_factor}\n"
            f"Seed: {seed}\n"
            f"Remaining policies: {remaining_policies}"
        )

    print(
        f"Selected policy: "
        f"model={model}, "
        f"load={load_factor}, "
        f"seed={seed}, "
        f"policy={remaining_policies[0]}"
    )

    return selected


def validate_episode_set(
    df: pd.DataFrame,
    file_path: Path,
    model: str,
    load_factor: float,
    seed: int,
) -> None:
    """
    Validate that a source file contains exactly 31 unique episodes,
    with episode IDs 0 through 30.
    """

    if len(df) != EXPECTED_EPISODES_PER_RUN:
        episode_counts = (
            df.groupby("episode_id")
            .size()
            .rename("copies")
            .reset_index()
        )

        raise ValueError(
            "\nUnexpected number of learned-policy TP6 episodes.\n"
            f"File:\n{file_path.resolve()}\n"
            f"Model: {model}\n"
            f"Load factor: {load_factor}\n"
            f"Seed: {seed}\n"
            f"Expected rows: {EXPECTED_EPISODES_PER_RUN}\n"
            f"Found rows: {len(df)}\n\n"
            f"Episode counts:\n"
            f"{episode_counts.to_string(index=False)}"
        )

    duplicate_mask = df["episode_id"].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_ids = sorted(
            df.loc[
                duplicate_mask,
                "episode_id",
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            "\nDuplicate episode IDs were found after policy filtering.\n"
            f"File:\n{file_path.resolve()}\n"
            f"Model: {model}\n"
            f"Load factor: {load_factor}\n"
            f"Seed: {seed}\n"
            f"Duplicate episode IDs: {duplicate_ids}"
        )

    actual_episode_ids = set(
        df["episode_id"]
        .astype(int)
        .tolist()
    )

    missing_episode_ids = sorted(
        EXPECTED_EPISODE_IDS
        - actual_episode_ids
    )

    extra_episode_ids = sorted(
        actual_episode_ids
        - EXPECTED_EPISODE_IDS
    )

    if missing_episode_ids or extra_episode_ids:
        raise ValueError(
            "\nEpisode ID validation failed.\n"
            f"File:\n{file_path.resolve()}\n"
            f"Model: {model}\n"
            f"Load factor: {load_factor}\n"
            f"Seed: {seed}\n"
            f"Missing episode IDs: {missing_episode_ids}\n"
            f"Unexpected episode IDs: {extra_episode_ids}"
        )


def load_episode_file(
    file_path: Path,
    model: str,
    load_factor: float,
    seed: int,
) -> pd.DataFrame:
    """
    Load one episode_summary.csv file and return one clean 31-row
    learned-policy TP6 dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "\nExpected episode-summary file was not found.\n"
            f"Model: {model}\n"
            f"Load factor: {load_factor}\n"
            f"Seed: {seed}\n"
            f"Expected path:\n{file_path.resolve()}"
        )

    source_df = pd.read_csv(file_path)

    validate_required_columns(
        source_df,
        file_path,
    )

    selected_df = select_learned_policy_rows(
        source_df=source_df,
        file_path=file_path,
        model=model,
        load_factor=load_factor,
        seed=seed,
    )

    selected_df = selected_df[
        ANALYSIS_COLUMNS
    ].copy()

    selected_df = convert_analysis_columns_to_numeric(
        selected_df,
        file_path,
    )

    selected_df = (
        selected_df
        .sort_values("episode_id")
        .reset_index(drop=True)
    )

    validate_episode_set(
        df=selected_df,
        file_path=file_path,
        model=model,
        load_factor=load_factor,
        seed=seed,
    )

    selected_df.insert(
        0,
        "seed",
        seed,
    )

    selected_df.insert(
        0,
        "load_factor",
        load_factor,
    )

    selected_df.insert(
        0,
        "model",
        model,
    )

    return selected_df[OUTPUT_COLUMNS]


# =============================================================================
# Load all source files
# =============================================================================

combined_frames = []
source_records = []


# -----------------------------------------------------------------------------
# Nominal load factor: lambda = 1.0
#
# Folder format:
# model_ieee33_tp_full_eval_runSEED_best
# -----------------------------------------------------------------------------

for model in MODELS:
    for seed in SEEDS:
        folder_name = (
            f"{model}_ieee33_tp_full_eval_"
            f"run{seed}_best"
        )

        file_path = (
            BASE_DIR
            / folder_name
            / "episode_summary.csv"
        )

        print(
            "\nLoading nominal file: "
            f"model={model}, "
            f"load=1.0, "
            f"seed={seed}"
        )

        frame = load_episode_file(
            file_path=file_path,
            model=model,
            load_factor=1.0,
            seed=seed,
        )

        combined_frames.append(frame)

        source_records.append(
            {
                "model": model,
                "load_factor": 1.0,
                "seed": seed,
                "rows": len(frame),
                "source": str(file_path),
            }
        )


# -----------------------------------------------------------------------------
# Stress load factors: lambda = 1.1 through 1.4
#
# Folder format:
# model_ieee33_tp6_loadTOKEN_runSEED
# -----------------------------------------------------------------------------

for model in MODELS:
    for load_factor, load_token in STRESS_LOAD_TOKENS.items():
        for seed in SEEDS:
            folder_name = (
                f"{model}_ieee33_tp6_"
                f"load{load_token}_run{seed}"
            )

            file_path = (
                BASE_DIR
                / folder_name
                / "episode_summary.csv"
            )

            print(
                "\nLoading stress file: "
                f"model={model}, "
                f"load={load_factor}, "
                f"seed={seed}"
            )

            frame = load_episode_file(
                file_path=file_path,
                model=model,
                load_factor=load_factor,
                seed=seed,
            )

            combined_frames.append(frame)

            source_records.append(
                {
                    "model": model,
                    "load_factor": load_factor,
                    "seed": seed,
                    "rows": len(frame),
                    "source": str(file_path),
                }
            )


# =============================================================================
# Combine
# =============================================================================

if len(combined_frames) != EXPECTED_NUMBER_OF_FILES:
    raise RuntimeError(
        "\nUnexpected number of source files loaded.\n"
        f"Expected: {EXPECTED_NUMBER_OF_FILES}\n"
        f"Loaded: {len(combined_frames)}"
    )

combined = pd.concat(
    combined_frames,
    ignore_index=True,
)

combined = (
    combined[OUTPUT_COLUMNS]
    .sort_values(
        by=[
            "model",
            "load_factor",
            "seed",
            "episode_id",
        ]
    )
    .reset_index(drop=True)
)


# =============================================================================
# Final validation
# =============================================================================

validation_errors = []

key_columns = [
    "model",
    "load_factor",
    "seed",
    "episode_id",
]


# -----------------------------------------------------------------------------
# Total row count
# -----------------------------------------------------------------------------

if len(combined) != EXPECTED_TOTAL_ROWS:
    validation_errors.append(
        f"Expected {EXPECTED_TOTAL_ROWS} rows, "
        f"but found {len(combined)}."
    )


# -----------------------------------------------------------------------------
# Duplicate combined keys
# -----------------------------------------------------------------------------

duplicate_key_mask = combined.duplicated(
    subset=key_columns,
    keep=False,
)

if duplicate_key_mask.any():
    duplicate_keys = (
        combined.loc[
            duplicate_key_mask,
            key_columns,
        ]
        .value_counts()
        .rename("copies")
        .reset_index()
    )

    validation_errors.append(
        "Duplicate combined keys were found:\n"
        + duplicate_keys.to_string(index=False)
    )


# -----------------------------------------------------------------------------
# Missing values
# -----------------------------------------------------------------------------

missing_counts = combined.isna().sum()

missing_counts = missing_counts[
    missing_counts > 0
]

if not missing_counts.empty:
    validation_errors.append(
        "Missing values were found:\n"
        + missing_counts.to_string()
    )


# -----------------------------------------------------------------------------
# Per-seed row count
# -----------------------------------------------------------------------------

per_seed_counts = (
    combined.groupby(
        [
            "model",
            "load_factor",
            "seed",
        ],
        observed=True,
    )
    .size()
    .rename("rows")
    .reset_index()
)

invalid_seed_counts = per_seed_counts[
    per_seed_counts["rows"]
    != EXPECTED_EPISODES_PER_RUN
]

if not invalid_seed_counts.empty:
    validation_errors.append(
        "Some model/load_factor/seed groups "
        "do not contain exactly 31 rows:\n"
        + invalid_seed_counts.to_string(index=False)
    )


# -----------------------------------------------------------------------------
# Per-model/load row count
# -----------------------------------------------------------------------------

per_model_load_counts = (
    combined.groupby(
        [
            "model",
            "load_factor",
        ],
        observed=True,
    )
    .size()
    .rename("rows")
    .reset_index()
)

invalid_model_load_counts = per_model_load_counts[
    per_model_load_counts["rows"]
    != EXPECTED_ROWS_PER_MODEL_LOAD
]

if not invalid_model_load_counts.empty:
    validation_errors.append(
        "Some model/load_factor groups "
        "do not contain exactly 93 rows:\n"
        + invalid_model_load_counts.to_string(
            index=False
        )
    )


# -----------------------------------------------------------------------------
# Expected model, load-factor, and seed sets
# -----------------------------------------------------------------------------

actual_models = set(
    combined["model"].unique()
)

actual_load_factors = set(
    combined["load_factor"].unique()
)

actual_seeds = set(
    combined["seed"].unique()
)

if actual_models != set(MODELS):
    validation_errors.append(
        "Model set mismatch.\n"
        f"Expected: {sorted(MODELS)}\n"
        f"Found: {sorted(actual_models)}"
    )

if actual_load_factors != EXPECTED_LOAD_FACTORS:
    validation_errors.append(
        "Load-factor set mismatch.\n"
        f"Expected: {sorted(EXPECTED_LOAD_FACTORS)}\n"
        f"Found: {sorted(actual_load_factors)}"
    )

if actual_seeds != set(SEEDS):
    validation_errors.append(
        "Seed set mismatch.\n"
        f"Expected: {sorted(SEEDS)}\n"
        f"Found: {sorted(actual_seeds)}"
    )


# -----------------------------------------------------------------------------
# Validate episode IDs within every group
# -----------------------------------------------------------------------------

for (
    model,
    load_factor,
    seed,
), group in combined.groupby(
    [
        "model",
        "load_factor",
        "seed",
    ],
    observed=True,
):
    actual_ids = set(
        group["episode_id"]
        .astype(int)
        .tolist()
    )

    if actual_ids != EXPECTED_EPISODE_IDS:
        missing_ids = sorted(
            EXPECTED_EPISODE_IDS
            - actual_ids
        )

        extra_ids = sorted(
            actual_ids
            - EXPECTED_EPISODE_IDS
        )

        validation_errors.append(
            "Episode set mismatch for "
            f"model={model}, "
            f"load_factor={load_factor}, "
            f"seed={seed}.\n"
            f"Missing IDs: {missing_ids}\n"
            f"Unexpected IDs: {extra_ids}"
        )


# -----------------------------------------------------------------------------
# Convergence check
# -----------------------------------------------------------------------------

nonconverged_rows = combined[
    combined["converged_rate"] != 1.0
].copy()


# =============================================================================
# Print summaries
# =============================================================================

source_summary = pd.DataFrame(
    source_records
)

print("\n" + "=" * 100)
print("SOURCE FILE SUMMARY")
print("=" * 100)
print(
    source_summary[
        [
            "model",
            "load_factor",
            "seed",
            "rows",
            "source",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 100)
print("ROWS PER MODEL, LOAD FACTOR, AND SEED")
print("=" * 100)
print(
    per_seed_counts.to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("ROWS PER MODEL AND LOAD FACTOR")
print("=" * 100)
print(
    per_model_load_counts.to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("CONVERGENCE CHECK")
print("=" * 100)

if nonconverged_rows.empty:
    print(
        f"Passed: converged_rate equals 1.0 "
        f"for all {len(combined)} rows."
    )
else:
    print(
        "Warning: some rows have converged_rate != 1.0."
    )

    print(
        nonconverged_rows[
            [
                "model",
                "load_factor",
                "seed",
                "episode_id",
                "converged_rate",
            ]
        ].to_string(index=False)
    )


# =============================================================================
# Save only if validation passes
# =============================================================================

if validation_errors:
    print("\n" + "=" * 100)
    print("VALIDATION FAILED")
    print("=" * 100)

    for index, error_message in enumerate(
        validation_errors,
        start=1,
    ):
        print(
            f"\n{index}. {error_message}"
        )

    raise RuntimeError(
        "\nCombined file was not written because validation failed."
    )

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =============================================================================
# Success message
# =============================================================================

print("\n" + "=" * 100)
print("VALIDATION PASSED")
print("=" * 100)
print(f"Source files loaded: {len(source_records)}")
print(f"Rows written: {len(combined)}")
print(f"Columns written: {len(combined.columns)}")
print(f"Output file: {OUTPUT_FILE.resolve()}")

print("\nOutput columns:")

for column in combined.columns:
    print(f"  - {column}")