from pathlib import Path
import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mpc_smpc"
    / "ieee69_15min_nodal_load_pv_full_year.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "mpc_smpc" / "forecasts"
MODEL_DIR = PROJECT_ROOT / "results" / "mpc_smpc" / "models" / "autogluon_69bus"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUM_BUSES = 69
DT_COL = "date_time"
PRICE_COL = "price"
LABEL = "y"

BASE_TRAIN_START = "2018-01-01"

WF_MONTHS = ["2018-09", "2018-10", "2018-11", "2018-12"]
RESIDUAL_MONTHS = ["2018-09", "2018-10", "2018-11"]
SCENARIO_MONTH = "2018-12"

TARGET_LAGS = [1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 672]
ROLL_WINDOWS = [4, 12, 24, 96, 672]


def month_start_end(month_str: str):
    start = pd.Timestamp(f"{month_str}-01 00:00:00")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def target_columns(target_type: str):
    load_cols = [f"load_node_{i}" for i in range(1, NUM_BUSES + 1)]
    pv_cols = [f"pv_node_{i}" for i in range(1, NUM_BUSES + 1)]

    if target_type == "load":
        return load_cols
    if target_type == "pv":
        return pv_cols
    if target_type == "both":
        return load_cols + pv_cols

    raise ValueError("target_type must be one of: load, pv, both")


def build_features_for_target(df: pd.DataFrame, target_col: str):
    data = pd.DataFrame(index=df.index)

    data[LABEL] = pd.to_numeric(df[target_col], errors="coerce")

    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    data["dow"] = data.index.dayofweek
    data["month"] = data.index.month
    data["dayofyear"] = data.index.dayofyear
    data["is_weekend"] = data["dow"].isin([5, 6]).astype(int)

    slot = data.index.hour * 4 + data.index.minute // 15
    data["slot_sin"] = np.sin(2 * np.pi * slot / 96)
    data["slot_cos"] = np.cos(2 * np.pi * slot / 96)
    data["dow_sin"] = np.sin(2 * np.pi * data["dow"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dow"] / 7)
    data["doy_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 366)
    data["doy_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 366)

    if PRICE_COL in df.columns:
        data["price"] = pd.to_numeric(df[PRICE_COL], errors="coerce").fillna(0.0)
        data["price_lag_1"] = data["price"].shift(1)
        data["price_lag_96"] = data["price"].shift(96)

    for lag in TARGET_LAGS:
        data[f"{target_col}_lag_{lag}"] = data[LABEL].shift(lag)

    for window in ROLL_WINDOWS:
        shifted = data[LABEL].shift(1)
        data[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window).mean()
        data[f"{target_col}_roll_std_{window}"] = shifted.rolling(window).std()
        data[f"{target_col}_roll_min_{window}"] = shifted.rolling(window).min()
        data[f"{target_col}_roll_max_{window}"] = shifted.rolling(window).max()

    data["lag_96_minus_lag_672"] = (
        data[f"{target_col}_lag_96"] - data[f"{target_col}_lag_672"]
    )

    data = data.dropna().copy()
    feature_cols = [c for c in data.columns if c != LABEL]

    return data, feature_cols


def get_model_path(month_str: str, target_col: str):
    month_tag = month_str.replace("-", "_")
    safe_target = target_col.replace("/", "_")
    return MODEL_DIR / month_tag / safe_target


def prepare_ag_df(df_in: pd.DataFrame, feature_cols):
    return df_in[[LABEL] + feature_cols].copy().reset_index(drop=True)


def run_forecast(args):
    print("=" * 80)
    print("IEEE69 WALK-FORWARD LOAD/PV FORECASTING")
    print("=" * 80)
    print(f"Input:      {INPUT_PATH}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Model dir:  {MODEL_DIR}")
    print(f"Target type: {args.target_type}")
    print(f"Train model: {args.train_model}")
    print("=" * 80)

    df = pd.read_csv(INPUT_PATH)

    if DT_COL not in df.columns:
        raise ValueError(f"Missing required datetime column: {DT_COL}")

    df[DT_COL] = pd.to_datetime(
        df[DT_COL],
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None)

    if df[DT_COL].isna().any():
        raise ValueError("Some date_time values could not be parsed.")

    df = df.sort_values(DT_COL).set_index(DT_COL)

    expected_index = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    missing = expected_index.difference(df.index)
    if len(missing) > 0:
        raise ValueError(
            f"Missing 15-min timestamps detected. First missing: {missing[:5]}"
        )

    targets = target_columns(args.target_type)

    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    if args.max_targets is not None:
        targets = targets[: args.max_targets]

    if args.clear_tmp and OUT_DIR.exists():
        tmp_dir = OUT_DIR / "monthly_tmp_69bus"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    tmp_dir = OUT_DIR / "monthly_tmp_69bus"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_prediction_frames = []
    all_metric_rows = []

    for target_col in targets:
        print("\n" + "=" * 80)
        print(f"TARGET: {target_col}")
        print("=" * 80)

        data_ml, feature_cols = build_features_for_target(df, target_col)

        for month_str in WF_MONTHS:
            test_start, test_end = month_start_end(month_str)

            train_df = data_ml.loc[
                (data_ml.index >= pd.Timestamp(BASE_TRAIN_START))
                & (data_ml.index < test_start)
            ].copy()

            test_df = data_ml.loc[
                (data_ml.index >= test_start)
                & (data_ml.index < test_end)
            ].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                raise ValueError(
                    f"Empty split for target={target_col}, month={month_str}. "
                    f"train={len(train_df)}, test={len(test_df)}"
                )

            print(f"\n[{target_col} | {month_str}]")
            print(
                f"  Train: {train_df.index.min()} -> {train_df.index.max()} "
                f"| rows={len(train_df)}"
            )
            print(
                f"  Test : {test_df.index.min()} -> {test_df.index.max()} "
                f"| rows={len(test_df)}"
            )
            print(f"  Features: {len(feature_cols)}")

            train_ag = prepare_ag_df(train_df, feature_cols)
            test_ag = prepare_ag_df(test_df, feature_cols)

            X_test = test_ag[feature_cols].copy()
            y_test = test_ag[LABEL].astype(float).to_numpy()

            model_path = get_model_path(month_str, target_col)

            t0 = time.time()

            if args.train_model:
                if model_path.exists():
                    shutil.rmtree(model_path, ignore_errors=True)

                predictor = TabularPredictor(
                    label=LABEL,
                    eval_metric="mean_absolute_error",
                    path=str(model_path),
                )

                predictor.fit(
                    train_data=train_ag,
                    time_limit=args.time_limit,
                    presets=args.presets,
                    dynamic_stacking=args.dynamic_stacking,
                    save_bag_folds=False,
                )
            else:
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Saved model not found for {target_col}, "
                        f"{month_str}: {model_path}"
                    )

                predictor = TabularPredictor.load(str(model_path))

            expected_features = predictor.feature_metadata_in.get_features()
            X_test = X_test[expected_features].copy()

            yhat = predictor.predict(X_test).astype(float).to_numpy()
            residual = y_test - yhat

            mae = safe_mae(y_test, yhat)
            rmse = safe_rmse(y_test, yhat)
            elapsed = time.time() - t0

            pred_df = pd.DataFrame(
                {
                    DT_COL: test_df.index,
                    "forecast_month": month_str,
                    "target": target_col,
                    "y_true": y_test,
                    "yhat": yhat,
                    "residual": residual,
                }
            )

            month_tag = month_str.replace("-", "_")
            pred_path = tmp_dir / f"predictions_{target_col}_{month_tag}.csv"
            pred_df.to_csv(pred_path, index=False)

            metric_row = {
                "forecast_month": month_str,
                "target": target_col,
                "train_start": train_df.index.min(),
                "train_end": train_df.index.max(),
                "test_start": test_df.index.min(),
                "test_end": test_df.index.max(),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "mae": mae,
                "rmse": rmse,
                "elapsed_sec": elapsed,
                "model_path": str(model_path),
                "n_features": len(expected_features),
                "selected_model": getattr(predictor, "model_best", "unknown"),
            }

            all_prediction_frames.append(pred_df)
            all_metric_rows.append(metric_row)

            print(f"  MAE:  {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Time: {elapsed:.2f} sec")

    all_preds_long = pd.concat(all_prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(all_metric_rows)

    all_preds_long[DT_COL] = pd.to_datetime(all_preds_long[DT_COL])
    all_preds_long = all_preds_long.sort_values(["target", DT_COL]).reset_index(
        drop=True
    )
    metrics_df = metrics_df.sort_values(["target", "forecast_month"]).reset_index(
        drop=True
    )

    all_long_path = OUT_DIR / "all_walkforward_predictions_sep_dec_69bus_long.csv"
    metrics_path = OUT_DIR / "monthly_metrics_sep_dec_69bus.csv"

    all_preds_long.to_csv(all_long_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    wide_parts = []

    for kind in ["load", "pv"]:
        kind_df = all_preds_long[
            all_preds_long["target"].str.startswith(f"{kind}_node_")
        ].copy()

        if len(kind_df) == 0:
            continue

        yhat_wide = kind_df.pivot_table(
            index=[DT_COL, "forecast_month"],
            columns="target",
            values="yhat",
            aggfunc="first",
        ).reset_index()

        ytrue_wide = kind_df.pivot_table(
            index=[DT_COL, "forecast_month"],
            columns="target",
            values="y_true",
            aggfunc="first",
        ).reset_index()

        resid_wide = kind_df.pivot_table(
            index=[DT_COL, "forecast_month"],
            columns="target",
            values="residual",
            aggfunc="first",
        ).reset_index()

        yhat_wide = yhat_wide.rename(
            columns={
                c: f"forecast_{c}"
                for c in yhat_wide.columns
                if c not in [DT_COL, "forecast_month"]
            }
        )
        ytrue_wide = ytrue_wide.rename(
            columns={
                c: f"true_{c}"
                for c in ytrue_wide.columns
                if c not in [DT_COL, "forecast_month"]
            }
        )
        resid_wide = resid_wide.rename(
            columns={
                c: f"residual_{c}"
                for c in resid_wide.columns
                if c not in [DT_COL, "forecast_month"]
            }
        )

        merged_kind = ytrue_wide.merge(yhat_wide, on=[DT_COL, "forecast_month"])
        merged_kind = merged_kind.merge(resid_wide, on=[DT_COL, "forecast_month"])
        wide_parts.append(merged_kind)

    if not wide_parts:
        raise ValueError("No wide forecast parts were produced.")

    wide_df = wide_parts[0]
    for part in wide_parts[1:]:
        wide_df = wide_df.merge(part, on=[DT_COL, "forecast_month"], how="outer")

    if PRICE_COL in df.columns:
        price_df = df[[PRICE_COL]].reset_index()
        wide_df = wide_df.merge(price_df, on=DT_COL, how="left")

    wide_df = wide_df.sort_values(DT_COL).reset_index(drop=True)

    wide_path = OUT_DIR / "forecast_69bus_load_pv_sep_dec_wide.csv"
    wide_df.to_csv(wide_path, index=False)

    residual_archive = wide_df[
        wide_df["forecast_month"].isin(RESIDUAL_MONTHS)
    ].copy()
    december_forecast = wide_df[
        wide_df["forecast_month"] == SCENARIO_MONTH
    ].copy()

    residual_path = OUT_DIR / "residual_archive_sep_nov_69bus_load_pv.csv"
    december_path = OUT_DIR / "december_forecast_69bus_load_pv.csv"

    residual_archive.to_csv(residual_path, index=False)
    december_forecast.to_csv(december_path, index=False)

    manifest = {
        "input_path": str(INPUT_PATH),
        "out_dir": str(OUT_DIR),
        "model_dir": str(MODEL_DIR),
        "wf_months": WF_MONTHS,
        "residual_months": RESIDUAL_MONTHS,
        "scenario_month": SCENARIO_MONTH,
        "num_buses": NUM_BUSES,
        "targets": targets,
        "target_lags": TARGET_LAGS,
        "roll_windows": ROLL_WINDOWS,
        "outputs": {
            "all_predictions_long": str(all_long_path),
            "monthly_metrics": str(metrics_path),
            "wide_forecast": str(wide_path),
            "residual_archive": str(residual_path),
            "december_forecast": str(december_path),
        },
    }

    manifest_path = OUT_DIR / "forecast_manifest_69bus.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("IEEE69 FORECAST PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Saved long predictions: {all_long_path}")
    print(f"Saved metrics:          {metrics_path}")
    print(f"Saved wide forecast:    {wide_path}")
    print(f"Saved residual archive: {residual_path}")
    print(f"Saved December forecast:{december_path}")
    print(f"Saved manifest:         {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_type",
        type=str,
        default="both",
        choices=["load", "pv", "both"],
    )
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--dynamic_stacking", action="store_true")
    parser.add_argument("--clear_tmp", action="store_true")
    parser.add_argument("--max_targets", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    run_forecast(parse_args())