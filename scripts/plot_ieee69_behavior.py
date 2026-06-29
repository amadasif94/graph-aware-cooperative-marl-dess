#!/usr/bin/env python3
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def agent_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix: {prefix}")
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def pick_episode(df, topology_case, episode_id=None):
    topology_case = str(topology_case).upper()
    d = df[df["topology_case"].astype(str).str.upper() == topology_case].copy()

    if d.empty:
        available = sorted(df["topology_case"].astype(str).str.upper().unique())
        raise ValueError(f"No rows for {topology_case}. Available: {available}")

    if episode_id is None:
        episode_id = int(d["episode_id"].iloc[0])

    d = d[d["episode_id"].astype(int) == int(episode_id)].copy()
    if d.empty:
        raise ValueError(f"No rows found for episode_id={episode_id}")

    start_index = int(d["start_index"].iloc[0])
    d = d[d["start_index"].astype(int) == start_index].copy()

    return d.sort_values("step").reset_index(drop=True)


def clean_episode_df(df):
    acc_cols = agent_cols(df, "accepted_action_agent_")
    req_cols = agent_cols(df, "requested_action_agent_")
    soc_cols = agent_cols(df, "soc_agent_")
    power_cols = agent_cols(df, "dess_power_kw_agent_")

    numeric_cols = (
        acc_cols + req_cols + soc_cols + power_cols +
        ["step", "throughput_mwh", "voltage_deviation", "grid_stress", "infeasible_action"]
    )

    return safe_numeric(df.copy(), numeric_cols)


def smooth_series(x, window):
    x = pd.Series(x, dtype="float64")
    window = int(window)

    if window <= 1:
        return x.to_numpy(dtype=float)

    return (
        x.rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def save_behavior_summary(df, out_dir, dt_hours):
    acc_cols = agent_cols(df, "accepted_action_agent_")
    req_cols = agent_cols(df, "requested_action_agent_")
    soc_cols = agent_cols(df, "soc_agent_")
    power_cols = agent_cols(df, "dess_power_kw_agent_")

    acc = df[acc_cols].to_numpy(float)
    req = df[req_cols].to_numpy(float)
    soc = df[soc_cols].to_numpy(float)
    power = df[power_cols].to_numpy(float)

    gap = np.abs(req - acc)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = np.corrcoef(acc.T)

    if corr.ndim == 2 and corr.shape[0] > 1:
        upper = corr[np.triu_indices_from(corr, k=1)]
        mean_pairwise_corr = float(np.nanmean(upper))
    else:
        mean_pairwise_corr = np.nan

    summary = {
        "steps": int(len(df)),
        "dt_hours": float(dt_hours),
        "mean_abs_action": float(np.nanmean(np.abs(acc))),
        "mean_action_correction_gap": float(np.nanmean(gap)),
        "max_action_correction_gap": float(np.nanmax(gap)),
        "infeasible_action_rate": float(np.nanmean(df["infeasible_action"])),
        "total_throughput_mwh": float(np.nansum(df["throughput_mwh"])),
        "total_charge_mwh": float(np.nansum(np.maximum(-power, 0.0)) * dt_hours / 1000.0),
        "total_discharge_mwh": float(np.nansum(np.maximum(power, 0.0)) * dt_hours / 1000.0),
        "soc_min": float(np.nanmin(soc)),
        "soc_max": float(np.nanmax(soc)),
        "soc_range": float(np.nanmax(soc) - np.nanmin(soc)),
        "mean_pairwise_action_corr": mean_pairwise_corr,
    }

    pd.DataFrame([summary]).to_csv(out_dir / "behavior_summary.csv", index=False)


def plot_requested_vs_accepted(df, out_dir, model_name, topology_case):
    acc_cols = agent_cols(df, "accepted_action_agent_")
    req_cols = agent_cols(df, "requested_action_agent_")

    req = df[req_cols].to_numpy(float).reshape(-1)
    acc = df[acc_cols].to_numpy(float).reshape(-1)

    mask = np.isfinite(req) & np.isfinite(acc)
    req = req[mask]
    acc = acc[mask]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(req, acc, alpha=0.35, s=14)

    lo = float(min(np.min(req), np.min(acc)))
    hi = float(max(np.max(req), np.max(acc)))

    ax.plot([lo, hi], [lo, hi], linewidth=2, label="Accepted = requested")

    ax.set_xlabel("Requested action")
    ax.set_ylabel("Accepted action")
    ax.set_title(f"{model_name}: requested vs accepted actions ({topology_case})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "requested_vs_accepted.png", dpi=300)
    plt.close(fig)


def plot_agent_action_correlation(df, out_dir, model_name, topology_case):
    acc_cols = agent_cols(df, "accepted_action_agent_")
    acc = df[acc_cols].to_numpy(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = np.corrcoef(acc.T)

    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr, vmin=-1, vmax=1)

    n = len(acc_cols)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"A{i}" for i in range(n)])
    ax.set_yticklabels([f"A{i}" for i in range(n)])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    ax.set_title(f"{model_name}: inter-agent action correlation ({topology_case})")

    fig.tight_layout()
    fig.savefig(out_dir / "agent_action_correlation.png", dpi=300)
    plt.close(fig)


def plot_soc_behavior(
    df,
    out_dir,
    model_name,
    topology_case,
    dt_hours,
    smooth_window,
    soc_lower,
    soc_upper,
):
    soc_cols = agent_cols(df, "soc_agent_")

    time_hours = np.arange(len(df)) * float(dt_hours)
    soc = df[soc_cols].to_numpy(float)

    mean_soc = np.nanmean(soc, axis=1)
    min_soc = np.nanmin(soc, axis=1)
    max_soc = np.nanmax(soc, axis=1)

    mean_soc_smooth = smooth_series(mean_soc, smooth_window)
    min_soc_smooth = smooth_series(min_soc, smooth_window)
    max_soc_smooth = smooth_series(max_soc, smooth_window)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.fill_between(
        time_hours,
        min_soc_smooth,
        max_soc_smooth,
        alpha=0.20,
        label=f"SOC range across agents, {smooth_window}-step mean",
    )

    ax.plot(
        time_hours,
        mean_soc_smooth,
        linewidth=2,
        label=f"Mean SOC, {smooth_window}-step mean",
    )

    if soc_lower is not None:
        ax.axhline(float(soc_lower), linewidth=1.5, linestyle="--", label="SOC lower limit")

    if soc_upper is not None:
        ax.axhline(float(soc_upper), linewidth=1.5, linestyle="--", label="SOC upper limit")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SOC")
    ax.set_title(f"{model_name}: SOC behavior across DESS agents ({topology_case})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "soc_behavior.png", dpi=300)
    plt.close(fig)


def plot_charge_discharge_soc_timeseries(
    df,
    out_dir,
    model_name,
    topology_case,
    smooth_window,
    dt_hours,
):
    power_cols = agent_cols(df, "dess_power_kw_agent_")
    soc_cols = agent_cols(df, "soc_agent_")

    time_hours = np.arange(len(df)) * float(dt_hours)

    power = df[power_cols].to_numpy(float)

    total_discharge_kw = np.maximum(power, 0.0).sum(axis=1)
    total_charge_kw = np.maximum(-power, 0.0).sum(axis=1)
    mean_soc = df[soc_cols].mean(axis=1).to_numpy(float)

    total_discharge_kw_smooth = smooth_series(total_discharge_kw, smooth_window)
    total_charge_kw_smooth = smooth_series(total_charge_kw, smooth_window)
    mean_soc_smooth = smooth_series(mean_soc, smooth_window)

    fig, ax1 = plt.subplots(figsize=(11.5, 4.8))

    ax1.plot(
        time_hours,
        total_discharge_kw_smooth,
        linewidth=2,
        label=f"Discharge power, {smooth_window}-step mean",
    )

    ax1.plot(
        time_hours,
        -total_charge_kw_smooth,
        linewidth=2,
        linestyle="--",
        label=f"Charge power, {smooth_window}-step mean",
    )

    ax1.axhline(0.0, linewidth=1)

    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Aggregated DESS power (kW)")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(
        time_hours,
        mean_soc_smooth,
        linewidth=2,
        linestyle=":",
        label=f"Mean SOC, {smooth_window}-step mean",
    )

    ax2.set_ylabel("Mean SOC")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(
        f"{model_name}: smoothed charge/discharge and SOC behavior ({topology_case})"
    )

    fig.tight_layout()
    fig.savefig(out_dir / "charge_discharge_soc_timeseries.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--step_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="MADDPG")
    parser.add_argument("--topology_case", default="TP1")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--episode_id", type=int, default=None)

    parser.add_argument("--dt_hours", type=float, default=0.25)
    parser.add_argument("--smooth_window", type=int, default=12)

    parser.add_argument("--soc_lower", type=float, default=0.10)
    parser.add_argument("--soc_upper", type=float, default=0.90)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_csv = Path(args.step_csv)

    if not step_csv.exists():
        raise FileNotFoundError(f"step_csv not found: {step_csv}")

    df = pd.read_csv(step_csv)

    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    if args.policy is not None and "policy" in df.columns:
        df = df[df["policy"].astype(str) == str(args.policy)].copy()

    if "topology_case" not in df.columns:
        df["topology_case"] = args.topology_case

    required_cols = [
        "episode_id",
        "start_index",
        "step",
        "throughput_mwh",
        "infeasible_action",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in step_csv: {missing}")

    ep_df = pick_episode(
        df=df,
        topology_case=args.topology_case,
        episode_id=args.episode_id,
    )

    ep_df = clean_episode_df(ep_df)

    plot_requested_vs_accepted(
        ep_df,
        out_dir,
        args.model_name,
        args.topology_case,
    )

    plot_agent_action_correlation(
        ep_df,
        out_dir,
        args.model_name,
        args.topology_case,
    )

    plot_soc_behavior(
        ep_df,
        out_dir,
        args.model_name,
        args.topology_case,
        args.dt_hours,
        args.smooth_window,
        args.soc_lower,
        args.soc_upper,
    )

    plot_charge_discharge_soc_timeseries(
        ep_df,
        out_dir,
        args.model_name,
        args.topology_case,
        args.smooth_window,
        args.dt_hours,
    )

    save_behavior_summary(
        ep_df,
        out_dir,
        args.dt_hours,
    )

    print("=" * 72)
    print("Saved RL behavior analysis")
    print("=" * 72)
    print(f"Input CSV     : {step_csv}")
    print(f"Output dir    : {out_dir}")
    print(f"Topology      : {args.topology_case}")
    print(f"Smooth window : {args.smooth_window} steps")
    print(f"Time step     : {args.dt_hours} hours")
    print("Files:")
    print(f"  - {out_dir / 'requested_vs_accepted.png'}")
    print(f"  - {out_dir / 'agent_action_correlation.png'}")
    print(f"  - {out_dir / 'soc_behavior.png'}")
    print(f"  - {out_dir / 'charge_discharge_soc_timeseries.png'}")
    print(f"  - {out_dir / 'behavior_summary.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()