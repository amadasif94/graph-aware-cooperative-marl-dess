"""
Daily learned-policy behavior plots for MARL-DESS evaluation.

Reads:
    step_metrics.csv

Works with:
    results/csv/<run_name>/step_metrics.csv
    results/topology_generalization/csv/<run_name>/step_metrics.csv

By default, the script automatically keeps the learned policy and removes:
    zero
    random

So it works for:
    GNN policies: maddpg
    MLP policy:  mlp_maddpg
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def get_agent_cols(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]

    def agent_index(col):
        try:
            return int(col.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(cols, key=agent_index)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def require_columns(df, cols, plot_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Skipping {plot_name}: missing columns {missing}")
        return False
    return True


def filter_policy(df, policy):
    if "policy" not in df.columns:
        return df.copy()

    df = df.copy()
    df["policy"] = df["policy"].astype(str).str.lower().str.strip()
    policy = str(policy).lower().strip()

    if policy == "learned":
        return df[~df["policy"].isin(["zero", "random"])].copy()

    return df[df["policy"] == policy].copy()


def plot_two_axis(
    x,
    y_left,
    y_right,
    left_label,
    right_label,
    title,
    out_path,
    right_style=":",
):
    plt.figure(figsize=(11, 4))

    ax1 = plt.gca()
    ax1.plot(x, y_left, label=left_label)
    ax1.axhline(0.0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel(left_label)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, linestyle=right_style, label=right_label)
    ax2.set_ylabel(right_label)
    ax2.legend(loc="upper right")

    plt.title(title)
    savefig(out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--policy",
        type=str,
        default="learned",
        help="Use 'learned' to automatically keep maddpg/mlp_maddpg and remove zero/random.",
    )
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--topology_case", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df = filter_policy(df, args.policy)

    if args.topology_case is not None and "topology_case" in df.columns:
        df = df[
            df["topology_case"].astype(str).str.upper()
            == str(args.topology_case).upper()
        ].copy()

    if "episode_id" not in df.columns:
        raise ValueError("CSV does not contain episode_id column.")

    df = df[df["episode_id"] == args.episode_id].copy()

    if "step" not in df.columns:
        raise ValueError("CSV does not contain step column.")

    df = df.sort_values("step")

    if df.empty:
        raise ValueError(
            "No rows found for the selected policy/topology/episode. "
            "Check --policy, --topology_case, and --episode_id."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = df["step"]

    # ------------------------------------------------------------
    # Build total DESS power if not already saved
    # ------------------------------------------------------------
    if "total_dess_power_kw" not in df.columns:
        dess_cols = get_agent_cols(df, "dess_power_kw_agent_")

        if len(dess_cols) == 0:
            raise ValueError("No DESS power columns found.")

        df["total_dess_power_kw"] = df[dess_cols].sum(axis=1)

    # ============================================================
    # 1. Total DESS power vs price
    # ============================================================
    if require_columns(df, ["total_dess_power_kw", "price"], "DESS power vs price"):
        plot_two_axis(
            x=x,
            y_left=df["total_dess_power_kw"],
            y_right=df["price"],
            left_label="Total DESS power (kW)",
            right_label="Price",
            title="Daily Behavior: Total DESS Power vs Price",
            out_path=out_dir / "01_total_dess_power_vs_price.png",
        )

    # ============================================================
    # 2. SOC trajectories
    # ============================================================
    soc_cols = get_agent_cols(df, "soc_agent_")

    if len(soc_cols) > 0:
        plt.figure(figsize=(11, 4))

        for col in soc_cols:
            plt.plot(x, df[col], label=col)

        plt.xlabel("Time step")
        plt.ylabel("SOC")
        plt.title("Daily SOC Trajectories")
        plt.legend(ncol=2, fontsize=8)
        savefig(out_dir / "02_soc_trajectories.png")
    else:
        print("Skipping SOC trajectories: no soc_agent_* columns found.")

    # ============================================================
    # 3. Requested vs accepted actions for each agent
    # ============================================================
    req_cols = get_agent_cols(df, "requested_action_agent_")
    acc_cols = get_agent_cols(df, "accepted_action_agent_")

    if len(req_cols) > 0 and len(acc_cols) > 0:
        for i, (req_col, acc_col) in enumerate(zip(req_cols, acc_cols)):
            plt.figure(figsize=(11, 4))
            plt.plot(x, df[req_col], label="Requested action")
            plt.plot(x, df[acc_col], linestyle="--", label="Accepted action")
            plt.axhline(0.0, linestyle=":", linewidth=1)
            plt.xlabel("Time step")
            plt.ylabel("Action")
            plt.title(f"Agent {i}: Requested vs Accepted Action")
            plt.legend()
            savefig(out_dir / f"03_requested_vs_accepted_agent_{i}.png")
    else:
        print("Skipping requested-vs-accepted plots: action columns not found.")

    # ============================================================
    # 4. Total DESS power vs minimum voltage
    # ============================================================
    if require_columns(
        df,
        ["total_dess_power_kw", "min_voltage_pu"],
        "DESS power vs minimum voltage",
    ):
        plot_two_axis(
            x=x,
            y_left=df["total_dess_power_kw"],
            y_right=df["min_voltage_pu"],
            left_label="Total DESS power (kW)",
            right_label="Minimum voltage (p.u.)",
            title="Daily Behavior: Total DESS Power vs Minimum Voltage",
            out_path=out_dir / "04_total_dess_power_vs_min_voltage.png",
        )

    # ============================================================
    # 5. Total DESS power vs voltage deviation
    # ============================================================
    if require_columns(
        df,
        ["total_dess_power_kw", "voltage_deviation"],
        "DESS power vs voltage deviation",
    ):
        plot_two_axis(
            x=x,
            y_left=df["total_dess_power_kw"],
            y_right=df["voltage_deviation"],
            left_label="Total DESS power (kW)",
            right_label="Voltage deviation",
            title="Daily Behavior: Total DESS Power vs Voltage Deviation",
            out_path=out_dir / "05_total_dess_power_vs_voltage_deviation.png",
        )

    # ============================================================
    # 6. Total DESS power vs net load
    # ============================================================
    if require_columns(
        df,
        ["total_dess_power_kw", "net_load_kw"],
        "DESS power vs net load",
    ):
        plot_two_axis(
            x=x,
            y_left=df["total_dess_power_kw"],
            y_right=df["net_load_kw"],
            left_label="Total DESS power (kW)",
            right_label="Net load (kW)",
            title="Daily Behavior: Total DESS Power vs Net Load",
            out_path=out_dir / "06_total_dess_power_vs_net_load.png",
        )

    # ============================================================
    # 7. Load, PV, and net load
    # ============================================================
    if require_columns(
        df,
        ["total_load_kw", "total_pv_kw"],
        "load/PV/net-load plot",
    ):
        plt.figure(figsize=(11, 4))
        plt.plot(x, df["total_load_kw"], label="Total load")
        plt.plot(x, df["total_pv_kw"], label="Total PV")

        if "net_load_kw" in df.columns:
            plt.plot(x, df["net_load_kw"], linestyle="--", label="Net load")

        plt.xlabel("Time step")
        plt.ylabel("Power (kW)")
        plt.title("Daily Load, PV, and Net Load")
        plt.legend()
        savefig(out_dir / "07_load_pv_net_load.png")

    # ============================================================
    # 8. DESS power per agent
    # ============================================================
    dess_cols = get_agent_cols(df, "dess_power_kw_agent_")

    if len(dess_cols) > 0:
        plt.figure(figsize=(11, 4))

        for col in dess_cols:
            plt.plot(x, df[col], label=col)

        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Time step")
        plt.ylabel("DESS power (kW)")
        plt.title("Per-Agent DESS Power")
        plt.legend(ncol=2, fontsize=8)
        savefig(out_dir / "08_per_agent_dess_power.png")
    else:
        print("Skipping per-agent DESS power: no dess_power_kw_agent_* columns found.")

    print(f"Saved behavior plots to: {out_dir}")


if __name__ == "__main__":
    main()