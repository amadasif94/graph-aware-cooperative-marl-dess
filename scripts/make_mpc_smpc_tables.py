from pathlib import Path
import pandas as pd


ROOT = Path("results/mpc_smpc")

SYSTEMS = {
    "IEEE33": ROOT / "optimization",
    "IEEE69": ROOT / "optimization_ieee69",
}

CONTROLLERS = ["mpc", "smpc", "uncontrolled"]


def find_summary_file(folder: Path):
    files = list(folder.glob("aggregate_summary*.csv"))
    if files:
        return files[0]

    files = list(folder.glob("summary*.csv"))
    if files:
        return files[0]

    files = list(folder.glob("episode_summary*.csv"))
    if files:
        return files[0]

    return None


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_one(system, base_dir, controller, topology):
    folder = base_dir / controller / topology
    f = find_summary_file(folder)

    if f is None:
        return None

    df = pd.read_csv(f)

    if df.empty:
        return None

    row = df.iloc[0]

    def val(candidates):
        c = pick_col(df, candidates)
        return row[c] if c is not None else None

    return {
        "System": system,
        "Topology": topology,
        "Controller": controller.upper(),
        "Reward": val([
            "mean_total_reward_mean",
            "total_reward_mean",
            "reward_mean",
        ]),
        "Energy Cost ($)": val([
            "mean_energy_cost",
            "total_energy_cost",
            "energy_cost",
        ]),
        "Grid Import (MWh)": val([
            "mean_grid_import_mwh",
            "total_grid_import_mwh",
            "grid_import_mwh",
        ]),
        "Curtailment (MWh)": val([
            "mean_curtailment_mwh",
            "total_curtailment_mwh",
            "curtailment_mwh",
        ]),
        "Throughput (MWh)": val([
            "mean_throughput_mwh",
            "total_throughput_mwh",
            "throughput_mwh",
        ]),
        "Voltage Dev. (p.u.)": val([
            "mean_voltage_deviation",
            "voltage_deviation",
        ]),
        "Feasible Rate": val([
            "feasible_rate",
            "mean_feasible_rate",
        ]),
        "Solve Time (s)": val([
            "mean_solve_time_sec",
            "solve_time_sec",
        ]),
        "Source File": str(f),
    }


def build_tables():
    rows = []

    for system, base_dir in SYSTEMS.items():
        if not base_dir.exists():
            print(f"Missing directory: {base_dir}")
            continue

        for controller in CONTROLLERS:
            cdir = base_dir / controller
            if not cdir.exists():
                continue

            topology_dirs = sorted(
                [p for p in cdir.iterdir() if p.is_dir() and p.name.upper().startswith("TP")],
                key=lambda p: int(p.name.upper().replace("TP", ""))
            )

            for tp_dir in topology_dirs:
                row = read_one(system, base_dir, controller, tp_dir.name)
                if row is not None:
                    rows.append(row)

    out = pd.DataFrame(rows)

    out_dir = ROOT / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / "mpc_smpc_optimization_baselines_all.csv"
    out.to_csv(full_csv, index=False)

    print(f"Saved full table: {full_csv}")

    # One table per system
    for system in out["System"].dropna().unique():
        sdf = out[out["System"] == system].copy()

        system_csv = out_dir / f"{system.lower()}_mpc_smpc_table.csv"
        sdf.to_csv(system_csv, index=False)

        print(f"Saved: {system_csv}")

        # LaTeX version
        latex_path = out_dir / f"{system.lower()}_mpc_smpc_table.tex"

        display_cols = [
            "Topology",
            "Controller",
            "Reward",
            "Energy Cost ($)",
            "Grid Import (MWh)",
            "Curtailment (MWh)",
            "Throughput (MWh)",
            "Voltage Dev. (p.u.)",
            "Feasible Rate",
            "Solve Time (s)",
        ]

        existing_cols = [c for c in display_cols if c in sdf.columns]

        latex = sdf[existing_cols].to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
        )

        latex_path.write_text(latex)

        print(f"Saved LaTeX: {latex_path}")


if __name__ == "__main__":
    build_tables()
