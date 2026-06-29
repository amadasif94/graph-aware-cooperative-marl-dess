"""
Create IEEE 33-bus Baran-Wu benchmark CSV files using pandapower.

This version exports:
    1. Original in-service radial branches with status = 1
    2. Normally-open tie-switch branches with status = 0

This supports Gao-style topology reconfiguration:
    - open one sectionalizing branch: status 1 -> 0
    - close one tie-switch branch:   status 0 -> 1

Outputs:
    data/network/ieee33/bus_data.csv
    data/network/ieee33/line_data.csv
    data/network/ieee33/dess_buses.csv
    data/network/ieee33/gridtensor_bus.csv
    data/network/ieee33/gridtensor_line.csv

Clean project files use zero-based bus indexing.
GridTensor files use one-based bus indexing.
"""

from pathlib import Path
import pandas as pd
import pandapower.networks as pn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "network" / "ieee33"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IEEE33 normally-open tie-switches
# ============================================================
# These are common Baran-Wu IEEE33 tie-lines.
# Zero-based bus pairs:
#   one-based 8-21  -> zero-based 7-20
#   one-based 9-15  -> zero-based 8-14
#   one-based 12-22 -> zero-based 11-21
#   one-based 18-33 -> zero-based 17-32
#   one-based 25-29 -> zero-based 24-28
#
# NOTE:
# pandapower.case33bw() may not include these as out-of-service lines.
# Therefore, we explicitly add them here.
#
# The impedance values should be replaced with source-specific values
# if your IEEE33 data source provides exact tie-line parameters.

TIE_LINES = [
    # one-based 8-21  -> zero-based 7-20
    {"from_bus": 7,  "to_bus": 20, "r_ohm": 0.0200, "x_ohm": 0.0100},

    # one-based 9-15  -> zero-based 8-14
    {"from_bus": 8,  "to_bus": 14, "r_ohm": 0.0300, "x_ohm": 0.0200},

    # one-based 12-22 -> zero-based 11-21
    {"from_bus": 11, "to_bus": 21, "r_ohm": 0.0200, "x_ohm": 0.0100},

    # one-based 18-33 -> zero-based 17-32
    {"from_bus": 17, "to_bus": 32, "r_ohm": 0.0100, "x_ohm": 0.0050},

    # one-based 25-29 -> zero-based 24-28
    {"from_bus": 24, "to_bus": 28, "r_ohm": 0.0100, "x_ohm": 0.0050},
]


DESS_BUSES = [11, 15, 24, 29, 32]


def normalize_edge(i, j):
    i = int(i)
    j = int(j)
    return min(i, j), max(i, j)


def edge_key(row):
    return normalize_edge(row["from_bus"], row["to_bus"])


def main():
    net = pn.case33bw()

    # ============================================================
    # Clean project-format bus data
    # ============================================================

    bus_rows = []

    for bus_idx in net.bus.index:
        load_at_bus = net.load[net.load["bus"] == bus_idx]

        load_kw = float(load_at_bus["p_mw"].sum() * 1000.0)
        load_kvar = float(load_at_bus["q_mvar"].sum() * 1000.0)

        bus_rows.append(
            {
                "bus": int(bus_idx),
                "load_kw": load_kw,
                "load_kvar": load_kvar,
                "pv_kw": 0.0,
            }
        )

    bus_df = pd.DataFrame(bus_rows).sort_values("bus").reset_index(drop=True)

    # ============================================================
    # Clean project-format line data
    # ============================================================
    # Export radial lines with status = 1.
    # Export tie-switch lines with status = 0.

    line_rows = []

    for _, row in net.line.iterrows():
        r_ohm = float(row["r_ohm_per_km"] * row["length_km"])
        x_ohm = float(row["x_ohm_per_km"] * row["length_km"])

        line_rows.append(
            {
                "from_bus": int(row["from_bus"]),
                "to_bus": int(row["to_bus"]),
                "r_ohm": r_ohm,
                "x_ohm": x_ohm,
                "status": 1 if bool(row["in_service"]) else 0,
                "line_type": "radial" if bool(row["in_service"]) else "tie",
            }
        )

    line_df = pd.DataFrame(line_rows)

    existing_edges = set(edge_key(row) for _, row in line_df.iterrows())

    for tie in TIE_LINES:
        key = normalize_edge(tie["from_bus"], tie["to_bus"])

        if key in existing_edges:
            mask = (
                ((line_df["from_bus"] == key[0]) & (line_df["to_bus"] == key[1]))
                | ((line_df["from_bus"] == key[1]) & (line_df["to_bus"] == key[0]))
            )
            line_df.loc[mask, "status"] = 0
            line_df.loc[mask, "line_type"] = "tie"
        else:
            line_df = pd.concat(
                [
                    line_df,
                    pd.DataFrame(
                        [
                            {
                                "from_bus": int(tie["from_bus"]),
                                "to_bus": int(tie["to_bus"]),
                                "r_ohm": float(tie["r_ohm"]),
                                "x_ohm": float(tie["x_ohm"]),
                                "status": 0,
                                "line_type": "tie",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    line_df = line_df.reset_index(drop=True)

    # ============================================================
    # DESS placement
    # ============================================================

    dess_df = pd.DataFrame({"bus": DESS_BUSES})

    # ============================================================
    # GridTensor-compatible bus file
    # ============================================================

    gridtensor_bus_rows = []

    for bus_idx in net.bus.index:
        gridtensor_bus_rows.append(
            {
                "NODES": int(bus_idx) + 1,
                "Tb": 1 if int(bus_idx) == 0 else 0,
            }
        )

    gridtensor_bus_df = (
        pd.DataFrame(gridtensor_bus_rows)
        .sort_values("NODES")
        .reset_index(drop=True)
    )

    # ============================================================
    # GridTensor-compatible line file
    # ============================================================
    # FROM and TO are one-based.
    # STATUS is copied from project-format line_data.csv.

    gridtensor_line_rows = []

    for _, row in line_df.iterrows():
        gridtensor_line_rows.append(
            {
                "FROM": int(row["from_bus"]) + 1,
                "TO": int(row["to_bus"]) + 1,
                "R": float(row["r_ohm"]),
                "X": float(row["x_ohm"]),
                "B": 0.0,
                "STATUS": int(row["status"]),
                "TAP": 1.0,
            }
        )

    gridtensor_line_df = pd.DataFrame(
        gridtensor_line_rows,
        columns=["FROM", "TO", "R", "X", "B", "STATUS", "TAP"],
    )

    # ============================================================
    # Save files
    # ============================================================

    bus_df.to_csv(OUT_DIR / "bus_data.csv", index=False)
    line_df.to_csv(OUT_DIR / "line_data.csv", index=False)
    dess_df.to_csv(OUT_DIR / "dess_buses.csv", index=False)
    gridtensor_bus_df.to_csv(OUT_DIR / "gridtensor_bus.csv", index=False)
    gridtensor_line_df.to_csv(OUT_DIR / "gridtensor_line.csv", index=False)

    # ============================================================
    # Summary
    # ============================================================

    active_lines = int((line_df["status"] == 1).sum())
    inactive_tie_lines = int((line_df["status"] == 0).sum())

    print("Created IEEE33 benchmark CSV files with tie-switches.")
    print()
    print("Output directory:", OUT_DIR)
    print()
    print("Clean project-format files:")
    print("bus_data:", OUT_DIR / "bus_data.csv")
    print("line_data:", OUT_DIR / "line_data.csv")
    print("dess_buses:", OUT_DIR / "dess_buses.csv")
    print()
    print("GridTensor-compatible files:")
    print("gridtensor_bus:", OUT_DIR / "gridtensor_bus.csv")
    print("gridtensor_line:", OUT_DIR / "gridtensor_line.csv")
    print()
    print("Number of buses:", len(bus_df))
    print("Number of total lines:", len(line_df))
    print("Number of active radial lines:", active_lines)
    print("Number of normally-open tie-lines:", inactive_tie_lines)
    print("Total load kW:", bus_df["load_kw"].sum())
    print("Total load kvar:", bus_df["load_kvar"].sum())
    print("DESS buses:", dess_df["bus"].tolist())
    print()
    print("Tie-lines written:")
    print(line_df[line_df["line_type"] == "tie"].to_string(index=False))
    print()
    print("GridTensor line STATUS counts:")
    print(gridtensor_line_df["STATUS"].value_counts().sort_index())


if __name__ == "__main__":
    main()