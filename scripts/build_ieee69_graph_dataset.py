"""
Create IEEE 69-bus benchmark CSV files from MATPOWER case69.m.

Source:
    data/matpower/case69.m

Outputs:
    data/network/ieee69/bus_data.csv
    data/network/ieee69/line_data.csv
    data/network/ieee69/dess_buses.csv
    data/network/ieee69/gridtensor_bus.csv
    data/network/ieee69/gridtensor_line.csv

Clean project files use zero-based bus indexing.
GridTensor files use one-based bus indexing.
"""

from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE69_PATH = PROJECT_ROOT / "data" / "matpower" / "case69.m"

OUT_DIR = PROJECT_ROOT / "data" / "network" / "ieee69"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IEEE69 DESS placement
# ============================================================
# One-based:
# [14, 16, 18, 20, 22, 24, 26, 27, 65]
#
# Zero-based:
# [13, 15, 17, 19, 21, 23, 25, 26, 64]

DESS_BUSES = [13, 15, 17, 19, 21, 23, 25, 26, 64]


def extract_matrix_block(text, matrix_name):
    """
    Extract MATPOWER matrix block.

    Example:
        mpc.bus = [
            ...
        ];
    """

    pattern = rf"mpc\.{matrix_name}\s*=\s*\[(.*?)\];"

    match = re.search(pattern, text, re.DOTALL)

    if match is None:
        raise ValueError(f"Could not find matrix block: {matrix_name}")

    return match.group(1)


def parse_numeric_matrix(block_text):
    """
    Parse MATPOWER numeric matrix.
    """

    rows = []

    for line in block_text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("%"):
            continue

        if "%" in line:
            line = line.split("%")[0].strip()

        line = line.rstrip(";").strip()

        if not line:
            continue

        values = [float(x) for x in line.split()]

        rows.append(values)

    return rows


def main():

    if not CASE69_PATH.exists():
        raise FileNotFoundError(
            f"Could not find MATPOWER file:\n{CASE69_PATH}"
        )

    print("Reading:", CASE69_PATH)

    with open(CASE69_PATH, "r") as f:
        text = f.read()

    # ============================================================
    # Extract MATPOWER matrices
    # ============================================================

    bus_block = extract_matrix_block(text, "bus")
    branch_block = extract_matrix_block(text, "branch")

    bus_rows_raw = parse_numeric_matrix(bus_block)
    branch_rows_raw = parse_numeric_matrix(branch_block)

    # ============================================================
    # Build bus_data.csv
    # ============================================================
    #
    # MATPOWER bus columns:
    #
    # 0 bus_i
    # 1 type
    # 2 Pd
    # 3 Qd
    #

    bus_rows = []

    for row in bus_rows_raw:

        bus_one_based = int(row[0])

        bus_zero_based = bus_one_based - 1

        load_kw = float(row[2])
        load_kvar = float(row[3])

        bus_rows.append(
            {
                "bus": bus_zero_based,
                "load_kw": load_kw,
                "load_kvar": load_kvar,
                "pv_kw": 0.0,
            }
        )

    bus_df = (
        pd.DataFrame(bus_rows)
        .sort_values("bus")
        .reset_index(drop=True)
    )

    # ============================================================
    # Build line_data.csv
    # ============================================================
    #
    # MATPOWER branch columns:
    #
    # 0 fbus
    # 1 tbus
    # 2 r
    # 3 x
    # 10 status
    #

    line_rows = []

    for row in branch_rows_raw:

        from_bus = int(row[0]) - 1
        to_bus = int(row[1]) - 1

        r_ohm = float(row[2])
        x_ohm = float(row[3])

        status = int(row[10])

        line_rows.append(
            {
                "from_bus": from_bus,
                "to_bus": to_bus,
                "r_ohm": r_ohm,
                "x_ohm": x_ohm,
                "status": status,
                "line_type": "radial" if status == 1 else "tie",
            }
        )

    line_df = pd.DataFrame(line_rows).reset_index(drop=True)

    # ============================================================
    # DESS placement
    # ============================================================

    dess_df = pd.DataFrame({"bus": DESS_BUSES})

    # ============================================================
    # GridTensor-compatible bus file
    # ============================================================

    gridtensor_bus_rows = []

    for bus_idx in range(len(bus_df)):

        gridtensor_bus_rows.append(
            {
                "NODES": bus_idx + 1,
                "Tb": 1 if bus_idx == 0 else 0,
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

    gridtensor_bus_df.to_csv(
        OUT_DIR / "gridtensor_bus.csv",
        index=False,
    )

    gridtensor_line_df.to_csv(
        OUT_DIR / "gridtensor_line.csv",
        index=False,
    )

    # ============================================================
    # Summary
    # ============================================================

    active_lines = int((line_df["status"] == 1).sum())
    inactive_lines = int((line_df["status"] == 0).sum())

    print()
    print("Created IEEE69 benchmark CSV files.")
    print()

    print("Output directory:")
    print(OUT_DIR)

    print()
    print("Files written:")
    print(OUT_DIR / "bus_data.csv")
    print(OUT_DIR / "line_data.csv")
    print(OUT_DIR / "dess_buses.csv")
    print(OUT_DIR / "gridtensor_bus.csv")
    print(OUT_DIR / "gridtensor_line.csv")

    print()
    print("Number of buses:", len(bus_df))
    print("Number of lines:", len(line_df))
    print("Active lines:", active_lines)
    print("Inactive lines:", inactive_lines)

    print()
    print("Total load kW:", bus_df["load_kw"].sum())
    print("Total load kvar:", bus_df["load_kvar"].sum())

    print()
    print("DESS buses (zero-based):")
    print(dess_df["bus"].tolist())

    print()
    print("DESS buses (one-based):")
    print([x + 1 for x in dess_df["bus"].tolist()])


if __name__ == "__main__":
    main()