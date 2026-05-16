import sys
from pathlib import Path
import copy
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.ieee33_config import IEEE33_CONFIG
from utility.data_loader import load_ieee33_all_data, select_time_series_row
from utility.grid import build_grid_from_static_data
from utility.power_flow import build_power_flow_solver


def scan_scale(scale, config, static_data, time_series, indices):
    grid = build_grid_from_static_data(static_data, config)
    pf = build_power_flow_solver(grid, config)

    min_voltage_seen = 999.0
    worst_index = None
    feasible_count = 0

    for idx in indices:
        row = select_time_series_row(time_series, idx)

        load_kw = row["load_kw"].astype(float) * scale
        pv_kw = row["pv_kw"].astype(float)

        base_load_kw = grid.get_base_load_kw().astype(float)
        base_load_kvar = grid.get_base_load_kvar().astype(float)

        total_base_load = np.sum(base_load_kw)
        load_scale_for_q = np.sum(load_kw) / total_base_load
        load_kvar = base_load_kvar * load_scale_for_q

        dess_power_kw = np.zeros(config["network"]["num_buses"], dtype=float)

        result = pf.run_power_flow(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_kw,
            dess_power_kw=dess_power_kw,
        )

        min_v = float(np.min(result["voltage_pu"]))

        if min_v < min_voltage_seen:
            min_voltage_seen = min_v
            worst_index = idx

        if result["converged"] and min_v >= config["network"]["v_min"]:
            feasible_count += 1

    return {
        "scale": scale,
        "min_voltage": min_voltage_seen,
        "worst_index": worst_index,
        "feasible_count": feasible_count,
        "total_count": len(indices),
        "all_feasible": feasible_count == len(indices),
    }


def main():
    config = copy.deepcopy(IEEE33_CONFIG)

    all_data = load_ieee33_all_data(config, require_time_series=True)
    static_data = all_data["static"]
    time_series = all_data["processed_time_series"]
    splits = all_data["splits"]

    indices = (
        splits["train_indices"]
        + splits["val_indices"]
        + splits["test_indices"]
    )

    print("Total baseline points:", len(indices))
    print("Voltage lower limit:", config["network"]["v_min"])
    print()

    candidate_scales = np.arange(1.00, 0.40, -0.05)

    best_scale = None
    results = []

    for scale in candidate_scales:
        out = scan_scale(
            scale=scale,
            config=config,
            static_data=static_data,
            time_series=time_series,
            indices=indices,
        )

        results.append(out)

        print(
            "scale={:.2f} | minV={:.4f} | feasible={}/{} | worst_index={}".format(
                out["scale"],
                out["min_voltage"],
                out["feasible_count"],
                out["total_count"],
                out["worst_index"],
            )
        )

        if out["all_feasible"] and best_scale is None:
            best_scale = scale

    print()
    print("==============================")

    if best_scale is None:
        print("No fully feasible scale found in tested range.")
        print("Try extending candidate_scales lower, e.g. down to 0.20.")
    else:
        print("Largest fully feasible load scale:", best_scale)

        target_scale = best_scale * 0.98
        print("Recommended conservative scale:", round(target_scale, 4))

    print("==============================")


if __name__ == "__main__":
    main()
