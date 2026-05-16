"""
Utilities for applying and validating IEEE 33-bus topology reconfiguration cases.

This module:
    - copies the base IEEE33 config,
    - applies TP open/close branch status changes,
    - writes temporary TP-specific network CSV files,
    - validates that each topology is connected and radial.

Assumptions:
    - line_data.csv uses zero-based bus indexing.
    - gridtensor_line.csv uses one-based bus indexing.
    - line_data.csv has status column.
    - gridtensor_line.csv has STATUS column.
"""

import copy
from pathlib import Path

import pandas as pd

from configs.ieee33_config import IEEE33_CONFIG
from experiments.topology_generalization.tp_configs import (
    get_topology_case,
    normalize_edge,
)


def _edge_mask_zero_based(df, edge):
    """
    Return mask for an undirected zero-based edge in line_data.csv.
    """

    i, j = normalize_edge(edge)

    return (
        ((df["from_bus"].astype(int) == i) & (df["to_bus"].astype(int) == j))
        | ((df["from_bus"].astype(int) == j) & (df["to_bus"].astype(int) == i))
    )


def _edge_mask_one_based(df, edge):
    """
    Return mask for an undirected edge in gridtensor_line.csv.

    Input edge is zero-based. GridTensor file is one-based.
    """

    i, j = normalize_edge(edge)
    i += 1
    j += 1

    return (
        ((df["FROM"].astype(int) == i) & (df["TO"].astype(int) == j))
        | ((df["FROM"].astype(int) == j) & (df["TO"].astype(int) == i))
    )


def _find_edges_with_status_zero_based(line_df, edges, required_status, label):
    """
    Verify that each edge exists in line_data.csv with required status.
    """

    for edge in edges:
        mask = _edge_mask_zero_based(line_df, edge)

        if not mask.any():
            raise ValueError(
                "{} edge {} does not exist in line_data.csv.".format(label, edge)
            )

        statuses = line_df.loc[mask, "status"].astype(float).tolist()

        if required_status not in statuses:
            raise ValueError(
                "{} edge {} exists, but does not have required status {}. "
                "Found statuses: {}".format(
                    label,
                    edge,
                    required_status,
                    statuses,
                )
            )


def _find_edges_with_status_one_based(gt_df, edges, required_status, label):
    """
    Verify that each edge exists in gridtensor_line.csv with required STATUS.
    """

    for edge in edges:
        mask = _edge_mask_one_based(gt_df, edge)

        if not mask.any():
            raise ValueError(
                "{} edge {} does not exist in gridtensor_line.csv.".format(
                    label,
                    edge,
                )
            )

        statuses = gt_df.loc[mask, "STATUS"].astype(float).tolist()

        if required_status not in statuses:
            raise ValueError(
                "{} edge {} exists, but does not have required STATUS {}. "
                "Found STATUS values: {}".format(
                    label,
                    edge,
                    required_status,
                    statuses,
                )
            )


def _apply_status_changes(line_df, gt_df, open_edges, close_edges):
    """
    Apply open/close status changes to both line files.
    """

    line_df = line_df.copy()
    gt_df = gt_df.copy()

    for edge in open_edges:
        mask = _edge_mask_zero_based(line_df, edge)
        line_df.loc[mask, "status"] = 0.0

        gt_mask = _edge_mask_one_based(gt_df, edge)
        gt_df.loc[gt_mask, "STATUS"] = 0

    for edge in close_edges:
        mask = _edge_mask_zero_based(line_df, edge)
        line_df.loc[mask, "status"] = 1.0

        gt_mask = _edge_mask_one_based(gt_df, edge)
        gt_df.loc[gt_mask, "STATUS"] = 1

    return line_df, gt_df


def _validate_no_duplicate_active_edges(line_df):
    """
    Ensure active topology has no duplicate undirected edges.
    """

    active = line_df[line_df["status"].astype(float) == 1.0].copy()

    edges = []

    for _, row in active.iterrows():
        edge = normalize_edge((int(row["from_bus"]), int(row["to_bus"])))
        edges.append(edge)

    if len(edges) != len(set(edges)):
        duplicates = sorted(
            edge for edge in set(edges) if edges.count(edge) > 1
        )
        raise ValueError(
            "Duplicate active undirected edges found: {}".format(duplicates)
        )


def _validate_connected_and_radial(line_df, num_buses):
    """
    Validate active topology using simple graph traversal.

    For radial connected graph:
        active_edges = num_buses - 1
        all buses reachable from slack/root
    """

    active = line_df[line_df["status"].astype(float) == 1.0].copy()

    active_edges = []

    for _, row in active.iterrows():
        edge = normalize_edge((int(row["from_bus"]), int(row["to_bus"])))
        active_edges.append(edge)

    if len(active_edges) != num_buses - 1:
        raise ValueError(
            "Topology is not radial by edge count. "
            "Expected {} active edges, found {}.".format(
                num_buses - 1,
                len(active_edges),
            )
        )

    adjacency = {i: set() for i in range(num_buses)}

    for i, j in active_edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    visited = set()
    stack = [0]

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        for nbr in adjacency[node]:
            if nbr not in visited:
                stack.append(nbr)

    if len(visited) != num_buses:
        missing = sorted(set(range(num_buses)) - visited)
        raise ValueError(
            "Topology is not connected. Missing buses: {}".format(missing)
        )


def validate_topology_files(line_df, gt_df, num_buses):
    """
    Validate project-format and GridTensor-format topology files.
    """

    if "status" not in line_df.columns:
        raise ValueError("line_data.csv must contain a 'status' column.")

    if "STATUS" not in gt_df.columns:
        raise ValueError("gridtensor_line.csv must contain a 'STATUS' column.")

    active_line_count = int((line_df["status"].astype(float) == 1.0).sum())
    active_gt_count = int((gt_df["STATUS"].astype(float) == 1.0).sum())

    if active_line_count != active_gt_count:
        raise ValueError(
            "Active line count mismatch: line_data has {}, gridtensor_line has {}.".format(
                active_line_count,
                active_gt_count,
            )
        )

    _validate_no_duplicate_active_edges(line_df)
    _validate_connected_and_radial(line_df, num_buses=num_buses)


def build_topology_config(
    case_name,
    base_config=None,
    output_root=None,
    validate=True,
):
    """
    Build a TP-specific config by writing temporary topology CSV files.

    Parameters
    ----------
    case_name : str
        TP case name, e.g., TP1, TP2, ..., TP7.

    base_config : dict or None
        Base config. If None, IEEE33_CONFIG is used.

    output_root : str or Path or None
        Directory where temporary topology files are written.
        If None:
            results/topology_generalization/generated_topologies/<case_name>

    validate : bool
        If True, validate active topology after applying status changes.

    Returns
    -------
    dict
        Modified config pointing to TP-specific line files.
    """

    config = copy.deepcopy(IEEE33_CONFIG if base_config is None else base_config)
    case = get_topology_case(case_name)

    case_name = str(case_name).upper()
    open_edges = [normalize_edge(edge) for edge in case["open_edges"]]
    close_edges = [normalize_edge(edge) for edge in case["close_edges"]]

    num_buses = int(config["network"]["num_buses"])

    line_path = Path(config["paths"]["line_data"])
    gt_path = Path(config["paths"]["gridtensor_line_data"])

    line_df = pd.read_csv(line_path)
    gt_df = pd.read_csv(gt_path)

    if "status" not in line_df.columns:
        line_df["status"] = 1.0

    if "STATUS" not in gt_df.columns:
        gt_df["STATUS"] = 1

    # ------------------------------------------------------------
    # Pre-check requested switches
    # ------------------------------------------------------------
    _find_edges_with_status_zero_based(
        line_df=line_df,
        edges=open_edges,
        required_status=1.0,
        label="Open",
    )

    _find_edges_with_status_zero_based(
        line_df=line_df,
        edges=close_edges,
        required_status=0.0,
        label="Close",
    )

    _find_edges_with_status_one_based(
        gt_df=gt_df,
        edges=open_edges,
        required_status=1.0,
        label="Open",
    )

    _find_edges_with_status_one_based(
        gt_df=gt_df,
        edges=close_edges,
        required_status=0.0,
        label="Close",
    )

    # ------------------------------------------------------------
    # Apply topology perturbation
    # ------------------------------------------------------------
    tp_line_df, tp_gt_df = _apply_status_changes(
        line_df=line_df,
        gt_df=gt_df,
        open_edges=open_edges,
        close_edges=close_edges,
    )

    if validate:
        validate_topology_files(
            line_df=tp_line_df,
            gt_df=tp_gt_df,
            num_buses=num_buses,
        )

    # ------------------------------------------------------------
    # Write generated topology files
    # ------------------------------------------------------------
    if output_root is None:
        output_root = (
            Path(config["paths"]["results_dir"])
            / "topology_generalization"
            / "generated_topologies"
            / case_name
        )
    else:
        output_root = Path(output_root) / case_name

    output_root.mkdir(parents=True, exist_ok=True)

    tp_line_path = output_root / "line_data.csv"
    tp_gt_path = output_root / "gridtensor_line.csv"

    tp_line_df.to_csv(tp_line_path, index=False)
    tp_gt_df.to_csv(tp_gt_path, index=False)

    # ------------------------------------------------------------
    # Update config paths
    # ------------------------------------------------------------
    config["case_name"] = "ieee33_{}".format(case_name)
    config["paths"]["line_data"] = tp_line_path
    config["paths"]["gridtensor_line_data"] = tp_gt_path

    config["topology_case"] = {
        "case_name": case_name,
        "description": case["description"],
        "open_edges": open_edges,
        "close_edges": close_edges,
        "generated_dir": output_root,
    }

    return config


def summarize_topology_case(case_name, base_config=None):
    """
    Build and summarize one topology case.
    """

    config = build_topology_config(
        case_name=case_name,
        base_config=base_config,
        validate=True,
    )

    line_df = pd.read_csv(config["paths"]["line_data"])

    active = line_df[line_df["status"].astype(float) == 1.0]
    inactive = line_df[line_df["status"].astype(float) == 0.0]

    return {
        "case_name": case_name,
        "description": config["topology_case"]["description"],
        "active_lines": int(len(active)),
        "inactive_lines": int(len(inactive)),
        "open_edges": config["topology_case"]["open_edges"],
        "close_edges": config["topology_case"]["close_edges"],
        "generated_dir": str(config["topology_case"]["generated_dir"]),
    }


def validate_all_topology_cases(case_names=None, base_config=None):
    """
    Validate all requested topology cases.

    Returns
    -------
    list[dict]
        Summary rows for each case.
    """

    from experiments.topology_generalization.tp_configs import list_topology_cases

    if case_names is None:
        case_names = list_topology_cases(include_baseline=True)

    rows = []

    for case_name in case_names:
        summary = summarize_topology_case(
            case_name=case_name,
            base_config=base_config,
        )
        rows.append(summary)

    return rows