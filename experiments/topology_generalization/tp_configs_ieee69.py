"""
Topology reconfiguration benchmark cases for IEEE 69-bus
graph-aware MADDPG evaluation.

All bus pairs use zero-based indexing.

These cases are adapted to the IEEE69 network file used in this project.
Each topology preserves:
    - 69 buses
    - 68 active branches
    - radial structure
    - network connectivity
    - identical DESS placement
    - identical observation dimensions
    - identical action dimensions

TP1 = baseline topology
TP2-TP3 = single moderate/local reconfiguration cases
TP4-TP5 = double reconfiguration cases
TP6-TP7 = single downstream/local reconfiguration cases
TP8-TP9 = stronger downstream stress-test reconfiguration cases
"""

TOPOLOGY_CASES = {
    "TP1": {
        "description": "Baseline IEEE69 radial topology.",
        "open_edges": [],
        "close_edges": [],
    },

    "TP2": {
        "description": "Single downstream reconfiguration: open 68-69, close 13-69.",
        "open_edges": [(67, 68)],
        "close_edges": [(12, 68)],
    },

    "TP3": {
        "description": "Single local reconfiguration: open 45-46, close 44-46.",
        "open_edges": [(44, 45)],
        "close_edges": [(43, 45)],
    },

    "TP4": {
        "description": "Double reconfiguration: TP2 + TP3.",
        "open_edges": [
            (67, 68),
            (44, 45),
        ],
        "close_edges": [
            (12, 68),
            (43, 45),
        ],
    },

    "TP5": {
        "description": "Double downstream reconfiguration: open 51-52 and 53-54; close 8-52 and 9-54.",
        "open_edges": [
            (50, 51),
            (52, 53),
        ],
        "close_edges": [
            (7, 51),
            (8, 53),
        ],
    },

    "TP6": {
        "description": "Single local reconfiguration: open 12-13, close 11-13.",
        "open_edges": [(11, 12)],
        "close_edges": [(10, 12)],
    },

    "TP7": {
        "description": "Single downstream reconfiguration: open 68-69, close 11-69.",
        "open_edges": [(67, 68)],
        "close_edges": [(10, 68)],
    },

    "TP8": {
        "description": "Strong downstream reconfiguration: open 64-65, close 11-65.",
        "open_edges": [(63, 64)],
        "close_edges": [(10, 64)],
    },

    "TP9": {
        "description": "Strong double downstream reconfiguration: open 64-65 and 68-69; close 11-65 and 13-69.",
        "open_edges": [
            (63, 64),
            (67, 68),
        ],
        "close_edges": [
            (10, 64),
            (12, 68),
        ],
    },
}


def normalize_edge(edge):
    i, j = edge
    i = int(i)
    j = int(j)

    if i == j:
        raise ValueError("Self-loop edge is invalid: {}".format(edge))

    return min(i, j), max(i, j)


def get_topology_case(case_name):
    case_name = str(case_name).upper()

    if case_name not in TOPOLOGY_CASES:
        raise ValueError(
            "Unknown topology case '{}'. Available cases: {}".format(
                case_name,
                sorted(TOPOLOGY_CASES.keys()),
            )
        )

    base_case = TOPOLOGY_CASES[case_name]

    return {
        "description": base_case["description"],
        "open_edges": [normalize_edge(edge) for edge in base_case["open_edges"]],
        "close_edges": [normalize_edge(edge) for edge in base_case["close_edges"]],
    }


def list_topology_cases(include_baseline=True):
    cases = list(TOPOLOGY_CASES.keys())

    if not include_baseline:
        cases = [case for case in cases if case != "TP1"]

    return cases


def describe_topology_cases():
    rows = []

    for case_name in list_topology_cases(include_baseline=True):
        case = get_topology_case(case_name)

        rows.append({
            "case": case_name,
            "description": case["description"],
            "open_edges": case["open_edges"],
            "close_edges": case["close_edges"],
        })

    return rows