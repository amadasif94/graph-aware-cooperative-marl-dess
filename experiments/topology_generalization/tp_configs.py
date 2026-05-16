"""
Topology reconfiguration benchmark cases for IEEE 33-bus
graph-aware MADDPG evaluation.

All bus pairs use zero-based indexing.

TP1 = baseline radial topology
TP2-TP3 = single-line reconfiguration cases
TP4-TP5 = double-line reconfiguration cases
TP6-TP7 = downstream reconfiguration cases

Each topology preserves:
    - 33 buses
    - radial network structure
    - network connectivity
    - identical DESS placement
    - identical observation dimensions
    - identical action dimensions
"""

TOPOLOGY_CASES = {
    "TP1": {
        "description": "Baseline IEEE33 radial topology.",
        "open_edges": [],
        "close_edges": [],
    },

    "TP2": {
        "description": "Single reconfiguration: open 6-7, close tie 8-21.",
        "open_edges": [(5, 6)],
        "close_edges": [(7, 20)],
    },

    "TP3": {
        "description": "Single reconfiguration: open 10-11, close tie 12-22.",
        "open_edges": [(9, 10)],
        "close_edges": [(11, 21)],
    },

    "TP4": {
        "description": "Double reconfiguration: TP2 + TP3.",
        "open_edges": [
            (5, 6),
            (9, 10),
        ],
        "close_edges": [
            (7, 20),
            (11, 21),
        ],
    },

    "TP5": {
        "description": "Double reconfiguration: open 6-7 and 27-28; close ties 8-21 and 25-29.",
        "open_edges": [
            (5, 6),
            (26, 27),
        ],
        "close_edges": [
            (7, 20),
            (24, 28),
        ],
    },

    "TP6": {
        "description": "Downstream reconfiguration: open 26-27, close tie 25-29.",
        "open_edges": [(25, 26)],              #  29, 30 
        "close_edges": [(24, 28)],             #17, 32
    },

    "TP7": {
        "description": "Downstream reconfiguration: open 27-28, close tie 25-29.",
        "open_edges": [(26, 27)],
        "close_edges": [(24, 28)],
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
