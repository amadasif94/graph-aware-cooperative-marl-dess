# ============================================================
# run_ieee69_optimization_baselines.py
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mpc_smpc.optimization import baseline_config_ieee69 as cfg
from experiments.mpc_smpc.optimization.rolling_horizon_eval_ieee69 import (
    run_controller_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IEEE69 MPC/SMPC optimization baselines."
    )

    parser.add_argument(
        "--controller",
        type=str,
        default="smpc",
        choices=["mpc", "smpc", "both"],
        help="Which baseline to run.",
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        default=[getattr(cfg, "DEFAULT_TOPOLOGY_CASE", "TP1")],
        help=(
            "Topology cases to evaluate, e.g. "
            "TP1 TP2 TP3 TP4 TP5 TP6 TP7 TP8 TP9."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=str,
        default="1",
        help="Number of episodes to evaluate, or 'all'.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=cfg.HORIZON_T,
        help="Rolling MPC/SMPC horizon length in 15-min steps.",
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=10,
        help="Number of scenarios for SMPC. Ignored for MPC.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="DESSEnv mode.",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=24,
        help="Print progress every N steps.",
    )

    return parser.parse_args()


def normalize_episodes(value: str):
    if str(value).lower() == "all":
        return "all"

    return int(value)


def normalize_topology_case(topology_case: str) -> str:
    if hasattr(cfg, "normalize_topology_case"):
        return cfg.normalize_topology_case(topology_case)

    return str(topology_case or "TP1").upper()


def normalize_topology_list(topologies: list[str]) -> list[str]:
    out = []

    for topology_case in topologies:
        topology_case = normalize_topology_case(topology_case)

        if topology_case not in out:
            out.append(topology_case)

    return out


def topology_output_dir(base_dir: Path, topology_case: str) -> Path:
    topology_case = normalize_topology_case(topology_case)

    if hasattr(cfg, "get_topology_output_dir"):
        return cfg.get_topology_output_dir(base_dir, topology_case)

    out_dir = Path(base_dir) / topology_case
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def print_runner_header(args, episodes, topologies):
    print("=" * 72)
    print("IEEE69 OPTIMIZATION BASELINE RUNNER")
    print("=" * 72)
    print(f"controller  : {args.controller}")
    print(f"topologies  : {topologies}")
    print(f"episodes    : {episodes}")
    print(f"horizon     : {args.horizon}")
    print(f"scenarios   : {args.scenarios}")
    print(f"seed        : {args.seed}")
    print(f"mode        : {args.mode}")
    print("=" * 72)


def run_one_controller_for_topology(
    controller: str,
    topology_case: str,
    episodes,
    args,
):
    topology_case = normalize_topology_case(topology_case)

    if controller == "mpc":
        output_dir = topology_output_dir(
            Path(cfg.MPC_RESULTS_DIR),
            topology_case,
        )

        return run_controller_evaluation(
            controller="mpc",
            episodes=episodes,
            horizon_t=args.horizon,
            n_scenarios=1,
            seed=args.seed,
            mode=args.mode,
            topology_case=topology_case,
            output_dir=output_dir,
            print_every=args.print_every,
        )

    if controller == "smpc":
        output_dir = topology_output_dir(
            Path(cfg.SMPC_RESULTS_DIR),
            topology_case,
        )

        return run_controller_evaluation(
            controller="smpc",
            episodes=episodes,
            horizon_t=args.horizon,
            n_scenarios=args.scenarios,
            seed=args.seed,
            mode=args.mode,
            topology_case=topology_case,
            output_dir=output_dir,
            print_every=args.print_every,
        )

    raise ValueError(f"Unknown controller: {controller}")


def main():
    args = parse_args()
    episodes = normalize_episodes(args.episodes)
    topologies = normalize_topology_list(args.topologies)

    print_runner_header(args, episodes, topologies)

    controllers = []

    if args.controller in ["mpc", "both"]:
        controllers.append("mpc")

    if args.controller in ["smpc", "both"]:
        controllers.append("smpc")

    results = {}

    for topology_case in topologies:
        for controller in controllers:
            key = f"{controller}_{topology_case}"

            print()
            print("#" * 72)
            print(f"RUNNING: controller={controller} | topology={topology_case}")
            print("#" * 72)

            results[key] = run_one_controller_for_topology(
                controller=controller,
                topology_case=topology_case,
                episodes=episodes,
                args=args,
            )

    print()
    print("=" * 72)
    print("IEEE69 RUN COMPLETE")
    print("=" * 72)

    for name, result in results.items():
        print()
        print(f"[{name.upper()}] Aggregate summary:")

        aggregate = result.get("aggregate_summary")

        if aggregate is not None and not aggregate.empty:
            print(aggregate.to_string(index=False))
        else:
            print("No aggregate summary produced.")


if __name__ == "__main__":
    main()