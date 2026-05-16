import numpy as np
from environments.dess_env import DESSEnv


def main():
    env = DESSEnv(mode="train", seed=123)

    bad_episodes = []

    for start_idx in env.episode_start_indices:
        env.episode_start_index = start_idx
        env.current_step = 0
        env.current_index = start_idx

        for battery in env.batteries:
            battery.reset()

        env.previous_actions = np.zeros(env.num_agents, dtype=np.float32)
        env.last_dess_power_kw = np.zeros(env.num_buses, dtype=np.float64)

        load_kw, load_kvar, pv_kw, price, dt = env._get_profiles(start_idx)

        pf = env.power_flow.run_power_flow(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_kw,
            dess_power_kw=np.zeros(env.num_buses),
        )

        if not pf["converged"] or not pf["feasible"]:
            bad_episodes.append((start_idx, dt, np.min(pf["voltage_pu"]), pf["max_voltage_violation"]))

    print("Total train episode starts:", len(env.episode_start_indices))
    print("Bad reset episodes:", len(bad_episodes))

    for item in bad_episodes[:20]:
        print(item)

    if bad_episodes:
        raise AssertionError("Some training reset states are infeasible.")

    print("ALL TRAIN RESET STATES FEASIBLE")


if __name__ == "__main__":
    main()