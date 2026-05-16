import numpy as np

from environments.dess_env import DESSEnv


def main():
    env = DESSEnv(mode="train", seed=123)

    obs, info = env.reset(seed=123)

    print("===================================")
    print("FULL 96-STEP ROLLOUT TEST")
    print("===================================")
    print("Start date_time:", info["date_time"])
    print("Episode length:", env.episode_length)
    print("Delta t hours:", env.delta_t_hours)
    print("Initial feasible:", info["feasible"])
    print("Initial min voltage:", info["min_voltage_pu"])
    print("Initial max voltage:", info["max_voltage_pu"])
    print("Initial max line current:", info["max_line_current_pu"])
    print()

    total_reward = np.zeros(env.num_agents, dtype=np.float64)
    infeasible_requested_count = 0
    accepted_infeasible_count = 0

    min_voltage_seen = float(info["min_voltage_pu"])
    max_voltage_seen = float(info["max_voltage_pu"])
    max_line_current_seen = float(info["max_line_current_pu"])

    final_info = info

    for step in range(env.episode_length):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        min_voltage_seen = min(min_voltage_seen, float(info["min_voltage_pu"]))
        max_voltage_seen = max(max_voltage_seen, float(info["max_voltage_pu"]))
        max_line_current_seen = max(
            max_line_current_seen,
            float(info["max_line_current_pu"]),
        )

        if info["infeasible_action"]:
            infeasible_requested_count += 1

        if not info["feasible"]:
            accepted_infeasible_count += 1
            raise AssertionError(
                f"Accepted transition became infeasible at step {step}. "
                f"date_time={info['date_time']}, "
                f"minV={info['min_voltage_pu']}, "
                f"maxV={info['max_voltage_pu']}, "
                f"maxI={info['max_line_current_pu']}"
            )

        if not info["converged"]:
            raise AssertionError(
                f"Power flow did not converge at step {step}. "
                f"date_time={info['date_time']}"
            )

        if not np.all(np.isfinite(reward)):
            raise AssertionError(f"Reward has NaN/Inf at step {step}.")

        if not np.all(np.isfinite(obs["x"])):
            raise AssertionError(f"Observation x has NaN/Inf at step {step}.")

        if not np.all(np.isfinite(obs["agent_obs"])):
            raise AssertionError(f"agent_obs has NaN/Inf at step {step}.")

        final_info = info

        print(
            f"step={step:03d} | "
            f"date={info['date_time']} | "
            f"feasible={info['feasible']} | "
            f"infeasible_requested={info['infeasible_action']} | "
            f"minV={info['min_voltage_pu']:.4f} | "
            f"maxV={info['max_voltage_pu']:.4f} | "
            f"maxI={info['max_line_current_pu']:.4f} | "
            f"reward_mean={np.mean(reward):.4f}"
        )

        if terminated or truncated:
            print()
            print("Episode ended.")
            print("terminated:", terminated)
            print("truncated:", truncated)
            break

    print()
    print("===================================")
    print("ROLLOUT SUMMARY")
    print("===================================")
    print("Final date_time:", final_info["date_time"])
    print("Total reward per agent:", total_reward)
    print("Mean total reward:", np.mean(total_reward))
    print("Infeasible requested actions corrected:", infeasible_requested_count)
    print("Accepted infeasible transitions:", accepted_infeasible_count)
    print("Minimum voltage seen:", min_voltage_seen)
    print("Maximum voltage seen:", max_voltage_seen)
    print("Maximum line current seen:", max_line_current_seen)
    print("Final feasible:", final_info["feasible"])
    print("Final converged:", final_info["converged"])
    print("===================================")
    print("FULL ROLLOUT TEST PASSED")
    print("===================================")


if __name__ == "__main__":
    main()