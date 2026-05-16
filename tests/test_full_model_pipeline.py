"""
Full graph-aware DESS MARL smoke test.

Checks:
    1. All train reset states are feasible
    2. Full 96-step rollout with random env actions
    3. Actor forward/select action
    4. Agent-specific critic forward
    5. GraphAwareMADDPG select_action()
    6. GraphAwareMADDPG update() on fake batch

Run:
    PYTHONPATH=. python tests/test_full_model_pipeline.py
"""

import numpy as np
import torch

from environments.dess_env import DESSEnv
from models.actor import GraphAwareDESSActors
from models.critic import GraphAwareAgentCritic
from models.maddpg import GraphAwareMADDPG


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def print_section(title):
    print("\n===================================")
    print(title)
    print("===================================")


def make_tensor_obs(obs):
    return {
        "x": torch.as_tensor(obs["x"], dtype=torch.float32),
        "edge_index": torch.as_tensor(obs["edge_index"], dtype=torch.long),
        "edge_attr": torch.as_tensor(obs["edge_attr"], dtype=torch.float32),
        "agent_obs": torch.as_tensor(obs["agent_obs"], dtype=torch.float32),
    }


def test_all_train_reset_states():
    print_section("TEST 1: ALL TRAIN RESET STATES FEASIBLE")

    env = DESSEnv(mode="train", seed=123)
    bad_episodes = []

    for start_idx in env.episode_start_indices:
        load_kw, load_kvar, pv_kw, price, dt = env._get_profiles(start_idx)

        pf = env.power_flow.run_power_flow(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_kw,
            dess_power_kw=np.zeros(env.num_buses),
        )

        if not pf["converged"] or not pf["feasible"]:
            bad_episodes.append(
                (
                    start_idx,
                    dt,
                    float(np.min(pf["voltage_pu"])),
                    float(pf["max_voltage_violation"]),
                    float(pf["max_line_current_violation"]),
                )
            )

    print("Total train episode starts:", len(env.episode_start_indices))
    print("Bad reset episodes:", len(bad_episodes))

    for item in bad_episodes[:20]:
        print(item)

    check(len(bad_episodes) == 0, "Some training reset states are infeasible.")

    print("ALL TRAIN RESET STATES FEASIBLE")


def test_full_random_rollout():
    print_section("TEST 2: FULL 96-STEP RANDOM ROLLOUT")

    env = DESSEnv(mode="train", seed=123)
    obs, info = env.reset(seed=123)

    print("Start date_time:", info["date_time"])
    print("Episode length:", env.episode_length)
    print("Delta t hours:", env.delta_t_hours)
    print("Initial feasible:", info["feasible"])
    print("Initial min voltage:", info["min_voltage_pu"])
    print("Initial max voltage:", info["max_voltage_pu"])
    print("Initial max line current:", info["max_line_current_pu"])

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

        check(info["converged"], f"Power flow did not converge at step {step}.")
        check(np.all(np.isfinite(reward)), f"Reward has NaN/Inf at step {step}.")
        check(
            np.all(np.isfinite(obs["x"])),
            f"Observation x has NaN/Inf at step {step}.",
        )
        check(
            np.all(np.isfinite(obs["agent_obs"])),
            f"agent_obs has NaN/Inf at step {step}.",
        )

        final_info = info

        if step < 5 or step % 20 == 0 or terminated or truncated:
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
            break

    print("Final date_time:", final_info["date_time"])
    print("Total reward per agent:", total_reward)
    print("Mean total reward:", np.mean(total_reward))
    print("Infeasible requested actions corrected:", infeasible_requested_count)
    print("Accepted infeasible transitions:", accepted_infeasible_count)
    print("Minimum voltage seen:", min_voltage_seen)
    print("Maximum voltage seen:", max_voltage_seen)
    print("Maximum line current seen:", max_line_current_seen)

    print("FULL RANDOM ROLLOUT TEST PASSED")

    return obs


def test_actor_critic_and_maddpg():
    print_section("TEST 3: ACTOR, AGENT-SPECIFIC CRITIC, AND MADDPG MODEL")

    env = DESSEnv(mode="train", seed=456)
    obs, info = env.reset(seed=456)

    num_agents = env.num_agents
    node_feature_dim = env.config["graph"]["node_feature_dim"]
    agent_obs_dim = env.agent_obs_dim

    obs_t = make_tensor_obs(obs)

    actor = GraphAwareDESSActors(
        node_feature_dim=node_feature_dim,
        agent_obs_dim=agent_obs_dim,
        dess_buses=env.dess_buses,
        action_dim_per_agent=1,
        gnn_hidden_dim=64,
        gnn_embedding_dim=64,
        gnn_num_layers=3,
        gnn_type="gcn",
        actor_hidden_dims=(256, 256),
        share_actor=False,
    )

    critic = GraphAwareAgentCritic(
        node_feature_dim=node_feature_dim,
        agent_obs_dim=agent_obs_dim,
        num_agents=num_agents,
        agent_bus_idx=env.dess_buses[0],
        action_dim_per_agent=1,
        gnn_hidden_dim=64,
        gnn_embedding_dim=64,
        gnn_num_layers=3,
        gnn_type="gcn",
        critic_hidden_dims=(256, 256),
    )

    actor.eval()
    critic.eval()

    with torch.no_grad():
        actions = actor(
            x=obs_t["x"],
            edge_index=obs_t["edge_index"],
            edge_attr=obs_t["edge_attr"],
            agent_obs=obs_t["agent_obs"],
        )

        q_value = critic(
            x=obs_t["x"],
            edge_index=obs_t["edge_index"],
            edge_attr=obs_t["edge_attr"],
            agent_obs=obs_t["agent_obs"],
            actions=actions,
        )

    check(
        actions.shape == (num_agents,),
        f"Actor action shape mismatch: {actions.shape}",
    )
    check(
        q_value.shape == (1, 1),
        f"Agent-specific critic Q shape mismatch: {q_value.shape}",
    )
    check(torch.isfinite(actions).all().item(), "Actor actions contain NaN/Inf.")
    check(torch.isfinite(q_value).all().item(), "Critic Q contains NaN/Inf.")

    print("Actor forward OK")
    print("Agent-specific critic forward OK")
    print("actions:", actions.detach().cpu().numpy())
    print("q_value:", q_value.detach().cpu().numpy())

    model = GraphAwareMADDPG(
        node_feature_dim=node_feature_dim,
        agent_obs_dim=agent_obs_dim,
        dess_buses=env.dess_buses,
        num_agents=num_agents,
        action_dim_per_agent=1,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        device="cpu",
        share_actor=False,
        gnn_type="gcn",
        gnn_hidden_dim=64,
        gnn_embedding_dim=64,
        gnn_num_layers=3,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
    )

    check(
        len(model.critics) == num_agents,
        f"Expected {num_agents} critics, got {len(model.critics)}.",
    )

    selected_action = model.select_action(obs, noise_std=0.10)

    check(
        selected_action.shape == (num_agents,),
        f"MADDPG select_action shape mismatch: {selected_action.shape}",
    )
    check(
        np.all(np.isfinite(selected_action)),
        "MADDPG selected action contains NaN/Inf.",
    )
    check(np.all(selected_action <= 1.0 + 1e-6), "Selected action above 1.")
    check(np.all(selected_action >= -1.0 - 1e-6), "Selected action below -1.")

    print("MADDPG select_action OK")
    print("selected_action:", selected_action)

    next_obs, reward, terminated, truncated, step_info = env.step(
        selected_action.astype(np.float32)
    )

    check(step_info["converged"], "MADDPG selected action caused PF non-convergence.")
    check(step_info["feasible"], "MADDPG accepted transition infeasible.")
    check(np.all(np.isfinite(reward)), "Reward after MADDPG action contains NaN/Inf.")

    print("MADDPG action env.step OK")
    print("reward:", reward)
    print("accepted_action:", step_info["accepted_action"])

    batch_size = 4

    obs_batch = {
        "x": np.stack([obs["x"]] * batch_size, axis=0),
        "edge_index": obs["edge_index"],
        "edge_attr": obs["edge_attr"],
        "agent_obs": np.stack([obs["agent_obs"]] * batch_size, axis=0),
        "actions": np.stack([selected_action] * batch_size, axis=0),
        "rewards": np.stack([reward] * batch_size, axis=0),
        "dones": np.zeros((batch_size, num_agents), dtype=np.float32),
        "next_x": np.stack([next_obs["x"]] * batch_size, axis=0),
        "next_edge_index": next_obs["edge_index"],
        "next_edge_attr": next_obs["edge_attr"],
        "next_agent_obs": np.stack([next_obs["agent_obs"]] * batch_size, axis=0),
    }

    update_info = model.update(obs_batch)

    required_keys = [
        "critic_loss",
        "actor_loss",
        "mean_q",
        "mean_target_q",
        "mean_actor_q",
    ]

    for key in required_keys:
        check(key in update_info, f"Missing update key: {key}")
        check(np.isfinite(update_info[key]), f"Update metric {key} is NaN/Inf.")

    for agent_idx in range(num_agents):
        for key in [
            f"critic_loss_agent_{agent_idx}",
            f"mean_q_agent_{agent_idx}",
            f"mean_target_q_agent_{agent_idx}",
            f"mean_actor_q_agent_{agent_idx}",
        ]:
            check(key in update_info, f"Missing per-agent update key: {key}")
            check(np.isfinite(update_info[key]), f"Update metric {key} is NaN/Inf.")

    print("MADDPG update OK")
    print("update_info:", update_info)

    print("ACTOR + AGENT-SPECIFIC CRITIC + MADDPG MODEL TEST PASSED")


def main():
    test_all_train_reset_states()
    test_full_random_rollout()
    test_actor_critic_and_maddpg()

    print("\n===================================")
    print("FULL MODEL PIPELINE TEST PASSED")
    print("===================================\n")


if __name__ == "__main__":
    main()