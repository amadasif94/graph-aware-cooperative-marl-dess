"""
Smoke test for GraphMADDPGTrainer.

Run:
    PYTHONPATH=. python tests/test_trainer.py
"""

import os
import tempfile
import numpy as np

from environments.dess_env import DESSEnv
from training.trainer import GraphMADDPGTrainer


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    print("\n==============================")
    print("TEST: GraphMADDPGTrainer")
    print("==============================")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_env = DESSEnv(mode="train", seed=123)
        eval_env = DESSEnv(mode="val", seed=456)

        config = {
            "batch_size": 8,
            "buffer_size": 200,
            "warmup_steps": 12,
            "total_steps": 20,
            "update_after": 12,
            "update_every": 1,
            "eval_every": 10,
            "save_every": 10,
            "noise_std": 0.10,
            "eval_episodes": 1,
            "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
            "store_buffer_on_device": False,
            "share_actor": False,
            "gnn_type": "gcn",
            "gnn_hidden_dim": 64,
            "gnn_embedding_dim": 64,
            "gnn_num_layers": 3,
            "actor_hidden_dims": (128, 128),
            "critic_hidden_dims": (128, 128),
        }

        trainer = GraphMADDPGTrainer(
            env=train_env,
            eval_env=eval_env,
            config=config,
            device="cpu",
        )

        obs, info = trainer._reset_env(train_env)

        check("x" in obs, "Reset obs missing x.")
        check("agent_obs" in obs, "Reset obs missing agent_obs.")
        check(info["converged"], "Initial reset did not converge.")
        check(info["feasible"], "Initial reset is infeasible.")

        print("Trainer init/reset OK")

        next_obs, reward, done, step_info = trainer.collect_step(
            obs,
            random_action=True,
        )

        check(len(trainer.replay_buffer) == 1, "Replay buffer should contain 1 transition.")
        check(np.all(np.isfinite(reward)), "Reward contains NaN/Inf.")
        check(step_info["converged"], "Collect step did not converge.")
        check(step_info["feasible"], "Collect step accepted infeasible transition.")

        print("collect_step(random_action=True) OK")
        print("buffer size:", len(trainer.replay_buffer))

        obs = trainer.warmup()

        check(
            len(trainer.replay_buffer) >= config["warmup_steps"],
            "Warmup did not fill replay buffer enough.",
        )

        print("Warmup OK")
        print("buffer size after warmup:", len(trainer.replay_buffer))
        print("global_step:", trainer.global_step)

        batch = trainer.replay_buffer.sample(config["batch_size"])
        update_info = trainer.agent.update(batch)

        for key, value in update_info.items():
            check(np.isfinite(value), f"Update metric {key} is NaN/Inf.")

        print("Manual trainer update OK")
        print("update_info:", update_info)

        eval_info = trainer.evaluate()

        check("mean_reward" in eval_info, "Eval missing mean_reward.")
        check(np.isfinite(eval_info["mean_reward"]), "Eval mean_reward is NaN/Inf.")
        check(eval_info["mean_length"] > 0, "Eval mean_length should be positive.")

        print("Evaluate OK")
        print("eval_info:", eval_info)

        logs = trainer.train()

        check(trainer.global_step >= config["total_steps"], "Trainer did not reach total_steps.")
        check(len(logs) >= 1, "Expected at least one eval log.")
        check(os.path.exists(os.path.join(tmpdir, "checkpoints", "maddpg_final.pt")),
              "Final checkpoint was not saved.")

        print("Trainer train() OK")
        print("final global_step:", trainer.global_step)
        print("logs:", logs)

    print("\n==============================")
    print("TRAINER SMOKE TEST PASSED")
    print("==============================\n")


if __name__ == "__main__":
    main()
