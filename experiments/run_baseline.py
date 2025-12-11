import numpy as np
import os

from env.data_generator import generate_synthetic_dataset
from env.cleaning_env import CleaningEnv
from rl.baseline_controller import BaselineController

def run_baseline(num_episodes: int = 3000):
    env = CleaningEnv(max_steps=5)
    # 0=Impute, 1=Dedup, 2=Outliers, 3=Normalize, 4=SchemaValidator
    baseline_sequence = [0, 1, 2, 3, 4]
    controller = BaselineController(
        tool_action_indices=baseline_sequence,
        finalize_action=env.finalize_action,
    )

    rewards = []
    for ep in range(num_episodes):
        df = generate_synthetic_dataset()
        total_reward = controller.run_episode(env, df)
        rewards.append(total_reward)
        if (ep + 1) % 100 == 0:
            print(f"[Baseline] Episode {ep+1}/{num_episodes}, reward={total_reward:.3f}")

    avg_reward = np.mean(rewards)
    print(f"[Baseline] Avg reward over {num_episodes} episodes: {avg_reward:.3f}")

    # save rewards for plotting
    os.makedirs("logs", exist_ok=True)
    np.save("logs/baseline_rewards.npy", np.array(rewards))
    return rewards

if __name__ == "__main__":
    run_baseline()
