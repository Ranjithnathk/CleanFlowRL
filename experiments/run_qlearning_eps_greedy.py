import numpy as np
import os

from env.data_generator import generate_synthetic_dataset
from env.cleaning_env import CleaningEnv
from rl.q_learning_agent import QLearningAgent
from rl.exploration import EpsilonGreedyStrategy
from utils.plotting import plot_learning_curve

def run_qlearning_eps_greedy(
    num_episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.99,
):
    env = CleaningEnv(max_steps=5)
    agent = QLearningAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        alpha=alpha,
        gamma=gamma,
    )
    explore = EpsilonGreedyStrategy(
        epsilon_start=1.0,
        epsilon_min=0.05,
        decay=0.995,
    )

    episode_rewards = []

    for ep in range(num_episodes):
        df = generate_synthetic_dataset()
        state = env.reset(df)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, explore_strategy=explore)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            print(f"[Q-learning ε-greedy] Episode {ep+1}/{num_episodes}, reward={total_reward:.3f}")

    avg_reward = np.mean(episode_rewards)
    print(f"[Q-learning ε-greedy] Avg reward over {num_episodes} episodes: {avg_reward:.3f}")
    
    # save rewards for comparison plotting
    os.makedirs("logs", exist_ok=True)
    np.save("logs/eps_greedy_rewards.npy", np.array(episode_rewards))
    
    plot_learning_curve(episode_rewards, title="Q-learning with ε-greedy", save_path="qlearning_eps_greedy.png")
    return episode_rewards

if __name__ == "__main__":
    run_qlearning_eps_greedy()
