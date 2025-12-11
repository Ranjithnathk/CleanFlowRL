import os
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skipping.")
        return None
    return np.load(path)


def moving_average(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_single_curve(rewards: np.ndarray, label: str, window: int = 50):
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = moving_average(rewards, window=window)
    smoothed_eps = np.arange(window, len(rewards) + 1)

    plt.plot(episodes, rewards, alpha=0.2, linewidth=0.5)  # raw (faint)
    plt.plot(smoothed_eps, smoothed, linewidth=1.5, label=f"{label} (smoothed)")


def main():
    os.makedirs("plots", exist_ok=True)

    baseline = load_rewards("logs/baseline_rewards.npy")
    eps = load_rewards("logs/eps_greedy_rewards.npy")
    ucb = load_rewards("logs/ucb_rewards.npy")

    # 1) Combined comparison plot
    plt.figure(figsize=(8, 5))
    if baseline is not None:
        plot_single_curve(baseline, "Baseline", window=50)
    if eps is not None:
        plot_single_curve(eps, "Q-learning ε-greedy", window=100)
    if ucb is not None:
        plot_single_curve(ucb, "Q-learning UCB", window=100)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curves: Baseline vs Q-learning Variants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/learning_curves_comparison.png", dpi=150)

    # 2) Separate plots (clean versions) for each agent (optional)
    if eps is not None:
        plt.figure(figsize=(8, 5))
        plot_single_curve(eps, "Q-learning ε-greedy", window=100)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-learning with ε-greedy (Smoothed)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/qlearning_eps_greedy_smooth.png", dpi=150)

    if ucb is not None:
        plt.figure(figsize=(8, 5))
        plot_single_curve(ucb, "Q-learning UCB", window=100)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-learning with UCB (Smoothed)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/qlearning_ucb_smooth.png", dpi=150)


if __name__ == "__main__":
    main()
