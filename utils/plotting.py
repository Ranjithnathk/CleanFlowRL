import matplotlib.pyplot as plt
from typing import List

def plot_learning_curve(rewards: List[float], title: str, save_path: str | None = None):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
