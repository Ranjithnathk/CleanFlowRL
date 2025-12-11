import numpy as np
from collections import defaultdict

class EpsilonGreedyStrategy:
    def __init__(self, epsilon_start: float = 1.0, epsilon_min: float = 0.05, decay: float = 0.995):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay = decay

    def choose_action(self, q_values: np.ndarray, state_key) -> int:
        if np.random.rand() < self.epsilon:
            action = np.random.randint(len(q_values))
        else:
            action = int(np.argmax(q_values))
        self._decay()
        return action

    def _decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


class UCBStrategy:
    def __init__(self, c: float = 1.0):
        self.c = c
        self.counts = defaultdict(lambda: np.zeros(0, dtype=int))
        self.total_counts = defaultdict(int)

    def choose_action(self, q_values: np.ndarray, state_key) -> int:
        n_actions = len(q_values)
        if self.counts[state_key].size == 0:
            self.counts[state_key] = np.zeros(n_actions, dtype=int)

        counts = self.counts[state_key]
        total = self.total_counts[state_key]

        # choose untried actions first
        for a in range(n_actions):
            if counts[a] == 0:
                self.counts[state_key][a] += 1
                self.total_counts[state_key] += 1
                return a

        # UCB formula
        ucb_values = q_values + self.c * np.sqrt(np.log(total + 1) / (counts + 1e-8))
        action = int(np.argmax(ucb_values))

        self.counts[state_key][action] += 1
        self.total_counts[state_key] += 1
        return action
