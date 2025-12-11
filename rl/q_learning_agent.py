import numpy as np

class QLearningAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma

        # For simplicity, start with a table for discretized states
        # Later we can swap this for a function approximator if we want.
        self.q_table = {}

    def _state_to_key(self, state: np.ndarray):
        # simple discretization placeholder — refine later
        return tuple(np.round(state, 2))

    def select_action(self, state: np.ndarray, explore_strategy=None) -> int:
        """
        Select an action given a state.
        If explore_strategy is None, act greedily (argmax Q-values).
        If explore_strategy is provided, delegate exploration behavior.
        """
        key = self._state_to_key(state)

        # Initialize Q row if missing
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.num_actions, dtype=float)

        q_values = self.q_table[key]

        # Greedy action if no exploration strategy is provided
        if explore_strategy is None:
            return int(np.argmax(q_values))

        # Otherwise use exploration strategy (epsilon-greedy or UCB)
        return explore_strategy.choose_action(q_values, key)

    def update(self, state, action, reward, next_state, done):
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.num_actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.num_actions)

        q_values = self.q_table[key]
        next_q_values = self.q_table[next_key]

        target = reward
        if not done:
            target += self.gamma * np.max(next_q_values)

        q_values[action] = q_values[action] + self.alpha * (target - q_values[action])
        self.q_table[key] = q_values
