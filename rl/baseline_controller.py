from typing import List

class BaselineController:
    """
    Fixed, hand-crafted cleaning sequence:
    e.g., Impute -> Dedup -> Outlier -> Normalize -> Validate -> Finalize
    """

    def __init__(self, tool_action_indices: List[int], finalize_action: int):
        self.sequence = tool_action_indices + [finalize_action]

    def run_episode(self, env, df):
        state = env.reset(df)
        total_reward = 0.0
        for action in self.sequence:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        return total_reward
