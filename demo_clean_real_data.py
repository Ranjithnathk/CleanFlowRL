import argparse
import numpy as np
import pandas as pd

from env.cleaning_env import CleaningEnv
from env.data_generator import generate_synthetic_dataset
from rl.q_learning_agent import QLearningAgent
from rl.exploration import EpsilonGreedyStrategy


ACTION_NAMES = {
    0: "Impute Missing Values",
    1: "Remove Duplicates",
    2: "Remove Outliers",
    3: "Normalize Numeric Features",
    4: "Validate & Clean Categorical Labels",
    5: "Finalize"
}


def quality_summary(df, title):
    print("\n==========", title, "==========")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Duplicates:", df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        z = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].std() + 1e-8)
        outliers = (np.abs(z) > 3).any(axis=1).sum()
        print("Approx numeric outliers:", outliers)
    else:
        print("No numeric columns available for outlier detection.")

    # Simple categorical inconsistency metric
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    inconsistency = 0
    for col in cat_cols:
        inconsistency += df[col].isna().sum()

    print("Categorical inconsistency score:", inconsistency)
    print("=========================================")


def train_agent(env, episodes=500):
    print("\n=== Training RL Agent on Synthetic Datasets (for generalization) ===")

    agent = QLearningAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        alpha=0.1,
        gamma=0.99,
    )

    explore = EpsilonGreedyStrategy(
        epsilon_start=1.0,
        epsilon_min=0.05,
        decay=0.995
    )

    for ep in range(episodes):
        df_syn = generate_synthetic_dataset()
        state = env.reset(df_syn)
        done = False

        while not done:
            action = agent.select_action(state, explore_strategy=explore)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} completed.")

    print("Training completed.\n")
    return agent


def run_agent_on_real_data(df_real):
    print("\n===== RAW INPUT PREVIEW =====")
    print(df_real.head())

    env = CleaningEnv(max_steps=6)

    # Train agent on synthetic datasets
    agent = train_agent(env, episodes=500)

    # BEFORE CLEANING
    quality_summary(df_real, "BEFORE CLEANING")

    state = env.reset(df_real)
    done = False
    step = 0

    print("\n===== AGENT ACTION TRACE =====")
    while not done:
        action = agent.select_action(state, explore_strategy=None)
        print(f"Step {step}: {ACTION_NAMES.get(action, str(action))}")

        next_state, reward, done, info = env.step(action)
        state = next_state
        step += 1

        if step > 10:  
            print("Stopping early to avoid infinite loop.")
            break

    # AFTER CLEANING
    cleaned_df = getattr(env, "current_df", None)
    if cleaned_df is None:
        cleaned_df = env.df  # fallback

    quality_summary(cleaned_df, "AFTER CLEANING")

    cleaned_df.to_csv("adult_cleaned_output.csv", index=False)
    print("\nSaved cleaned dataset as adult_cleaned_output.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    # Ensure missing values are NaN
    df = df.replace(" ?", np.nan)

    run_agent_on_real_data(df)


if __name__ == "__main__":
    main()
