# CleanFlowRL
## An Agentic Workflow System for Data Cleaning with Reinforcement Learning

### Overview
CleanFlowRL is an Agentic Workflow System where a Q-learning agent improves through experience. Each episode starts with a noisy dataset generated synthetically (or loaded from a real file in demo mode). The agent observes dataset quality metrics, selects cleaning tools, and receives rewards based on improvement.

This project was developed for:
INFO 7375 — Reinforcement Learning for Agentic AI Systems 

---

### Key Features
- Agentic workflow planning and tool selection
- Q-learning with ε-greedy and UCB exploration strategies
- State representation from dataset quality metrics
- Multiple cleaning tools as discrete RL actions
- Synthetic and real dataset support
- Baseline vs RL agent comparisons over 3000 episodes

---

### Cleaning Tools (Actions)
- Impute Missing Values  
- Remove Duplicates  
- Remove Outliers  
- Normalize Numeric Features  
- Validate Categorical Labels  
- Finalize (Stop Episode)  

---

### Project Structure

CleanFlowRL/
|
├── env/
│ ├── cleaning_env.py # MDP environment + reward function
│ ├── data_generator.py # Synthetic dataset generator
│ └── state_encoder.py # Extracts numeric state features
│
├── cleaning_tools/
│ ├── impute.py
│ ├── dedup.py
│ ├── outliers.py
│ ├── normalize.py
│ └── categorical_cleaner.py
│
├── rl/
│ ├── q_learning_agent.py
│ ├── exploration.py # ε-greedy and UCB exploration
│ └── baseline_controller.py
│
├── experiments/
│ ├── run_baseline.py
│ ├── run_qlearning_eps_greedy.py
│ ├── run_qlearning_ucb.py
│ └── plot_learning_curves.py
│
├── logs/ # Saved rewards
├── plots/ # Learning curve images
└── README.md

---

### Installation
```bash
git clone https://github.com/<your-username>/CleanFlowRL.git
cd CleanFlowRL

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

### Running Experiments

#### Baseline (fixed workflow)
```bash
python -m experiments.run_baseline
```

#### Q-learning with ε-greedy exploration
```bash
python -m experiments.run_qlearning_eps_greedy
```

#### Q-learning with UCB exploration
```bash
python -m experiments.run_qlearning_ucb
```

#### Generate learning curve comparison plots
```bash
python -m experiments.plot_learning_curves
```

The plots will appear in the plots/ folder.

---

### Results Summary

Average reward over 3000 episodes:

| Method                  | Avg Reward |
|------------------------|------------|
| Baseline               | 0.067      |
| Q-learning (ε-greedy)  | 0.524      |
| Q-learning (UCB)       | 0.304      |

**Key observations:**
- ε-greedy converges to the most effective tool-selection strategy.
- UCB explores more aggressively and converges slower.
- Baseline remains flat because it cannot adapt to dataset variability.

---

### How CleanFlowRL Works (MDP)

- **State**: Vector of dataset quality metrics (missing rate, outlier ratio, duplicate count, variance score, categorical inconsistency, etc.).
- **Actions**: Data-cleaning tools (impute, dedup, remove outliers, normalize, validate categorical labels, finalize).
- **Reward**: Improvement in dataset quality minus penalties for unnecessary or harmful operations.
- **Transition**: Each cleaning tool deterministically modifies the dataset.
- **Goal**: Maximize total cleaning quality improvement while minimizing unnecessary steps.

---

### Using Real Data (Demo)

To clean a real CSV file, place it in the project root and run:

```bash
python demo_clean_real_data.py --file yourdata.csv
```
The RL agent will:

- Load and inspect the dataset
- Encode dataset quality into a state vector
- Apply its learned cleaning sequence
- Print before/after cleaning statistics

---

### Technical Report
- The full PDF report includes:
- System architecture diagram
- Mathematical MDP formulation
- Q-learning and exploration strategy descriptions
- Reward engineering details
- Experimental setup and analysis
- Learning curves and performance interpretation
- Challenges and solutions
- Ethical considerations
- Future improvements

File: CleanFlowRL_TechnicalReport.pdf

---

### Future Work
- Deep Q-Networks (DQN) for continuous state approximation
- Multi-agent cleaning workflows
- Curriculum learning for progressively harder datasets
- Meta-RL for adapting to unseen data distributions
- Integration with real ETL and data engineering pipelines

---

### Citation

If referencing this project:
```bash
Karunanidhi, Ranjithnath. (2025). CleanFlowRL: An Agentic Workflow System for Data Cleaning with Reinforcement Learning
```

---