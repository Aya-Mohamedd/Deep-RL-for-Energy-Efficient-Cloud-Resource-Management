# Deep RL for Energy-Efficient Cloud Scheduling

Reinforcement learning for multi-objective cloud task scheduling. Two agents: **DQN** (value-based) and **PPO** (policy-gradient actor-critic), both from **Stable-Baselines3** are trained inside a custom `gymnasium.Env` that simulates a heterogeneous multi-server cluster under stochastic task arrivals, and benchmarked against four classic heuristics: **Round Robin**, **SJF** (shortest job first), **Greedy Min-Min**, and **Random**.

The reward is a scalarized combination of three competing objectives: **throughput**, **latency**, and **energy consumption**  controlled by weights `(w1, w2, w3)`. This repo explores how well the learned policies trade these objectives off compared to static heuristics, and how they generalize across workload intensity, server counts, heterogeneous power profiles, and transient failures.

## Repo Contents

| Notebook | Description |
|---|---|
| `experiment00.ipynb` | **Master notebook.** Baseline 4-server cluster: environment definition, heuristic baselines, DQN/PPO training, evaluation sweep across arrival rates, and a reward-weight sensitivity sweep. |
| `experiment03-energy_asymmetry.ipynb` | **Experiment 3 - Energy Asymmetry.** Replaces the uniform power model with a per-VM affine power model (`CloudSchedulingEnvE3`, `P_i(u_i) = static + dynamic * u_i`), mixing energy-efficient "micro" VMs with power-hungry "performance" VMs to test whether the agents learn to exploit the cheaper hardware. |
| `experiment04.ipynb` | **Experiment 4 - Transient Server Failure Resilience.** Injects a temporary VM crash mid-episode to test whether trained agents recover and generalize better than static heuristics. |
| `server_cloud_scheduler.ipynb` | **8-Server Scaling Study.** Re-instantiates the environment with `n_servers = 8` (double the baseline) with a wider network (`net_arch=[256, 256]`) to test how agents and heuristics generalize to a larger action space. |
| `stress_test_cloud_scheduler_.ipynb` | **Stress Test - Heavy-Tailed Workload.** Widens the task-execution-length distribution (`exec_length_range = (100, 10000)`) to induce head-of-line blocking / convoy effects, isolating how PPO's credit assignment compares to queue-aware heuristics under high-variance load. |

Each notebook is self-contained and can be run independently (dependency install → environment setup → baselines → training → evaluation → plots).

## Environment

- **Framework:** `gymnasium.Env`, vectorized/trained via `stable-baselines3`
- **Observation space:** `5·N + 6`, where `N` is the number of servers (per-server queue/utilization/power features plus global state)
- **Action space:** discrete, assign the incoming task to one of `N` servers
- **Task arrivals:** Poisson process with configurable arrival rate `λ`
- **Reward:** `w1 * throughput_term - w2 * latency_term - w3 * energy_term`, with weights swept in the sensitivity study

## Baselines

- **Round Robin**: cycles through servers in order
- **SJF**: shortest job first, queue-load variant
- **Greedy Min-Min**: greedily assigns to the least-loaded server
- **Random**: uniform random server assignment

## Agents

- **DQN**: value-based, discrete action space
- **PPO**: policy-gradient actor-critic

Both are trained via Stable-Baselines3 under the scalarized reward, then evaluated deterministically against the heuristics.

## Reward-Weight Sensitivity Sweep

Trained agents are fine-tuned under five weight configurations to map the throughput/latency/energy trade-off surface:

| Configuration | `(w1, w2, w3)` |
|---|---|
| `throughput_priority` | (0.70, 0.20, 0.10) |
| `balanced` | (0.50, 0.30, 0.20) |
| `equal` | (0.33, 0.33, 0.34) |
| `latency_priority` | (0.20, 0.60, 0.20) |
| `energy_priority` | (0.20, 0.20, 0.60) |

## Running

1. Open a notebook (Jupyter / Colab).
2. Run the setup cells to install dependencies:
   ```
   pip install gymnasium stable-baselines3 numpy pandas matplotlib
   ```
3. Set the run profile at the top of the config cell:
   - `MODE = "demo"`, fast pipeline check (short training, fewer episodes/arrival rates)
   - `MODE = "full"`, paper-scale replication (200k training timesteps, 30k sensitivity fine-tune steps, 20 eval episodes per config, full arrival-rate sweep `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]`)
4. Run all cells top-to-bottom. Outputs are written to:
   - `models/` , trained DQN/PPO checkpoints (`.zip`)
   - `results/` , evaluation summary tables and CSVs
   - `results/figures/` , sensitivity-sweep figures
   - `tb_logs/` , TensorBoard training logs

## Requirements

- Python 3.9+
- `gymnasium`
- `stable-baselines3`
- `numpy`, `pandas`, `matplotlib`

## Key Findings (Baseline Experiment)

- Heuristics (SJF / Greedy Min-Min in particular) dominate raw reward and throughput at high workload intensity, since they optimize purely for task completion without regard to power draw.
- DQN and PPO trade off some throughput/reward for lower energy consumption, with the gap widening at high arrival rates (`λ = 3`).
- The reward-weight sweep confirms a clear Pareto frontier: `energy_priority` configurations substantially cut power draw at the cost of large latency increases, while `throughput_priority` configurations recover near-heuristic performance.
- Extension experiments (energy asymmetry, server scaling, transient failure, heavy-tailed workload) test how robust these trade-offs are under more realistic infrastructure conditions.

See each notebook's "Result Observations" / "Conclusion" sections for the full quantitative breakdown and figures.
