# Topology-Aware Multi-Agent Reinforcement Learning for Distributed Energy Storage in Reconfigurable Distribution Networks

This repository contains the official implementation of the paper:

> **Topology-Aware Multi-Agent Reinforcement Learning for Distributed Energy Storage in Reconfigurable Distribution Networks**

The framework develops topology-aware graph-based multi-agent reinforcement learning (MARL) controllers for coordinated operation of distributed energy storage systems (DESSs) in reconfigurable radial distribution networks. The proposed approach combines Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with graph neural networks (GNNs), enabling decentralized controllers to exploit electrical network topology while maintaining scalable execution.

---

## Highlights

- Graph-based MARL framework for coordinated DESS control
- Supports IEEE 33-bus and IEEE 69-bus benchmark distribution systems
- Zero-shot topology generalization across unseen feeder reconfigurations
- Comparison with topology-blind MLP-MADDPG baseline
- Comparison with deterministic MPC and stochastic MPC (SMPC)
- Includes statistical significance analysis using paired Wilcoxon signed-rank tests with Holm correction
- Includes computational efficiency benchmarking

---

# Repository Structure

```
configs/          Configuration files
data/             Network data and time-series processing
environments/     Reinforcement learning environment
experiments/      Training and evaluation pipelines
models/           GNN and MADDPG models
results/          Experimental results and evaluation outputs
scripts/          Utility and analysis scripts
training/         MARL training implementation
utility/          Helper functions and graph utilities
```

---

# Implemented Methods

### Graph-Based Controllers

- GCN-MADDPG
- GAT-MADDPG
- TAGConv-MADDPG

### Baseline

- MLP-MADDPG

### Optimization-Based Controllers

- Deterministic Model Predictive Control (MPC)
- Stochastic Model Predictive Control (SMPC)

---

# Benchmark Systems

Experiments are performed on

- IEEE 33-bus distribution feeder
- IEEE 69-bus distribution feeder

The trained policies are evaluated under multiple unseen feeder reconfiguration scenarios to assess zero-shot topology generalization.

---

# Repository Contents

The repository includes

- Source code
- Training implementation
- Evaluation scripts
- Trained model configurations
- Statistical analysis scripts
- Computational benchmarking scripts
- Experimental result summaries

---

# Requirements

Typical dependencies include

- Python 3.11+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- SciPy
- NetworkX
- Gymnasium
- Gurobi (for MPC/SMPC experiments)

Install dependencies using your preferred Python environment.

---

# Data
Processed 15-minute node-level load and photovoltaic (PV) time-series
data for the IEEE 33-bus and IEEE 69-bus systems are included in this
repository under `data/`. These profiles were derived from the SMART-DS
dataset provided by the National Renewable Energy Laboratory (NREL):
https://data.openei.org/submissions/2981

Electricity prices are based on NYISO day-ahead Locational Based Marginal
Prices (LBMPs).

---

# Citation

If you use this repository in your research, please cite:

```
Citation information will be added after publication.
```

---

# License

This repository is released under the MIT License.

---

# Contact

**Muhammad Asif**

Department of Electrical and Computer Engineering

University at Albany, SUNY

Email: aasif@albany.edu
