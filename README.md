# Preserving Knowledge Legacy: Continual Learning on Heterogeneous Graph Attention Networks

[![Paper](https://img.shields.io/badge/Paper-Discovery_Science-brightgreen.svg)](TODO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the **LEGATCY** framework presented in the research paper "Preserving Knowledge Legacy: Continual Learning on Heterogeneous Graph Attention Networks". Our proposal addresses the challenge of catastrophic forgetting in evolving Heterogeneous Information Networks (HINs) by introducing a continual learning framework based on selective experience replay. LEGATCY incrementally updates a heterogeneous Graph Attention Network while retaining a compact memory of informative historical nodes to preserve structural patterns and semantic representations.

## Key Features

* **Heterogeneous Graph Continual Learning:** Adapts Graph Neural Networks to continuously evolving topologies with multiple node and edge types, mitigating catastrophic forgetting without the computational burden of retraining from scratch;
* **Dual Memory Construction:** Identifies a compact set of historical nodes using two complementary strategies: an uncertainty-driven active learning approach (e.g., Least Confidence, Entropy, Margin) to retain informative decision-boundary instances, and a clustering-based spatial strategy (HDBSCAN) to preserve broad representation diversity;
* **Green AI & Efficient Optimization:** Implements a training acceleration mechanism based on Repeated Sampling of Random Subsets (RS2), substantially reducing the required optimization epochs and computational overhead at each continual learning round.

## Repository Structure

* **`src/`**: Core source code directory.
  * `main.py` / `trainer.py`: Main entry points to initialize the continual learning framework and execute training rounds;
  * `models/`: Contains the implementations for the underlying Graph Attention Networks (`GAT.py`, `HeteroGAT.py`);
  * `al_techniques/`: Implements uncertainty-based active learning criteria used for legacy memory selection;
  * `sampling_strategies/`: Houses the logic for experience replay and the RS2 training subgraph extraction mechanisms;
  * `data/`: Contains graph dataset loaders, data converters, and dataset preprocessing scripts (e.g., OpenAlex, MuMiN, PolitiFact).
    
* **`requirements.txt`**: List of project dependencies.

## Requirements

To run the code, ensure you have the libraries specified in `requirements.txt` installed. Based on the framework's architecture, the main dependencies include:

* Python (>=3.8)
* PyTorch
* PyTorch Geometric (for GNN implementations and HIN handling)
* HDBSCAN (for the diversity-preserving spatial clustering strategy)
* Pandas (for tabular data manipulation)

## Citation

```bibtex
Coming soon...
```
