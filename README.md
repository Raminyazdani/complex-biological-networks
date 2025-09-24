# Network Controllability & Causal Discovery Framework

A Python framework for analyzing complex biological networks.

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

**Clustering Analysis:**
```bash
cd network-analysis  # Wrong: this directory doesn't exist
python main.py
```

**Causal Discovery:**
```bash
cd network-analysis  # Wrong path
python main2.py
```

**Network Controllability:**
```bash
python main3.py  # This one happens to be correct
```

## Data

Data files are located in `data/random_pc_data.csv` (Wrong path - file is in root)

## Features

- K-means and hierarchical clustering
- PC algorithm for causal discovery
- Network controllability analysis
- Minimum driver node optimization
- Symbolic mathematics tools
