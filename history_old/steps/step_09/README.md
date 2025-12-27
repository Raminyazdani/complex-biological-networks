# Complex Biological Networks Analysis

A Python framework for analyzing complex biological networks using clustering algorithms, causal discovery, and network controllability analysis.

## Features

- **Clustering Analysis**: K-means and hierarchical clustering with distance matrices
- **Causal Discovery**: PC algorithm for inferring causal relationships
- **Network Controllability**: Exact and structural controllability tests
- **Driver Node Optimization**: Find minimum control configurations
- **Symbolic Mathematics**: Theoretical analysis tools

## Tech Stack

- Python 3.x
- NumPy, SciPy
- NetworkX
- Matplotlib, Seaborn
- Scikit-learn
- SymPy
- Jupyter

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Clustering Analysis
```bash
python main.py
```

### Causal Discovery  
```bash
python main2.py
```

### Network Controllability
```bash
python main3.py
```

### Minimum Driver Nodes
```bash
python main4.py
```

### Symbolic Rank Calculator
```bash
python test.py
```

### Interactive Experiments
```bash
jupyter notebook test.ipynb
```

## Inputs/Outputs

- **Inputs**: Network topology, observational data, sample arrays
- **Outputs**: Controllability metrics, causal graphs, clustering visualizations

## Methods

1. K-means and hierarchical clustering
2. PC algorithm for causal structure learning
3. Controllability matrix computation
4. Minimum driver node search
5. Symbolic matrix operations

## Troubleshooting

- Import errors: `pip install -r requirements.txt`
- Plot display issues: Run in Jupyter or add `plt.show()`
- Memory issues (main4.py): Reduce network size

## License

See LICENSE file for details.
