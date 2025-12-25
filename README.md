# Complex Biological Networks Analysis (شبکه های پیچیده زیستی)

**Project Type:** University Project  
**Primary Stack:** Python

## Description

This is a project focused on analyzing complex biological networks using clustering algorithms, network analysis, and data visualization techniques. The project includes implementations of hierarchical clustering, K-means clustering, and network distance analysis.

## Tech Stack

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook (optional)

## Folder Structure

```
شبکه های پیچیده زیستی/
├── main.py                      # Main analysis script (clustering & heatmaps)
├── main2.py                     # Alternative analysis script
├── main3.py                     # Additional analysis script
├── main4.py                     # Further analysis variations
├── test.py                      # Test script
├── test.ipynb                   # Jupyter notebook for testing
├── random_pc_data.csv           # Random principal component data
├── report.docx                  # Project report (Persian)
├── __pycache__/                 # Python cache directory
└── README.md                    # This file
```

## Setup / Installation

Install required dependencies:
```bash
pip install numpy scipy matplotlib seaborn scikit-learn jupyter pandas
```

Or using requirements file:
```bash
pip install -r requirements.txt
```

## How to Run

### Run Main Analysis
```bash
cd "شبکه های پیچیده زیستی"
python main.py
```

This will:
- Perform K-means clustering (3 clusters)
- Generate hierarchical clustering dendrogram
- Create Euclidean distance heatmap
- Display clustering results

### Run Alternative Analyses
```bash
python main2.py
python main3.py
python main4.py
```

### Run Test Notebook
```bash
jupyter notebook test.ipynb
```

## Inputs/Outputs

**Inputs:**
- `random_pc_data.csv` - Principal component analysis data
- Sample data defined within scripts (e.g., array A = [1, 7, 10, 11, 14, 20])

**Outputs:**
- Console output showing clustering results
- Euclidean distance heatmap visualization
- Dendrogram plots for hierarchical clustering
- K-means cluster assignments
- Saved figures (if configured in scripts)

## Analysis Methods

The project implements:
1. **K-means Clustering**: Partitions data into K clusters
2. **Hierarchical Clustering**: Creates dendrograms using Ward's method
3. **Distance Matrix Calculation**: Computes Euclidean distances
4. **Heatmap Visualization**: Visualizes distance matrices with Seaborn

## Notes

- Multiple Python scripts suggest iterative development and different analysis approaches
- All file paths are relative to the project directory
- Uses inverted blue shades for distance heatmaps
- Clustering analysis with 3 clusters by default
- Persian language documentation in report.docx
- Focuses on network topology and clustering analysis

## Troubleshooting

- If you get import errors: `pip install numpy scipy matplotlib seaborn scikit-learn`
- For display issues with plots, try: `plt.show()` or run in Jupyter
- If CSV reading fails, check file encoding: `pd.read_csv(..., encoding='utf-8')`
- For dendrogram issues, ensure scipy is properly installed
- If memory issues occur, reduce dataset size
