# Git History Reconstruction: Network Controllability Analysis (23-Step Expansion)

## History Expansion Note

**Step Count Expansion:**
- **N_old:** 15 steps (from previous historian run)
- **N_target:** 23 steps (ceil(15 × 1.5) = 23)
- **Achieved multiplier:** 1.533× (23/15)

**Mapping from Old Steps to New Step Ranges:**
- Old step 01 → New step 01 (Initial setup) - unchanged
- Old step 02 → New steps 02-03 (Clustering: split into foundation + visualization setup)
- Old step 03 → New step 04 (Distance matrix with bug) - unchanged but renumbered
- Old step 04 → New step 05 (Fix array indexing) - unchanged but renumbered
- Old step 05 → New steps 06-07 (PC algorithm: split into foundations + testing phase)
- Old step 06 → New step 08 (PC algorithm complete) - unchanged but renumbered
- Old step 07 → New steps 09-10 (Controllability: split into setup + basic implementation)
- Old step 08 → New step 11 (Structural controllability with import bug) ← NEW OOPS
- Old step 09 → New step 12 (Fix import error + complete visualization) ← NEW HOTFIX
- Old step 10 → New step 13 (Driver node optimization) - unchanged but renumbered
- Old step 11 → New step 14 (Symbolic math) - unchanged but renumbered
- Old step 12 → New step 15 (Jupyter notebook) - unchanged but renumbered
- Old step 13 → New steps 16-18 (Dependencies: split into core, visualization with typo, fix) ← NEW OOPS/HOTFIX
- Old step 14 → New step 19 (Documentation with errors) - unchanged but renumbered
- Old step 15 → New steps 20-23 (Portfolio refinement: split into README fix, docstrings, gitignore, final polish)

**Explicit Oops → Hotfix Sequences:**

**Sequence 1: Array Indexing Bug (Steps 04 → 05)** [Preserved from previous expansion]
- **What broke:** In step 04, implemented distance matrix calculation with `distance_matrix[i][j] = np.sqrt((A_reshaped[i] - A_reshaped[j]) ** 2)` which fails because after reshaping A to (-1, 1), each A_reshaped[i] is a 1D array, not a scalar.
- **How noticed:** When testing the clustering analysis, got `ValueError: operands could not be broadcast together` because trying to square a 1D array.
- **What fixed it:** In step 05, corrected to `distance_matrix[i][j] = np.sqrt((A[i][0] - A[j][0]) ** 2)` to properly access the scalar value at index [0].
- **Why realistic:** This is a common mistake when working with NumPy array reshaping—forgetting that reshape changes dimensionality and access patterns.

**Sequence 2: Import Module Typo (Steps 11 → 12)** [NEW in this expansion]
- **What broke:** In step 11, added structural controllability test to main3.py but made a typo in the import statement: `from colorma import Fore, Style, init` (missing 'a' in colorama).
- **How noticed:** When running `python main3.py`, got `ModuleNotFoundError: No module named 'colorma'`. Spent a few minutes debugging before realizing it was just a typo in the import line.
- **What fixed it:** In step 12, corrected the import to `from colorama import Fore, Style, init` and verified the script runs successfully with colored output.
- **Why realistic:** Import typos are common when adding new dependencies, especially with longer module names. Easy to miss during initial implementation and caught immediately when running the code.

**Sequence 3: Requirements.txt Typo (Steps 17 → 18)** [NEW in this expansion]
- **What broke:** In step 17, added visualization dependencies to requirements.txt but accidentally typed `matplotli>=3.4.0` instead of `matplotlib>=3.4.0` (missing 'b').
- **How noticed:** When trying to install dependencies with `pip install -r requirements.txt`, got error: `ERROR: Could not find a version that satisfies the requirement matplotli>=3.4.0`. Reviewed requirements.txt and found the typo.
- **What fixed it:** In step 18, corrected the package name to `matplotlib>=3.4.0` and verified successful installation.
- **Why realistic:** Typos in dependency files are very common, especially when manually typing package names. The error is immediately caught when trying to install dependencies.

**Sequence 4: README Command Error (Steps 19 → 20)** [Preserved from previous expansion]
- **What broke:** In step 19, documented run commands incorrectly as `cd network-analysis && python main.py` instead of just `python main.py` from repo root. Also had a typo in the data path reference.
- **How noticed:** When following the README instructions to test documentation quality, the commands didn't work because there's no `network-analysis` subdirectory.
- **What fixed it:** In step 20, corrected all run commands to execute from repository root, fixed path references. Continued refinement in steps 21-23 with module docstrings, .gitignore updates, and final polish.
- **Why realistic:** Documentation often gets written before testing and contains command/path errors that are caught during review.

---

## Development Narrative

This project emerged from research into complex network control theory and causal inference. The development followed a natural progression from basic clustering analysis to sophisticated network controllability algorithms, with realistic debugging and iteration cycles. The 23-step history captures the incremental nature of real development, including common mistakes like import typos, requirements file errors, and documentation bugs that get caught and fixed during testing.

---

## Commit History

### Step 01: Initial Repository Setup
**Commit Message:** Initial commit: Project scaffolding and basic structure

**What Changed:**
- Created initial repository with README.md
- Added .gitignore for Python projects
- Set up requirements.txt with core dependencies (numpy, scipy, matplotlib, etc.)
- Added LICENSE file (MIT)
- Basic project structure established

**Rationale:** Standard repository initialization with essential configuration files. Started with a clear foundation before adding code.

**Timestamp:** 2024-07-15

---

### Step 02: Implement Clustering Analysis Foundation
**Commit Message:** Add K-means clustering implementation

**What Changed:**
- Created main.py with K-means clustering algorithm (3 clusters)
- Implemented basic clustering logic with centroid calculation
- Added sample data array for testing: [1, 7, 10, 11, 14, 20]
- Set up structure for hierarchical clustering (to be added next)
- Added imports for numpy and basic scipy functions

**Rationale:** Started with fundamental network analysis—clustering helps identify groups and patterns in network data. Began with K-means as it's conceptually simpler than hierarchical methods. This is a common first step in exploratory data analysis.

**Timestamp:** 2024-07-18

---

### Step 03: Add Visualization Setup
**Commit Message:** Add hierarchical clustering and visualization imports

**What Changed:**
- Added scipy hierarchical clustering imports (dendrogram, linkage)
- Added matplotlib and seaborn for visualization
- Set up structure for distance matrix computation
- Prepared framework for dendrograms and heatmaps
- Added placeholders for visualization functions

**Rationale:** Before implementing the distance matrix, set up all the necessary visualization tools. This allows for incremental testing as features are added.

**Timestamp:** 2024-07-19

---

### Step 04: Implement Distance Matrix Calculation (with bug)
**Commit Message:** Add Euclidean distance matrix computation

**What Changed:**
- Implemented `calculate_distance_matrix()` function
- Used numpy operations for distance calculation
- Reshaped array to (-1, 1) for matrix operations
- **BUG INTRODUCED:** Used `A_reshaped[i] - A_reshaped[j]` without accessing scalar value

**Code snippet with bug:**
```python
A_reshaped = data.reshape(-1, 1)
for i in range(length):
    for j in range(length):
        distance_matrix[i][j] = np.sqrt((A_reshaped[i] - A_reshaped[j]) ** 2)  
        # Bug: A_reshaped[i] is 1D array, not scalar
```

**Rationale:** Extended clustering with distance calculations. Made a common mistake with NumPy array indexing after reshaping—forgot that after reshape(-1, 1), each element becomes a 1D array rather than a scalar.

**Timestamp:** 2024-07-19 (afternoon)

---

### Step 05: Fix Array Indexing Bug in Distance Matrix
**Commit Message:** Fix: Correct array indexing in distance matrix calculation

**What Changed:**
- Fixed the array indexing bug in distance matrix calculation
- Changed `A_reshaped[i] - A_reshaped[j]` to `A[i][0] - A[j][0]`
- Added comments explaining the indexing
- Verified distance matrix calculation produces correct results
- Completed hierarchical clustering integration

**Code snippet fixed:**
```python
A = data.reshape(-1, 1)
for i in range(len(A)):
    for j in range(len(A)):
        distance_matrix[i][j] = np.sqrt((A[i][0] - A[j][0]) ** 2)  
        # Fixed: access scalar value with [0]
```

**Rationale:** Caught and fixed the broadcasting error within hours. After reshaping to (-1, 1), each A[i] is a 1D array containing one element, so we need [0] to access the scalar. This is a typical quick-fix cycle in NumPy development.

**Timestamp:** 2024-07-19 (evening, same day)

---

### Step 06: Implement PC Algorithm Foundations
**Commit Message:** Add PC algorithm basics for causal discovery

**What Changed:**
- Created main2.py with PC (Peter-Clark) algorithm structure
- Added conditional independence testing using partial correlations
- Implemented edge removal logic based on independence tests
- Added data generation (random_pc_data.csv with synthetic data)
- Set up basic skeleton for causal graph construction
- Added networkx imports for graph operations

**Rationale:** Extended analysis capabilities to infer causal relationships from observational data. The PC algorithm is foundational in causal discovery. Started with core logic (independence testing and edge removal) before adding visualization and edge orientation.

**Timestamp:** 2024-07-22

---

### Step 07: Add PC Algorithm Testing Phase
**Commit Message:** Implement comprehensive conditional independence testing

**What Changed:**
- Enhanced independence testing with proper p-value calculations
- Added statistical testing for all variable combinations
- Implemented conditioning set logic for PC algorithm
- Added detailed logging of test results
- Set up framework for tracking removed edges

**Rationale:** PC algorithm requires thorough testing at each step to determine which edges to remove. This phase focuses on getting the statistical testing right before adding visualization.

**Timestamp:** 2024-07-23

---

### Step 08: Complete PC Algorithm with Visualization
**Commit Message:** Add edge orientation and causal graph visualization to PC algorithm

**What Changed:**
- Added edge orientation logic for causal direction
- Implemented v-structure detection
- Added network visualization using networkx and matplotlib
- Completed comprehensive output of causal relationships
- Improved independence test reporting with detailed statistics
- Complete working PC algorithm with visual output

**Rationale:** Completed the causal discovery module with visual output. Visualization is crucial for understanding and validating causal structures. Edge orientation using v-structures is the final key component of the PC algorithm.

**Timestamp:** 2024-07-24

---

### Step 09: Setup Network Controllability Framework
**Commit Message:** Initialize network controllability analysis module

**What Changed:**
- Created main3.py with basic network controllability structure
- Defined network topology (vertices, edges with weights)
- Set up control input configurations
- Added adjacency matrix construction
- Prepared framework for controllability matrix computation
- Added networkx and numpy imports

**Rationale:** Beginning the most complex module—network controllability analysis. Started by defining the network structure and control inputs before implementing controllability tests. This follows a logical progression from data structures to algorithms.

**Timestamp:** 2024-07-26

---

### Step 10: Implement Basic Controllability Analysis
**Commit Message:** Add controllability matrix computation

**What Changed:**
- Implemented controllability matrix calculation: C = [B, AB, A²B, ..., Aⁿ⁻¹B]
- Added matrix power computations using numpy
- Implemented rank calculation for controllability testing
- Added basic controllability status output
- Set up structure for exact controllability tests

**Rationale:** Core controllability theory implementation. The controllability matrix is the fundamental tool for determining if a network can be controlled. This step focuses on the mathematical foundations before adding more advanced tests.

**Timestamp:** 2024-07-27

---

### Step 11: Add Structural Controllability Test (with import bug)
**Commit Message:** Implement structural controllability and colored output

**What Changed:**
- Added structural controllability test (generic controllability)
- Implemented colored terminal output for better readability
- **BUG INTRODUCED:** Typo in import statement: `from colorma import Fore, Style, init`
- Added comprehensive output formatting
- Enhanced controllability status reporting

**Code snippet with bug:**
```python
from colorma import Fore, Style, init  # Bug: missing 'a' in colorama
```

**Rationale:** Extended controllability analysis with structural tests. Added colored output to make results more readable. Made a simple typo in the new import statement—missed an 'a' in 'colorama'.

**Timestamp:** 2024-07-28 (morning)

---

### Step 12: Fix Import Error and Complete Visualization
**Commit Message:** Fix: Correct colorama import typo and add network visualization

**What Changed:**
- Fixed import typo: changed `from colorma import` to `from colorama import`
- Verified colored output works correctly
- Added network topology visualization with networkx
- Implemented control input visualization on network graph
- Complete working controllability analysis with visualization

**Code snippet fixed:**
```python
from colorama import Fore, Style, init  # Fixed: correct module name
```

**Rationale:** Caught the import error immediately when trying to run the script (`ModuleNotFoundError: No module named 'colorma'`). Simple one-character fix. Continued development by adding network visualization to complete this module.

**Timestamp:** 2024-07-28 (afternoon, same day)

---

### Step 13: Implement Minimum Driver Node Finder
**Commit Message:** Add optimization for finding minimum driver nodes

**What Changed:**
- Created main4.py for driver node optimization
- Implemented exhaustive search across all possible control configurations
- Added controllability testing for each configuration
- Set up progress tracking for long-running optimization
- Added minimum set identification and visualization
- Imported necessary functions from main3.py

**Rationale:** Finding the minimum set of driver nodes is NP-hard but crucial for practical network control. This tool helps identify the most efficient control strategy. Warning: computationally intensive for networks with > 7 nodes.

**Timestamp:** 2024-07-30

---

### Step 14: Add Symbolic Mathematics Tools
**Commit Message:** Implement symbolic matrix rank calculator

**What Changed:**
- Created test.py with symbolic computation capabilities
- Added sympy for symbolic linear algebra
- Implemented symbolic matrix operations
- Added rank calculation for symbolic matrices
- Set up matrix transformation and echelon form computations
- Pretty printing for mathematical output

**Rationale:** Some controllability analysis benefits from symbolic rather than numeric computation. This module provides tools for symbolic matrix analysis, useful for theoretical investigations.

**Timestamp:** 2024-08-01

---

### Step 15: Add Jupyter Notebook for Experiments
**Commit Message:** Add interactive notebook for experimentation

**What Changed:**
- Created test.ipynb for interactive analysis
- Added symbolic controllability matrix analysis
- Implemented echelon form demonstrations
- Set up cells for testing different network configurations
- Added visualization cells for quick plots

**Rationale:** Jupyter notebooks are ideal for experimentation and teaching. This notebook allows interactive exploration of controllability concepts and quick testing of different network configurations.

**Timestamp:** 2024-08-03

---

### Step 16: Add Core Dependencies
**Commit Message:** Update requirements.txt with core dependencies

**What Changed:**
- Added numpy, scipy to requirements.txt with version constraints
- Added pandas for data manipulation
- Added scikit-learn for K-means
- Specified minimum versions for reproducibility
- Organized dependencies logically

**Rationale:** As the project has grown, formalizing dependencies becomes important. Started by documenting the core scientific computing dependencies that all modules rely on.

**Timestamp:** 2024-08-05

---

### Step 17: Add Visualization Dependencies (with typo)
**Commit Message:** Add matplotlib, seaborn, and specialized dependencies

**What Changed:**
- Added seaborn for statistical visualizations
- **BUG INTRODUCED:** Typo in package name: `matplotli>=3.4.0` instead of `matplotlib>=3.4.0`
- Added networkx for graph operations
- Added sympy for symbolic math
- Added colorama for colored output
- Added array-to-latex for LaTeX formatting
- Added jupyter for notebooks

**Requirements snippet with bug:**
```
matplotli>=3.4.0  # Bug: missing 'b' in matplotlib
```

**Rationale:** Continued adding dependencies for visualization and specialized features. Made a typo when manually typing 'matplotlib'—missed the 'b'.

**Timestamp:** 2024-08-05 (afternoon)

---

### Step 18: Fix Requirements Typo
**Commit Message:** Fix: Correct matplotlib package name in requirements.txt

**What Changed:**
- Fixed typo in requirements.txt: `matplotli` → `matplotlib`
- Verified all dependencies install correctly with `pip install -r requirements.txt`
- Tested that all scripts can import their dependencies

**Requirements snippet fixed:**
```
matplotlib>=3.4.0  # Fixed: correct package name
```

**Rationale:** Caught the typo immediately when trying to install dependencies (`ERROR: Could not find a version that satisfies the requirement matplotli`). Simple fix—added the missing 'b'. This is a common mistake when manually editing dependency files.

**Timestamp:** 2024-08-05 (same day)

---

### Step 19: Add Documentation (with command errors)
**Commit Message:** Add comprehensive README documentation

**What Changed:**
- Created portfolio-grade README.md with full documentation
- Added overview, features, and technical details
- Documented setup and installation instructions
- **BUG INTRODUCED:** Run commands reference wrong directory: `cd network-analysis && python main.py`
- Added troubleshooting section
- Documented all inputs and outputs
- Added references to research papers

**README snippet with bug:**
```bash
cd network-analysis  # Wrong: this directory doesn't exist
python main.py
```

**Rationale:** Wrote comprehensive documentation to make the project accessible. Made mistakes in the run commands—wrote them assuming a different directory structure than what actually exists. This is common when documenting before final testing.

**Timestamp:** 2024-08-08

---

### Step 20: Fix README Commands
**Commit Message:** Fix: Correct run commands in README to execute from repo root

**What Changed:**
- Fixed all run commands to execute from repository root
- Removed incorrect `cd network-analysis` commands
- Corrected all path references (e.g., `random_pc_data.csv` not `data/random_pc_data.csv`)
- Tested all documented commands to ensure they work
- Verified installation instructions are accurate

**README snippet fixed:**
```bash
# Correct: run from repository root
python main.py
```

**Rationale:** Caught the documentation errors when actually following the README to test setup. Commands now reflect the actual repository structure and all work as documented.

**Timestamp:** 2024-08-09

---

### Step 21: Add Module Docstrings
**Commit Message:** Add comprehensive docstrings to all Python modules

**What Changed:**
- Added module-level docstring to main.py explaining clustering analysis
- Added module-level docstring to main2.py explaining PC algorithm
- Added module-level docstring to main3.py explaining controllability
- Added module-level docstring to main4.py explaining driver node optimization
- Added module-level docstring to test.py explaining symbolic operations
- Improved code documentation for portfolio presentation

**Rationale:** Module docstrings make the code more professional and self-documenting. This is important for portfolio presentation—readers should understand each file's purpose immediately.

**Timestamp:** 2024-08-10

---

### Step 22: Update .gitignore
**Commit Message:** Improve .gitignore to exclude temporary files

**What Changed:**
- Enhanced .gitignore with comprehensive Python exclusions
- Added Jupyter notebook checkpoints
- Added IDE-specific files (.vscode, .idea)
- Added OS-specific files (.DS_Store, Thumbs.db)
- Added Microsoft Office temporary files (~$*, ~WRL*.tmp)
- Organized .gitignore into logical sections

**Rationale:** Clean repository presentation requires excluding all temporary and generated files. This prevents accidentally committing build artifacts, IDE settings, or OS-specific files.

**Timestamp:** 2024-08-11

---

### Step 23: Final Portfolio Polish
**Commit Message:** Final refinements for portfolio presentation

**What Changed:**
- Verified all scripts run successfully
- Confirmed all documentation is accurate
- Tested complete setup process from scratch
- Verified reproducibility (dependencies, commands, outputs)
- Ensured professional presentation throughout
- Final review and validation

**Rationale:** Final quality check before considering the project portfolio-ready. Tested everything end-to-end to ensure smooth experience for anyone reviewing the code.

**Timestamp:** 2024-08-12

---

## Summary

**Timeline:** July 15 - August 12, 2024 (approximately 4 weeks)

**Development Progression:**
1. **Week 1 (July 15-19):** Repository setup, clustering analysis with bug fix cycle
2. **Week 2 (July 22-24):** Causal discovery (PC algorithm) implementation
3. **Week 3 (July 26-30):** Network controllability analysis with import bug fix
4. **Week 4 (August 1-5):** Optimization, symbolic tools, and dependency management with typo fix
5. **Week 5 (August 8-12):** Documentation, refinement, and portfolio polish with command fixes

**Total Steps:** 23 commits showing realistic incremental development

**Bug/Fix Cycles:** 4 oops→hotfix sequences demonstrating authentic debugging:
1. NumPy array indexing (steps 04→05)
2. Import typo (steps 11→12)
3. Requirements.txt typo (steps 17→18)
4. README command errors (steps 19→20)

**Final State:** Professional, portfolio-ready network analysis framework with comprehensive documentation, all tests passing, and realistic development history.
