Codes for paper **GETS: Nodewise Ensemble Calibration for Graph Neural Networks**

ICLR 2025 submission number: 2271


### Graph ensemble calibration
This repository includes the implementation of uncertainty calibration methods for graph neural networks based on ensemble strategies. Both input ensemble and model ensemble are included. The goal is to improve the calibration of a classifier's predicted class probabilities; well-calibrated predictions are those that match the true class probabilities. 

### Installation

Python version: `3.12.5`

```Console
$ conda create --name GETS python=3.12.5
$ conda activate GETS
```

To install all the required packages, kindly run
```Console
$ chmod +x install.sh
$ ./install.sh
```

### Run codes

We store parameters in `./config` folder to run each dataset. 

To run a simple test model, just:
```python
python main.py --dataset=cora --gpu=0 --n_runs=10
```

To run all the methods and all codes with logs stored in `./log`, results stored in `./output`:
```Console
$ ./run_all.sh
```


### Structure of codes

GETS/
- **dataset/**: Dataset processing module
  - `dataset.py`: Script for loading and processing datasets.
  
- **exp/**: Experiment management and solvers
  - `expManager.py`: Manages experiment setup and execution.
  - `solver.py`: Handles optimization or solver logic for training models.
  
- **model/**: Model implementations
  - `calibrator.py`: Implements model calibration methods.
  - `gnns.py`: Graph Neural Networks model definitions.
  - `GETS.py`: Our method based on Mixture of Experts model.
  
- **utils/**: Utility functions for logging and tracking
  - `logger.py`: Manages logging of project execution.
  - `recorder.py`: Tracks and records experiment metrics.
  - `utils.py`: Miscellaneous helper functions.
  
- **README.md**: Project documentation and usage instructions.
  
- **automl.py**: Automated Machine Learning script for optimizing models.
  
- **install.sh**: Installation script for setting up the environment.

- **main.py**: Main entry point for running the project.

- **visualize.py**: Script for visualizing results or data.

### Results table

Current results: (ECE(1e-2); Most of them achieve SOTA; Note that results may vary in different machines)
| **Dataset**        | **Classifier** | **Uncal**               | **TS**                 | **VS**                 | **ETS**                | **CaGCN**              | **GATS**               | **GETS**               |
|--------------------|----------------|-------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| **Citeseer**        | GCN            | 20.51 ± 2.94            | 3.10 ± 0.34            | 2.82 ± 0.56            | 3.06 ± 0.33            | 2.79 ± 0.37            | 3.36 ± 0.68            | **2.50 ± 1.42**        |
|                    | GAT            | 16.06 ± 0.62            | 2.18 ± 0.20            | 2.89 ± 0.28            | 2.38 ± 0.25            | 2.49 ± 0.47            | 2.50 ± 0.36            | **1.98 ± 0.30**        |
|                    | GIN            | 4.12 ± 2.83             | 2.39 ± 0.32            | 2.55 ± 0.34            | 2.58 ± 0.34            | 2.96 ± 1.49            | 2.03 ± 0.23            | **1.86 ± 0.22**        |
| **Computers**       | GCN            | 5.63 ± 1.06             | 3.29 ± 0.48            | 2.80 ± 0.29            | 3.45 ± 0.56            | **1.90 ± 0.42**        | 3.28 ± 0.52            | 2.03 ± 0.35            |
|                    | GAT            | 6.71 ± 1.57             | 1.83 ± 0.27            | 2.19 ± 0.15            | 1.89 ± 0.24            | 1.88 ± 0.36            | 1.89 ± 0.34            | **1.77 ± 0.27**        |
|                    | GIN            | 4.00 ± 2.84             | 3.92 ± 2.59            | 2.57 ± 1.06            | 2.91 ± 2.07            | **2.37 ± 1.87**        | 3.34 ± 2.08            | 3.14 ± 3.70            |
| **Cora**            | GCN            | 22.62 ± 0.84            | 2.31 ± 0.53            | 2.43 ± 0.41            | 2.56 ± 0.47            | 3.22 ± 0.81            | 2.60 ± 0.76            | **2.29 ± 0.52**        |
|                    | GAT            | 16.70 ± 0.75            | 1.82 ± 0.39            | 1.71 ± 0.28            | **1.67 ± 0.33**        | 2.85 ± 0.66            | 2.58 ± 0.71            | 2.00 ± 0.48            |
|                    | GIN            | 3.59 ± 0.65             | 2.51 ± 0.33            | **2.20 ± 0.38**        | 2.29 ± 0.30            | 2.79 ± 0.26            | 2.64 ± 0.39            | 2.63 ± 0.87            |
| **Cora-full**       | GCN            | 27.73 ± 0.22            | 4.43 ± 0.10            | 3.35 ± 0.22            | 4.34 ± 0.09            | 4.08 ± 0.25            | 4.46 ± 0.17            | **3.32 ± 1.24**        |
|                    | GAT            | 37.51 ± 0.22            | 2.27 ± 0.33            | 3.55 ± 0.13            | **1.37 ± 0.25**        | 3.96 ± 1.15            | 2.47 ± 0.35            | 1.52 ± 2.27            |
|                    | GIN            | 10.69 ± 3.55            | 3.33 ± 0.22            | 2.43 ± 0.19            | 2.18 ± 0.16            | 4.74 ± 0.66            | 3.98 ± 0.35            | **1.95 ± 0.37**        |
| **CS**              | GCN            | 1.50 ± 0.10             | 1.49 ± 0.07            | 1.39 ± 0.10            | 1.44 ± 0.08            | 2.00 ± 0.98            | 1.51 ± 0.10            | **1.34 ± 0.10**        |
|                    | GAT            | 4.30 ± 0.42             | 2.98 ± 0.15            | 3.02 ± 0.18            | 2.98 ± 0.15            | 2.57 ± 0.38            | 2.56 ± 0.15            | **1.05 ± 0.27**        |
|                    | GIN            | 4.36 ± 0.31             | 4.31 ± 0.31            | 4.28 ± 0.28            | 3.04 ± 1.19            | 1.81 ± 1.09            | 1.24 ± 0.45            | **1.15 ± 0.32**        |
| **Ogbn-arxiv**      | GCN            | 9.90 ± 0.77             | 9.17 ± 1.10            | 9.29 ± 0.88            | 9.60 ± 0.92            | 2.32 ± 0.33            | 2.97 ± 0.34            | **1.85 ± 0.22**        |
|                    | GAT            | 6.90 ± 0.47             | 3.56 ± 0.10            | 4.39 ± 0.08            | 3.55 ± 0.10            | 3.52 ± 0.13            | 3.90 ± 0.77            | **2.34 ± 0.12**        |
|                    | GIN            | 11.56 ± 0.27            | 11.35 ± 0.27           | 11.26 ± 0.43           | 11.38 ± 0.40           | 6.01 ± 0.28            | 6.41 ± 0.29            | **3.33 ± 0.36**        |
| **Photo**           | GCN            | 4.05 ± 0.42             | 2.14 ± 0.46            | 2.11 ± 0.23            | 2.21 ± 0.34            | 2.42 ± 0.76            | 2.24 ± 0.30            | **2.07 ± 0.31**        |
|                    | GAT            | 4.97 ± 0.75             | 2.06 ± 0.57            | 1.71 ± 0.23            | 2.57 ± 0.59            | 1.69 ± 0.14            | 2.05 ± 0.45            | **1.10 ± 0.25**        |
|                    | GIN            | 3.37 ± 2.40             | 3.12 ± 1.85            | 3.77 ± 1.10            | 3.34 ± 1.70            | 2.42 ± 1.90            | 3.37 ± 1.85            | **1.43 ± 0.71**        |
| **Physics**         | GCN            | 0.99 ± 0.10             | 0.96 ± 0.05            | 0.96 ± 0.05            | 0.87 ± 0.05            | 1.34 ± 0.45            | 0.91 ± 0.04            | **0.87 ± 0.09**        |
|                    | GAT            | 1.52 ± 0.29             | 0.37 ± 0.10            | 0.44 ± 0.05            | 0.47 ± 0.22            | 0.69 ± 0.11            | 0.48 ± 0.23            | **0.29 ± 0.05**        |
|                    | GIN            | 2.11 ± 0.14             | 2.08 ± 0.14            | 2.08 ± 0.14            | 1.64 ± 0.09            | 2.36 ± 0.52            | 0.90 ± 0.66            | **0.44 ± 0.10**        |
| **Pubmed**          | GCN            | 13.26 ± 1.20            | 2.45 ± 0.30            | 2.34 ± 0.27            | 2.05 ± 0.26            | 1.93 ± 0.36            | 2.19 ± 0.27            | **1.90 ± 0.40**        |
|                    | GAT            | 9.84 ± 0.16             | 0.80 ± 0.07            | 0.95 ± 0.08            | 0.81 ± 0.09            | 1.04 ± 0.08            | 0.81 ± 0.07            | **0.78 ± 0.15**        |
|                    | GIN            | 1.43 ± 0.15             | 1.01 ± 0.06            | 1.04 ± 0.06            | 1.00 ± 0.04            | 1.40 ± 0.47            | 1.00 ± 0.06            | **0.92 ± 0.13**        |
| **Reddit**          | GCN            | 6.97 ± 0.10             | 1.72 ± 0.07            | 2.02 ± 0.07            | 1.72 ± 0.07            | 1.50 ± 0.08            | OOM                    | **1.49 ± 0.07**        |
|                    | GAT            | 4.75 ± 0.15             | 3.26 ± 0.08            | 3.43 ± 0.10            | 3.52 ± 0.10            | 0.81 ± 0.09            | OOM                    | **0.62 ± 0.08**        |
|                    | GIN            | 3.22 ± 0.09             | 3.17 ± 0.13            | 3.19 ± 0.10            | 3.25 ± 0.14            | 1.63 ± 0.21            | OOM                    | **1.57 ± 0.12**        |