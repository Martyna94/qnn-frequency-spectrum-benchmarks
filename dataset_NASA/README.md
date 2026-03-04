# NASA Bearing Dataset (Kaggle)

This folder contains notebooks and scripts to (1) prepare the NASA bearing dataset, (2) run QNN experiments on it, and (3) summarize/plot experiment results.

## Where the data lives

- `dataset/`: put the raw Kaggle dataset folder here (not committed). The notebook `nasa_data_generation.ipynb` expects a local path to the downloaded Kaggle data.
- `results/`: CSV/JSON artifacts from grid searches and experiment runs.
- `figures/`: generated plots used in reports/papers.

## File guide (what each file is for)

- `nasa_data_generation.ipynb`: creates a "combined" dataset from the raw Kaggle files (writes `combined_dataset2.csv`) and computes an RMS time series (`df_rms` in-memory) that is used by the QNN notebooks/scripts.
- `Exploratory_and_Data_Analysis_CLassical_Model.ipynb`: exploratory analysis of the NASA data and a simple classical NN baseline; reads `combined_dataset2.csv`.
- `Quantum_Model.ipynb`: end-to-end example of training/evaluating a QNN on the NASA RMS features; reads `df_rms.csv` and writes a model-run JSON (e.g. `test_qnn_output.json`).
- `Quantum_Model.py`: script version of the QNN training/evaluation run; reads `df_rms.csv` and appends model configuration + training losses to a JSON (default `test.json`) and writes metrics to a CSV (default `test.csv`).
- `QuantumModelGridSearch.py`: runs repeated experiments across a list of configurations (R/L/layers/iters/step size, plus encoding and batch size); appends detailed run data to JSON (default `exp.json`) and writes aggregated CSVs (default `exp_all.csv`, `exp_best_metrics_.csv`).
- `Quantum_Plots.ipynb`: turns `results/exp.json` and/or `results/exp_all.csv` into summary tables (e.g. `qnn_experiments_summary.csv`) and generates plots saved under `figures/`.
- `Utils.py`: helper functions for windowing (`create_X_Y_set`), splitting/scaling + SMOTE balancing (`create_train_and_test_set`), JSON append/load helpers, and plotting utilities.

## Notes / known quirks

- Some scripts currently import `data_NASA.Utils` (legacy folder name). In this repository the file is `dataset_NASA/Utils.py`, so you may need to adjust the import before running the scripts.
- `df_rms.csv` is referenced by `Quantum_Model.ipynb` and the Python scripts. If you do not have it yet, you can generate the RMS dataframe in `nasa_data_generation.ipynb` and export it to CSV.
