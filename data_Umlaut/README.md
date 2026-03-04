# Umlaut / Fischertechnik Leak Dataset

This folder contains the notebooks and scripts used to prepare, analyze, and model the umlaut/fischertechnik factory dataset (time series current/power with leak labels), and to store the resulting plots and experiment summaries.

## Where the data and artifacts live

- `dataset/`: data artifacts used for training. This repo currently includes `segments_output.zip`; extract it to `dataset/segments_output/` so the training scripts can load the per-segment CSV files.
- `results/`: experiment outputs and summary spreadsheets/CSVs (best-metrics exports, etc.).
- `figures/`: generated plots used in reports/papers.

## File guide (what each file is for)

- `Exploratory_Analysis_Data.ipynb`: end-to-end exploratory analysis and data preparation (visualization, cleaning, segmentation). It also contains code to generate the final segment CSVs used for training.
- `Quantum_Model.py`: runs a single QNN training/evaluation on the prepared segment data (windowing + split + train + metrics). It writes a JSON with model configuration/loss history (default `test.json`) and appends metrics to a text file (default `test.txt`).
- `Quantum_Model_GridSearch.py`: runs a small grid search over QNN configurations and writes per-run and best-run metrics to CSV (filenames are set at the top of the script) and appends detailed run data to a JSON.
- `Quantum_Plots.ipynb`: plotting/figure notebook for the umlaut dataset and QNN experiments (e.g. loss comparisons, strongly-entangling-layer visualization), saving PNGs under `figures/`.
- `Utils.py`: shared helpers for loading the raw run CSVs, segmenting the time series into training windows, smoothing/denoising helpers, train/test split + scaling, JSON append/load helpers, and plotting utilities.

## Notes / known quirks

- The scripts assume that segmented CSVs exist as a directory of files named like `0.csv`, `1.csv`, ... (after extracting `segments_output.zip`). If you generate segments in the exploratory notebook, consider writing them into `dataset/segments_output/` to match `Quantum_Model_GridSearch.py`.
