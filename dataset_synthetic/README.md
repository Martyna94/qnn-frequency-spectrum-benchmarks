# Synthetic dataset

This folder contains tools, examples, and sample data used to generate synthetic datasets for experiments.

Structure
- `generate_data.ipynb` — notebook that creates datasets with `experiments.DatasetManager`, saves JSON, and plots results.
- `dataset_12degree_100sample.json` — example generated dataset (degree 12, 100 samples).


Usage

1. Open `generate_data.ipynb` and run the cells in order. The notebook:
   - Creates a dataset with `degree=12`, `num_sample=100`, `x` in `[0, 2π]` and `x_test` in `[-2π, 0]`.
   - Applies normalization strategies: `L2_NORMALIZATION`, `MIN_MAX_NORMALIZATION`, and `HALF_RANGE_NORMALIZATION`.
   - Saves `dataset_12degree_100sample.json`.
   - Visualizes train/test functions and writes `Normalization.png`.
   - Plots amplitude spectra with `compute_dft_and_plot_amplitude_spectra`.

Notes
- The notebook includes a second, commented example that uses `degree=5` and 1000 samples; uncomment if needed.
- Commit generated figures or large datasets only if they are small; prefer storing large artifacts in external storage.

Contact
-
If you have questions about the synthetic data format, open an issue or contact the repository maintainers.
