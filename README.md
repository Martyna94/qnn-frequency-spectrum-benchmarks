# Womanium

## EniQmA

[Project homepage](https://eniqma-quantum.de)

This repository contains code and experiments for the EniQmA project around the *frequency spectrum* of quantum neural networks (QNNs). The main goals are:

1. Benchmark different QNN ansatze (German: Ansaetze/Ansätze) and data encodings (building on our theory paper) and turn the results into an experiments paper.
2. Build and evaluate (hybrid) QML models on two "real data" datasets:
   - A dataset delivered by the project partner [umlaut](https://www.accenture.com/us-en/services/industry-x/umlaut) based on a [fischertechnik factory](https://www.fischertechnik.de/de-de/produkte/industrie-und-hochschulen/simulationsmodelle/536634-fabrik-simulation-24v). It contains time series measurements of electric current from a pneumatic component, including runs with an induced leak; the task is leak detection.
   - The [NASA Bearing Dataset](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset).

## Quickstart (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run commands from the repository root so `qnn/` and `experiments/` are importable without installing the project as a package.

Run the unit tests:

```powershell
python -m unittest discover -s test -p "test_*.py"
```

## The `qnn` package and existing code
The repository includes a small `qnn` package with a scikit-learn-like `QNN` class (fit/predict) implemented with JAX + PennyLane (`default.qubit`).

Encodings currently implemented in code:

- Hamming
- Binary
- Ternary
- Exponential
- Golomb
- Turnpike

Example usage lives in `demos/` (scripts + notebooks).

## Repository structure (top-level)
Short "what is in this folder" map based on the current repository contents:

- `qnn/`: core QNN implementation (model, encodings, Fourier utilities).
- `experiments/`: shared utilities used by notebooks/scripts (dataset management, Fourier series helpers, metrics tracking, plotting).
- `demos/`: runnable demos and notebooks (e.g. `qnn_demo.py`, `qnn_mnist_demo.py`, `qnn_experiments.ipynb`).
- `dataset_NASA/`: NASA bearing dataset workbench (preprocessing notebook, quantum model scripts/notebooks, grid search script, plus `results/` and `figures/` artifacts).
- `data_Umlaut/`: umlaut/fischertechnik leak dataset workbench (exploratory notebook, quantum model scripts, grid search, plus `dataset/` archive and `results/`/`figures/` artifacts).
- `data_fabrik/`: factory dataset archive (`cleaned.zip`) as delivered/cleaned for experiments.
- `dataset_synthetic/`: synthetic dataset generator (`generate_data.ipynb`) plus sample JSON and plotting notebooks.
- `test/`: unit tests (`unittest` + `parameterized`).
- `requirements.txt`: Python dependencies for local runs/CI.


## Background QNNs

Our main paper [Spectral invariance and maximality properties of the frequency spectrum of 
quantum neural networks](https://arxiv.org/abs/2402.14515). 
It covers the theory of the frequency spectrum of QNNs. It is mainly theoretical. We want to extend the results with a second 
paper where we benchmark different QNN ansatze and possibly compare them against classical NNs.

## Relevant Papers

Here is a list of papers that could be relevant or source of inspiration for benchmarks.

* [Effect of data encoding on the expressive power of variational quantum-machine-learning models](https://arxiv.org/abs/2008.08605)
* [An exponentially-growing family of universal quantum circuits](https://arxiv.org/pdf/2212.00736)
* [Learning capability of parametrized quantum circuits](https://arxiv.org/pdf/2209.10345) (interesting benchmarking)
* [Generalization despite overfitting in quantum machine learning models ](https://arxiv.org/pdf/2209.05523)
* [Fourier expansion in variational quantum algorithms](https://arxiv.org/pdf/2304.03787)
* [Multidimensional Fourier series with quantum circuits](https://arxiv.org/pdf/2302.03389)
* [Exponential data encoding for quantum supervised learning](https://arxiv.org/pdf/2206.12105)

Here is a link to all Papers on [semantic scholar](https://www.semanticscholar.org/shared/library/folder/10660864?utm_source=direct_link).

## Possible Research questions:
* How to benchmark QNNs against classical NNs such that the result is comparable? (Compare size of the network, number of parameters, number of layers, etc.)
* What Fourier coefficients could be reached for each ansatz? We guess that richer frequency spectrums lead to less freedom in the Fourier coefficients.
* How to measure time complexity and convergence of QNNs and compare them fairly against NNs?
* Do approximation quality and expressiveness depend mainly on the area $A = R\\cdot L$? This is important for choosing an ansatz in practice.
* How do approximation quality and expressiveness depend on the parameter encoding layers $W$? What can be said about dependency on number of parameters and circuit choice for parameter encoding?
* ...
