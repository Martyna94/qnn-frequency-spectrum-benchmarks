import json
import numpy as np
import jax.numpy as jnp

import experiments.constants as const_exp
from experiments.FourierSeries import RealFourierSeries


class DatasetManager:
    def __init__(self, x, x_test, num_sample, degree, normalization_strategies: list, json_file_path=None):
        self.x = x
        self.x_test = x_test
        self.num_sample = num_sample
        self.degree = degree
        self.normalization_strategies = normalization_strategies
        self.json_file_path = json_file_path

        # Determine whether to load from file or create a new dataset
        if self.json_file_path:
            self.dataset = self.load_dataset_from_json(self.json_file_path)
        else:
            self.dataset = self.create_and_evaluate_series()

    @staticmethod
    def generate_random_coeffs(degree, scaling=1):
        """Generate random complex coefficients for Fourier series."""
        real_parts = scaling * np.random.randn(degree)
        imag_parts = np.random.randn(degree)
        c_0 = np.random.uniform(-0.7, 0.7)
        return [c_0] + list(real_parts + 1j * imag_parts)

    @staticmethod
    def evaluate_series(series, x):
        """Evaluate Fourier series over a dataset."""
        return {name: f(x) for name, f in series.items()}

    def create_fourier_series(self, coeffs):
        """Create a dictionary of Fourier series based on different normalization strategies."""
        return {name: RealFourierSeries(coeffs, normalization_strategy=name) for name in self.normalization_strategies}

    def create_and_evaluate_series(self):
        """Create and evaluate Fourier series multiple times based on the given degree and strategies."""
        dataset = {
            name: {'fourier_series': [], 'y_series': [], 'y_series_test': []} for name in self.normalization_strategies
        }

        for idx in range(self.num_sample):
            coeffs = self.generate_random_coeffs(self.degree)

            # Create and evaluate Fourier series
            fourier_series = self.create_fourier_series(list(coeffs))
            y_series = self.evaluate_series(fourier_series, self.x)

            if self.x_test is not None:
                y_series_test = self.evaluate_series(fourier_series, self.x_test)

            # Append the results to the dataset
            for name, series in fourier_series.items():
                dataset[name]['fourier_series'].append(series)
                dataset[name]['y_series'].append(y_series[name])  # Convert JAX/NumPy array to list
                if self.x_test is not None:
                    dataset[name]['y_series_test'].append(y_series_test[name])  # Convert JAX/NumPy array to list

        return dataset

    def get_series(self, normalization):
        """Retrieve the training and test series for a given normalization."""
        return self.dataset[normalization][const_exp.Y_SERIES], self.dataset[normalization][const_exp.Y_SERIES_TEST]

    def get_fourier_series(self,normalization):
        return self.dataset[normalization][const_exp.FOURIER_SERIES]

    def load_dataset_from_json(self, file_path):
        """Load dataset from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract x and x_test from the file
        self.x = jnp.array(data['x'])
        self.x_test = jnp.array(data['x_test'])
        self.degree = int(data['degree'])
        self.num_sample = int(data['num_sample'])
        self.normalization_strategies = list(data['dataset'].keys())
        # Load dataset
        dataset = {}
        for normalization, item in data['dataset'].items():
            dataset[normalization] = {
                const_exp.FOURIER_SERIES: [RealFourierSeries([complex(c[const_exp.REAL], c[const_exp.IMAG]) for c in coeffs], normalization) for coeffs in item[const_exp.FOURIER_SERIES]],
                const_exp.Y_SERIES: [jnp.array(y) for y in item[const_exp.Y_SERIES]],
                const_exp.Y_SERIES_TEST: [jnp.array(y) for y in item[const_exp.Y_SERIES_TEST]]
            }

        return dataset


    def convert_complex(self, value):
        """Convert complex numbers to a JSON-serializable format."""
        if isinstance(value, complex):
            return {const_exp.REAL: value.real, const_exp.IMAG: value.imag}
        return value

    def save_dataset_to_json(self, filename):
        """Save the dataset to a JSON file, handling complex numbers and storing x and x_test."""
        data_to_save = {
            'x': self.x.tolist(),
            'x_test': self.x_test.tolist(),  # Convert JAX/NumPy array to list
            'degree': self.degree,
            'num_sample': self.num_sample,
            'dataset': {}
        }

        # Save the dataset (including Fourier series, y_series, y_series_test)
        for normalization, item in self.dataset.items():
            data_to_save['dataset'][normalization] = {
                const_exp.FOURIER_SERIES: [
                    [self.convert_complex(c) for c in
                     fourier.coefficients.tolist()[fourier.size:]] for
                    fourier in item[const_exp.FOURIER_SERIES]
                ],
                const_exp.Y_SERIES: [jnp.array(y).tolist()
                                     for y in item[const_exp.Y_SERIES]],  # Convert JAX/NumPy array to list
                const_exp.Y_SERIES_TEST: [jnp.array(y).tolist() for y in item[const_exp.Y_SERIES_TEST]]  # Convert JAX/NumPy array to list
            }

        with open(filename, 'w') as file:
            json.dump(data_to_save, file, indent=4)