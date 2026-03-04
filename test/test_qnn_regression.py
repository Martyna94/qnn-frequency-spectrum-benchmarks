import unittest
from itertools import product

import numpy as np
from parameterized import parameterized

from experiments.constants import HAMMING, EXPONENTIAL, BINARY, TERNARY
from qnn import QNN
from qnn.constants import MSE, GOLOMB, SEQUENTIAL, PARALLEL, TURNPIKE
from qnn.fourier import RealFourierSeries
from test.utils import timeit


class TestFrequencySpectrumOfTrainedUnivariateQNN(unittest.TestCase):
    """
    Here we test the training of a QNN model and the computation of its frequency spectrum. As data, we use
    a small Fourier series with 5 coefficients. We then train the QNN model to predict this series and calculate the
    first 100 Fourier coefficients of the model. We compare the computed coefficients with the original ones.
    """

    @parameterized.expand([
        ("hamming", 3),
        ("exponential", 1),
        ("binary", 1),
        ("ternary", 1),
    ])
    @timeit
    def test_trained_qnn_frequency_spectrum(self, encoding: str, decimal: int):
        coeffs = [1 / 30 + k * 1j / 30 for k in range(5)]
        f = RealFourierSeries(coeffs)
        x = np.linspace(0, 2 * np.pi, 1000)
        y = f(x)
        for R in (1, 2, 3, 6):
            for ansatz in (SEQUENTIAL, PARALLEL):
                L = 6 // R
                print(f"Test {encoding=} with {(R, L)=}, {ansatz=}")
                qnn = QNN(R=R,
                          L=L,
                          N=1,
                          ansatz=ansatz,
                          encoding=encoding,
                          loss_fn=MSE,
                          trainable_block_layers=5,
                          save_weights=False,
                          seed=13,
                          max_iter=200,
                          batch_size=128,
                          step_size=0.01,
                          verbose=False)
                qnn.fit(x.reshape(-1, 1), y)
                N = 100
                fourier_coeffs = qnn.fourier_coefficients(N)
                np.testing.assert_array_almost_equal(fourier_coeffs, coeffs + [0] * (N - len(coeffs) + 1),
                                                     decimal=decimal)

    @parameterized.expand([
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
    ])
    @timeit
    def test_trained_qnn_frequency_spectrum_golomb(self, R: int, L: int):
        coeffs = [1 / 30 + k * 1j / 30 for k in range(5)]
        f = RealFourierSeries(coeffs)
        x = np.linspace(0, 2 * np.pi, 1000)
        y = f(x)
        for ansatz in (SEQUENTIAL, PARALLEL):
            print(f"Test golomb with {(R, L)=}, {ansatz=}")
            qnn = QNN(R=R,
                      L=L,
                      N=1,
                      ansatz=ansatz,
                      encoding=GOLOMB,
                      loss_fn=MSE,
                      trainable_block_layers=5,
                      save_weights=False,
                      seed=13,
                      s=[1, 1, -1, -1],
                      max_iter=100,
                      step_size=0.01,
                      verbose=False)
            qnn.fit(x.reshape(-1, 1), y)
            N = 100
            fourier_coeffs = qnn.fourier_coefficients(N)
            np.testing.assert_array_almost_equal(fourier_coeffs, coeffs + [0] * (N - len(coeffs) + 1), decimal=1)

    @parameterized.expand([
        (4, 1),
        (4, 2),
        (8, 1),
        (8, 2),
    ])
    @timeit
    def test_trained_qnn_frequency_spectrum_turnpike(self, R: int, L: int):
        coeffs = [1 / 30 + k * 1j / 30 for k in range(5)]
        f = RealFourierSeries(coeffs)
        x = np.linspace(0, 2 * np.pi, 1000)
        y = f(x)
        for ansatz in (SEQUENTIAL, PARALLEL):
            print(f"Test turnpike with {(R, L)=}, {ansatz=}")
            qnn = QNN(R=R,
                      L=L,
                      N=1,
                      ansatz=SEQUENTIAL,
                      encoding=TURNPIKE,
                      loss_fn=MSE,
                      trainable_block_layers=5,
                      save_weights=False,
                      seed=13,
                      s=[1, 1, -1, -1],
                      max_iter=100,
                      step_size=0.01,
                      verbose=False)
            qnn.fit(x.reshape(-1, 1), y)
            N = 100
            fourier_coeffs = qnn.fourier_coefficients(N)
            np.testing.assert_array_almost_equal(fourier_coeffs, coeffs + [0] * (N - len(coeffs) + 1), decimal=1)


class TestFrequencySpectrumOfTrainedMultivariateQNN(unittest.TestCase):

    @parameterized.expand(list(product([HAMMING, EXPONENTIAL,BINARY, TERNARY],
                                       [SEQUENTIAL, PARALLEL]
                                       )
                               )
                          )
    @timeit
    def test_trained_qnn_sequential(self, encoding: str, ansatz: str):
        x = np.array(list(product(np.linspace(0, 2 * np.pi, 30), repeat=2)))

        def f(row):
            xm, ym = row
            return (1
                    + (2 + 3j) * np.exp(1j * xm)
                    + (2 - 3j) * np.exp(-1j * xm)
                    + (1 + 3j) * np.exp(1j * ym)
                    + (1 - 3j) * np.exp(-1j * ym)
                    + (1 + 2j) * np.exp(1j * (xm + ym))
                    + (1 - 2j) * np.exp(-1j * (xm + ym))
                    ).real / 30

        y = np.apply_along_axis(f, 1, x)
        for R in (1,):
            L = 4 // R
            print(f"Test {encoding=} with {(R, L)=} and {ansatz=}")
            qnn = QNN(R=R,
                      L=L,
                      N=2,
                      ansatz=ansatz,
                      encoding=encoding,
                      trainable_block_layers=5,
                      save_weights=False,
                      seed=13,
                      max_iter=100,
                      step_size=0.01,
                      batch_size=64,
                      verbose=False)
            qnn.fit(x, y)
            loss = qnn.loss_score(x, y)
            self.assertLess(loss, 10 ** (1 - R))  # This choice of tolerance is arbitrary


if __name__ == '__main__':
    unittest.main()
