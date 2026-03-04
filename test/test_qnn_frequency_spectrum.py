import unittest

import numpy as np
from parameterized import parameterized

from experiments.constants import HAMMING, EXPONENTIAL, BINARY, TERNARY
from qnn import QNN
from qnn.constants import SEQUENTIAL, PARALLEL, GOLOMB, TURNPIKE
from test.utils import timeit


class TestFrequencySpectrumOfRandomUnivariateQNN(unittest.TestCase):
    """
    Here we test the frequency spectrum of a QNN with randomly initialized weights. Depending on the encoding, we check
    if the frequency range of the model is as expected. Since the frequency spectrum is stochastic, we run the test
    multiple times until the expected frequency range is measured.
    """

    @parameterized.expand([
        ("hamming", 2, 1),
        ("hamming", 3, 2),
        ("hamming", 3, 3),
        ("hamming", 4, 6),
        ("exponential", 1, 2),
        ("exponential", 2, 1),
        ("exponential", 2, 2),
        ("exponential", 2, 3),
        ("exponential", 3, 1),
        ("exponential", 3, 2),
        ("exponential", 3, 3),
        ("exponential", 4, 2),
        ("binary", 1, 2),
        ("binary", 2, 1),
        ("binary", 2, 2),
        ("binary", 2, 3),
        ("binary", 3, 1),
        ("binary", 3, 2),
        ("binary", 3, 3),
        ("binary", 4, 2),
        ("ternary", 1, 2),
        ("ternary", 2, 1),
        ("ternary", 2, 2),
        ("ternary", 2, 3),
        ("ternary", 3, 1),
        ("ternary", 3, 2),
        ("ternary", 3, 3),
        ("ternary", 4, 2),
    ])
    @timeit
    def test_random_qnn_frequency_spectrum(self, encoding: str, R: int, L: int):
        if encoding == HAMMING:
            expected = R * L
        elif encoding == EXPONENTIAL:
            expected = 2 ** (R * L)
        elif encoding == BINARY:
            expected = 2 ** (R * L) - 1
        elif encoding == TERNARY:
            expected = (3 ** (R * L) - 1) // 2
        for ansatz in (SEQUENTIAL, PARALLEL):
            print(f"Test {encoding=} with {(R, L)=}, {ansatz=}, expected frequency range = {expected}")
            x = np.linspace(0, 2 * np.pi, 10)  # Dummy data
            y = np.linspace(0, 2 * np.pi, 10)  # Dummy data
            max_frequency = 0
            for idx_round in range(10):  # Run multiple times to get the maximum frequency since it is stochastic
                qnn = QNN(R=R,
                          L=L,
                          N=1,
                          ansatz=ansatz,
                          encoding=encoding,
                          trainable_block_layers=5,
                          seed=13 + idx_round,
                          max_iter=0,  # No training
                          verbose=False)
                qnn.fit(x.reshape(-1, 1), y)  # Just to initialize weights
                n_coeffs_computed = expected + 10
                fourier_coeffs = qnn.fourier_coefficients(n_coeffs_computed - 1)
                max_frequency = max(np.argmax(np.isclose(np.absolute(fourier_coeffs), 0)) - 1, max_frequency)
                print(f"Round {idx_round}: {max_frequency=}")
                if max_frequency == expected:
                    break  # Stop if the expected frequency was measured once
            self.assertEqual(max_frequency, expected)

    @parameterized.expand([(2, 1, [0, 1, 4, 6], 1000),
                           (2, 2, [0, 1, 4, 6], 1000),
                           (4, 1, [0, 1, 4, 6], 1000),
                           (4, 2, [0, 1, 4, 6], 15_000),
                           (6, 1, [0, 1, 4, 6], 10_000),
                           (8, 1, [0, 1, 4, 6], 15_000),
                           (3, 1, [0, 1, 4, 9, 15, 22, 32, 34], 1000),
                           (3, 2, [0, 1, 4, 9, 15, 22, 32, 34], 10_000),
                           (6, 1, [0, 1, 4, 9, 15, 22, 32, 34], 10_000),
                           ])
    @timeit
    def test_random_qnn_frequency_spectrum_golomb(self, R: int, L: int, s: list[int], n_coeffs_computed: int):
        q = round(np.log2(len(s)))
        expected = ((4 ** q - 2 ** q + 1) ** (R * L // q) - 1) // 2 + 1
        x = np.linspace(0, 2 * np.pi, 10)  # Dummy data
        y = np.linspace(0, 2 * np.pi, 10)  # Dummy data
        for ansatz in ("sequential", "parallel"):
            print(f"Test golomb with {(R, L)=}, {s=}, {ansatz=} expected frequency count = {expected}")
            for idx_round in range(10):  # Run multiple times to get the maximum frequency since it is stochastic
                qnn = QNN(R=R,
                          L=L,
                          N=1,
                          ansatz=ansatz,
                          encoding=GOLOMB,
                          trainable_block_layers=5,
                          seed=13 + idx_round,
                          max_iter=0,  # No training
                          s=s,
                          verbose=False)
                qnn.fit(x.reshape(-1, 1), y)  # Just to initialize weights
                print(f"{n_coeffs_computed=}")
                fourier_coeffs = qnn.fourier_coefficients(n_coeffs_computed - 1)
                count = np.sum(1 - np.isclose(np.absolute(fourier_coeffs), 0))
                print(f"Round {idx_round}: {count=}")
                if count == expected:
                    break  # Stop if the expected frequency was measured once
            self.assertEqual(count, expected)

    @parameterized.expand([(2, 1, [0, 1, 4, 6], 6, 1000),
                           (2, 2, [0, 1, 4, 6], 6, 1000),
                           (4, 1, [0, 1, 4, 6], 6, 1000),
                           (4, 2, [0, 1, 4, 6], 6, 15000),
                           (6, 1, [0, 1, 4, 6], 6, 2000),
                           (8, 1, [0, 1, 4, 6], 6, 15000),
                           (3, 1, [0, 8, 15, 17, 20, 21, 31, 39], 24, 2000),
                           (3, 2, [0, 8, 15, 17, 20, 21, 31, 39], 24, 2000),
                           (6, 1, [0, 8, 15, 17, 20, 21, 31, 39], 24, 2000),
                           ])
    @timeit
    def test_random_qnn_frequency_spectrum_turnpike(self,
                                                    R: int,
                                                    L: int,
                                                    s: list[int],
                                                    K: int,
                                                    n_coeffs_computed: int):
        q = round(np.log2(len(s)))
        expected = ((2 * K + 1) ** (R * L // q) - 1) // 2
        x = np.linspace(0, 2 * np.pi, 10)  # Dummy data
        y = np.linspace(0, 2 * np.pi, 10)  # Dummy data
        for ansatz in ("sequential", "parallel"):
            print(f"Test turnpike with {(R, L)=}, {s=}, {K=}, {ansatz=} expected K' = {expected}")
            max_frequency = 0
            for idx_round in range(10):  # Run multiple times to get the maximum frequency since it is stochastic
                qnn = QNN(R=R,
                          L=L,
                          N=1,
                          ansatz=ansatz,
                          encoding=TURNPIKE,
                          trainable_block_layers=5,
                          seed=13 + idx_round,
                          max_iter=0,  # No training
                          s=s,
                          verbose=False)
                qnn.fit(x.reshape(-1, 1), y)  # Just to initialize weights
                print(f"{n_coeffs_computed=}")
                fourier_coeffs = qnn.fourier_coefficients(n_coeffs_computed - 1)
                max_frequency = max(np.argmax(np.isclose(np.absolute(fourier_coeffs), 0, atol=10 ** -7)) - 1,
                                    max_frequency)
                print(f"Round {idx_round}: {max_frequency=}")
                if max_frequency >= expected:
                    break  # Stop if the expected frequency was measured once
            self.assertGreaterEqual(max_frequency, expected)
