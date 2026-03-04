import unittest

import numpy as np
from parameterized import parameterized

from qnn.fourier import RealFourierSeries, fourier_coefficients


class TestFourierSeriesRealCoeffs(unittest.TestCase):
    """
    Here we test the __call__ method of the RealFourierSeries class with only real coefficients,
    which is used to represent a Fourier series with real coefficients.
    """

    @parameterized.expand([
        ([0], 10),
        ([2], 10),
        ([2, 1], 10),
        ([1, 2, 3], 10),
        ([1, 2, 3, 4], 10),
        ([1, 2, 3, 4, 5], 10),
        ([1, 2, 3, 4, 5, 6], 10),
    ])
    def test_series(self, coeffs: list[float], decimal: int):
        series = RealFourierSeries(coeffs)
        x = np.linspace(0, 2 * np.pi, 100)
        y = coeffs[0] + sum(
            [coeffs[k] * np.exp(1j * k * x) + coeffs[k] * np.exp(-1j * k * x) for k in range(1, len(coeffs))])
        # Assert the result is as expected
        np.testing.assert_array_almost_equal(series(x), y, decimal=decimal)


class TestFourierSeriesComplexCoeffs(unittest.TestCase):
    """
    Here we test the __call__ method of the RealFourierSeries class with complex coefficients.
    """

    @parameterized.expand([
        ([2, 1j], 10),
        ([1, 2 + 1j, 3 - 1j], 10),
        ([1, 2 + 1j, 3 - 1j, 4 + 2j], 10),
        ([1, 2 + 1j, 3 - 1j, 4 + 2j, 5 - 3j], 10),
        ([1, 2 + 1j, 3 - 1j, 4 + 2j, 5 - 3j, 6 + 4j], 10),
    ])
    def test_series(self, coeffs: list[float], decimal: int):
        series = RealFourierSeries(coeffs)
        x = np.linspace(0, 2 * np.pi, 100)
        y = coeffs[0] + sum(
            [coeffs[k] * np.exp(1j * k * x) + coeffs[k].conjugate() * np.exp(-1j * k * x) for k in
             range(1, len(coeffs))])
        # Assert the result is as expected
        np.testing.assert_array_almost_equal(series(x), y, decimal=decimal)


class TestFourierCoefficientComputation(unittest.TestCase):
    """
    Here we test the fourier_coefficients function, which computes the first K + 1 Fourier coefficients
    of a 2*pi periodic function.
    """

    @parameterized.expand([
        ([0], 10),
        ([2], 10),
        ([2, 1j], 10),
        ([1, 2 + 1j, 3 - 1j], 10),
        ([1, 2 + 1j, 3 - 1j, 4 + 2j], 10),
        ([1, 2 + 1j, 3 - 1j, 4 + 2j, 5 - 3j], 10),
        ([1, 2 + 1j, 3 - 1j, 4 + 2j, 5 - 3j, 6 + 4j], 10),
    ])
    def test_fourier_computation(self, coeffs: list[float], decimal: int):
        f = RealFourierSeries(coeffs)
        for N in range(len(coeffs), 20):
            fourier_coeffs = fourier_coefficients(f, N)
            # Assert the result is as expected
            np.testing.assert_array_almost_equal(fourier_coeffs, coeffs + [0] * (N - len(coeffs) + 1), decimal=decimal)


if __name__ == '__main__':
    unittest.main()
