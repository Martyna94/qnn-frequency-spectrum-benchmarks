from jax import numpy as jnp


def fourier_coefficients(f, K):
    """
    Computes the first K + 1 Fourier coefficients of a 2*pi periodic function.
    """
    n_coeffs = 2 * K + 1
    t = jnp.linspace(0, 2 * jnp.pi, n_coeffs, endpoint=False)
    y = jnp.fft.rfft(f(t)) / t.size
    return y


class RealFourierSeries:
    def __init__(self, pos_coefficients: list[complex]):
        r"""
        Takes the  complex coefficients c_0, c_1,... of the Fourier series. The actual series contains c_{-1},...c_{-N},
        which is the complex conjugate of the coefficients with a positive index. Note that c_0 has to be real.

        The Fourier series is then given by: f(x) = c_0 + \sum_{n=-N}^{N} c_n * exp(i * n * x)

        :param pos_coefficients: A list of coefficients.
        """
        neg_coeffs = reversed(pos_coefficients[1:])
        neg_coeffs = [complex(c).conjugate() for c in neg_coeffs]
        self.coefficients = jnp.array(neg_coeffs + pos_coefficients)

    @property
    def size(self) -> int:
        return (len(self.coefficients) - 1) // 2

    def __call__(self, x):
        """
        Evaluates the Fourier series at x. If x is a scalar, the function returns the value of the series at x. If x is a list of values, the function returns f(v) for each value v in x.
        """
        if isinstance(x, (int, float, complex)):
            exponentials = jnp.exp(1j * x * jnp.array(range(-self.size, self.size + 1)))
            return float(jnp.sum(exponentials * self.coefficients).real)
        else:
            exponentials = [jnp.exp(1j * x_ * jnp.array(range(-self.size, self.size + 1))) for x_ in x]
            return jnp.array([jnp.sum(e * self.coefficients).real for e in exponentials])
