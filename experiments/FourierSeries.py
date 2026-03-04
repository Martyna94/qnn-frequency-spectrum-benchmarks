import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable
import experiments.constants as const

jax.config.update("jax_enable_x64", True)


def no_normalization(result: jnp.ndarray, coeffs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """No normalization."""
    return result, coeffs


def l2_normalization(result: jnp.ndarray, coeffs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """L2 normalization of the result using the Euclidean norm."""
    norm = jnp.linalg.norm(result)
    if norm == 0:
        raise ValueError("Cannot normalize when the norm of result is zero.")
    result = result / norm
    new_coeffs = coeffs / norm
    return result, new_coeffs


def min_max_normalization(result: jnp.ndarray, coeffs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Min-max normalization (scales result to [-1, 1]) and adjust coefficients accordingly."""
    min_val = jnp.min(result)
    max_val = jnp.max(result)
    if max_val == min_val:
        raise ValueError("Cannot normalize when all results are the same.")
    result = 2 * (result - min_val) / (max_val - min_val) - 1
    scale_factor = 2 / (max_val - min_val)


    new_coeffs = coeffs * scale_factor
    # Adjust the zero-frequency coefficient for the shift
    # shift = - (2 * min_val / (max_val - min_val) + 1)
    # new_coeffs = new_coeffs.at[self.size].set(new_coeffs[self.size] + shift)
    return result, new_coeffs

def half_range_normalization(result: jnp.ndarray, coeffs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalization to [-1/2, 1/2] and adjust coefficients accordingly."""
    min_val = jnp.min(result)
    max_val = jnp.max(result)
    if max_val == min_val:
        raise ValueError("Cannot normalize when all results are the same.")

    result = (result - min_val) / (max_val - min_val) - 0.5
    scale_factor = 1 / (max_val - min_val)

    new_coeffs = coeffs * scale_factor
    # Adjust the zero-frequency coefficient for the shift
    # shift = - (min_val / (max_val - min_val) + 0.5)
    # new_coeffs = new_coeffs.at[self.size].set(new_coeffs[self.size] + shift)
    return result, new_coeffs

class RealFourierSeries:

    normalization_map = {
        const.NO_NORMALIZATION: no_normalization,
        const.L2_NORMALIZATION: l2_normalization,
        const.MIN_MAX_NORMALIZATION: min_max_normalization,
        const.HALF_RANGE_NORMALIZATION: half_range_normalization
    }

    def __init__(self, pos_coefficients: List[complex], normalization_strategy: str = "none"):
        """
        Initializes a real-valued Fourier series.

        Parameters:
        - pos_coefficients: A list of complex coefficients [c_0, c_1, ..., c_N].
                            c_0 must be real. c_{-n} is the complex conjugate of c_n.
        - normalization_strategy: A string that represents the normalization strategy ("none", "l2", "min_max").
        """
        if not isinstance(pos_coefficients, list):
            raise TypeError("pos_coefficients must be a list of complex numbers.")
        if not isinstance(pos_coefficients[0], (int, float, complex)):
            raise TypeError("Each coefficient must be a complex number.")
        if not jnp.isclose(jnp.imag(pos_coefficients[0]), 0):
            raise ValueError("The zero-frequency coefficient c_0 must be real.")

        if normalization_strategy not in RealFourierSeries.normalization_map:
            raise ValueError(f"Unknown normalization strategy: {normalization_strategy}")
        self.normalization_strategy = normalization_strategy

        pos_coefficients = jnp.array(pos_coefficients, dtype=jnp.complex64)
        neg_coeffs = jnp.array([c.conjugate() for c in reversed(pos_coefficients[1:])])

        self.coefficients = jnp.concatenate([neg_coeffs, pos_coefficients])
        self.size = (len(self.coefficients) - 1) // 2
        self.frequencies = jnp.arange(-self.size, self.size + 1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates the Fourier series at given x values.

        Parameters:
        - x: A scalar or a JAX array of x values.

        Returns:
        - f(x): The evaluated Fourier series at x, normalized based on the strategy.
        """
        exponentials = jnp.exp(1j * jnp.expand_dims(x, axis=-1) * self.frequencies)
        result = jnp.sum(exponentials * self.coefficients, axis=-1).real

        if jnp.min(result) < -1 or jnp.max(result) > 1:
            result, new_coeffs = RealFourierSeries.normalization_map[self.normalization_strategy](result, self.coefficients)
            self.coefficients = new_coeffs

        return result
