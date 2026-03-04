import jax.numpy as jnp


def compute_dft(y, x):
    """
    Compute the Discrete Fourier Transform of a signal.

    Parameters:
    - y: The signal in time domain.
    - x: The array of x values corresponding to y.

    Returns:
    - freq: Frequencies corresponding to the DFT components.
    - Y: The magnitudes of the DFT components.
    """
    Y = jnp.fft.fft(y)
    freq = jnp.fft.fftfreq(len(y), d=(x[1] - x[0]) / (2 * jnp.pi))  # Adjust for interval [-π, π]
    # Shift zero frequency component to center
    Y = jnp.fft.fftshift(Y)
    freq = jnp.fft.fftshift(freq)
    return freq, jnp.abs(Y)


#%%
