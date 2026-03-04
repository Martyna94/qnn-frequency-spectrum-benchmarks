from pathlib import Path
import sys

import pennylane as qml
from jax import numpy as jnp
from matplotlib import pyplot as plt

from qnn import QNN

# Initialize the QNN model with specified parameters
model = QNN(R=2,
            L=3,
            N=1,
            ansatz="sequential",
            encoding="ternary",
            trainable_block_layers=10,
            save_weights=True,
            save_losses=True,
            max_iter=4_000,
            step_size=0.05,
            verbose=True)

# Generate training data
x = jnp.linspace(-jnp.pi, jnp.pi, 500)
x_train = x.reshape(-1, 1)
print(x_train.shape)

# Generate target data using a sum of sine functions
y = sum([1 / 80 * jnp.sin(n * x) for n in range(1, 100)])

# Fit the model to the training data
model.fit(x_train, y)

# Visualize the internal model using PennyLane's drawing function
qml.draw_mpl(model.internal_model, style="pennylane", decimals=2)(x=x_train, weights=model.trained_weights_)
plt.show()

# Plot the loss values after the first 200 iterations
plt.plot(model.losses[200:])
plt.show()

# Plot the model predictions against the true values
plt.plot(x, model.predict(x_train), label="predict")
plt.plot(x, y, label="true")
plt.legend()
plt.show()

# Print the final loss score and the minimum loss value
print(model.loss_score(x_train, y), min(model.losses))

def fourier_coefficients(f, K):
    """
    Computes the first 2*K+1 Fourier coefficients of a 2*pi periodic function.

    Parameters:
    f (function): The function to compute the Fourier coefficients for.
    K (int): The number of coefficients to compute.

    Returns:
    jnp.ndarray: The computed Fourier coefficients.
    """
    n_coeffs = 2 * K + 1
    t = jnp.linspace(0, 2 * jnp.pi, n_coeffs, endpoint=False)
    y = jnp.fft.rfft(f(t.reshape(-1, 1))) / t.size
    return y

# Compute and print the Fourier coefficients of the model's predictions
print(fourier_coefficients(model.predict, 3**4))
