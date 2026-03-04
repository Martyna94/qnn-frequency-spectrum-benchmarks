from pathlib import Path
import sys

import numpy as np
import pennylane as qml
from jax import numpy as jnp
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

# Allow running as `python demos/qnn_mnist_demo.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnn import QNN

# Fetch the MNIST dataset
mnist = fetch_openml("mnist_784", as_frame=False)
mnist_data = mnist.data

# Reshape the first image in the dataset to 28x28
image = jnp.reshape(mnist_data[0], (28, 28))

# Initialize lists for coordinates and pixel values
X = []
y = []
for x_coord in range(28):
    for y_coord in range(28):
        X.append((x_coord, y_coord))
        y.append(image[x_coord, y_coord] / (2 * 255))
X = jnp.array(X)
y = jnp.array(y)

# Initialize the QNN model with specified parameters
model = QNN(R=2,
            L=6,
            N=2,
            ansatz="sequential",
            encoding="ternary",
            trainable_block_layers=5,
            save_weights=True,
            save_losses=True,
            max_iter=2_000,
            step_size=0.3,
            verbose=True)

# Fit the model to the training data
model.fit(X, y)

# Predict the pixel values using the trained model
y_pred = model.predict(X)

# Reshape the predicted values to form the image
image_pred = np.zeros((28, 28))
i = 0
for x_coord in range(28):
    for y_coord in range(28):
        image_pred[x_coord, y_coord] = y_pred[i]
        i += 1

# Plot the loss values after the first 200 iterations
plt.plot(model.losses[200:])
plt.show()

# Visualize the internal model using PennyLane's drawing function
qml.draw_mpl(model.internal_model, style="pennylane", decimals=2)(x=X, weights=model.trained_weights_)
plt.show()

# Plot the original and predicted images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image, cmap="gray")
axs[1].imshow(image_pred * 255 * 2, cmap="gray")
fig.suptitle(f"Encoding: {model.encoding}, Loss: {min(model.losses)}")
plt.show()
