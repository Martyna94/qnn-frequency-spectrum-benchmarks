from functools import partial
from typing import Callable, Optional

import jax
import jaxopt
import pennylane as qml
from jax import numpy as jnp
from tqdm import tqdm

from qnn.constants import LOSS_FNS, SEQUENTIAL, MSE, BINARY_CROSS_ENTROPY
from qnn.constants import HIGHER_DIMENSIONAL_ENCODINGS
from qnn.encodings import S_parallel, S_sequential, W, get_encoding_fn
from qnn.encodings import S_sequential_turnpike_golomb, S_parallel_turnpike_golomb
from qnn.fourier import fourier_coefficients
from qnn.utils import allow_1d_input

jax.config.update("jax_enable_x64", True)


class QNN:
    def __init__(self,
                 R: int,
                 L: int,
                 N: int,
                 ansatz: str = SEQUENTIAL,
                 encoding: str | Callable = 'constant',
                 loss_fn: str = MSE,
                 seed: int = 13,
                 trainable_block_layers: int = 3,
                 save_weights: bool = True,
                 save_losses: bool = True,
                 max_iter: int = 100,
                 step_size: float = 0.1,
                 batch_size: Optional[int] = None,
                 verbose: bool = True,
                 s: Optional[list] = None,
                 ) -> None:
        self.R = R
        self.L = L
        self.N = N
        self.ansatz = ansatz
        self.encoding = encoding
        self.loss_fn = loss_fn
        self.key = jax.random.PRNGKey(seed)
        self.trainable_block_layers = trainable_block_layers
        self.save_weights = save_weights
        self.save_losses = save_losses
        self.max_iter = max_iter
        self.step_size = step_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.s = s if s is not None else [0, 1]  # only used for higher-dimensional encodings
        self.q = round(jnp.log2(len(self.s)))  # only used for higher-dimensional encodings

        self.dev = qml.device("default.qubit", wires=R if ansatz == 'sequential' else R * N)
        self.internal_model = self._create_internal_model()
        self.trained_weights_ = None

        if save_weights:
            self.weights = []  # History of weights
        if save_losses:
            self.losses = []  # History of losses

        if self.encoding in HIGHER_DIMENSIONAL_ENCODINGS:
            if self.R % self.q != 0:
                raise ValueError(f"q = {self.q} does not divide R = {self.R}")

        if self.loss_fn not in LOSS_FNS:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def _check_shape(self, X):
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional. Got {X.ndim}-dimensional instead.")
        if X.shape[1] != self.N:
            raise ValueError(f"X must have N={self.N} features columns, but {X.shape[1]} given.")

    def _create_internal_model(self) -> Callable:
        encoding_fn = get_encoding_fn(self.encoding) if isinstance(self.encoding, str) else self.encoding
        if self.ansatz == 'parallel':
            @qml.qnode(self.dev)
            def qnn(x, weights, encoding_fn, R, L, N, s, encoding):
                for l, weight in enumerate(weights[:-1]):
                    W(weight, R=R * N)
                    if encoding in HIGHER_DIMENSIONAL_ENCODINGS:
                        S_parallel_turnpike_golomb(x, encoding_fn, l, R, L, N, s)
                    else:
                        S_parallel(x, encoding_fn, l, R, L, N)
                W(weights[-1], R=R * N)
                return qml.expval(qml.PauliZ(wires=0))
        elif self.ansatz == 'sequential':
            @qml.qnode(self.dev)
            def qnn(x, weights, encoding_fn, R, L, N, s, encoding):
                for n in range(N):
                    for l, weight in enumerate(weights[n * L:(n + 1) * L]):
                        W(weight, R=R)
                        if encoding in HIGHER_DIMENSIONAL_ENCODINGS:
                            S_sequential_turnpike_golomb(x, encoding_fn, l, R, L, n, s)
                        else:
                            S_sequential(x, encoding_fn, l, R, L, n)
                W(weights[-1], R=R)
                return qml.expval(qml.PauliZ(wires=0))
        else:
            raise NotImplementedError(f"Unknown ansatz: {self.ansatz}")
        return partial(qnn, encoding_fn=encoding_fn, R=self.R, L=self.L, N=self.N, s=self.s, encoding=self.encoding)

    def _create_loss_fn(self) -> Callable:
        if self.loss_fn == MSE:
            @jax.jit
            def loss(params, X, y):
                predictions = self.internal_model(x=X, weights=params)
                loss_value = jnp.sum((y - predictions) ** 2) / len(X)
                return loss_value
        elif self.loss_fn == BINARY_CROSS_ENTROPY:
            @jax.jit
            def loss(params, X, y):
                logit = self.internal_model(x=X, weights=params)
                predictions = jax.nn.sigmoid(6 * logit)  # The scaling factor 6 is chosen because the raw model outputs
                # values in the range of (-1, 1) and sigmoid(6) is close enough to 1
                predictions = jnp.clip(predictions, min=1e-10, max=1 - 1e-10)  # Clip for numerical stability
                loss_value = -jnp.sum(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions)) / len(X)
                return loss_value
        else:
            raise NotImplementedError(f"Unknown loss function: {self.loss_fn}")

        return loss

    def fit(self, X, y):
        self._check_shape(X)

        weights = self._get_initial_weights()
        # If the fit method is called with max_iter = 0 to initialize the weights,
        # we only initialize the weights and return immediately
        if self.max_iter == 0:
            self.trained_weights_ = weights
            return
        loss_fn = self._create_loss_fn()
        best_weights = weights
        best_loss = loss_fn(weights, X, y)
        if self.save_weights:
            self.weights.append(weights)
        if self.save_losses:
            self.losses.append(loss_fn(weights, X, y))
        gd = jaxopt.GradientDescent(loss_fn, maxiter=self.max_iter, stepsize=self.step_size)
        gd_state = gd.init_state(weights)
        training_rounds = tqdm(range(self.max_iter)) if self.verbose else range(self.max_iter)
        for i in training_rounds:
            if self.batch_size is None:
                weights, gd_state = gd.update(weights, gd_state, X, y)
            else:
                # Shuffle X
                self.key, subkey = jax.random.split(self.key)
                indices = jax.random.permutation(subkey, jnp.arange(len(X)))
                X = X[indices]
                y = y[indices]
                # Split into batches
                num_samples = X.shape[0]
                if self.batch_size <= 0:
                    raise ValueError("batch_size must be >= 1")
                if self.batch_size >= num_samples:
                    weights, gd_state = gd.update(weights, gd_state, X, y)
                else:
                    num_full_batches = num_samples // self.batch_size
                    for batch_index in range(num_full_batches):
                        start = batch_index * self.batch_size
                        end = start + self.batch_size
                        weights, gd_state = gd.update(weights, gd_state, X[start:end], y[start:end])
                    remainder = num_samples % self.batch_size
                    if remainder:
                        # Train on leftover samples so no data is dropped.
                        weights, gd_state = gd.update(weights, gd_state,
                                                      X[num_full_batches * self.batch_size:],
                                                      y[num_full_batches * self.batch_size:])
            loss_value = loss_fn(weights, X, y)
            if self.save_weights:
                self.weights.append(weights)
            if self.save_losses:
                self.losses.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_weights = weights
                if self.verbose:
                    training_rounds.set_description(f"Best loss: {best_loss:.6f}")
        self.trained_weights_ = best_weights

    def _get_initial_weights(self):
        if self.ansatz == 'sequential':
            weights = 2 * jnp.pi * jax.random.uniform(key=self.key,
                                                      shape=(self.N * self.L + 1,
                                                             self.trainable_block_layers,
                                                             self.R,
                                                             3)
                                                      )
        else:
            weights = 2 * jnp.pi * jax.random.uniform(key=self.key,
                                                      shape=(self.L + 1,
                                                             self.trainable_block_layers,
                                                             self.R * self.N,
                                                             3)
                                                      )
        return weights

    def predict(self, X):
        self._check_shape(X)
        if self.loss_fn == MSE:
            return self.internal_model(x=X, weights=self.trained_weights_)
        elif self.loss_fn == BINARY_CROSS_ENTROPY:
            logit = self.internal_model(x=X, weights=self.trained_weights_)
            return jax.nn.sigmoid(6 * logit)  # See the comment in the loss function for an explanation of the factor 6
        else:
            raise NotImplementedError(f"Unknown loss function: {self.loss_fn}")

    def loss_score(self, X, y) -> float:
        loss_fn = self._create_loss_fn()
        return loss_fn(self.trained_weights_, X, y)

    def fourier_coefficients(self, K):
        """
        Computes the first K + 1 Fourier coefficients of a 2*pi periodic function.
        """
        if self.N > 1:
            raise NotImplementedError(f"Computing Fourier coefficients is not implemented for N > 1.")
        return fourier_coefficients(allow_1d_input(self.predict), K=K)
