from typing import Callable

import numpy as np
from jax import numpy as jnp
import pennylane as qml
from pennylane import StronglyEntanglingLayers

from qnn.turnpike import find_value_of_k


def beta_hamming(r: int, l: int, R: int, L: int) -> float:
    return 1


def beta_ternary(r: int, l: int, R: int, L: int) -> float:
    return 3 ** (l + L * r)


def beta_exponential(r: int, l: int, R: int, L: int) -> float:
    if (r == (R - 1)) and (l == (L - 1)):
        return 2 ** (R * L - 1) + 1
    else:
        return 2 ** (l + L * r)


def beta_binary(r: int, l: int, R: int, L: int) -> float:
    return 2 ** (l + L * r)


def beta_golomb(r: int, l: int, R: int, L: int, s: list) -> float:
    length = max(s) - min(s)
    beta = 2 * length + 1
    return beta ** (l + L * r)


def beta_turnpike(r: int, l: int, R: int, L: int, s: list) -> float:
    K = find_value_of_k(s)
    return (K + 1) ** (l + L * r)


def get_encoding_fn(name: str) -> Callable:
    return {"hamming": beta_hamming,
            "exponential": beta_exponential,
            "binary": beta_binary,
            "ternary": beta_ternary,
            "golomb": beta_golomb,
            "turnpike": beta_turnpike}[name]


def S_parallel(x, beta_r_l: Callable, l: int, R: int, L: int, N: int):
    """Data encoding circuit block."""
    for n in range(N):
        for r in range(R):
            qml.RZ(beta_r_l(r, l, R, L) * x[:, n], wires=r + n * R)


def S_sequential(x, beta_r_l: Callable, l: int, R: int, L: int, n: int):
    """Data encoding circuit block."""
    for r in range(R):
        qml.RZ(beta_r_l(r, l, R, L) * x[:, n], wires=r)


def S_parallel_turnpike_golomb(x, beta_r_l: Callable, l: int, R: int, L: int, N: int, s: list):
    """Data encoding circuit block."""
    k = len(s)
    q = round(np.log2(k))
    for n in range(N):
        for r in range(0, R // q):
            beta_r = beta_r_l(r, l, R, L, s)
            diagonal_exponent = -1j * beta_r * x[:, n].reshape(-1, 1) @ jnp.array(s).reshape(1, -1)
            diagonal_unitary = jnp.exp(diagonal_exponent)
            qml.DiagonalQubitUnitary(diagonal_unitary, wires=list(range(r * q + n * R, (r + 1) * q + n * R)))


def S_sequential_turnpike_golomb(x, beta_r_l: Callable, l: int, R: int, L: int, n: int, s: list):
    """Data encoding circuit block."""
    k = len(s)
    q = round(np.log2(k))
    for r in range(0, R // q):
        beta_r = beta_r_l(r, l, R, L, s)
        diagonal_exponent = -1j * beta_r * x[:, n].reshape(-1, 1) @ jnp.array(s).reshape(1, -1)
        diagonal_unitary = jnp.exp(diagonal_exponent)
        qml.DiagonalQubitUnitary(diagonal_unitary, wires=list(range(r * q, (r + 1) * q)))


def W(theta, R: int):
    """Trainable circuit block."""
    StronglyEntanglingLayers(theta, wires=range(R))
