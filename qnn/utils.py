from functools import wraps


def allow_1d_input(f):
    @wraps(f)
    def wrapper(x, *args, **kwargs):
        if x.ndim == 1:
            return f(x.reshape(-1, 1), *args, **kwargs)
        else:
            return f(x, *args, **kwargs)

    return wrapper


# Re-export useful Fourier helpers from qnn.fourier for convenience
from .fourier import fourier_coefficients, RealFourierSeries
