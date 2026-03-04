import json
import os
import experiments.constants as const


def convert_complex(value):
    if isinstance(value, complex):
        return {const.REAL: value.real, const.IMAG: value.imag}
    return value


def save_model_to_json(model, fourier_coeffs, normalization, filename):
    result = {
        const.MODEL_R: model.R,
        const.MODEL_L: model.L,
        const.MODEL_TRAINABLE_BLOCK_LAYERS: model.trainable_block_layers,
        const.MODEL_MAX_ITER: model.max_iter,
        const.MODEL_MAX_ITER: model.step_size,
        const.MODEL_S: model.s,
        const.MODEL_ENCODING: model.encoding,
        const.MODEL_TRAINED_WEIGHTS: model.trained_weights_.tolist(),
        const.MODEL_LOSSES: list(map(float, model.losses)),
        const.FOURIER_COEFFS: [convert_complex(c) for c in fourier_coeffs.tolist()],
        const.NORMALIZATION: normalization
    }

    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)

    with open(filename, 'a') as file:
        json.dump(result, file, indent=4)
        file.write('\n')

