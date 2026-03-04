import json
import os

import pandas as pd
import experiments.constants as const_exp


class MetricsTracker:
    def __init__(self):
        self.df_metrics = pd.DataFrame(columns=[
            const_exp.METRICS_ENCODING,
            const_exp.METRICS_NORMALIZATION,
            const_exp.MODEL_R,
            const_exp.MODEL_L,
            const_exp.MODEL_N,
            const_exp.MODEL_S,
            const_exp.MODEL_TRAINABLE_BLOCK_LAYERS,
            const_exp.MODEL_ANSATZ,
            const_exp.MODEL_MAX_ITER,
            const_exp.MODEL_STEP_SIZE,
            const_exp.METRICS_LEARNING_CAPABILITY,
            const_exp.METRICS_STANDARD_DEVIATION
        ])
        self.test_loss_dict = {}

    def update_metrics(self, ansatz, name, model, mean_loss, std_loss):
        new_row = pd.DataFrame({
            const_exp.METRICS_ENCODING: [ansatz],
            const_exp.METRICS_NORMALIZATION: [name],
            const_exp.MODEL_R: [model.R],
            const_exp.MODEL_L: [model.L],
            const_exp.MODEL_N: [model.N],
            const_exp.MODEL_S: [model.s],
            const_exp.MODEL_TRAINABLE_BLOCK_LAYERS: [model.trainable_block_layers],
            const_exp.MODEL_ANSATZ: [model.ansatz],
            const_exp.MODEL_MAX_ITER: [model.max_iter],
            const_exp.MODEL_STEP_SIZE: [model.step_size],
            const_exp.METRICS_LEARNING_CAPABILITY: [mean_loss],
            const_exp.METRICS_STANDARD_DEVIATION: [std_loss]
        })

        self.df_metrics = pd.concat([self.df_metrics, new_row], ignore_index=True)

    def save_test_loss(self, name, ansatz, test_losses, filename):

        if name not in self.test_loss_dict:
            self.test_loss_dict[name] = {}
        self.test_loss_dict[name][ansatz] = test_losses.tolist()

        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'w') as f:
            json.dump(self.test_loss_dict[name][ansatz], f, indent=4)
            print(f"Test losses saved as {filename}.json")

    def save_metrics(self, folder):
        filename = f'{folder}/df_metrics.csv'  # Define the full filename with .csv extension

        if os.path.exists(filename):
            # If the file exists, load the existing CSV into a DataFrame
            existing_df = pd.read_csv(filename, sep=';')
            updated_df = pd.concat([existing_df, self.df_metrics], ignore_index=True)
            updated_df.to_csv(filename, sep=';', index=False)
            print(f"DataFrame appended to {filename}")
        else:
            # If the file doesn't exist, just save the current DataFrame
            self.df_metrics.to_csv(filename, sep=';', index=False)
            print(f"DataFrame saved as {filename}")
