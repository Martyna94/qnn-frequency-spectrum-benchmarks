import jax.numpy as jnp
import gc

from experiments.DatasetManager import DatasetManager
from experiments.MetricsTracker import MetricsTracker
import experiments.constants as const_exp

from qnn import QNN
from experiments.logging import save_model_to_json
from experiments.plotting import plot_learning_capabilities, plot_losses, plot_train_loss_and_test_function

#  Dataset parameters
degree = 12
num_sample = 100

# Input data for the model
x = jnp.linspace(0, 2 * jnp.pi, 100)
x_test = jnp.linspace(-2 * jnp.pi, 0, 100)

# Model parameters
R = 6
L = 2
N = 1
ansatz = const_exp.MODEL_SEQUENTIAL
trainable_block_layers = 5
max_iter = 3000
step_size = 0.05
#s = [0, 1, 4, 6]

# List of different encoding methods for experiments
encodings = [const_exp.GOLOMB, const_exp.TERNARY]

# Normalization strategies for the dataset
normalization_strategies = [const_exp.L2_NORMALIZATION,const_exp.MIN_MAX_NORMALIZATION,const_exp.HALF_RANGE_NORMALIZATION]


# Folder for logging outputs
folder = 'exp_7'
##############################################################################################
# if we should generate data
#dataset_manager = DatasetManager(x=x, x_test=x_test, num_sample=2, degree=degree, normalization_strategies=normalization_strategies)
# if only load
dataset_manager = DatasetManager(x=None, x_test=None, num_sample=None, degree=None, normalization_strategies=None, json_file_path='dataset_12degree_100sample.json')

metrics_tracker = MetricsTracker()

models = {encoding: QNN(R=R,
                        L=L,
                        N=N,
                        ansatz=ansatz,
                        encoding=encoding,
                        trainable_block_layers=trainable_block_layers,
                        save_weights=True,
                        save_losses=True,
                        max_iter=max_iter,
                        step_size=step_size,
                        verbose=True) for encoding in encodings}


for ansatz, model in models.items():
    for normalization in normalization_strategies:
        y_series, y_series_test = dataset_manager.get_series(normalization)
        fourier_series = dataset_manager.get_fourier_series(normalization)

        test_losses = jnp.zeros(num_sample)
        gc.collect()

        for idx, y in enumerate(y_series):
            model.losses = []
            if idx%10 ==0:
                print(idx)
            model.fit(x.reshape(-1, 1), y)

            predictions = model.predict(x_test.reshape(-1, 1))
            test_loss = model.loss_score(x_test.reshape(-1, 1), y_series_test[idx])
            test_losses = test_losses.at[idx].set(test_loss)

            save_model_to_json(model=model, fourier_coeffs=fourier_series[idx].coefficients, normalization=normalization, filename=os.path.join(folder, ansatz, "models", f"{idx}_{normalization}"))

            #plot_train_loss_and_test_function(losses=model.losses, test_loss=test_loss, test_set=(x_test, y_series_test[idx]), predictions=predictions, save_plot=True, filename=f'{folder}\\{ansatz}\\Fourier_series\\{idx+1}_{normalization}_Loss')

        mean_loss = jnp.mean(test_losses)
        std_loss = jnp.std(test_losses)

        # Calculate the lowest value and its index
        min_index = jnp.argmin(test_losses)

        mean_index = jnp.argmin(jnp.abs(test_losses - mean_loss))

        max_index = jnp.argmax(test_losses)

        plot_losses(test_losses, mean_loss=mean_loss, std_loss=std_loss, mark_index= True ,min_index=min_index, mean_index=mean_index, max_index=max_index, save_plot=True, filename=f'{folder}\\{ansatz}\\{normalization}_Losses_Mark_Index', verbose=True)
        plot_losses(test_losses, mean_loss=mean_loss, std_loss=std_loss, save_plot=True, filename=os.path.join(folder, ansatz, f"{normalization}_Losses"), verbose=True)

        metrics_tracker.save_test_loss(normalization, ansatz, test_losses, filename = f'{folder}/{ansatz}/{normalization}_Test_Loss_Values')
        metrics_tracker.update_metrics(ansatz, normalization, model, mean_loss, std_loss)
        print(f"{normalization=:20} {ansatz=:15} Mean: {mean_loss:.2e}, Std: {std_loss:.2e}")

print(metrics_tracker.df_metrics)


# Plot and save the figure
plot_learning_capabilities(metrics_tracker.df_metrics, normalization_strategies, encodings, models[encodings[0]], save_plot=True, save_model_params=True, filename=os.path.join(folder, "LC_all.png"))
#plot_learning_capabilities(metrics_tracker.df_metrics, normalization_strategies, [const_exp.HAMMING], models[const_exp.HAMMING], save_plot=True, save_model_params=True, filename=f'{folder}\\LC_Hamming')

# Save the metrics
metrics_tracker.save_metrics(folder)
#%%
