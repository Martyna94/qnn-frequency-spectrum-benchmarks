import os
from operator import index

import matplotlib.pyplot as plt
import numpy as np
import experiments.constants as const_exp
import matplotlib.lines as mlines
from experiments.utils import compute_dft


def plot_time_domain_signals(x, signals, titles, y_label="f(x)", x_label="x", y_limits=None):
    """
    Plot time-domain signals with customizable titles and the number of subplots.

    Parameters:
    - x: Array of x values.
    - signals: List of time-domain signals to plot.
    - titles: List of titles for each subplot.
    - y_label: Label for the y-axis (default is 'f(x)').
    - x_label: Label for the x-axis (default is 'x').
    - y_limits: Tuple specifying the y-axis limits (default is (-1.1, 1.1)).
    """
    n_signals = len(signals)

    if len(titles) != n_signals:
        raise ValueError("The number of titles must match the number of signals.")

    fig, axs = plt.subplots(1, n_signals, figsize=(6 * n_signals, 4))

    if n_signals == 1:
        axs = [axs]

    for i in range(n_signals):
        axs[i].plot(x, signals[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel(y_label)
        if None != y_limits:
            axs[i].set_ylim(y_limits)

    plt.tight_layout()
    plt.show()


def plot_amplitude_spectra(freq_dfts, Y_dfts, titles, degree, x_label="Frequency", y_label="Amplitude"):
    """
    Plot amplitude spectra with customizable titles and number of subplots.

    Parameters:
    - freq_dfts: List of frequency arrays from the DFT of signals.
    - Y_dfts: List of magnitude arrays from the DFT of signals.
    - titles: List of titles for each subplot.
    - degree: Degree of the Fourier series (used for setting x-axis limits).
    - x_label: Label for the x-axis (default is 'Frequency').
    - y_label: Label for the y-axis (default is 'Amplitude').
    """
    n_spectra = len(freq_dfts)

    if len(titles) != n_spectra or len(Y_dfts) != n_spectra:
        raise ValueError("The number of titles and DFT arrays must match the number of spectra.")

    fig, axs = plt.subplots(1, n_spectra, figsize=(6 * n_spectra, 4))


    if n_spectra == 1:
        axs = [axs]

    for i in range(n_spectra):
        axs[i].stem(freq_dfts[i], Y_dfts[i], linefmt="r", markerfmt=" ", basefmt="-b")
        axs[i].set_title(titles[i])
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel(y_label)
        axs[i].set_xlim(-degree, degree)

    plt.tight_layout()
    plt.show()


def compute_dft_and_plot_amplitude_spectra(y_series, x, titles, size):
    """Plot the frequency spectra"""
    freq_dfts, Y_dfts = [], []
    for name, y in zip(titles, y_series):
        freq_dft, Y_dft = compute_dft(y, x)
        freq_dfts.append(freq_dft)
        Y_dfts.append(Y_dft)
    plot_amplitude_spectra(freq_dfts, Y_dfts, titles, size)


def create_model_params_table(ax, model):
    """
    Create a table of model parameters below the plot.

    Parameters:
    - ax: The Axes object where the table will be created.
    - models: Dictionary of model parameters for displaying in a table.
    """

    table_data = [
        [const_exp.MODEL_R, model.R],
        [const_exp.MODEL_L, model.L],
        [const_exp.MODEL_N, model.N],
        [const_exp.MODEL_S, model.s],
        [const_exp.MODEL_TRAINABLE_BLOCK_LAYERS, model.trainable_block_layers],
        [const_exp.MODEL_ANSATZ, model.ansatz],
        [const_exp.MODEL_MAX_ITER, model.max_iter],
        [const_exp.MODEL_STEP_SIZE, model.step_size]
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        cellLoc='center',
        loc='bottom',
        bbox=[0.1, -0.5, 0.8, 0.3]
    )
    return table


def plot_learning_capabilities(df_metrics, normalization_strategies, encodings, model, save_plot=False, save_model_params=False, filename='Learning_Capacities.png', verbose=False):
    """
    Plot learning capabilities for different normalization strategies and encodings.

    Parameters:
    - df_metrics: DataFrame containing learning capacities, normalization strategies, and encodings.
    - normalization_strategies: List of normalization strategy names.
    - encodings: List of encoding names.
    - model: Dictionary of model parameters for displaying in a table.
    - save_plot: Boolean to specify whether to save the plot to a file (default is False).
    - filename: The filename to save the plot (default is 'Learning_Capacities.png').
    """
    x = np.arange(len(encodings))
    num_strategies = len(normalization_strategies)
    width = 0.8 / num_strategies
    fig, ax = plt.subplots(figsize=(10, 12))

    # Pivot the df_metrics for faster access
    capabilities_pivot = df_metrics.pivot(index=const_exp.METRICS_ENCODING,
                                          columns=const_exp.METRICS_NORMALIZATION,
                                          values=const_exp.METRICS_LEARNING_CAPABILITY)
    std_pivot = df_metrics.pivot(index=const_exp.METRICS_ENCODING,
                                 columns=const_exp.METRICS_NORMALIZATION,
                                 values=const_exp.METRICS_STANDARD_DEVIATION)

    for i, strategy in enumerate(normalization_strategies):
        # Fetch capacities for all encodings for the given strategy at once
        capabilities = capabilities_pivot[strategy].reindex(encodings).fillna(0).values
        std_devs = std_pivot[strategy].reindex(encodings).fillna(0).values

        offset = (i - (num_strategies / 2)) * width

        bars = ax.bar(x + offset, capabilities, width,  yerr=std_devs, label=f'{strategy}', capsize=5)

        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, yval + 0.1*yval,
                    f'{yval:.2e}', ha='center', va='bottom', fontsize=10
                )
                plt.scatter(bar.get_x() + bar.get_width() / 2, yval, color=bar.get_facecolor(), zorder=5)

    ax.set_xlabel('Encoding')
    ax.set_ylabel('Learning Capability')
    ax.set_title('Learning Capability by Normalization Strategy')

    ax.set_xticks(x)
    ax.set_xticklabels(encodings)

    std_proxy = mlines.Line2D([], [], color='black', linestyle='-', marker='', label='Standard Deviation')

    ax.legend(handles=[*ax.get_legend_handles_labels()[0], std_proxy])
    if save_model_params:
        create_model_params_table(ax, model)
    # Adjust the layout to leave space for the table
    plt.subplots_adjust(bottom=0.35, top=0.85)

    if save_plot:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename, dpi=100)

    if verbose:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


def plot_train_loss_and_test_function(losses, test_loss, test_set, predictions, save_plot=False, filename='Loss', verbose = False):
    """
    This function takes an array of losses and plots the values.
    X-axis: Index + 1
    Y-axis: Loss values
    """
    if not losses:
        print("The losses array is empty. No plot will be created.")
        return

    x_values = range(1, len(losses) + 1)
    x_test_values, y_test_values = test_set

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].plot(x_values, losses)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss Value')
    axs[0].set_title(f"Training Loss Progress Over Epochs")

    min_loss = min(losses)
    min_index = losses.index(min_loss)

    axs[0].plot(min_index, min_loss, 'ro')  # Plot the point in red
    axs[0].annotate(f"Min Loss: {min_loss:.2e}",
                xy=(min_index, min_loss),
                xytext=(min_index-0.2*min_index, 0.2*min_loss),  # Adjust the text position
                fontsize=10,
                color = 'green')

    axs[1].plot(x_test_values, predictions)
    axs[1].plot(x_test_values, y_test_values, linestyle="dotted")
    axs[1].set_xlabel('X Values')
    axs[1].set_ylabel('Y Values')
    axs[1].set_title(f"Test Function: {test_loss: .2e} loss value")

    plt.tight_layout()

    if save_plot:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(filename, dpi=300)

    if verbose:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def plot_losses(losses, mean_loss: float, std_loss: float,mark_index = False,min_index=0,mean_index=0,max_index=0, save_plot: bool = False, filename: str = 'Loss', verbose: bool = False) -> None:
    """
    This function takes an array of losses and plots the values.

    Parameters:
    - losses: A jax.numpy array or list of loss values.
    - save_plot: Whether to save the plot to a file (default: False).
    - filename: The filename to save the plot (if save_plot is True).
    - verbose: Whether to display the plot (default: False).

    Returns:
    - None: The function generates a plot but does not return any value.
    """
    losses = np.array(losses)
    function_index = np.arange(1, losses.size + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(function_index, losses, 'bo-', label='Loss Values')
    plt.axhline(y=mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.2e}')
    plt.fill_between(function_index, mean_loss - std_loss, mean_loss + std_loss, color='gray', alpha=0.2, label=f'Std: {std_loss:.2e}')


    if mark_index:
        plt.scatter(min_index + 1, losses[min_index], color='green', s=100, zorder=5, label=f'Min Loss: {losses[min_index]:.2e}, index: {min_index}')
        plt.scatter(mean_index + 1, losses[mean_index], color='purple', s=100, zorder=5, label=f'Closest to Mean: {losses[mean_index]:.2e}, index: {mean_index}')
        plt.scatter(max_index + 1, losses[max_index], color='orange', s=100, zorder=5, label=f'Max Loss: {losses[max_index]:.2e}, index: {max_index}')


    plt.title('Loss Values Across Different Functions with Mean and Std Dev')
    plt.xlabel('Function Index')
    plt.ylabel('Loss Value')
    plt.legend()

    if save_plot:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename, dpi=300)

    if verbose:
        plt.show()

    plt.close()




#%%
