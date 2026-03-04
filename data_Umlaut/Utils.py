import json
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_csv_files(folder_path):
    runs_metadata = None
    other_files = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if os.path.splitext(filename)[0] == "runs_metadata":
                runs_metadata = df
            else:
                file_key = os.path.splitext(filename)[0]
                other_files[file_key] = df

    return runs_metadata, other_files


def plot_all_runs_with_highlight(runs_files, nearest_times_df=None, time_column='Time', column='Current'):
    # Use the default Matplotlib style
    plt.style.use('default')

    num_runs = len(runs_files)
    num_cols = 3  # Number of columns
    num_rows = math.ceil(num_runs / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, (run_key, data) in enumerate(runs_files.items()):
        run_id = run_key.split('_')[-1]

        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])

        ax = axes[idx]
        ax.plot(data[time_column], data[column], color='tab:blue', label=column)

        # Add nearest_time highlight if nearest_times_df is provided
        if nearest_times_df is not None and run_id in nearest_times_df['run_ID'].astype(str).values:
            nearest_time = nearest_times_df.loc[nearest_times_df['run_ID'] == int(run_id), 'first_leak_time'].values[0]
            ax.axvline(x=nearest_time, color='red', linestyle='--', label='Leak Time')

        ax.set_title(f'Run {run_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel(column)

        # Adjust x-axis formatting based on the time range
        if (data[time_column].max() - data[time_column].min()).days > 1:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.legend()
        ax.grid(True)

    # Remove empty subplots
    for idx in range(len(runs_files), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_all_segments_with_leak_index(segments_dict, columns=['Current'], leak_column='leak', y_scale=None):
    """
    Plots all segments with optional leak indicators and Y-axis scaling.

    Parameters:
    - segments_dict: Dictionary of segments to plot.
    - columns: List of column names to plot.
    - leak_column: Column indicating leak occurrences (default: 'leak').
    - y_scale: Tuple (ymin, ymax) to set fixed Y-axis scale; if None, scales dynamically.
    """
    num_segments = len(segments_dict)
    num_cols = 3  # Number of columns for plots
    num_rows = math.ceil(num_segments / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, (segment_key, segment_df) in enumerate(segments_dict.items()):
        ax = axes[idx]

        for column in columns:
            if column in segment_df.columns and pd.api.types.is_numeric_dtype(segment_df[column]):
                ax.plot(segment_df.index, segment_df[column], label=column)

        if leak_column in segment_df.columns and segment_df[leak_column].any():
            first_leak_index = segment_df.loc[segment_df[leak_column]].index[0]
            ax.axvline(x=first_leak_index, color='red', linestyle='--', label='First Leak')

        ax.set_title(f'Segment {segment_key}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Values')

        if y_scale is not None:
            ax.set_ylim(y_scale)

        ax.legend()
        ax.grid(True)

    for idx in range(num_segments, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_loss_function(losses, cutoff=100, ansatz_label = 'hamming'):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(losses[cutoff:])), losses[cutoff:], label=f"Ansatz: {ansatz_label}")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()


def plot_current_and_power(run_id, runs_files):
    """
    Plots 'Current' and 'Power' versus 'Time' for the specified run_id,
    with a vertical red dashed line indicating the leak time.

    Parameters:
        run_id (int or str): The run_ID to be plotted.
        runs_files (dict): Dictionary containing run data DataFrames.
                           Keys should be formatted as 'run_<run_id>'.
    """
    plt.style.use('default')
    run_key = f"run_{run_id}"

    if run_key not in runs_files:
        print(f"Run with run_id '{run_id}' not found in runs_files.")
        return

    df = runs_files[run_key]

    for col in ['Time', 'Current', 'Power', 'leak']:
        if col not in df.columns:
            print(f"Column '{col}' not found in the DataFrame for run {run_id}.")
            return

    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'])

    leak_times = df.loc[df['leak'] == True, 'Time']
    leak_time = leak_times.min() if not leak_times.empty else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    plot_color = "tab:blue"

    ax1.plot(df['Time'], df['Current'], label='Current', color=plot_color)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Current', fontsize=14)
    ax1.set_title(f"Current vs Time", fontsize=16)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45, labelsize=12)

    if leak_time:
        ax1.axvline(leak_time, color='red', linestyle='--', linewidth=2, label='Leak Time')
        ax1.legend()

    ax2.plot(df['Time'], df['Power'], label='Power', color=plot_color)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Power', fontsize=14)
    ax2.set_title(f"Power vs Time", fontsize=16)
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45, labelsize=12)

    if leak_time:
        ax2.axvline(leak_time, color='red', linestyle='--', linewidth=2, label='Leak Time')
        ax2.legend()

    plt.tight_layout()
    plt.show()


def clean_data(runs_metadata,runs_files):
    # Convert 'Time' columns in runs_files to timezone-naive
    for name, df in runs_files.items():
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize(None)

    runs_metadata = runs_metadata.query('4045 <= run_ID <= 4080')

    # Convert 'leak_time' column in runs_metadata to timezone-naive
    if 'leak_time' in runs_metadata.columns:
        runs_metadata['leak_time'] = pd.to_datetime(runs_metadata['leak_time']).dt.tz_localize(None)
    # Remove the 'Target' column from all DataFrames in runs_files
    for name, df in runs_files.items():
        if 'Target' in df.columns:
            df.drop(columns=['Target'], inplace=True)
    # Create 'leak' column
    for name, df in runs_files.items():
        df['leak'] = False

    # Remove the 'Target' column from all DataFrames in runs_files
    for name, df in runs_files.items():
        if 'Target' in df.columns:
            df.drop(columns=['Target'], inplace=True)

    # Correct the column name for 'leak time' based on the actual column name in runs_metadata
    leak_data = runs_metadata[(runs_metadata['leak'] == True)]

    # Extract 'run_ID' and 'leak time' columns from the filtered data
    run_ids = leak_data['run_ID'].values
    leak_times = leak_data['leak time'].values

    # Update the 'leak' column in runs_files based on 'leak time' from runs_metadata
    for run_id, leak_time in zip(run_ids, leak_times):
        run_key = f'run_{run_id}'
        if run_key in runs_files and not pd.isna(leak_time):
            # Find rows where 'Time' is greater than or equal to 'leak_time'
            runs_files[run_key].loc[runs_files[run_key]['Time'] >= leak_time, 'leak'] = True

    return runs_metadata,runs_files


def create_segments(runs_files):
    all_runs_combined = pd.concat(runs_files.values(), ignore_index=True)
    segments_dict = {}
    segment_counter = 0

    current_segment = []

    for _, row in all_runs_combined.iterrows():
        if 10 <= row['Current']:
            current_segment.append(row)
        else:
            if current_segment:  # If a segment exists, save it
                segments_dict[f'{segment_counter}'] = pd.DataFrame(current_segment).reset_index(drop=True)
                segment_counter += 1
                current_segment = []

    if current_segment:
        segments_dict[f'segment_{segment_counter}'] = pd.DataFrame(current_segment).reset_index(drop=True)

    for key, segment in segments_dict.items():
        segments_dict[key] = segment[(segment['Current'] <= 50) & (segment['Current'] >= 20)].reset_index(drop=True)
    # At startup, the motor often causes large fluctuations in current. We will manually remove them
    for key in segments_dict:
        segments_dict[key] = segments_dict[key].iloc[1000:].reset_index(drop=True)

    for key in segments_dict:
        segments_dict[key]['leak'] = segments_dict[key]['leak'].apply(lambda x: 1 if x else 0)

    segments_dict['15'] = segments_dict['15'].iloc[4000:].reset_index(drop=True)

    return segments_dict


def gaussian_filter(segments_dict):
    for key in segments_dict:
        segments_dict[key]['Gaussian_Smoothed_Current'] = gaussian_filter1d(segments_dict[key]['Current'].values, sigma=100)
        segments_dict[key]['Gaussian_Smoothed_Power'] = gaussian_filter1d(segments_dict[key]['Power'].values, sigma=100)
    return segments_dict


def wavelet_denoising(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet)
    threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(data)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    return pywt.waverec(coeffs, wavelet)[:len(data)]


def create_X_Y_set(combined_data, window_size=10, stride=5, feature_names=['Gaussian_Smoothed_Current']):
    """
    Create X, Y datasets from the combined_data DataFrame.

    Args:
        combined_data (pd.DataFrame): Input data containing features and labels.
        window_size (int): Size of the sliding window.
        stride (int): Step size for the sliding window.
        feature_names (list): List of feature column names to include in the dataset.

    Returns:
        X (np.array): Array of features with shape (num_windows, window_size * num_features).
        y (np.array): Array of labels with shape (num_windows,).
    """
    windows = []
    labels = []

    for i in range(0, len(combined_data) - window_size, stride):
        window = combined_data.iloc[i:i + window_size]
        features = window[feature_names].values

        label = window['leak'].any()

        windows.append(features)
        labels.append(int(label))

    X = np.array(windows)

    X = X.reshape(X.shape[0], -1)
    y = np.array(labels)
    return X, y


def create_train_and_test_set(X, y, test_size=0.2):
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_balance = np.bincount(y_train) / len(y_train)
    test_balance = np.bincount(y_test) / len(y_test)

    print(f"Balancing Train data: {train_balance}")
    print(f"Balancing Test data: {test_balance}")

    return X_train, X_test, y_train, y_test


def append_to_json(file_path, new_data):
    """
    Safely appends a new entry to a JSON file, ensuring the JSON structure remains valid.

    Args:
        file_path (str): Path to the JSON file.
        new_data (dict): The new data to append.
    """
    data = []

    # If file exists, load existing data
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                print("Error: JSON file is corrupted or empty.")

    data.append(new_data)

    # Save updated JSON
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data appended to {file_path}")


def load_qnn_output(file_path):
    """
    Load the QNN output JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries containing QNN configurations and losses.
    qnn_data = load_qnn_output(output_file)
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # Ensure the data is in a list format
        if not isinstance(data, list):
            data = [data]
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return []



