import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from imblearn.over_sampling import SMOTE


def create_X_Y_set(data, window_size=10, stride=5, feature_names=['RMS_Bearing_1']):
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

    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i + window_size]
        features = window[feature_names].values

        label = window['Label'].any()

        windows.append(features)
        labels.append(int(label))

    X = np.array(windows)

    X = X.reshape(X.shape[0], -1)
    y = np.array(labels)
    return X, y


def create_train_and_test_set(X, y, test_size=0.2):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Scale the data
    X_train_smote = scaler.fit_transform(X_train_smote)
    X_test = scaler.transform(X_test)

    # Check class balance after SMOTE
    train_balance = np.bincount(y_train_smote) / len(y_train_smote)
    test_balance = np.bincount(y_test) / len(y_test)

    print(f"Balancing Train data after SMOTE: {train_balance}")
    print(f"Balancing Test data: {test_balance}")

    return X_train_smote, X_test, y_train_smote, y_test


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


def load_qnn_output(file_path: object) -> object:
    """
    Load the QNN output JSON file with detailed debugging.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)

        # Ensure it's in a list format
        if not isinstance(data, list):
            data = [data]
        return data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

    except json.JSONDecodeError as e:
        print(f"JSON decoding error at line {e.lineno}, column {e.colno}: {e.msg}")
        print("Invalid JSON content snippet:\n", content[max(e.pos-50, 0):e.pos+50])
        return []


def plot_losses(qnn_data):
    """
    Plot the losses for each QNN configuration stored in the JSON file.

    Args:
        qnn_data (list): A list of dictionaries containing QNN configurations and losses.
    plot_losses(qnn_data)
    """
    if not qnn_data:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    for idx, entry in enumerate(qnn_data):
        losses = entry.get("Loss_Values", [])
        config_label = (
            f"R={entry['Model_Configuration']['R']}, "
            f"L={entry['Model_Configuration']['L']}, "
            f"Encoding={entry['Model_Configuration']['Encoding']}"
        )
        plt.plot(losses, label=f"Config {idx + 1}: {config_label}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Values During QNN Training")
    plt.legend()
    plt.grid(True)
    plt.show()

