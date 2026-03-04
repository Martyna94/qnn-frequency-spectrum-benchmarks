import os
import sys
import time
# Add the data_Umlaut directory to the path
sys.path.insert(0, os.path.join(os.getcwd(), 'data_Umlaut'))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Utils import create_X_Y_set, create_train_and_test_set, append_to_json, load_csv_files
from qnn import QNN
from qnn.constants import BINARY_CROSS_ENTROPY

# Configuration options
ENCODING_OPTIONS = ['hamming']
# Use [None] for full-batch training; empty list skips all runs.
BATCH_SIZE_OPTIONS = [None]
CONFIGURATIONS = [
    (2, 1, 3, 500, 0.001),  # R,L, TRAINABLE_BLOCK_LAYERS, MAX_ITER, STEP_SIZE
]

# Higher-dimensional encoding parameter s
# For golomb and turnpike, we need q = log2(len(s)) to divide R
# Using s = [0, 1, 2, 3] gives q = 2, so R must be divisible by 2
ENCODING_S_PARAMETER = [0, 1, 2, 3]

# File paths
BEST_METRICS_FILE = "Best_metrics_exp_ent_3_area_2.csv"
MODEL_OUTPUT_FILE = "exp_ent_3_area_2.json"
ALL_RUN_METRICS_FILE = "exp_ent_3_area_2.csv"
# Load the dataset
print("Loading dataset...")
# Handle both regular Python execution and Jupyter/IPython environments
try:
    # For regular Python script execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # For Jupyter/IPython execution - assume we're in the data_Umlaut directory
    script_dir = os.getcwd()
    # If we're not in data_Umlaut, try to find it
    if not os.path.basename(script_dir) == 'data_Umlaut':
        # Try common locations
        possible_paths = [
            os.path.join(script_dir, 'data_Umlaut'),
            os.path.join(os.path.dirname(script_dir), 'data_Umlaut'),
            'C:\\Users\\marty\\Projects_All\\womanium\\data_Umlaut'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                script_dir = path
                break

segments_output_path = os.path.join(script_dir, 'dataset\\segments_output')
print(f"Looking for data in: {segments_output_path}")

# Check if the path exists
if not os.path.exists(segments_output_path):
    print(f"ERROR: segments_output directory not found at {segments_output_path}")
    print(f"Current working directory: {os.getcwd()}")
    print("Please make sure you're running this script from the correct directory.")
    exit(1)

#df_rms = pd.read_csv("df_rms.csv")
_, data_frames = load_csv_files(segments_output_path)
print(f"Loaded {len(data_frames)} CSV files")
combined_data = pd.concat(data_frames.values(), ignore_index=True)
print(f"Combined data shape: {combined_data.shape}")
print(f"Combined data columns: {combined_data.columns.tolist()}")

all_run_metrics_list = []
best_metrics_list = []

# Initialize mean metrics storage
mean_metrics_list = []

# Create X and y
print("Creating X and y datasets...")
X, y = create_X_Y_set(combined_data, window_size=10, stride=5, feature_names=['Gaussian_Smoothed_Current'])
print(f"Data shapes after windows- X: {X.shape}, y: {y.shape}") # X: (num_windows,num_features * window_size)
print(f"Unique y values: {np.unique(y)}")
print(f"Y value counts: {np.bincount(y.astype(int))}")

# Split and preprocess the data
print("Splitting data...")
X_train, X_test, y_train, y_test = create_train_and_test_set(X, y, test_size=0.2)
print(f"Data shapes - N: {X_train.shape[1]}, y: {y_train.shape}") # X: (num_windows,num_features * window_size)
print(f"X_train min/max: {X_train.min():.4f}/{X_train.max():.4f}")
print(f"X_test min/max: {X_test.min():.4f}/{X_test.max():.4f}")

# Check for class imbalance issues
train_balance = np.bincount(y_train.astype(int)) / len(y_train)
test_balance = np.bincount(y_test.astype(int)) / len(y_test)
print(f"Training set class balance: {train_balance}")
print(f"Test set class balance: {test_balance}")

# Warning if data is severely imbalanced
if len(train_balance) == 1:
    print("WARNING: Only one class in training data - this may cause training issues!")
    print("Consider using a different dataset or adjusting the window/stride parameters.")

for encoding in ENCODING_OPTIONS:
    for R, L, TRAINABLE_BLOCK_LAYERS, MAX_ITER, STEP_SIZE in CONFIGURATIONS:
        for batch_size_val in BATCH_SIZE_OPTIONS: # Dodana pętla po batch_size
            print(f"Running for Encoding={encoding}, R={R}, L={L}, Layers={TRAINABLE_BLOCK_LAYERS}, Iter={MAX_ITER}, Step={STEP_SIZE}, BatchSize={batch_size_val}")
            # Run the configuration 3 times and store metrics
            metrics_runs = []
            for run in range(1):
                print(f"    Run {run + 1}/1...")
                print(f"    Initializing QNN with encoding={encoding}, R={R}, L={L}...")
                start_time = time.time()  # Start measuring time
                # Initialize and train the QNN model
                qnn = QNN(
                    R=R,
                    L=L,
                    N=X_train.shape[1],
                    ansatz='sequential',
                    encoding=encoding,
                    loss_fn=BINARY_CROSS_ENTROPY,
                    trainable_block_layers=TRAINABLE_BLOCK_LAYERS,
                    save_weights=True,
                    save_losses=True,
                    seed=42 + run,  # Change seed for each run
                    max_iter=MAX_ITER,
                    batch_size=batch_size_val,
                    verbose=True,
                    s=ENCODING_S_PARAMETER  # Required for golomb and turnpike encodings
                )
                print(f"    QNN initialized successfully. Starting training...")
                qnn.fit(X_train, y_train)
                print(f"    Training completed!")
                end_time = time.time()  # End measuring time
                fit_time = end_time - start_time  # Calculate total fit time
                print(f"Total time for fitting: {fit_time:.2f} seconds,Run {run}, R: {R}, L: {L}, encoding: {encoding}, max iter: {MAX_ITER}, step_size: {STEP_SIZE}")

                # Evaluate on the test set
                y_test_pred = qnn.predict(X_test) > 0.5
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred)
                test_recall = recall_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                test_roc_auc = roc_auc_score(y_test, qnn.predict(X_test))

                # Store metrics for this run
                metrics_run = {
                    "R": R,
                    "L": L,
                    "Encoding": encoding,
                    "Max iter": MAX_ITER,
                    "Step size": STEP_SIZE,
                    "Run": run + 1,
                    "Batch_Size" : batch_size_val,
                    "Fit_time": fit_time,
                    "Train_Accuracy": accuracy_score(y_train, qnn.predict(X_train) > 0.5) * 100,
                    "Test_Accuracy": test_accuracy * 100,
                    "Test_Precision": test_precision,
                    "Test_Recall": test_recall,
                    "Test_F1": test_f1,
                    "Test_ROC_AUC": test_roc_auc,
                    "Min_Loss": float(min(qnn.losses)),
                    "Trained_weights_": qnn.trained_weights_.tolist(),
                    "Loss_Values": [float(loss) for loss in qnn.losses],
                }
                metrics_runs.append(metrics_run)
                all_run_metrics_list.append(metrics_run)

        # Find the best metrics based on Test Accuracy
            best_metrics = max(metrics_runs, key=lambda x: x["Test_Accuracy"])
            best_metrics_list.append(best_metrics)
            # Save intermediate results to JSON
            append_to_json(MODEL_OUTPUT_FILE, {
                "Encoding": encoding,
                "Configuration": {
                    "R": R,
                    "L": L,
                    "Layers": TRAINABLE_BLOCK_LAYERS,
                    "Max_Iter": MAX_ITER,
                    "Step_Size": STEP_SIZE,
                    "Runs": metrics_runs
                }
            })

# Save all best metrics to a CSV file
best_metrics_df = pd.DataFrame(best_metrics_list)
write_header = not os.path.exists(BEST_METRICS_FILE)
best_metrics_df.to_csv(BEST_METRICS_FILE, sep=";", index=False, mode='a', header=write_header)
print(f"Best metrics saved to {BEST_METRICS_FILE}")

# Save all run metrics to a single CSV file
all_run_metrics_df = pd.DataFrame(all_run_metrics_list)
write_header_all = not os.path.exists(ALL_RUN_METRICS_FILE)
all_run_metrics_df.to_csv(ALL_RUN_METRICS_FILE, sep=";", index=False, mode='a', header=write_header_all )
print(f"All run metrics saved to {ALL_RUN_METRICS_FILE}")

#%%
