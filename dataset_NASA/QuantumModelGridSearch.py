import os
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from data_NASA.Utils import create_X_Y_set, create_train_and_test_set, append_to_json
from qnn import QNN
from qnn.constants import BINARY_CROSS_ENTROPY

# Configuration options
ENCODING_OPTIONS = ['turnpike']
BATCH_SIZE_OPTIONS = [64] # None means full batch
CONFIGURATIONS = [
    #(1, 2, 5, 3000, 0.001),
    #(1, 2, 5, 3000, 0.005),
    (6, 1, 5, 3000, 0.001),
    (6, 1, 5, 3000, 0.005),
    #(1, 2, 5, 3000, 0.005),
    #(1, 6, 5, 5000, 0.005),
    #(1, 6, 5, 3000, 0.01),
    #(1, 6, 3, 5000, 0.001),
    #(1, 6, 3, 5000, 0.005),
    #(1, 6, 3, 3000, 0.01),
    #(2, 3, 5, 5000, 0.001),
    #(2, 3, 5, 5000, 0.005),
    #(2, 3, 5, 3000, 0.01),
    #(2, 3, 3, 5000, 0.001),
    #(2, 3, 3, 5000, 0.005),
    #(2, 3, 3, 3000, 0.01),
    #(3, 2, 5, 5000, 0.001),
    #(3, 2, 5, 5000, 0.005),
    #(3, 2, 5, 3000, 0.01),
    #(3, 2, 3, 5000, 0.001),
    #(3, 2, 3, 5000, 0.005),
    #(3, 2, 3, 3000, 0.01),
    #(6, 1, 5, 5000, 0.001),
    #(6, 1, 5, 5000, 0.005),
    #(6, 1, 5, 3000, 0.01),
    #(6, 1, 3, 5000, 0.001),
    #(6, 1, 3, 5000, 0.005),
    #(6, 1, 3, 3000, 0.01),
]

# File paths
BEST_METRICS_FILE = "exp_best_metrics_.csv"
MODEL_OUTPUT_FILE = "exp.json"
ALL_RUN_METRICS_FILE = "exp_all.csv"
# Load the dataset
df_rms = pd.read_csv("df_rms.csv")
all_run_metrics_list = []
best_metrics_list = []

mean_metrics_list = []
# Create X and y
X, y = create_X_Y_set(df_rms, window_size=1, stride=1, feature_names=['RMS_Bearing_1', 'RMS_Bearing_2', 'RMS_Bearing_3', 'RMS_Bearing_4'])
print(f"Data shapes after windows- X: {X.shape}, y: {y.shape}") # X: (num_windows,num_features * window_size)
# Split and preprocess the data
X_train, X_test, y_train, y_test = create_train_and_test_set(X, y, test_size=0.2)
print(f"Data shapes - N: {X_train.shape[1]}, y: {y_train.shape}") # X: (num_windows,num_features * window_size)

for encoding in ENCODING_OPTIONS:
    for R, L, TRAINABLE_BLOCK_LAYERS, MAX_ITER, STEP_SIZE in CONFIGURATIONS:
        for batch_size_val in BATCH_SIZE_OPTIONS:
            print(f"Running for Encoding={encoding}, R={R}, L={L}, Layers={TRAINABLE_BLOCK_LAYERS}, Iter={MAX_ITER}, Step={STEP_SIZE}, BatchSize={batch_size_val}")
            metrics_runs = []
            for run in range(3):
                print(f"    Run {run + 1}/3...")

                start_time = time.time()
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
                    seed=42 + run,
                    max_iter=MAX_ITER,
                    step_size=STEP_SIZE,
                    batch_size=batch_size_val,
                    verbose=True,
                    s = [0, 8, 15, 17, 20, 21, 31, 39]
                )
                qnn.fit(X_train, y_train)
                end_time = time.time()
                fit_time = end_time - start_time
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
                    "Batch_size": batch_size_val,
                    "Fit_time": fit_time,
                    "Run": run + 1,
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
                    "Batch_Size": batch_size_val,
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
