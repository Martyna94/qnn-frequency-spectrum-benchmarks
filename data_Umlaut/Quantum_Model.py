from data_Umlaut.Utils import load_csv_files, create_X_Y_set, create_train_and_test_set, \
    append_to_json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

from qnn.constants import BINARY_CROSS_ENTROPY
from qnn import QNN

# ---------------------------------
# Configuration
# ---------------------------------
# Data parameters
FEATURE_NAMES = ['Gaussian_Smoothed_Current']
WINDOW_SIZE = 10
STRIDE = 5
TEST_SIZE = 0.2

# Model architecture
ENCODING = 'hamming'
R = 1
L = 6
ansatz = 'sequential'

# Training parameters
TRAINABLE_BLOCK_LAYERS = 5
MAX_ITER = 3000
STEP_SIZE = 0.001
SEED = 42

# Metrics parameters
METRICS_OUTPUT_FILE_NAME = 'test.txt'
MODEL_OUTPUT_FILE_NAME = 'test.json'

# if needed
FILE_PATHS = [
    'segments_output/0.csv',
    'segments_output/1.csv',
    'segments_output/14.csv',
    'segments_output/13.csv'
]

# ---------------------------------
# Load and prepare data
# ---------------------------------

# 1. Loaded all segments
_, data_frames = load_csv_files('../data_Umlaut/segments_output')
combined_data = pd.concat(data_frames.values(), ignore_index=True)

# Optional: (uncomment if needed) List of uploaded files
#data_frames = [pd.read_csv(file_path) for file_path in FILE_PATHS]
#combined_data = pd.concat(data_frames, ignore_index=True)

X,y = create_X_Y_set(combined_data, window_size=WINDOW_SIZE, stride=STRIDE, feature_names=FEATURE_NAMES)
print(f"Data shapes - X: {X.shape}, y: {y.shape}") # X: (num_windows,num_features * window_size)

# ---------------------------------
#  Train/Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = create_train_and_test_set(X, y, test_size=TEST_SIZE)
print(f"Data shapes - X: {X_train.shape[1]}, y: {y_train.shape}") # X: (num_windows,num_features * window_size)

# ---------------------------------
# Initialize, train the QNN
# ---------------------------------
qnn = QNN(
    R=R,
    L=L,
    N=X_train.shape[1],
    ansatz=ansatz,
    encoding=ENCODING,
    loss_fn=BINARY_CROSS_ENTROPY,
    trainable_block_layers=TRAINABLE_BLOCK_LAYERS ,
    save_weights=True,
    save_losses=True,
    seed=SEED,
    max_iter=MAX_ITER,
    step_size=STEP_SIZE,
    verbose=True
)
import gc
gc.collect()

qnn.fit(X_train, y_train)

# ---------------------------------
# Evaluate on the training set
# ---------------------------------
y_train_pred = qnn.predict(X_train) > 0.5
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print(f"Train Confusion matrix {train_conf_matrix}")

# ---------------------------------
# Evaluate on the test set
# ---------------------------------
y_test_pred = qnn.predict(X_test) > 0.5
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, qnn.predict(X_test))  # Use probabilities for ROC-AUC
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f"Test Metrics:\n")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"    Precision: {test_precision:.4f}\n")
print(f"    Recall: {test_recall:.4f}\n")
print(f"    F1-Score: {test_f1:.4f}\n")
print(f"    ROC-AUC: {test_roc_auc:.4f}\n")
print(f"    Test Confusion matrix {test_conf_matrix}")

# Optional: Plot the loss function (uncomment if needed)
#plot_loss_function(qnn.losses,cutoff=200,ansatz_label=encoding)

# ---------------------------------
# Log the results to a text file
# ---------------------------------
new_data = {
    "Model_Configuration": {
        "R": qnn.R,
        "L": qnn.L,
        "Encoding": qnn.encoding,
        "Step_Size": qnn.step_size,
        "Max_Iter": qnn.max_iter,
        "Trained_weights_": qnn.trained_weights_.tolist(),
        "Loss_Values": [float(loss) for loss in qnn.losses],  # Loss values during training
    },

}

append_to_json(MODEL_OUTPUT_FILE_NAME, new_data)

output_file = METRICS_OUTPUT_FILE_NAME

with open(output_file, "a") as f:
    f.write(f"Model: R: {qnn.R}, L: {qnn.L}, encoding: {qnn.encoding}, step_size: {qnn.step_size}\n")
    f.write(f"Loss value:\n{min(qnn.losses)}\n")
    f.write(f"Training Accuracy: {train_accuracy * 100:.2f}%\n")
    f.write(f"Training Accuracy: {train_accuracy * 100:.2f}%\n")
    f.write(f"Train Confusion Matrix:\n{train_conf_matrix}\n")

    # Test metrics
    f.write(f"Test Metrics:\n")
    f.write(f"Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"F1-Score: {test_f1:.4f}\n")
    f.write(f"ROC-AUC: {test_roc_auc:.4f}\n")
    f.write(f"Confusion Matrix:\n{test_conf_matrix}\n\n")

    f.write("\n")  # Add a newline for separation

#%%
