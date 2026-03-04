from data_NASA.Utils import create_X_Y_set, create_train_and_test_set, append_to_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

WINDOW_SIZE = 1
STRIDE = 1
FEATURE_NAMES = ['RMS_Bearing_1','RMS_Bearing_2','RMS_Bearing_3','RMS_Bearing_4']

ENCODING = 'ternary'
R = 2
L = 2
TRAINABLE_BLOCK_LAYERS = 5
MAX_ITER = 3000
STEP_SIZE = 0.01


METRICS_OUTPUT_FILE = "test.csv"
MODEL_OUTPUT_FILE = 'test.json'


df_rms = pd.read_csv('df_rms.csv')

X,y = create_X_Y_set(df_rms, window_size=WINDOW_SIZE,stride=STRIDE,feature_names=FEATURE_NAMES)
print(f"Data shapes - X: {X.shape}, y: {y.shape}")

X_train, X_test, y_train, y_test = create_train_and_test_set(X, y, test_size=0.4)

from qnn import QNN
from qnn.constants import BINARY_CROSS_ENTROPY
qnn = QNN(
    R=R,
    L=L,
    N=X_train.shape[1],
    ansatz='sequential',
    encoding=ENCODING,
    loss_fn=BINARY_CROSS_ENTROPY,
    trainable_block_layers=TRAINABLE_BLOCK_LAYERS,
    save_weights=True,
    save_losses=True,
    seed=42,
    max_iter=MAX_ITER,
    step_size=STEP_SIZE,
    verbose=True
)

qnn.fit(X_train, y_train)

# --------------------------------
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
#plot_loss_function(qnn.losses,cutoff=20,ansatz_label=ENCODING)

model_output_file = MODEL_OUTPUT_FILE
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

append_to_json(model_output_file, new_data)
print(f"Data appended to {model_output_file}")

# ---------------------------------
# Prepare metrics for CSV output
# ---------------------------------
metrics = {
    "R": qnn.R,
    "L": qnn.L,
    "Encoding": qnn.encoding,
    "Step_Size": qnn.step_size,
    "Max_Iter": qnn.max_iter,
    "Min_Loss": min(qnn.losses),
    "Train_Accuracy": train_accuracy * 100,
    "Test_Accuracy": test_accuracy * 100,
    "Test_Precision": test_precision,
    "Test_Recall": test_recall,
    "Test_F1": test_f1,
    "Test_ROC_AUC": test_roc_auc
}

metrics_df = pd.DataFrame([metrics])

metrics_output_csv = METRICS_OUTPUT_FILE

# Save metrics to a CSV file (append mode to keep history, header only if file doesn't exist)
metrics_df.to_csv(metrics_output_csv, sep=";", index=False, mode='a', header=not pd.io.common.file_exists(metrics_output_csv))

print(f"Metrics saved to {metrics_output_csv}")

#%%
