import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

PREDICTIONS_LOG = 'monitoring/predictions_log.csv'


df = pd.read_csv(PREDICTIONS_LOG)

# Ensure required columns exist
def check_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

check_columns(df, ['true_label', 'predicted_label'])

# If predicted_proba is missing, set to 0.5 for all (so ROC AUC can still run)
if 'predicted_proba' not in df.columns:
    df['predicted_proba'] = 0.5

y_true = df['true_label']
y_pred = df['predicted_label']
y_proba = df['predicted_proba']

# Compute metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Log to MLflow
with mlflow.start_run(run_name='periodic_monitoring'):
    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('precision', prec)
    mlflow.log_metric('recall', rec)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('roc_auc', roc_auc)
    mlflow.log_artifact(PREDICTIONS_LOG) 