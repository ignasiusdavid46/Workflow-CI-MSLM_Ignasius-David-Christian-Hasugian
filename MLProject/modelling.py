"""
Dijalankan oleh MLflow Project via GitHub Actions CI.
Mendukung DagsHub remote tracking dan menyimpan artefak model.

Environment variables yang dibutuhkan (via GitHub Secrets):
    DAGSHUB_USERNAME
    DAGSHUB_REPO
    MLFLOW_TRACKING_USERNAME
    MLFLOW_TRACKING_PASSWORD
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

# Konfigurasi
TRAIN_PATH = "winequality_preprocessing/winequality_train.csv"
TEST_PATH  = "winequality_preprocessing/winequality_test.csv"
TARGET     = "quality_label"
EXPERIMENT = "WineQuality-CI"

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "ignasiusdavid46")
DAGSHUB_REPO     = os.getenv("DAGSHUB_REPO",
                              "Eksperimen_SML_Ignasius-David-Christian-Hasugian")

# DagsHub + MLflow
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)

mlflow.set_experiment(EXPERIMENT)

# Load Data
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test  = test.drop(columns=[TARGET])
y_test  = test[TARGET]

FEATURES = X_train.columns.tolist()
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Artefak helpers
def plot_confusion_matrix(y_true, y_pred) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Good", "Good"],
                yticklabels=["Not Good", "Good"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return path


def plot_feature_importance(model, feature_names) -> str:
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(imp)), imp[idx], color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right", fontsize=8)
    ax.set_title("Feature Importance", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "feature_importance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return path


def save_classification_report(y_true, y_pred) -> str:
    report = classification_report(y_true, y_pred,
                                   target_names=["Not Good", "Good"],
                                   output_dict=True)
    path = os.path.join(tempfile.gettempdir(), "classification_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


# GridSearch + Training
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [6, 8, 10],
    "min_samples_split": [2, 5],
}

print("Memulai GridSearchCV ...")
gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
)
gs.fit(X_train, y_train)

best_model  = gs.best_estimator_
best_params = gs.best_params_
print(f"Best params: {best_params}  |  CV F1: {gs.best_score_:.4f}")

# Evaluasi
y_pred      = best_model.predict(X_test)
y_prob      = best_model.predict_proba(X_test)[:, 1]
cv_acc      = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")

metrics = {
    "accuracy":         accuracy_score(y_test, y_pred),
    "f1_score":         f1_score(y_test, y_pred),
    "precision":        precision_score(y_test, y_pred),
    "recall":           recall_score(y_test, y_pred),
    "roc_auc":          roc_auc_score(y_test, y_prob),
    "cv_f1_mean":       gs.best_score_,
    "cv_accuracy_mean": cv_acc.mean(),
    "cv_accuracy_std":  cv_acc.std(),
}
for k, v in metrics.items():
    print(f"  {k:22s}: {v:.4f}")

# MLflow Manual Logging
existing_run_id = os.getenv("MLFLOW_RUN_ID")

with mlflow.start_run(run_name="RF-CI-Advanced") as run:

    # Params
    mlflow.log_params(best_params)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Metrics
    mlflow.log_metrics(metrics)

    # Model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="WineQuality-RF-CI",
        input_example=X_test.iloc[:3],
    )

    # Artefak tambahan
    mlflow.log_artifact(plot_confusion_matrix(y_test, y_pred), "plots")
    mlflow.log_artifact(plot_feature_importance(best_model, FEATURES), "plots")
    mlflow.log_artifact(save_classification_report(y_test, y_pred), "reports")

    mlflow.set_tags({
        "author":    "Ignasius-David-Christian-Hasugian",
        "framework": "scikit-learn",
        "task":      "binary-classification",
        "trigger":   "github-actions-ci",
    })

    print(f"\nRun ID: {run_id}")
    print("Semua artefak tersimpan ke DagsHub MLflow!")

# Simpan model lokal untuk Docker build
MODEL_DIR = "model_output"
os.makedirs(MODEL_DIR, exist_ok=True)
mlflow.sklearn.save_model(best_model, MODEL_DIR)
print(f"Model disimpan lokal: {MODEL_DIR}/")
