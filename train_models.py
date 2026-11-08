"""
Train five classifiers on the combined UCI Heart Disease dataset.

Run:
    python download_data.py
    python train_models.py

Outputs:
    models/*.joblib
    test_data.csv
    outputs/model_comparison.csv
    outputs/classification_reports.txt
    outputs/dataset_summary.txt
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "heart_disease_combined.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "target"
SOURCE_COLUMN = "source"

# The assignment requires at least 12 input features. These are the 13
# commonly used clinical predictors in the processed UCI Heart Disease data.
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Run `python download_data.py` first."
    )

data = pd.read_csv(DATA_PATH)
missing_columns = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Source is retained in the master file for provenance but is NOT a model
# input. The target is also excluded from X.
X = data[FEATURES].copy()
y = data[TARGET].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError(
        f"Expected binary target labels 0 and 1; found {sorted(y.unique())}"
    )

# Stratification preserves the class proportions in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Imputation and scaling/encoding are fitted only on the training set.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
], remainder="drop")

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, min_samples_leaf=2
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(
        probability=True, random_state=42, kernel="rbf"
    )
}

rows = []
report_sections = []

for display_name, classifier in classifiers.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }
    rows.append(metrics)

    safe_name = (
        display_name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )
    model_path = MODEL_DIR / f"{safe_name}.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "model_name": display_name,
        "positive_label": 1
    }, model_path)

    report_sections.append(
        f"\n{'=' * 70}\n{display_name}\n{'=' * 70}\n"
        + classification_report(
            y_test, predictions, target_names=["No disease (0)", "Disease (1)"],
            zero_division=0
        )
    )
    print(f"Saved {display_name}: {model_path}")
    print(metrics)

comparison = pd.DataFrame(rows).sort_values(
    by=["F1", "AUC"], ascending=False
)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

# The assignment app evaluates a held-out test CSV; it does not retrain.
test_output = X_test.copy()
test_output[TARGET] = y_test
test_output.to_csv(ROOT / "test_data.csv", index=False)

with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_sections))

summary = {
    "total_records": int(len(data)),
    "number_of_predictors": int(len(FEATURES)),
    "train_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "target_distribution": {
        str(k): int(v) for k, v in y.value_counts().sort_index().items()
    },
    "random_state": 42,
    "test_size": 0.20,
    "features": FEATURES,
    "excluded_from_model": [SOURCE_COLUMN, "num"]
}
with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(OUTPUT_DIR / "dataset_summary.txt", "w", encoding="utf-8") as f:
    f.write("Combined UCI Heart Disease dataset summary\n")
    f.write(json.dumps(summary, indent=2))

print("\nMODEL COMPARISON (held-out test set)")
print(comparison.to_string(index=False))
print("\nTest data saved to:", ROOT / "test_data.csv")
print("Metrics saved to:", OUTPUT_DIR / "model_comparison.csv")
print(
    "\nUse the actual generated metrics in your submitted report. "
    "Do not copy example scores from another implementation."
)
