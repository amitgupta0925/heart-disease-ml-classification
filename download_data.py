"""
Download and combine the four processed UCI Heart Disease datasets.

Sources:
- Cleveland: 303 records
- Hungary: 294 records
- Switzerland: 123 records
- VA Long Beach: 200 records

The expected combined total is 920 records. The 13 predictors are the
commonly used 13 clinical variables; `num` is converted to binary `target`.
"""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
)
SOURCES = {
    "cleveland": ("processed.cleveland.data", 303),
    "hungary": ("processed.hungarian.data", 294),
    "switzerland": ("processed.switzerland.data", 123),
    "va_long_beach": ("processed.va.data", 200),
}

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

frames = []
for source_name, (filename, expected_rows) in SOURCES.items():
    local_path = RAW_DIR / filename

    if not local_path.exists():
        url = BASE_URL + filename
        print(f"Downloading {source_name}: {url}")
        try:
            urlretrieve(url, local_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename}. Download it manually from "
                f"the UCI Heart Disease files page and save it to {local_path}. "
                f"Original error: {exc}"
            ) from exc

    # UCI processed files are comma-separated and have no header row.
    frame = pd.read_csv(
        local_path,
        header=None,
        names=COLUMNS,
        na_values=["?", " ?"],
        skipinitialspace=True
    )
    if frame.shape[1] != 14:
        raise ValueError(
            f"{filename} should contain 14 columns; found {frame.shape[1]}."
        )

    frame["source"] = source_name
    print(
        f"{source_name}: loaded {len(frame)} rows "
        f"(UCI reference count: {expected_rows})"
    )
    frames.append(frame)

combined = pd.concat(frames, ignore_index=True)

# Convert all clinical columns to numeric; invalid tokens become missing.
for col in COLUMNS:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")

# Remove rows only if the outcome itself is missing; predictors are imputed
# inside the training pipeline to avoid data leakage.
combined = combined.dropna(subset=["num"]).copy()

# Convert the original UCI outcome: 0 = absence; 1,2,3,4 = presence.
combined["target"] = (combined["num"] > 0).astype(int)
combined = combined.drop(columns=["num"])

# Zero is not physiologically plausible for these two measurements and is
# treated as missing, consistent with common cleaning of this dataset.
for col in ["trestbps", "chol"]:
    combined.loc[combined[col] == 0, col] = pd.NA

output = OUT_DIR / "heart_disease_combined.csv"
combined.to_csv(output, index=False)

print("\nCombined dataset saved:", output)
print("Rows:", len(combined))
print("Predictor columns:", 13)
print("Target distribution:")
print(combined["target"].value_counts(dropna=False).sort_index())
print("Source distribution:")
print(combined["source"].value_counts())
print("\nMissing values per column:")
print(combined.isna().sum())
if len(combined) != 920:
    print(
        "\nWARNING: The combined row count is not 920. Check the four source "
        "files and missing target values before using the assignment results."
    )
