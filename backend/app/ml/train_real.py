"""
PhishGuard AI — Real Dataset ML Training Pipeline
Trains the URL phishing detection model using the UCI PhiUSIIL Phishing URL Dataset.

Key design:
  - Uses the EXISTING extract_url_features() function from url_detector.py
  - This guarantees training features = inference features
  - The model is saved to the EXISTING model path used by the FastAPI backend

UCI PhiUSIIL Dataset labels:
  Label 1 = legitimate
  Label 0 = phishing

Our internal convention:
  label 1 = phishing
  label 0 = legitimate

We invert the labels during loading.
"""

import os
import sys
import zipfile
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

# Add project root to path so imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.config import URL_MODEL_PATH, URL_SCALER_PATH
from backend.app.ml.train import URL_FEATURE_COLUMNS
from backend.app.detectors.url_detector import extract_url_features


# --- Configuration ---
DATASET_ZIP = "data/phiusiil_dataset.zip"
DATASET_CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
EXTRACTED_DIR = "data/phiusiil_extracted"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def extract_dataset():
    """Extract the downloaded ZIP file."""
    if not os.path.exists(DATASET_ZIP):
        print(f"[ERROR] Dataset ZIP not found at {DATASET_ZIP}")
        print("[ERROR] Download it first from: https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip")
        sys.exit(1)

    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    # Find the CSV — might be directly in zip or in a subdirectory
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        print(f"[ML] ZIP contains: {zf.namelist()[:20]}...")
        print(f"[ML] CSV files found: {csv_files}")
        zf.extractall(EXTRACTED_DIR)

    # Find the actual CSV path
    for root, dirs, files in os.walk(EXTRACTED_DIR):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)

    print("[ERROR] No CSV file found in the extracted ZIP.")
    sys.exit(1)


def load_and_inspect_dataset(csv_path: str) -> pd.DataFrame:
    """Load the UCI dataset and inspect its structure."""
    print(f"\n{'='*60}")
    print("Loading UCI PhiUSIIL Phishing URL Dataset")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)

    print(f"\n[ML] Dataset shape: {df.shape}")
    print(f"[ML] Columns ({len(df.columns)}): {list(df.columns[:10])}...")
    print(f"\n[ML] First 5 rows (URL + label columns):")

    # Find URL and label columns
    url_col = None
    label_col = None

    for col in df.columns:
        if col.lower() == "url":
            url_col = col
        if col.lower() == "label":
            label_col = col

    if url_col is None:
        # Try common alternatives
        for col in df.columns:
            if "url" in col.lower():
                url_col = col
                break

    if label_col is None:
        for col in df.columns:
            if "label" in col.lower() or col.lower() == "phishing":
                label_col = col
                break

    print(f"[ML] URL column: '{url_col}'")
    print(f"[ML] Label column: '{label_col}'")

    if url_col and label_col:
        print(df[[url_col, label_col]].head(10).to_string())
        print(f"\n[ML] Label distribution (raw UCI labels):")
        print(df[label_col].value_counts().to_string())
    else:
        print("[ML] All columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col} (dtype={df[col].dtype}, sample={df[col].iloc[0]})")

    return df


def clean_data(df: pd.DataFrame, url_col: str, label_col: str) -> pd.DataFrame:
    """Clean the dataset: remove duplicates, invalid rows."""
    print(f"\n{'='*60}")
    print("Data Cleaning")
    print(f"{'='*60}")

    initial_count = len(df)
    print(f"[ML] Initial records: {initial_count}")

    # Remove rows with missing URL or label
    df = df.dropna(subset=[url_col, label_col])
    print(f"[ML] After removing NaN URL/label: {len(df)}")

    # Remove empty URLs
    df = df[df[url_col].str.strip().str.len() > 0]
    print(f"[ML] After removing empty URLs: {len(df)}")

    # Remove duplicate URLs
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[url_col], keep="first")
    print(f"[ML] After removing duplicate URLs: {len(df)} (removed {before_dedup - len(df)})")

    # Ensure labels are valid (0 or 1)
    valid_labels = df[label_col].isin([0, 1])
    df = df[valid_labels]
    print(f"[ML] After filtering valid labels: {len(df)}")

    print(f"[ML] Total removed: {initial_count - len(df)} records")
    return df.reset_index(drop=True)


def extract_features_batch(urls: pd.Series, batch_size: int = 5000) -> pd.DataFrame:
    """Extract features from URLs using the EXISTING extract_url_features() function.
    This ensures training features exactly match inference features."""
    print(f"\n{'='*60}")
    print("Feature Extraction (using existing extract_url_features)")
    print(f"{'='*60}")
    print(f"[ML] Processing {len(urls)} URLs...")
    print(f"[ML] Features: {URL_FEATURE_COLUMNS}")

    all_features = []
    total = len(urls)
    start_time = time.time()

    for i, url in enumerate(urls):
        try:
            features = extract_url_features(str(url))
            # Extract only the features we need, in the correct order
            row = {col: features.get(col, 0) for col in URL_FEATURE_COLUMNS}
            all_features.append(row)
        except Exception:
            # If feature extraction fails for a URL, use zeros
            all_features.append({col: 0 for col in URL_FEATURE_COLUMNS})

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"[ML] Processed {i+1}/{total} URLs ({(i+1)/total*100:.1f}%) — "
                  f"{rate:.0f} URLs/sec, ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"[ML] Feature extraction complete: {total} URLs in {elapsed:.1f}s "
          f"({total/elapsed:.0f} URLs/sec)")

    return pd.DataFrame(all_features)


def train_model(X_train, y_train):
    """Train Random Forest classifier."""
    print(f"\n{'='*60}")
    print("Training Random Forest Classifier")
    print(f"{'='*60}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    print(f"[ML] Model config: n_estimators=200, max_depth=20, "
          f"min_samples_split=5, min_samples_leaf=2, class_weight=balanced")
    print(f"[ML] Training on {len(X_train)} samples...")

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"[ML] Training complete in {elapsed:.1f}s")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test set."""
    print(f"\n{'='*60}")
    print("Model Evaluation (on unseen test data)")
    print(f"{'='*60}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Legitimate", "Phishing"])

    print(f"\n[ML] Accuracy:  {acc:.4f}")
    print(f"[ML] Precision: {prec:.4f}")
    print(f"[ML] Recall:    {rec:.4f}")
    print(f"[ML] F1-score:  {f1:.4f}")
    print(f"\n[ML] Confusion Matrix:")
    print(f"  TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"  FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")
    print(f"\n[ML] Classification Report:\n{report}")

    # Feature importance
    importances = dict(zip(URL_FEATURE_COLUMNS, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("[ML] Feature Importances:")
    for feat, imp in sorted_imp:
        print(f"  {feat:<30} {imp:.4f}")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
    }


def save_model(model, scaler):
    """Save model and scaler to the existing paths."""
    print(f"\n{'='*60}")
    print("Saving Model")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(URL_MODEL_PATH), exist_ok=True)
    joblib.dump(model, URL_MODEL_PATH)
    joblib.dump(scaler, URL_SCALER_PATH)

    model_size = os.path.getsize(URL_MODEL_PATH) / (1024 * 1024)
    scaler_size = os.path.getsize(URL_SCALER_PATH) / 1024

    print(f"[ML] Model saved to: {URL_MODEL_PATH} ({model_size:.1f} MB)")
    print(f"[ML] Scaler saved to: {URL_SCALER_PATH} ({scaler_size:.1f} KB)")


def main():
    """Main training pipeline."""
    print("\n" + "#" * 60)
    print("PhishGuard AI — Real Dataset ML Training Pipeline")
    print("Dataset: UCI PhiUSIIL Phishing URL Dataset")
    print("#" * 60)

    # Step 1: Extract dataset
    csv_path = extract_dataset()
    print(f"[ML] CSV found at: {csv_path}")

    # Step 2: Load and inspect
    df = load_and_inspect_dataset(csv_path)

    # Identify columns
    url_col = None
    label_col = None
    for col in df.columns:
        if col.lower() == "url":
            url_col = col
        if col.lower() == "label":
            label_col = col
    if url_col is None:
        for col in df.columns:
            if "url" in col.lower() and "length" not in col.lower():
                url_col = col
                break
    if label_col is None:
        for col in df.columns:
            if "label" in col.lower():
                label_col = col
                break

    if url_col is None or label_col is None:
        print(f"[ERROR] Could not find URL column ({url_col}) or label column ({label_col})")
        print(f"[ERROR] Available columns: {list(df.columns)}")
        sys.exit(1)

    # Step 3: Clean data
    df = clean_data(df, url_col, label_col)

    # Step 4: Invert labels (UCI: 1=legit, 0=phish → Our: 1=phish, 0=legit)
    print(f"\n[ML] Inverting labels: UCI(1=legit,0=phish) → Internal(1=phish,0=legit)")
    df["internal_label"] = 1 - df[label_col].astype(int)

    n_phishing = df["internal_label"].sum()
    n_legitimate = len(df) - n_phishing
    print(f"[ML] After label inversion:")
    print(f"  Phishing (label=1): {n_phishing}")
    print(f"  Legitimate (label=0): {n_legitimate}")

    # Step 5: Extract features using existing extract_url_features()
    feature_df = extract_features_batch(df[url_col])

    # Verify feature schema matches
    print(f"\n[ML] Feature columns used: {list(feature_df.columns)}")
    print(f"[ML] Expected columns:     {URL_FEATURE_COLUMNS}")
    assert list(feature_df.columns) == URL_FEATURE_COLUMNS, \
        "Feature column mismatch between training and inference!"

    X = feature_df.values
    y = df["internal_label"].values

    # Step 6: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 7: Stratified train/test split
    print(f"\n[ML] Splitting data: {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test, "
          f"random_state={RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[ML] Train: {len(X_train)} samples "
          f"({sum(y_train)}/{len(y_train)} phishing = {sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"[ML] Test:  {len(X_test)} samples "
          f"({sum(y_test)}/{len(y_test)} phishing = {sum(y_test)/len(y_test)*100:.1f}%)")

    # Step 8: Train model
    model = train_model(X_train, y_train)

    # Step 9: Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)

    # Step 10: Save model and scaler
    save_model(model, scaler)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset:     UCI PhiUSIIL Phishing URL Dataset")
    print(f"Total URLs:  {len(df)}")
    print(f"Phishing:    {n_phishing}")
    print(f"Legitimate:  {n_legitimate}")
    print(f"Features:    {len(URL_FEATURE_COLUMNS)} (extracted by extract_url_features)")
    print(f"Split:       {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%} stratified")
    print(f"Model:       Random Forest (200 trees, max_depth=20)")
    print(f"Accuracy:    {metrics['accuracy']}")
    print(f"Precision:   {metrics['precision']}")
    print(f"Recall:      {metrics['recall']}")
    print(f"F1-score:    {metrics['f1_score']}")
    print(f"Model path:  {URL_MODEL_PATH}")
    print(f"\nThe backend will load this model automatically on startup.")

    return metrics


if __name__ == "__main__":
    main()
