"""
PhishGuard AI — ML Training Pipeline
Trains separate Random Forest models for URL and message classification.
NOTE: This uses a hackathon prototype dataset. Results require validation
on larger real-world datasets before production deployment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from backend.app.config import (
    URL_MODEL_PATH, MESSAGE_MODEL_PATH,
    URL_SCALER_PATH, MESSAGE_VECTORIZER_PATH,
)
from backend.app.ml.dataset import generate_url_dataset, generate_message_dataset


# Feature columns for URL model (must match url_detector.extract_url_features keys)
URL_FEATURE_COLUMNS = [
    "url_length", "has_https", "has_ip_address", "has_at_symbol",
    "dot_count", "hyphen_count", "subdomain_count", "path_length",
    "query_length", "has_suspicious_tld", "suspicious_keyword_count",
    "is_shortened", "has_brand_impersonation", "digit_count_in_domain",
    "has_double_slash_redirect", "special_char_count",
]


def train_url_model() -> dict:
    """Train a Random Forest classifier for URL phishing detection."""
    print("\n" + "=" * 60)
    print("Training URL Phishing Detection Model")
    print("=" * 60)

    # Load or generate dataset
    dataset_path = "data/url_dataset.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"[ML] Loaded existing dataset: {len(df)} samples")
    else:
        df = generate_url_dataset(dataset_path)

    X = df[URL_FEATURE_COLUMNS].values
    y = df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])

    print(f"\n[ML] URL Model Accuracy: {accuracy:.4f}")
    print(f"\n[ML] Classification Report:\n{report}")

    # Feature importance
    importances = dict(zip(URL_FEATURE_COLUMNS, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("[ML] Top Feature Importances:")
    for feat, imp in sorted_imp[:5]:
        print(f"  {feat}: {imp:.4f}")

    # Save model and scaler
    os.makedirs(os.path.dirname(URL_MODEL_PATH), exist_ok=True)
    joblib.dump(model, URL_MODEL_PATH)
    joblib.dump(scaler, URL_SCALER_PATH)
    print(f"\n[ML] URL model saved to {URL_MODEL_PATH}")
    print(f"[ML] URL scaler saved to {URL_SCALER_PATH}")

    return {
        "model_type": "Random Forest (URL)",
        "dataset_size": len(df),
        "test_accuracy": round(accuracy, 4),
        "note": "Hackathon prototype — requires validation on real-world data",
    }


def train_message_model() -> dict:
    """Train a TF-IDF + Random Forest classifier for message phishing detection."""
    print("\n" + "=" * 60)
    print("Training Message Phishing Detection Model")
    print("=" * 60)

    # Load or generate dataset
    dataset_path = "data/message_dataset.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"[ML] Loaded existing dataset: {len(df)} samples")
    else:
        df = generate_message_dataset(dataset_path)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(df["message"].values)
    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])

    print(f"\n[ML] Message Model Accuracy: {accuracy:.4f}")
    print(f"\n[ML] Classification Report:\n{report}")

    # Save model and vectorizer
    os.makedirs(os.path.dirname(MESSAGE_MODEL_PATH), exist_ok=True)
    joblib.dump(model, MESSAGE_MODEL_PATH)
    joblib.dump(vectorizer, MESSAGE_VECTORIZER_PATH)
    print(f"\n[ML] Message model saved to {MESSAGE_MODEL_PATH}")
    print(f"[ML] Message vectorizer saved to {MESSAGE_VECTORIZER_PATH}")

    return {
        "model_type": "TF-IDF + Random Forest (Message)",
        "dataset_size": len(df),
        "test_accuracy": round(accuracy, 4),
        "note": "Hackathon prototype — requires validation on real-world data",
    }


def train_all():
    """Train both URL and message models."""
    print("\n" + "#" * 60)
    print("PhishGuard AI — ML Training Pipeline")
    print("#" * 60)
    print("\nNOTE: This is a hackathon prototype using demonstration datasets.")
    print("Results require validation on larger real-world datasets.\n")

    url_result = train_url_model()
    message_result = train_message_model()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nURL Model: accuracy={url_result['test_accuracy']}, samples={url_result['dataset_size']}")
    print(f"Message Model: accuracy={message_result['test_accuracy']}, samples={message_result['dataset_size']}")
    print(f"\nModels saved. Backend will load these automatically on startup.")

    return {"url": url_result, "message": message_result}


if __name__ == "__main__":
    train_all()
