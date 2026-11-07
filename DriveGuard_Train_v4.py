"""
DriveGuard Train v4
One-snapshot-per-drive experiment with train/validation/test separation.

Usage:
    python DriveGuard_Train_v4.py --data-dir driveguard_extracted_v4

Requires:
    pandas numpy scikit-learn imbalanced-learn matplotlib joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42
MISSINGNESS_LIMIT = 0.95
THRESHOLD_GRID = np.arange(0.01, 0.501, 0.01)


def load_xy(path: Path):
    df = pd.read_csv(path, low_memory=False)
    if "failure" not in df:
        raise ValueError(f"{path} has no 'failure' target column.")
    y = pd.to_numeric(df.pop("failure"), errors="coerce").astype(int)
    # Only SMART telemetry is used as a predictor. Identifiers/date/model/capacity
    # are deliberately excluded to prevent metadata leakage.
    feature_cols = [c for c in df.columns if c.startswith("smart_")]
    if not feature_cols:
        raise ValueError(f"No SMART features found in {path}.")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X, y, feature_cols


def select_threshold(y_true, probabilities):
    # F2 weights recall more heavily than precision. This selection uses
    # validation data only; test data is never used to tune the threshold.
    best = None
    for threshold in THRESHOLD_GRID:
        pred = (probabilities >= threshold).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        denom = 4 * p + r
        f2 = (5 * p * r / denom) if denom else 0.0
        candidate = (f2, threshold, p, r)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return {
        "threshold": float(best[1]),
        "validation_f2": float(best[0]),
        "validation_precision": float(best[2]),
        "validation_recall": float(best[3])
    }


def evaluate(name, y, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": float(threshold),
        "test_rows": int(len(y)),
        "test_failure_rows": int(np.sum(y)),
        "test_prevalence": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="driveguard_extracted_v4")
    parser.add_argument("--output-dir", default="driveguard_model_v4")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train, cols = load_xy(data_dir / "driveguard_train_v4.csv")
    X_val, y_val, val_cols = load_xy(data_dir / "driveguard_validation_v4.csv")
    X_test, y_test, test_cols = load_xy(data_dir / "driveguard_test_v4.csv")

    if set(cols) != set(val_cols) or set(cols) != set(test_cols):
        raise ValueError("SMART feature columns differ between partitions.")
    X_val = X_val[cols]
    X_test = X_test[cols]

    if y_train.nunique() != 2 or y_val.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Each train/validation/test partition must contain both classes.")

    # Feature eligibility and all learned preprocessing are based on training only.
    missing_fraction = X_train.isna().mean()
    retained = missing_fraction[missing_fraction <= MISSINGNESS_LIMIT].index.tolist()
    dropped = [c for c in cols if c not in retained]
    if not retained:
        raise ValueError("No SMART features remain after missingness filtering.")

    X_train = X_train[retained]
    X_val = X_val[retained]
    X_test = X_test[retained]

    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xva = scaler.transform(imputer.transform(X_val))
    Xte = scaler.transform(imputer.transform(X_test))

    common_rf = dict(
        n_estimators=100, max_depth=12, criterion="gini",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    models = {}
    # A: class weighting on the original, naturally imbalanced training data.
    cost_model = RandomForestClassifier(class_weight="balanced", **common_rf)
    cost_model.fit(Xtr, y_train)
    models["Cost-Sensitive Random Forest"] = cost_model

    # B: SMOTE only on training data, no simultaneous class_weight balancing.
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    Xtr_smote, ytr_smote = smote.fit_resample(Xtr, y_train)
    smote_model = RandomForestClassifier(class_weight=None, **common_rf)
    smote_model.fit(Xtr_smote, ytr_smote)
    models["SMOTE Random Forest"] = smote_model

    results = []
    threshold_info = {}
    for name, model in models.items():
        val_prob = model.predict_proba(Xva)[:, 1]
        chosen = select_threshold(y_val, val_prob)
        threshold_info[name] = chosen
        test_prob = model.predict_proba(Xte)[:, 1]
        results.append(evaluate(name, y_test, test_prob, chosen["threshold"]))

        # Save deployable bundle with preprocessing and exact feature order.
        filename = (
            "driveguard_cost_sensitive_v4.joblib"
            if name.startswith("Cost") else "driveguard_smote_v4.joblib"
        )
        joblib.dump({
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            "feature_names": retained,
            "threshold": chosen["threshold"],
            "target": "failure",
            "snapshot_policy": "one snapshot per physical drive"
        }, out / filename)

    metrics = pd.DataFrame(results)
    metrics.to_csv(out / "metrics_comparison_v4.csv", index=False)

    summary = {
        "train_rows": int(len(y_train)),
        "train_failures": int(y_train.sum()),
        "validation_rows": int(len(y_val)),
        "validation_failures": int(y_val.sum()),
        "test_rows": int(len(y_test)),
        "test_failures": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "smart_features_available": int(len(cols)),
        "smart_features_retained": int(len(retained)),
        "dropped_features": dropped,
        "threshold_selection": "F2 maximization on validation set only; test set not used for tuning",
        "thresholds": threshold_info,
        "smote": {"random_state": RANDOM_STATE, "k_neighbors": 5},
        "random_forest": common_rf,
        "metrics": results,
        "warning": "Failure-positive counts are small. Metrics have high uncertainty. failure=1 is the last operational day before failure, not a multi-day warning label."
    }
    (out / "experiment_summary_v4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # ROC/PR curves and confusion matrices use the fixed thresholds selected on validation.
    for name, model in models.items():
        test_prob = model.predict_proba(Xte)[:, 1]
        threshold = threshold_info[name]["threshold"]
        pred = (test_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm)
        ax.set_title(f"{name}\nConfusion Matrix (threshold={threshold:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Healthy", "Failure"])
        ax.set_yticks([0, 1], ["Healthy", "Failure"])
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out / (
            "cost_sensitive_confusion_v4.png" if name.startswith("Cost")
            else "smote_confusion_v4.png"
        ), dpi=160)
        plt.close(fig)

    print(metrics.to_string(index=False))
    print(f"\nSaved experiment outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
