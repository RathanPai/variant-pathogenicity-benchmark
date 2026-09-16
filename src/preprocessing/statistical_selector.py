"""
QNN FEATURE SELECTION
=====================

Input:
    Data/processed/train_processed.csv
    Data/processed/validation_processed.csv
    Data/processed/test_processed.csv

Purpose:
    Rank features using TRAIN ONLY and determine a suitable
    QNN-compatible number of features using VALIDATION.

TEST IS NOT USED FOR MODEL SELECTION.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("Data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "selected"

TRAIN_FILE = PROCESSED_DIR / "train_processed.csv"
VAL_FILE = PROCESSED_DIR / "validation_processed.csv"
TEST_FILE = PROCESSED_DIR / "test_processed.csv"

TARGET = "label"

# Candidate dimensions for the future QNN
K_VALUES = [4, 6, 8, 10, 12, 16]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("QNN FEATURE SELECTION")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTRAIN      : {train.shape}")
print(f"VALIDATION : {val.shape}")
print(f"TEST       : {test.shape}")


# ============================================================
# SPLIT X / y
# ============================================================

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_val = val.drop(columns=[TARGET])
y_val = val[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

feature_names = X_train.columns.to_numpy()

print(f"\nCandidate features: {X_train.shape[1]}")


# ============================================================
# SANITY CHECK
# ============================================================

if list(X_train.columns) != list(X_val.columns):
    raise ValueError(
        "TRAIN and VALIDATION features do not match."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "TRAIN and TEST features do not match."
    )


# ============================================================
# MUTUAL INFORMATION
#
# IMPORTANT:
# Ranking is calculated using TRAIN ONLY.
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING MUTUAL INFORMATION")
print("=" * 70)

print("\nRanking 495 features using TRAIN only...")

mi_scores = mutual_info_classif(
    X_train,
    y_train,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

ranking = pd.DataFrame({
    "feature": feature_names,
    "mutual_information": mi_scores
})

ranking = ranking.sort_values(
    "mutual_information",
    ascending=False
).reset_index(drop=True)

ranking["rank"] = np.arange(
    1,
    len(ranking) + 1
)

ranking = ranking[
    [
        "rank",
        "feature",
        "mutual_information"
    ]
]

ranking_file = OUTPUT_DIR / "feature_ranking.csv"

ranking.to_csv(
    ranking_file,
    index=False
)

print("\nTop 30 features:")
print(
    ranking.head(30).to_string(
        index=False
    )
)


# ============================================================
# TEST DIFFERENT FEATURE COUNTS
#
# Validation is used here.
# Test is NOT used.
#
# Logistic Regression is only a cheap probe model.
# It is NOT our final model.
# ============================================================

print("\n" + "=" * 70)
print("CHOOSING FEATURE COUNT USING VALIDATION")
print("=" * 70)

results = []

for k in K_VALUES:

    selected_features = (
        ranking
        .head(k)["feature"]
        .tolist()
    )

    X_train_k = X_train[selected_features]
    X_val_k = X_val[selected_features]

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_k,
        y_train
    )

    val_pred = model.predict(
        X_val_k
    )

    val_prob = model.predict_proba(
        X_val_k
    )[:, 1]

    accuracy = accuracy_score(
        y_val,
        val_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        val_pred
    )

    f1 = f1_score(
        y_val,
        val_pred
    )

    mcc = matthews_corrcoef(
        y_val,
        val_pred
    )

    auc = roc_auc_score(
        y_val,
        val_prob
    )

    results.append({
        "k": k,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": auc
    })

    print(
        f"\nk = {k}"
        f"\n  Accuracy          : {accuracy:.4f}"
        f"\n  Balanced Accuracy : {balanced_accuracy:.4f}"
        f"\n  F1                : {f1:.4f}"
        f"\n  MCC               : {mcc:.4f}"
        f"\n  ROC-AUC           : {auc:.4f}"
    )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(results)

results_file = (
    OUTPUT_DIR /
    "feature_count_validation_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# CHOOSE BEST K
#
# MCC is useful here because our classes are imbalanced.
# ============================================================

best_row = results.loc[
    results["mcc"].idxmax()
]

best_k = int(
    best_row["k"]
)

print("\n" + "=" * 70)
print("SELECTED DIMENSION")
print("=" * 70)

print(
    f"\nBest validation MCC obtained with "
    f"k = {best_k}"
)

print(
    f"MCC     : {best_row['mcc']:.4f}"
)

print(
    f"ROC-AUC : {best_row['roc_auc']:.4f}"
)


# ============================================================
# SELECT FINAL FEATURES
# ============================================================

selected_features = (
    ranking
    .head(best_k)["feature"]
    .tolist()
)

print("\nSelected features:")

for i, feature in enumerate(
    selected_features,
    start=1
):
    score = ranking.loc[
        ranking["feature"] == feature,
        "mutual_information"
    ].iloc[0]

    print(
        f"{i:>2}. "
        f"{feature:<60} "
        f"MI={score:.6f}"
    )


# ============================================================
# CREATE REDUCED DATASETS
# ============================================================

train_selected = X_train[
    selected_features
].copy()

train_selected[TARGET] = (
    y_train.values
)

val_selected = X_val[
    selected_features
].copy()

val_selected[TARGET] = (
    y_val.values
)

test_selected = X_test[
    selected_features
].copy()

test_selected[TARGET] = (
    y_test.values
)


# ============================================================
# SAVE
# ============================================================

train_output = (
    OUTPUT_DIR /
    "train_selected.csv"
)

val_output = (
    OUTPUT_DIR /
    "validation_selected.csv"
)

test_output = (
    OUTPUT_DIR /
    "test_selected.csv"
)

selected_features_output = (
    OUTPUT_DIR /
    "selected_features.txt"
)

train_selected.to_csv(
    train_output,
    index=False
)

val_selected.to_csv(
    val_output,
    index=False
)

test_selected.to_csv(
    test_output,
    index=False
)

with open(
    selected_features_output,
    "w",
    encoding="utf-8"
) as f:

    for feature in selected_features:
        f.write(feature + "\n")


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASETS")
print("=" * 70)

print(
    f"TRAIN      : "
    f"{train_selected.shape}"
)

print(
    f"VALIDATION : "
    f"{val_selected.shape}"
)

print(
    f"TEST       : "
    f"{test_selected.shape}"
)

print("\nFiles created:")

print(f"  {ranking_file}")
print(f"  {results_file}")
print(f"  {train_output}")
print(f"  {val_output}")
print(f"  {test_output}")
print(f"  {selected_features_output}")

print("\nIMPORTANT:")
print("  Feature ranking       : TRAIN only")
print("  Number of features    : chosen using VALIDATION")
print("  TEST labels           : never used")
print("  TEST performance      : not evaluated yet")

print("\nNext:")
print("  Classical neural network baseline")
print("  + QNN using exactly the same selected features")