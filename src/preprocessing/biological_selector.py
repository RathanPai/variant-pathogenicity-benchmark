"""
BIOLOGICAL FEATURE SELECTION
=============================

Purpose:
    Select QNN input features at the ORIGINAL biological-feature level,
    rather than selecting individual one-hot columns independently.

Data:
    Data/processed/train_processed.csv
    Data/processed/validation_processed.csv
    Data/processed/test_processed.csv

The TEST set is NOT used for feature selection.

Output:
    Data/biological_selected/
        feature_importance.csv
        feature_groups.csv
        correlation_matrix.csv
        train_selected.csv
        validation_selected.csv
        test_selected.csv
        selected_features.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("Data")
INPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "biological_selected"

TRAIN_FILE = INPUT_DIR / "train_processed.csv"
VAL_FILE = INPUT_DIR / "validation_processed.csv"
TEST_FILE = INPUT_DIR / "test_processed.csv"

TARGET = "label"

RANDOM_STATE = 42

# Candidate numbers of ORIGINAL biological features
K_VALUES = [4, 6, 8, 10, 12, 16]

# Correlation threshold for redundancy detection
CORRELATION_THRESHOLD = 0.90


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BIOLOGICAL FEATURE SELECTION")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTRAIN      : {train.shape}")
print(f"VALIDATION : {val.shape}")
print(f"TEST       : {test.shape}")


# ============================================================
# SPLIT
# ============================================================

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_val = val.drop(columns=[TARGET])
y_val = val[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


# ============================================================
# GROUP PROCESSED FEATURES BACK TO BIOLOGICAL FEATURES
# ============================================================

def get_original_feature(processed_name):

    """
    Convert sklearn ColumnTransformer feature names such as:

        numeric__protein_length

    into:

        protein_length

    and:

        categorical__vep_impact_HIGH

    into:

        vep_impact

    This lets us score the underlying biological variable rather
    than individual one-hot columns.
    """

    if "__" in processed_name:
        name = processed_name.split("__", 1)[1]
    else:
        name = processed_name

    # Known categorical variables.
    # Longest names first to avoid partial matches.
    categorical_prefixes = [
        "Molecular consequence",
        "vep_primary_consequence",
        "vep_consequence",
        "variant_type_simple",
        "vep_variant_class",
        "Variant type",
        "vep_impact",
    ]

    for prefix in categorical_prefixes:

        if name == prefix:
            return prefix

        if name.startswith(prefix + "_"):
            return prefix

    return name


feature_groups = {}

for feature in X_train.columns:

    original = get_original_feature(feature)

    feature_groups.setdefault(
        original,
        []
    ).append(feature)


print("\n" + "=" * 70)
print("BIOLOGICAL FEATURE GROUPS")
print("=" * 70)

print(
    f"\n495 processed columns correspond to "
    f"{len(feature_groups)} underlying feature groups."
)

for group, members in feature_groups.items():

    print(
        f"{group:<40}"
        f"{len(members):>4} encoded columns"
    )


# ============================================================
# CALCULATE MUTUAL INFORMATION
#
# TRAIN ONLY
# ============================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION")
print("=" * 70)

print("\nCalculating MI for individual encoded columns...")

mi_values = mutual_info_classif(
    X_train,
    y_train,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

encoded_mi = pd.Series(
    mi_values,
    index=X_train.columns
)


# ============================================================
# AGGREGATE MI TO BIOLOGICAL FEATURE LEVEL
# ============================================================

biological_scores = []

for group, members in feature_groups.items():

    scores = encoded_mi[
        members
    ]

    biological_scores.append({

        "biological_feature": group,

        # Maximum category-level information
        "max_mutual_information":
            scores.max(),

        # Total information across encoded categories
        "sum_mutual_information":
            scores.sum(),

        # Number of encoded columns
        "encoded_columns":
            len(members),

        # Encoded columns belonging to this group
        "encoded_features":
            " | ".join(members)
    })


importance = pd.DataFrame(
    biological_scores
)

importance = importance.sort_values(
    "max_mutual_information",
    ascending=False
).reset_index(drop=True)

importance["rank"] = np.arange(
    1,
    len(importance) + 1
)

importance = importance[
    [
        "rank",
        "biological_feature",
        "max_mutual_information",
        "sum_mutual_information",
        "encoded_columns",
        "encoded_features"
    ]
]

importance_file = (
    OUTPUT_DIR /
    "feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)

print("\nTop biological features:")

print(
    importance.head(40).to_string(
        index=False
    )
)


# ============================================================
# BUILD BIOLOGICAL-LEVEL DATA
#
# For categorical groups:
#   retain all their encoded columns as one biological group.
#
# For numerical groups:
#   there is normally one column.
# ============================================================

biological_data = {}

for group, members in feature_groups.items():

    if len(members) == 1:

        biological_data[group] = (
            X_train[members[0]]
        )

    else:

        # Collapse one-hot group into a single score.
        #
        # Since exactly one category is normally active,
        # using the category's MI-weighted representation
        # would be unnecessarily complicated.
        #
        # Instead, use the encoded category values as a
        # group for redundancy analysis.
        #
        # We retain the strongest encoded category as the
        # representative signal.
        strongest = encoded_mi[
            members
        ].idxmax()

        biological_data[group] = (
            X_train[strongest]
        )

biological_train = pd.DataFrame(
    biological_data
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("REDUNDANCY ANALYSIS")
print("=" * 70)

correlation = biological_train.corr(
    method="spearman"
)

correlation_file = (
    OUTPUT_DIR /
    "correlation_matrix.csv"
)

correlation.to_csv(
    correlation_file
)

# Find highly correlated pairs

correlated_pairs = []

columns = correlation.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        corr = correlation.iloc[i, j]

        if abs(corr) >= CORRELATION_THRESHOLD:

            correlated_pairs.append({

                "feature_1": columns[i],

                "feature_2": columns[j],

                "spearman_correlation": corr
            })


correlated_pairs_df = pd.DataFrame(
    correlated_pairs
)

if len(correlated_pairs_df) > 0:

    print(
        f"\nHighly correlated biological "
        f"feature pairs (|r| >= "
        f"{CORRELATION_THRESHOLD}):"
    )

    print(
        correlated_pairs_df.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo biological feature pairs exceeded "
        f"|r| >= {CORRELATION_THRESHOLD}."
    )


# ============================================================
# REDUNDANCY-AWARE RANKING
#
# Start with highest MI.
# If a feature is highly correlated with an already selected
# feature, skip it.
# ============================================================

ranked_features = importance[
    "biological_feature"
].tolist()

selected_pool = []

for feature in ranked_features:

    if len(selected_pool) == 16:
        break

    keep = True

    for selected in selected_pool:

        corr = abs(
            correlation.loc[
                feature,
                selected
            ]
        )

        if corr >= CORRELATION_THRESHOLD:

            keep = False

            break

    if keep:

        selected_pool.append(
            feature
        )


print("\n" + "=" * 70)
print("REDUNDANCY-AWARE FEATURE POOL")
print("=" * 70)

for i, feature in enumerate(
    selected_pool,
    start=1
):

    score = importance.loc[
        importance["biological_feature"] == feature,
        "max_mutual_information"
    ].iloc[0]

    print(
        f"{i:>2}. "
        f"{feature:<45}"
        f"MI={score:.6f}"
    )


# ============================================================
# CREATE REDUCED MATRICES
# ============================================================

def make_biological_matrix(
    X,
    selected_groups
):

    output = {}

    for group in selected_groups:

        members = feature_groups[group]

        if len(members) == 1:

            output[group] = X[
                members[0]
            ]

        else:

            # Use the strongest MI category as
            # representative.
            strongest = encoded_mi[
                members
            ].idxmax()

            output[group] = X[
                strongest
            ]

    return pd.DataFrame(output)


# ============================================================
# VALIDATION MODEL COMPARISON
#
# Test remains untouched.
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FEATURE-COUNT EXPERIMENT")
print("=" * 70)

results = []

for k in K_VALUES:

    if k > len(selected_pool):
        continue

    selected = selected_pool[:k]

    train_k = make_biological_matrix(
        X_train,
        selected
    )

    val_k = make_biological_matrix(
        X_val,
        selected
    )

    # Logistic regression is used only as a
    # lightweight feature-selection probe.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        train_k,
        y_train
    )

    val_pred = model.predict(
        val_k
    )

    val_prob = model.predict_proba(
        val_k
    )[:, 1]

    balanced_acc = balanced_accuracy_score(
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

        "balanced_accuracy":
            balanced_acc,

        "f1":
            f1,

        "mcc":
            mcc,

        "roc_auc":
            auc
    })

    print(
        f"\nk={k}"
        f"\n  Balanced Accuracy : "
        f"{balanced_acc:.4f}"
        f"\n  F1                : "
        f"{f1:.4f}"
        f"\n  MCC               : "
        f"{mcc:.4f}"
        f"\n  ROC-AUC           : "
        f"{auc:.4f}"
    )


results_df = pd.DataFrame(
    results
)

results_file = (
    OUTPUT_DIR /
    "feature_count_validation_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# SELECT BEST K
# ============================================================

best_row = results_df.loc[
    results_df["mcc"].idxmax()
]

best_k = int(
    best_row["k"]
)

final_features = selected_pool[
    :best_k
]

print("\n" + "=" * 70)
print("FINAL BIOLOGICAL FEATURE SET")
print("=" * 70)

print(
    f"\nSelected biological features: "
    f"{best_k}"
)

for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i:>2}. {feature}"
    )


# ============================================================
# CONVERT TRAIN / VAL / TEST
# ============================================================

train_final = make_biological_matrix(
    X_train,
    final_features
)

val_final = make_biological_matrix(
    X_val,
    final_features
)

test_final = make_biological_matrix(
    X_test,
    final_features
)

train_final[TARGET] = y_train.values
val_final[TARGET] = y_val.values
test_final[TARGET] = y_test.values


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

features_output = (
    OUTPUT_DIR /
    "selected_features.txt"
)

train_final.to_csv(
    train_output,
    index=False
)

val_final.to_csv(
    val_output,
    index=False
)

test_final.to_csv(
    test_output,
    index=False
)

with open(
    features_output,
    "w",
    encoding="utf-8"
) as f:

    for feature in final_features:

        f.write(
            feature + "\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"\nTRAIN      : "
    f"{train_final.shape}"
)

print(
    f"VALIDATION : "
    f"{val_final.shape}"
)

print(
    f"TEST       : "
    f"{test_final.shape}"
)

print("\nFiles created:")

print(f"  {importance_file}")
print(f"  {correlation_file}")
print(f"  {results_file}")
print(f"  {train_output}")
print(f"  {val_output}")
print(f"  {test_output}")
print(f"  {features_output}")

print("\nData usage:")
print("  MI ranking       : TRAIN only")
print("  Correlation      : TRAIN only")
print("  Feature count    : VALIDATION")
print("  TEST             : untouched for selection")

print("\nNext stage:")
print("  Build classical NN baseline")
print("  Build QNN")
print("  Evaluate both on the same final features")