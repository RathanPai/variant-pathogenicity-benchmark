"""
======================================================================
FEATURE AUDIT
======================================================================

Project:
Quantum-Enhanced Genetic Variant Pathogenicity Prediction

Input:
    Data/variant_features_engineered.csv

Outputs:
    Data/feature_audit.csv
    Data/correlation_matrix.csv

Purpose:
    Audit engineered features BEFORE preprocessing/model training.

Checks:
    1. Data types
    2. Missing values
    3. Unique values
    4. Constant features
    5. Near-constant features
    6. Numerical distributions
    7. Categorical cardinality
    8. Correlations
    9. Label association
   10. Potential leakage
   11. Feature recommendations

IMPORTANT:
    This script does NOT modify the dataset.
======================================================================
"""

import os
import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

INPUT_FILE = "Data/variant_features_engineered.csv"

AUDIT_FILE = "Data/feature_audit.csv"

CORRELATION_FILE = "Data/correlation_matrix.csv"

TARGET = "label"

# Thresholds

MISSING_DROP_THRESHOLD = 0.98

NEAR_CONSTANT_THRESHOLD = 0.995

HIGH_CARDINALITY_THRESHOLD = 100

HIGH_CORRELATION_THRESHOLD = 0.95

LEAKAGE_CORRELATION_THRESHOLD = 0.90


# ======================================================================
# LOAD
# ======================================================================

print("=" * 70)
print("FEATURE AUDIT")
print("=" * 70)

print("\nLoading engineered dataset...")

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns):,}"
)


# ======================================================================
# VERIFY TARGET
# ======================================================================

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ======================================================================
# FEATURE LIST
# ======================================================================

feature_columns = [
    column
    for column in df.columns
    if column != TARGET
]

print(
    f"\nNumber of features: {len(feature_columns)}"
)


# ======================================================================
# BASIC AUDIT
# ======================================================================

print("\nBuilding feature audit...")


audit_rows = []


for column in feature_columns:

    series = df[column]

    dtype = str(series.dtype)

    missing_count = int(
        series.isna().sum()
    )

    missing_fraction = (
        missing_count / len(df)
    )

    non_missing = series.dropna()

    unique_count = (
        non_missing.nunique()
    )

    # --------------------------------------------------------------
    # Constant
    # --------------------------------------------------------------

    is_constant = (
        unique_count <= 1
    )

    # --------------------------------------------------------------
    # Most common value proportion
    # --------------------------------------------------------------

    if len(non_missing) > 0:

        value_counts = (
            non_missing
            .value_counts(normalize=True)
        )

        most_common_fraction = float(
            value_counts.iloc[0]
        )

    else:

        most_common_fraction = 1.0

    is_near_constant = (
        most_common_fraction
        >= NEAR_CONSTANT_THRESHOLD
    )

    # --------------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        minimum = numeric_series.min()

        maximum = numeric_series.max()

        mean = numeric_series.mean()

        median = numeric_series.median()

        std = numeric_series.std()

        # Number of zero values

        zero_fraction = (
            (numeric_series == 0)
            .sum()
            /
            len(numeric_series)
        )

    else:

        minimum = np.nan
        maximum = np.nan
        mean = np.nan
        median = np.nan
        std = np.nan
        zero_fraction = np.nan

    # --------------------------------------------------------------
    # Categorical cardinality
    # --------------------------------------------------------------

    is_high_cardinality = (
        unique_count
        > HIGH_CARDINALITY_THRESHOLD
    )

    # --------------------------------------------------------------
    # Initial recommendation
    # --------------------------------------------------------------

    if missing_fraction >= MISSING_DROP_THRESHOLD:

        recommendation = "REVIEW"

        reason = (
            "Extremely high missingness"
        )

    elif is_constant:

        recommendation = "DROP"

        reason = (
            "Constant feature"
        )

    elif is_near_constant:

        recommendation = "REVIEW"

        reason = (
            "Near-constant feature"
        )

    elif is_high_cardinality and not pd.api.types.is_numeric_dtype(series):

        recommendation = "REVIEW"

        reason = (
            "High-cardinality categorical feature"
        )

    else:

        recommendation = "KEEP"

        reason = ""

    audit_rows.append({

        "feature": column,

        "dtype": dtype,

        "missing_count": missing_count,

        "missing_fraction": missing_fraction,

        "unique_count": unique_count,

        "most_common_fraction":
            most_common_fraction,

        "zero_fraction":
            zero_fraction,

        "minimum":
            minimum,

        "maximum":
            maximum,

        "mean":
            mean,

        "median":
            median,

        "std":
            std,

        "high_cardinality":
            is_high_cardinality,

        "constant":
            is_constant,

        "near_constant":
            is_near_constant,

        "recommendation":
            recommendation,

        "reason":
            reason
    })


audit = pd.DataFrame(
    audit_rows
)


# ======================================================================
# CORRELATION ANALYSIS
# ======================================================================

print("\nCalculating numerical correlations...")

numeric_columns = [
    column
    for column in feature_columns
    if pd.api.types.is_numeric_dtype(
        df[column]
    )
]

print(
    f"Numerical features: {len(numeric_columns)}"
)


if len(numeric_columns) > 0:

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    correlation_matrix.to_csv(
        CORRELATION_FILE
    )

else:

    correlation_matrix = pd.DataFrame()


# ======================================================================
# HIGHLY CORRELATED FEATURE PAIRS
# ======================================================================

print("\nFinding highly correlated feature pairs...")

high_correlation_pairs = []

if not correlation_matrix.empty:

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            feature_a = columns[i]

            feature_b = columns[j]

            correlation = (
                correlation_matrix
                .loc[
                    feature_a,
                    feature_b
                ]
            )

            if pd.notna(correlation):

                if abs(correlation) >= HIGH_CORRELATION_THRESHOLD:

                    high_correlation_pairs.append({

                        "feature_a":
                            feature_a,

                        "feature_b":
                            feature_b,

                        "correlation":
                            correlation

                    })


high_correlation_pairs_df = pd.DataFrame(
    high_correlation_pairs
)


# ======================================================================
# LABEL ASSOCIATION
# ======================================================================
#
# IMPORTANT:
#
# This is NOT feature importance.
#
# It is only a diagnostic check to identify features that are suspiciously
# associated with the label.
#
# Very high association may indicate:
#   - legitimate biological signal
#   - a proxy for ClinVar evidence
#   - data leakage
#
# We will manually investigate these.
# ======================================================================

print("\nChecking feature-label associations...")

label_associations = []


for column in numeric_columns:

    if column == TARGET:
        continue

    try:

        correlation = (
            df[column]
            .corr(df[TARGET])
        )

    except Exception:

        correlation = np.nan

    label_associations.append({

        "feature":
            column,

        "label_correlation":
            correlation,

        "absolute_label_correlation":
            abs(correlation)
            if pd.notna(correlation)
            else np.nan

    })


label_associations_df = pd.DataFrame(
    label_associations
)


# Merge into audit

audit = audit.merge(
    label_associations_df,
    on="feature",
    how="left"
)


# ======================================================================
# POSSIBLE LEAKAGE
# ======================================================================

print("\nChecking for possible leakage...")

audit["possible_leakage"] = False

audit["leakage_reason"] = ""


# --------------------------------------------------------------
# Explicit suspicious naming patterns
# --------------------------------------------------------------

leakage_keywords = [

    "label",
    "clinical_significance",
    "clinical significance",
    "review_status",
    "review status",
    "pathogenic",
    "benign",
    "clinvar_significance",
    "clinvar significance",
    "classification",
    "submitter",
    "submission",
    "assertion"
]


for index, row in audit.iterrows():

    feature = str(
        row["feature"]
    ).lower()

    for keyword in leakage_keywords:

        if keyword in feature:

            audit.loc[
                index,
                "possible_leakage"
            ] = True

            audit.loc[
                index,
                "leakage_reason"
            ] = (
                f"Feature name contains "
                f"suspicious keyword: {keyword}"
            )

            break


# --------------------------------------------------------------
# Extremely high label correlation
# --------------------------------------------------------------

for index, row in audit.iterrows():

    correlation = row[
        "absolute_label_correlation"
    ]

    if pd.notna(correlation):

        if correlation >= LEAKAGE_CORRELATION_THRESHOLD:

            audit.loc[
                index,
                "possible_leakage"
            ] = True

            existing_reason = audit.loc[
                index,
                "leakage_reason"
            ]

            new_reason = (
                "Very high label correlation"
            )

            if existing_reason:

                existing_reason += (
                    "; "
                    + new_reason
                )

            else:

                existing_reason = new_reason

            audit.loc[
                index,
                "leakage_reason"
            ] = existing_reason


# ======================================================================
# OVERRIDE RECOMMENDATIONS FOR LEAKAGE
# ======================================================================

audit.loc[
    audit["possible_leakage"] == True,
    "recommendation"
] = "LEAKAGE"


audit.loc[
    audit["possible_leakage"] == True,
    "reason"
] = audit.loc[
    audit["possible_leakage"] == True,
    "leakage_reason"
]


# ======================================================================
# SAVE AUDIT
# ======================================================================

audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ======================================================================
# PRINT SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)


print("\nRecommendation counts:")

print(
    audit["recommendation"]
    .value_counts()
)


# ======================================================================
# CONSTANT FEATURES
# ======================================================================

constant_features = audit.loc[
    audit["constant"],
    "feature"
].tolist()

print(
    f"\nConstant features: "
    f"{len(constant_features)}"
)

for feature in constant_features:

    print(
        f"  DROP: {feature}"
    )


# ======================================================================
# NEAR CONSTANT FEATURES
# ======================================================================

near_constant_features = audit.loc[
    audit["near_constant"]
    &
    ~audit["constant"],
    [
        "feature",
        "most_common_fraction"
    ]
]

print(
    f"\nNear-constant features: "
    f"{len(near_constant_features)}"
)

for _, row in near_constant_features.iterrows():

    print(
        f"  REVIEW: "
        f"{row['feature']} "
        f"({row['most_common_fraction']:.4%} "
        f"same value)"
    )


# ======================================================================
# HIGH MISSINGNESS
# ======================================================================

high_missing = audit.loc[
    audit["missing_fraction"]
    >= MISSING_DROP_THRESHOLD,
    [
        "feature",
        "missing_fraction"
    ]
].sort_values(
    "missing_fraction",
    ascending=False
)

print(
    f"\nFeatures with >= "
    f"{MISSING_DROP_THRESHOLD:.0%} missing:"
)

if len(high_missing) == 0:

    print("  None")

else:

    for _, row in high_missing.iterrows():

        print(
            f"  REVIEW: "
            f"{row['feature']} "
            f"({row['missing_fraction']:.2%})"
        )


# ======================================================================
# HIGH CORRELATIONS
# ======================================================================

print(
    f"\nHighly correlated feature pairs "
    f"(absolute correlation >= "
    f"{HIGH_CORRELATION_THRESHOLD}):"
)

if len(high_correlation_pairs_df) == 0:

    print("  None")

else:

    print(
        high_correlation_pairs_df
        .sort_values(
            "correlation",
            key=lambda x: abs(x),
            ascending=False
        )
        .head(50)
        .to_string(index=False)
    )


# ======================================================================
# LABEL ASSOCIATION
# ======================================================================

print(
    "\nStrongest numerical feature-label associations:"
)

strong_label_features = (
    label_associations_df
    .sort_values(
        "absolute_label_correlation",
        ascending=False
    )
    .head(20)
)

print(
    strong_label_features
    .to_string(index=False)
)


# ======================================================================
# POSSIBLE LEAKAGE
# ======================================================================

possible_leakage = audit.loc[
    audit["possible_leakage"],
    [
        "feature",
        "leakage_reason",
        "label_correlation"
    ]
]

print(
    "\nPossible leakage features:"
)

if len(possible_leakage) == 0:

    print("  None detected")

else:

    print(
        possible_leakage
        .to_string(index=False)
    )


# ======================================================================
# CATEGORICAL FEATURES
# ======================================================================

categorical_columns = [
    column
    for column in feature_columns
    if not pd.api.types.is_numeric_dtype(
        df[column]
    )
]


print(
    f"\nCategorical features: "
    f"{len(categorical_columns)}"
)

for column in categorical_columns:

    unique_count = (
        df[column]
        .nunique(dropna=True)
    )

    print(
        f"  {column:<40} "
        f"{unique_count:>6} categories"
    )


# ======================================================================
# TOP MISSING FEATURES
# ======================================================================

print(
    "\nTop 20 missing features:"
)

print(
    audit[
        [
            "feature",
            "missing_fraction",
            "unique_count"
        ]
    ]
    .sort_values(
        "missing_fraction",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ======================================================================
# OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    f"\nFeature audit:"
    f"\n  {AUDIT_FILE}"
)

print(
    f"\nCorrelation matrix:"
    f"\n  {CORRELATION_FILE}"
)

print("\nIMPORTANT:")
print(
    "The original engineered dataset was NOT modified."
)

print(
    "\nNext step:"
)

print(
    "Review the audit results before preprocessing."
)

print("=" * 70)