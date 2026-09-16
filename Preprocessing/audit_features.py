"""
BIOLOGICAL FEATURE / LABEL LEAKAGE AUDIT
=========================================

Audits the ORIGINAL biological features directly.

IMPORTANT:
    This script uses the RAW train/validation/test CSV files
    for biological interpretation.

    It does NOT modify any dataset.

    TRAIN is used for feature-level analysis.
    VALIDATION is used only for distribution comparison.
    TEST is inspected for distribution/duplicate problems,
    but its labels are NOT used for feature selection.

Input:
    Data/train(1).csv
    Data/validation(1).csv
    Data/test(1).csv

Selected features:
    Data/biological_selected/selected_features.txt

Output:
    Data/audit/
        feature_audit.csv
        split_distribution_audit.csv
        duplicate_audit.txt
        audit_summary.txt
"""


from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("Data")

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

SELECTED_FEATURE_FILE = (
    DATA_DIR
    / "biological_selected"
    / "selected_features.txt"
)

AUDIT_DIR = DATA_DIR / "audit"

TARGET = "label"

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BIOLOGICAL FEATURE / LABEL LEAKAGE AUDIT")
print("=" * 70)


print("\nLoading raw datasets...")

train = pd.read_csv(
    TRAIN_FILE
)

val = pd.read_csv(
    VAL_FILE
)

test = pd.read_csv(
    TEST_FILE
)


print(
    f"TRAIN      : {train.shape}"
)

print(
    f"VALIDATION : {val.shape}"
)

print(
    f"TEST       : {test.shape}"
)


# ============================================================
# LOAD SELECTED FEATURES
# ============================================================

if not SELECTED_FEATURE_FILE.exists():

    raise FileNotFoundError(
        f"Selected feature file not found:\n"
        f"{SELECTED_FEATURE_FILE.resolve()}"
    )


with open(
    SELECTED_FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    selected_features = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    f"\nSelected biological features: "
    f"{len(selected_features)}"
)

for i, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{i:>2}. {feature}"
    )


# ============================================================
# CHECK FEATURES EXIST
# ============================================================

missing_train = [
    f
    for f in selected_features
    if f not in train.columns
]

missing_val = [
    f
    for f in selected_features
    if f not in val.columns
]

missing_test = [
    f
    for f in selected_features
    if f not in test.columns
]


if missing_train:

    raise ValueError(
        "Selected features missing from TRAIN:\n"
        + "\n".join(missing_train)
    )


if missing_val:

    raise ValueError(
        "Selected features missing from VALIDATION:\n"
        + "\n".join(missing_val)
    )


if missing_test:

    raise ValueError(
        "Selected features missing from TEST:\n"
        + "\n".join(missing_test)
    )


# ============================================================
# EXTRACT SELECTED FEATURES
# ============================================================

X_train = train[
    selected_features
]

X_val = val[
    selected_features
]

X_test = test[
    selected_features
]

y_train = train[
    TARGET
]

y_val = val[
    TARGET
]

y_test = test[
    TARGET
]


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"\nTRAIN positive rate      : "
    f"{y_train.mean():.4f}"
)

print(
    f"VALIDATION positive rate : "
    f"{y_val.mean():.4f}"
)

print(
    f"TEST positive rate       : "
    f"{y_test.mean():.4f}"
)


# ============================================================
# FEATURE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-LEVEL AUDIT")
print("=" * 70)


audit_rows = []


for feature in selected_features:

    series = X_train[feature]

    positive = series[
        y_train == 1
    ]

    negative = series[
        y_train == 0
    ]

    # --------------------------------------------------------
    # DATA TYPE
    # --------------------------------------------------------

    dtype = str(
        series.dtype
    )

    # --------------------------------------------------------
    # UNIQUE VALUES
    # --------------------------------------------------------

    unique_values = (
        series
        .nunique(
            dropna=True
        )
    )

    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    missing_rate = (
        series.isna().mean()
    )

    positive_missing_rate = (
        positive.isna().mean()
    )

    negative_missing_rate = (
        negative.isna().mean()
    )

    missing_difference = (
        positive_missing_rate
        -
        negative_missing_rate
    )

    # --------------------------------------------------------
    # NUMERICAL FEATURE ANALYSIS
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        train_mean = series.mean()

        train_std = series.std()

        positive_mean = positive.mean()

        negative_mean = negative.mean()

        train_median = series.median()

        # Fill missing values only for this
        # univariate diagnostic.
        filled = series.fillna(
            train_median
        )

        try:

            auc = roc_auc_score(
                y_train,
                filled
            )

            auc_strength = max(
                auc,
                1 - auc
            )

        except Exception:

            auc = np.nan
            auc_strength = np.nan

        valid_mask = (
            series.notna()
        )

        if valid_mask.sum() > 2:

            correlation, p_value = (
                spearmanr(
                    series[
                        valid_mask
                    ],
                    y_train[
                        valid_mask
                    ]
                )
            )

        else:

            correlation = np.nan
            p_value = np.nan

    # --------------------------------------------------------
    # CATEGORICAL FEATURE ANALYSIS
    # --------------------------------------------------------

    else:

        train_mean = np.nan
        train_std = np.nan
        positive_mean = np.nan
        negative_mean = np.nan
        train_median = np.nan

        auc = np.nan
        auc_strength = np.nan
        correlation = np.nan
        p_value = np.nan

    # --------------------------------------------------------
    # MOST COMMON VALUES
    # --------------------------------------------------------

    value_counts = (
        series
        .value_counts(
            dropna=False
        )
        .head(10)
    )

    top_values = "; ".join(
        [
            f"{str(value)}={count}"
            for value, count
            in value_counts.items()
        ]
    )

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    audit_rows.append({

        "feature":
            feature,

        "dtype":
            dtype,

        "unique_values":
            unique_values,

        "missing_rate":
            missing_rate,

        "positive_missing_rate":
            positive_missing_rate,

        "negative_missing_rate":
            negative_missing_rate,

        "missing_rate_difference":
            missing_difference,

        "train_mean":
            train_mean,

        "train_std":
            train_std,

        "positive_mean":
            positive_mean,

        "negative_mean":
            negative_mean,

        "train_median":
            train_median,

        "spearman_correlation":
            correlation,

        "spearman_pvalue":
            p_value,

        "roc_auc":
            auc,

        "auc_strength":
            auc_strength,

        "top_values":
            top_values
    })


audit_df = pd.DataFrame(
    audit_rows
)


# ============================================================
# SORT
# ============================================================

numeric_auc_df = audit_df[
    audit_df["auc_strength"].notna()
].sort_values(
    "auc_strength",
    ascending=False
)


# ============================================================
# SAVE FEATURE AUDIT
# ============================================================

feature_audit_file = (
    AUDIT_DIR /
    "feature_audit.csv"
)

audit_df.to_csv(
    feature_audit_file,
    index=False
)


# ============================================================
# DISPLAY NUMERICAL RESULTS
# ============================================================

print(
    "\nNumerical features ranked by "
    "univariate predictive strength:"
)

if len(numeric_auc_df):

    print(
        numeric_auc_df[
            [
                "feature",
                "missing_rate",
                "missing_rate_difference",
                "spearman_correlation",
                "roc_auc",
                "auc_strength"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No numerical features found."
    )


# ============================================================
# DISPLAY CATEGORICAL DISTRIBUTIONS
# ============================================================

categorical_features = [
    f
    for f in selected_features
    if not pd.api.types.is_numeric_dtype(
        train[f]
    )
]


print("\n" + "=" * 70)
print("CATEGORICAL FEATURE DISTRIBUTIONS")
print("=" * 70)


for feature in categorical_features:

    print(
        f"\n--- {feature} ---"
    )

    distribution = pd.crosstab(
        train[feature].fillna(
            "<MISSING>"
        ),
        y_train,
        normalize="index"
    )

    distribution.columns = [
        "label_0_fraction",
        "label_1_fraction"
    ]

    counts = pd.crosstab(
        train[feature].fillna(
            "<MISSING>"
        ),
        y_train
    )

    result = pd.concat(
        [
            counts,
            distribution
        ],
        axis=1
    )

    print(
        result.to_string()
    )


# ============================================================
# SUSPICIOUS FEATURES
# ============================================================

print("\n" + "=" * 70)
print("POTENTIALLY SUSPICIOUS FEATURES")
print("=" * 70)


suspicious = audit_df[
    (
        audit_df["auc_strength"]
        >= 0.90
    )
    |
    (
        audit_df[
            "missing_rate_difference"
        ].abs()
        >= 0.25
    )
]


if len(suspicious):

    print(
        "\nThese are NOT automatically leakage."
    )

    print(
        "They simply deserve biological/data-provenance "
        "investigation:"
    )

    print(
        suspicious[
            [
                "feature",
                "missing_rate",
                "missing_rate_difference",
                "roc_auc",
                "auc_strength"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nNo feature crossed the preliminary "
        "suspicion thresholds."
    )


# ============================================================
# TRAIN / VALIDATION / TEST DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST DISTRIBUTION")
print("=" * 70)


distribution_rows = []


for feature in selected_features:

    row = {
        "feature": feature
    }

    for split_name, df in [
        ("train", train),
        ("validation", val),
        ("test", test)
    ]:

        series = df[feature]

        row[
            f"{split_name}_missing_rate"
        ] = series.isna().mean()

        row[
            f"{split_name}_unique"
        ] = series.nunique(
            dropna=True
        )

        if pd.api.types.is_numeric_dtype(
            series
        ):

            row[
                f"{split_name}_mean"
            ] = series.mean()

            row[
                f"{split_name}_std"
            ] = series.std()

            row[
                f"{split_name}_median"
            ] = series.median()

        else:

            row[
                f"{split_name}_mean"
            ] = np.nan

            row[
                f"{split_name}_std"
            ] = np.nan

            row[
                f"{split_name}_median"
            ] = np.nan

    distribution_rows.append(
        row
    )


distribution_df = pd.DataFrame(
    distribution_rows
)


distribution_file = (
    AUDIT_DIR /
    "split_distribution_audit.csv"
)

distribution_df.to_csv(
    distribution_file,
    index=False
)


print(
    distribution_df.to_string(
        index=False
    )
)


# ============================================================
# EXACT DUPLICATE ROW AUDIT
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ROW AUDIT")
print("=" * 70)


train_duplicates = (
    train[
        selected_features
    ]
    .duplicated()
    .sum()
)

val_duplicates = (
    val[
        selected_features
    ]
    .duplicated()
    .sum()
)

test_duplicates = (
    test[
        selected_features
    ]
    .duplicated()
    .sum()
)


print(
    f"\nDuplicate TRAIN rows      : "
    f"{train_duplicates:,}"
)

print(
    f"Duplicate VALIDATION rows : "
    f"{val_duplicates:,}"
)

print(
    f"Duplicate TEST rows       : "
    f"{test_duplicates:,}"
)


# ============================================================
# CROSS-SPLIT EXACT FEATURE VECTOR OVERLAP
# ============================================================

print("\nChecking identical selected-feature vectors...")


train_hash = pd.util.hash_pandas_object(
    train[selected_features],
    index=False
)

val_hash = pd.util.hash_pandas_object(
    val[selected_features],
    index=False
)

test_hash = pd.util.hash_pandas_object(
    test[selected_features],
    index=False
)


train_hash_set = set(
    train_hash
)

val_hash_set = set(
    val_hash
)

test_hash_set = set(
    test_hash
)


train_val_overlap = len(
    train_hash_set &
    val_hash_set
)

train_test_overlap = len(
    train_hash_set &
    test_hash_set
)

val_test_overlap = len(
    val_hash_set &
    test_hash_set
)


print(
    f"TRAIN ↔ VALIDATION : "
    f"{train_val_overlap:,}"
)

print(
    f"TRAIN ↔ TEST       : "
    f"{train_test_overlap:,}"
)

print(
    f"VALIDATION ↔ TEST  : "
    f"{val_test_overlap:,}"
)


# ============================================================
# RAW IDENTIFIER AUDIT
# ============================================================

print("\n" + "=" * 70)
print("RAW VARIANT IDENTIFIER AUDIT")
print("=" * 70)


candidate_identifiers = [

    "vep_hgvsc",

    "vep_hgvsp",

    "Protein change",

    "Gene(s)",

    "vep_gene",

    "vep_mane_select",

    "vep_amino_acids",

    "Chromosome",

    "chromosome",

    "Position",

    "position",

    "start",

    "end",

    "ref",

    "alt"
]


available_identifiers = [
    col
    for col in candidate_identifiers
    if (
        col in train.columns
        and
        col in val.columns
        and
        col in test.columns
    )
]


print(
    "\nAvailable identifiers:"
)


if available_identifiers:

    for col in available_identifiers:

        print(
            f"  {col}"
        )

else:

    print(
        "  No standard identifiers found."
    )


# ============================================================
# IDENTIFIER OVERLAP
# ============================================================

identifier_results = []


for col in available_identifiers:

    train_ids = set(
        train[col]
        .dropna()
        .astype(str)
    )

    val_ids = set(
        val[col]
        .dropna()
        .astype(str)
    )

    test_ids = set(
        test[col]
        .dropna()
        .astype(str)
    )

    identifier_results.append({

        "identifier":
            col,

        "train_unique":
            len(train_ids),

        "validation_unique":
            len(val_ids),

        "test_unique":
            len(test_ids),

        "train_validation_overlap":
            len(
                train_ids &
                val_ids
            ),

        "train_test_overlap":
            len(
                train_ids &
                test_ids
            ),

        "validation_test_overlap":
            len(
                val_ids &
                test_ids
            )
    })


identifier_df = pd.DataFrame(
    identifier_results
)


if len(identifier_df):

    identifier_file = (
        AUDIT_DIR /
        "identifier_overlap.csv"
    )

    identifier_df.to_csv(
        identifier_file,
        index=False
    )

    print(
        "\nIdentifier overlap:"
    )

    print(
        identifier_df.to_string(
            index=False
        )
    )


# ============================================================
# SUMMARY
# ============================================================

summary = []

summary.append(
    "BIOLOGICAL FEATURE / LABEL LEAKAGE AUDIT"
)

summary.append(
    "=" * 60
)

summary.append(
    f"Selected features: "
    f"{len(selected_features)}"
)

summary.append(
    f"Training rows: "
    f"{len(train):,}"
)

summary.append(
    f"Validation rows: "
    f"{len(val):,}"
)

summary.append(
    f"Test rows: "
    f"{len(test):,}"
)

summary.append("")

summary.append(
    "Top numerical features by univariate AUC:"
)


for _, row in numeric_auc_df.head(10).iterrows():

    summary.append(
        f"  {row['feature']}: "
        f"AUC={row['roc_auc']:.4f}, "
        f"strength={row['auc_strength']:.4f}"
    )


summary.append("")

summary.append(
    f"Exact selected-feature-vector overlap:"
)

summary.append(
    f"  Train/Validation: "
    f"{train_val_overlap:,}"
)

summary.append(
    f"  Train/Test: "
    f"{train_test_overlap:,}"
)

summary.append(
    f"  Validation/Test: "
    f"{val_test_overlap:,}"
)

summary.append("")

summary.append(
    "Interpretation:"
)

summary.append(
    "High predictive power does not automatically mean leakage."
)

summary.append(
    "Each highly predictive feature must be evaluated "
    "for biological validity and data provenance."
)

summary.append(
    "Test labels were not used for feature selection."
)


summary_file = (
    AUDIT_DIR /
    "audit_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)


print(
    "\nFiles created:"
)

print(
    f"  {feature_audit_file}"
)

print(
    f"  {distribution_file}"
)

print(
    f"  {summary_file}"
)

if len(identifier_df):

    print(
        f"  {AUDIT_DIR / 'identifier_overlap.csv'}"
    )


print("\nNo datasets were modified.")

print(
    "\nNext step:"
)

print(
    "Review the audit results before finalizing "
    "the NN/QNN feature representation."
)