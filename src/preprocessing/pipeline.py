"""
PREPROCESS VARIANT DATASET
==========================

Input:
    Data/train(1).csv
    Data/validation(1).csv
    Data/test(1).csv

Output:
    Data/processed/train_processed.csv
    Data/processed/validation_processed.csv
    Data/processed/test_processed.csv
    Data/processed/preprocessor.joblib
    Data/processed/feature_names.txt
    Data/processed/preprocessing_metadata.json

IMPORTANT:
    The preprocessing transformer is FIT ONLY on TRAIN.
    Validation and test are TRANSFORMED using the training-fitted transformer.
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("Data")
OUTPUT_DIR = DATA_DIR / "processed"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

TARGET = "label"

# Do NOT one-hot encode these.
# They are variant/gene identifiers or sequence-like fields
# with very high cardinality.
HIGH_CARDINALITY_EXCLUDE = {
    "Gene(s)",
    "vep_gene",
    "vep_hgvsc",
    "vep_hgvsp",
    "Protein change",
    "vep_amino_acids",
}

# Object/string columns above this number of unique values
# are excluded rather than one-hot encoded.
MAX_CATEGORICAL_CARDINALITY = 500


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"'{TARGET}' not found in {name}."
        )

    print(f"{name:<15}: {df.shape}")

    return df


# ============================================================
# FEATURE CLASSIFICATION
# ============================================================

def classify_features(X):

    numeric_features = []
    categorical_features = []
    excluded_features = []

    for col in X.columns:

        # Explicitly excluded biological identifiers
        if col in HIGH_CARDINALITY_EXCLUDE:
            excluded_features.append(col)
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_features.append(col)
            continue

        # Boolean columns
        if pd.api.types.is_bool_dtype(X[col]):
            numeric_features.append(col)
            continue

        # Remaining string/object/category columns
        unique_count = X[col].nunique(dropna=True)

        if unique_count <= MAX_CATEGORICAL_CARDINALITY:
            categorical_features.append(col)
        else:
            excluded_features.append(col)

    return (
        numeric_features,
        categorical_features,
        excluded_features
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VARIANT FEATURE PREPROCESSING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    train = load_data(TRAIN_FILE, "TRAIN")
    val = load_data(VAL_FILE, "VALIDATION")
    test = load_data(TEST_FILE, "TEST")

    # --------------------------------------------------------
    # 2. CHECK COLUMN CONSISTENCY
    # --------------------------------------------------------

    train_features = [
        c for c in train.columns
        if c != TARGET
    ]

    val_features = [
        c for c in val.columns
        if c != TARGET
    ]

    test_features = [
        c for c in test.columns
        if c != TARGET
    ]

    if train_features != val_features:
        raise ValueError(
            "TRAIN and VALIDATION feature columns/order differ."
        )

    if train_features != test_features:
        raise ValueError(
            "TRAIN and TEST feature columns/order differ."
        )

    # --------------------------------------------------------
    # 3. SEPARATE FEATURES AND LABEL
    # --------------------------------------------------------

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].copy()

    X_val = val.drop(columns=[TARGET])
    y_val = val[TARGET].copy()

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].copy()

    # --------------------------------------------------------
    # 4. CLASSIFY FEATURES
    # --------------------------------------------------------

    (
        numeric_features,
        categorical_features,
        excluded_features
    ) = classify_features(X_train)

    print("\n" + "=" * 70)
    print("FEATURE CLASSIFICATION")
    print("=" * 70)

    print(f"Original features : {len(train_features)}")
    print(f"Numeric features  : {len(numeric_features)}")
    print(f"Categorical       : {len(categorical_features)}")
    print(f"Excluded          : {len(excluded_features)}")

    print("\n--- CATEGORICAL FEATURES ---")

    for col in categorical_features:
        print(
            f"{col:<40}"
            f"unique={X_train[col].nunique(dropna=True):,}"
        )

    print("\n--- EXCLUDED FEATURES ---")

    for col in excluded_features:
        print(
            f"{col:<40}"
            f"unique={X_train[col].nunique(dropna=True):,}"
        )

    # --------------------------------------------------------
    # 5. MISSING VALUE REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING MISSING VALUES")
    print("=" * 70)

    missing = X_train.isna().sum()
    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if len(missing) == 0:
        print("No missing values.")
    else:
        for col, count in missing.items():
            percentage = 100 * count / len(X_train)

            print(
                f"{col:<45}"
                f"{count:>8,} "
                f"({percentage:6.2f}%)"
            )

    # --------------------------------------------------------
    # 6. NUMERICAL PIPELINE
    #
    # Median and mean/std are learned ONLY from TRAIN.
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # 7. CATEGORICAL PIPELINE
    #
    # Categories are learned ONLY from TRAIN.
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 8. COMBINE
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    # --------------------------------------------------------
    # 9. FIT ONLY ON TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FITTING PREPROCESSOR")
    print("=" * 70)

    print("Fitting on TRAIN only...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    print("TRAIN preprocessing complete.")

    # --------------------------------------------------------
    # 10. TRANSFORM VALIDATION
    # --------------------------------------------------------

    print("\nTransforming VALIDATION...")

    X_val_processed = preprocessor.transform(
        X_val
    )

    print("VALIDATION preprocessing complete.")

    # --------------------------------------------------------
    # 11. TRANSFORM TEST
    # --------------------------------------------------------

    print("\nTransforming TEST...")

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("TEST preprocessing complete.")

    # --------------------------------------------------------
    # 12. GET FEATURE NAMES
    # --------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    print("\n" + "=" * 70)
    print("FINAL FEATURE DIMENSIONS")
    print("=" * 70)

    print(
        f"TRAIN      : "
        f"{X_train_processed.shape}"
    )

    print(
        f"VALIDATION : "
        f"{X_val_processed.shape}"
    )

    print(
        f"TEST       : "
        f"{X_test_processed.shape}"
    )

    print(
        f"\nFinal feature count: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # 13. CHECK FOR NaN / INF
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)

    import numpy as np

    datasets = {
        "TRAIN": X_train_processed,
        "VALIDATION": X_val_processed,
        "TEST": X_test_processed
    }

    for name, X in datasets.items():

        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()

        print(
            f"{name:<12} "
            f"NaN={nan_count:,} "
            f"Inf={inf_count:,}"
        )

        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # --------------------------------------------------------
    # 14. CREATE DATAFRAMES
    # --------------------------------------------------------

    train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    train_processed[TARGET] = y_train.values

    val_processed = pd.DataFrame(
        X_val_processed,
        columns=feature_names
    )

    val_processed[TARGET] = y_val.values

    test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )

    test_processed[TARGET] = y_test.values

    # --------------------------------------------------------
    # 15. SAVE
    # --------------------------------------------------------

    train_output = (
        OUTPUT_DIR / "train_processed.csv"
    )

    val_output = (
        OUTPUT_DIR / "validation_processed.csv"
    )

    test_output = (
        OUTPUT_DIR / "test_processed.csv"
    )

    preprocessor_output = (
        OUTPUT_DIR / "preprocessor.joblib"
    )

    feature_names_output = (
        OUTPUT_DIR / "feature_names.txt"
    )

    metadata_output = (
        OUTPUT_DIR / "preprocessing_metadata.json"
    )

    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    print("Saving TRAIN...")
    train_processed.to_csv(
        train_output,
        index=False
    )

    print("Saving VALIDATION...")
    val_processed.to_csv(
        val_output,
        index=False
    )

    print("Saving TEST...")
    test_processed.to_csv(
        test_output,
        index=False
    )

    print("Saving preprocessor...")
    joblib.dump(
        preprocessor,
        preprocessor_output
    )

    print("Saving feature names...")

    with open(
        feature_names_output,
        "w",
        encoding="utf-8"
    ) as f:
        for name in feature_names:
            f.write(str(name) + "\n")

    # --------------------------------------------------------
    # 16. SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "target": TARGET,

        "input_features": len(train_features),

        "numeric_features": numeric_features,

        "categorical_features": categorical_features,

        "excluded_features": excluded_features,

        "final_feature_count": len(feature_names),

        "train_shape": list(train_processed.shape),

        "validation_shape": list(val_processed.shape),

        "test_shape": list(test_processed.shape),

        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder",
            "unknown_categories": "ignored",
            "fit_dataset": "TRAIN ONLY"
        }
    }

    with open(
        metadata_output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 17. LABEL CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    print("\nTRAIN:")
    print(
        y_train.value_counts(
            normalize=False
        ).sort_index()
    )

    print("\nVALIDATION:")
    print(
        y_val.value_counts(
            normalize=False
        ).sort_index()
    )

    print("\nTEST:")
    print(
        y_test.value_counts(
            normalize=False
        ).sort_index()
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print("\nFiles created:")

    print(f"  {train_output}")
    print(f"  {val_output}")
    print(f"  {test_output}")
    print(f"  {preprocessor_output}")
    print(f"  {feature_names_output}")
    print(f"  {metadata_output}")

    print("\nIMPORTANT:")
    print("  Preprocessor FIT       : TRAIN only")
    print("  Validation             : TRANSFORM only")
    print("  Test                   : TRANSFORM only")

    print("\nNext stage:")
    print("  Feature selection / dimensionality reduction")
    print("  → reduce to QNN-compatible number of features")


if __name__ == "__main__":
    main()