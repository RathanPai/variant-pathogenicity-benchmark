import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Data_Final/variant_features_cleaned.csv"

TRAIN_FILE = "Data_Final/train.csv"
VAL_FILE = "Data_Final/validation.csv"
TEST_FILE = "Data_Final/test.csv"

RANDOM_STATE = 42

os.makedirs("Data_Final", exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")

if "label" not in df.columns:
    raise ValueError("ERROR: 'label' column not found.")

print("\nOriginal label distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]


# ============================================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# SECOND SPLIT
# TEMPORARY → 15% VALIDATION + 15% TEST
# ============================================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


# ============================================================
# REBUILD DATAFRAMES
# ============================================================

train_df = X_train.copy()
train_df["label"] = y_train.values

val_df = X_val.copy()
val_df["label"] = y_val.values

test_df = X_test.copy()
test_df["label"] = y_test.values


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(
    f"\nTRAIN      : {len(train_df):,} rows "
    f"({len(train_df)/len(df)*100:.2f}%)"
)

print(
    f"VALIDATION : {len(val_df):,} rows "
    f"({len(val_df)/len(df)*100:.2f}%)"
)

print(
    f"TEST       : {len(test_df):,} rows "
    f"({len(test_df)/len(df)*100:.2f}%)"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    counts = data["label"].value_counts().sort_index()
    proportions = data["label"].value_counts(
        normalize=True
    ).sort_index()

    print(f"\n{name}")

    for label in [0, 1]:

        print(
            f"  label {label}: "
            f"{counts.get(label, 0):,} "
            f"({proportions.get(label, 0) * 100:.3f}%)"
        )


# ============================================================
# CHECK FOR OVERLAPPING ROWS
# ============================================================

print("\n" + "=" * 70)
print("SPLIT OVERLAP CHECK")
print("=" * 70)

# Use original row indices to make sure no source row
# appears in multiple partitions.
#
# Because X_train/X_val/X_test retain the original indices,
# this is a reliable check.

train_indices = set(X_train.index)
val_indices = set(X_val.index)
test_indices = set(X_test.index)

print(
    f"\nTrain ∩ Validation: "
    f"{len(train_indices & val_indices)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_indices & test_indices)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(val_indices & test_indices)}"
)

if (
    len(train_indices & val_indices) == 0
    and len(train_indices & test_indices) == 0
    and len(val_indices & test_indices) == 0
):
    print("\n✓ No row overlap detected.")
else:
    raise RuntimeError(
        "ERROR: Dataset overlap detected!"
    )


# ============================================================
# CHECK TOTAL
# ============================================================

total = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print(
    f"\nTotal rows across splits: {total:,}"
)

if total != len(df):
    raise RuntimeError(
        "ERROR: Split sizes do not add up to original dataset."
    )

print("\n✓ All original rows accounted for.")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print("\nCreated:")

print(f"  {TRAIN_FILE}")
print(f"  {VAL_FILE}")
print(f"  {TEST_FILE}")

print("\nNext step:")
print("  TRAIN-ONLY PREPROCESSING")
print("  → categorical encoding")
print("  → missing-value handling")
print("  → numerical scaling")
print("  → preprocessing pipeline")