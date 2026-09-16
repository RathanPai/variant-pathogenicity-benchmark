import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("Data/clinvar_dataset.tsv")
OUTPUT_FILE = Path("Data/clinvar_clean.csv")

# ============================================================
# 1. LOAD CLINVAR
# ============================================================

print("=" * 70)
print("CLINVAR PREPROCESSING")
print("=" * 70)

print("\nLoading:", INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str,
    low_memory=False,
    index_col=False
)

print(f"Original rows: {len(df):,}")
print(f"Original columns: {len(df.columns)}")

# ============================================================
# 2. KEEP ONLY GERMLINE PATHOGENIC/BENIGN CLASSIFICATIONS
# ============================================================

classification_col = "Germline classification"

# Exact classifications we want
positive_labels = {
    "Pathogenic": 1,
    "Likely pathogenic": 1,
    "Pathogenic/Likely pathogenic": 1
}

negative_labels = {
    "Benign": 0,
    "Likely benign": 0,
    "Benign/Likely benign": 0
}

allowed_labels = set(positive_labels) | set(negative_labels)

before = len(df)

df = df[
    df[classification_col].isin(allowed_labels)
].copy()

print("\nAfter classification filtering:")
print(f"Rows: {len(df):,}")
print(f"Removed: {before - len(df):,}")

# ============================================================
# 3. CREATE BINARY TARGET
# ============================================================

df["label"] = df[classification_col].map(
    {**positive_labels, **negative_labels}
)

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nLabel percentages:")

print(
    (df["label"].value_counts(normalize=True)
     .sort_index() * 100)
    .round(2)
)

# ============================================================
# 4. REMOVE THE ONE HAPLOTYPE
# ============================================================

print("\nVariant types BEFORE filtering:")

print(
    df["Variant type"]
    .value_counts()
    .to_string()
)

# Keep SNVs and indels/deletions/insertions for now.
# Remove the single haplotype record.

df = df[
    df["Variant type"] != "Haplotype"
].copy()

print("\nRemoved Haplotype variants.")
print(f"Remaining rows: {len(df):,}")

# ============================================================
# 5. CREATE SIMPLIFIED VARIANT TYPE
# ============================================================

def simplify_variant_type(x):

    if pd.isna(x):
        return "Unknown"

    x = x.strip().lower()

    if x == "single nucleotide variant":
        return "SNV"

    if x == "deletion":
        return "Deletion"

    if x == "duplication":
        return "Duplication"

    if x == "indel":
        return "Indel"

    if x == "insertion":
        return "Insertion"

    if x == "haplotype":
        return "Haplotype"

    return "Other"


df["variant_type_simple"] = (
    df["Variant type"]
    .apply(simplify_variant_type)
)

# ============================================================
# 6. CREATE SIMPLIFIED MOLECULAR CONSEQUENCE FEATURES
# ============================================================

# ClinVar sometimes contains multiple consequences separated by "|".
#
# Example:
# missense variant|intron variant
#
# Instead of throwing these records away, we create
# binary feature columns.

consequence_features = {
    "is_missense": "missense variant",
    "is_frameshift": "frameshift variant",
    "is_nonsense": "nonsense",
    "is_splice_donor": "splice donor variant",
    "is_splice_acceptor": "splice acceptor variant",
    "is_synonymous": "synonymous variant",
    "is_intron": "intron variant",
    "is_utr_5": "5 prime UTR variant",
    "is_utr_3": "3 prime UTR variant",
    "is_non_coding": "non-coding transcript variant",
    "is_initiator": "initiator_codon_variant",
    "is_stop_lost": "stop lost",
    "is_inframe_insertion": "inframe_insertion",
    "is_inframe_deletion": "inframe_deletion",
    "is_inframe_indel": "inframe_indel"
}

consequence_col = "Molecular consequence"

for new_col, consequence in consequence_features.items():

    df[new_col] = (
        df[consequence_col]
        .fillna("")
        .str.lower()
        .str.contains(consequence.lower(), regex=False)
        .astype(int)
    )

# ============================================================
# 7. BASIC GENE CLEANING
# ============================================================

# Keep missing genes for now.
# We don't want to throw away ~12% of the dataset unnecessarily.

df["gene_available"] = (
    df["Gene(s)"]
    .notna()
    .astype(int)
)

# ============================================================
# 8. PROTEIN CHANGE AVAILABILITY
# ============================================================

df["protein_change_available"] = (
    df["Protein change"]
    .notna()
    .astype(int)
)

# ============================================================
# 9. CHROMOSOME NORMALIZATION
# ============================================================

for col in ["GRCh37Chromosome", "GRCh38Chromosome"]:

    if col in df.columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace("chr", "", case=False, regex=False)
            .str.strip()
        )

# ============================================================
# 10. REMOVE COMPLETELY IRRELEVANT COLUMNS
# ============================================================

columns_to_remove = [

    # Somatic information
    "Somatic clinical impact",
    "Somatic clinical impact date last evaluated",
    "Somatic clinical impact review status",

    # Oncogenicity information
    "Oncogenicity classification",
    "Oncogenicity date last evaluated",
    "Oncogenicity review status",

    # We don't need the date as an ML feature
    "Germline date last evaluated",

    # We'll keep review status for analysis,
    # but NOT use it as a predictive feature yet.
]

columns_to_remove = [
    c for c in columns_to_remove
    if c in df.columns
]

df = df.drop(columns=columns_to_remove)

# ============================================================
# 11. REMOVE EXACT DUPLICATE ROWS
# ============================================================

before = len(df)

df = df.drop_duplicates()

print("\nDuplicate removal:")
print(f"Removed: {before - len(df):,}")
print(f"Remaining: {len(df):,}")

# ============================================================
# 12. CHECK DUPLICATE VARIATION IDs
# ============================================================

print("\nDuplicate Variation IDs:")

duplicate_ids = df["VariationID"].duplicated().sum()

print(f"{duplicate_ids:,}")

# IMPORTANT:
# We are NOT removing duplicate VariationIDs yet.
#
# A ClinVar variation can potentially have multiple records/
# representations associated with it.
#
# We'll investigate this before deciding how to deduplicate.

# ============================================================
# 13. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nLabels:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nVariant types:")

print(
    df["variant_type_simple"]
    .value_counts()
    .to_string()
)

print("\nGene availability:")

print(
    df["gene_available"]
    .value_counts()
    .to_string()
)

print("\nProtein change availability:")

print(
    df["protein_change_available"]
    .value_counts()
    .to_string()
)

# ============================================================
# 14. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nClean dataset saved as:")
print(OUTPUT_FILE)

print("\nYou can now use this file for feature engineering.")