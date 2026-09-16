import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("Data/clinvar_dataset.tsv")
OUTPUT_FILE = Path("Data/clinvar_clean.csv")

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("CLINVAR CLEANING")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    dtype=str,
    low_memory=False,
    index_col=False
)

print(f"\nOriginal rows    : {len(df):,}")
print(f"Original columns : {len(df.columns)}")

# ============================================================
# 2. CLEAN CLASSIFICATION TEXT
# ============================================================

classification_col = "Germline classification"

df[classification_col] = (
    df[classification_col]
    .astype("string")
    .str.strip()
)

# ============================================================
# 3. CREATE BINARY LABEL
# ============================================================

PATHOGENIC = {
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic"
}

BENIGN = {
    "benign",
    "likely benign",
    "benign/likely benign"
}


def create_label(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    if value in PATHOGENIC:
        return 1

    if value in BENIGN:
        return 0

    # Ambiguous classifications
    return pd.NA


df["label"] = df[classification_col].apply(create_label)

# ============================================================
# 4. REMOVE AMBIGUOUS CLASSIFICATIONS
# ============================================================

before = len(df)

df = df[df["label"].notna()].copy()

df["label"] = df["label"].astype(int)

print("\nClassification filtering")
print("-" * 70)
print(f"Removed ambiguous variants : {before - len(df):,}")
print(f"Remaining variants         : {len(df):,}")

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nLabel percentages:")
print(
    (df["label"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

# ============================================================
# 5. REMOVE HAPLOTYPE
# ============================================================

before = len(df)

df = df[
    df["Variant type"].str.lower().ne("haplotype")
].copy()

print("\nHaplotype filtering")
print("-" * 70)
print(f"Removed haplotypes : {before - len(df):,}")
print(f"Remaining          : {len(df):,}")

# ============================================================
# 6. NORMALIZE VARIANT TYPE
# ============================================================

def simplify_variant_type(value):

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    mapping = {
        "single nucleotide variant": "SNV",
        "deletion": "Deletion",
        "duplication": "Duplication",
        "indel": "Indel",
        "insertion": "Insertion",
        "haplotype": "Haplotype"
    }

    return mapping.get(value, "Other")


df["variant_type_simple"] = (
    df["Variant type"]
    .apply(simplify_variant_type)
)

print("\nVariant types")
print("-" * 70)
print(
    df["variant_type_simple"]
    .value_counts()
    .to_string()
)

# ============================================================
# 7. MOLECULAR CONSEQUENCE FEATURES
# ============================================================

consequence_col = "Molecular consequence"

consequence_features = {

    "is_missense":
        "missense variant",

    "is_frameshift":
        "frameshift variant",

    "is_nonsense":
        "nonsense",

    "is_splice_donor":
        "splice donor variant",

    "is_splice_acceptor":
        "splice acceptor variant",

    "is_synonymous":
        "synonymous variant",

    "is_intron":
        "intron variant",

    "is_5prime_utr":
        "5 prime UTR variant",

    "is_3prime_utr":
        "3 prime UTR variant",

    "is_non_coding":
        "non-coding transcript variant",

    "is_initiator_codon":
        "initiator_codon_variant",

    "is_stop_lost":
        "stop lost",

    "is_inframe_insertion":
        "inframe_insertion",

    "is_inframe_deletion":
        "inframe_deletion",

    "is_inframe_indel":
        "inframe_indel",

    "is_genic_upstream":
        "genic upstream transcript variant",

    "is_genic_downstream":
        "genic downstream transcript variant"
}


for new_column, consequence in consequence_features.items():

    df[new_column] = (
        df[consequence_col]
        .fillna("")
        .str.lower()
        .str.contains(
            consequence.lower(),
            regex=False
        )
        .astype(int)
    )

# ============================================================
# 8. GENE AVAILABILITY
# ============================================================

df["gene_available"] = (
    df["Gene(s)"]
    .notna()
    .astype(int)
)

# ============================================================
# 9. PROTEIN CHANGE AVAILABILITY
# ============================================================

# Treat common ClinVar placeholders as missing information.

protein = (
    df["Protein change"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

invalid_protein_values = {
    "",
    "not provided",
    "not specified",
    "unknown"
}

df["protein_change_available"] = (
    ~protein.isin(invalid_protein_values)
).astype(int)

# ============================================================
# 10. NORMALIZE GRCh38 CHROMOSOME
# ============================================================

if "GRCh38Chromosome" in df.columns:

    df["GRCh38Chromosome"] = (
        df["GRCh38Chromosome"]
        .astype(str)
        .str.replace(
            "chr",
            "",
            case=False,
            regex=False
        )
        .str.strip()
    )

# ============================================================
# 11. CONVERT GRCh38 LOCATION TO NUMERIC
# ============================================================

if "GRCh38Location" in df.columns:

    df["GRCh38Location"] = pd.to_numeric(
        df["GRCh38Location"],
        errors="coerce"
    )

# ============================================================
# 12. KEEP ONLY USEFUL COLUMNS
# ============================================================

columns_to_keep = [

    # Variant identity
    "Name",
    "VariationID",
    "AlleleID(s)",
    "dbSNP ID",
    "Canonical SPDI",

    # Gene / protein
    "Gene(s)",
    "Protein change",

    # Genomic location
    "GRCh38Chromosome",
    "GRCh38Location",

    # Original variant information
    "Variant type",
    "variant_type_simple",
    "Molecular consequence",

    # Review information
    "Germline review status",

    # Engineered biological features
    "is_missense",
    "is_frameshift",
    "is_nonsense",
    "is_splice_donor",
    "is_splice_acceptor",
    "is_synonymous",
    "is_intron",
    "is_5prime_utr",
    "is_3prime_utr",
    "is_non_coding",
    "is_initiator_codon",
    "is_stop_lost",
    "is_inframe_insertion",
    "is_inframe_deletion",
    "is_inframe_indel",
    "is_genic_upstream",
    "is_genic_downstream",

    # Availability indicators
    "gene_available",
    "protein_change_available",

    # TARGET — keep this LAST
    "label"
]

# Only keep columns that actually exist
columns_to_keep = [
    col for col in columns_to_keep
    if col in df.columns
]

df = df[columns_to_keep].copy()

# ============================================================
# 13. REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

print("\nDuplicate removal")
print("-" * 70)
print(f"Exact duplicates removed : {before - len(df):,}")
print(f"Remaining variants       : {len(df):,}")

# ============================================================
# 14. CHECK DUPLICATE VARIATION IDs
# ============================================================

duplicate_variation_ids = (
    df["VariationID"].duplicated().sum()
)

print(
    f"\nDuplicate VariationIDs : "
    f"{duplicate_variation_ids:,}"
)

# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

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

print("\nMissing values:")
print(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

# ============================================================
# 16. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"\nSaved cleaned dataset to:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Feature engineering with external biological databases.")