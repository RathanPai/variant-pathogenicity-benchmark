import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Data/variant_features.csv"

OUTPUT_CLEAN = "Data_Final/variant_features_cleaned.csv"
OUTPUT_METADATA = "Data_Final/variant_metadata.csv"
OUTPUT_ANOMALIES = "Data_Final/protein_position_anomalies.csv"
OUTPUT_REPORT = "Data_Final/cleaning_report.txt"

os.makedirs("Data_Final", exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Return the first candidate column that exists.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def drop_if_exists(df, columns, reason, report):
    """
    Drop columns only if they actually exist.
    """
    existing = [c for c in columns if c in df.columns]

    if existing:
        df.drop(columns=existing, inplace=True)

        report.append(
            f"\n{reason}\n"
            + "\n".join(f"  DROP: {c}" for c in existing)
        )

    return existing


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CLEANING ORIGINAL VARIANT DATASET")
print("=" * 70)

print(f"\nLoading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print(f"Rows:    {original_rows:,}")
print(f"Columns: {original_columns}")


# ============================================================
# REPORT
# ============================================================

report = []

report.append("=" * 70)
report.append("VARIANT DATASET CLEANING REPORT")
report.append("=" * 70)

report.append(f"\nInput file: {INPUT_FILE}")
report.append(f"Original rows: {original_rows:,}")
report.append(f"Original columns: {original_columns}")


# ============================================================
# 1. CHECK LABEL
# ============================================================

print("\n" + "=" * 70)
print("1. LABEL CHECK")
print("=" * 70)

if "label" not in df.columns:
    raise ValueError("ERROR: 'label' column not found.")

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False).sort_index())

invalid_labels = df["label"].isna() | ~df["label"].isin([0, 1])

print(f"\nInvalid/missing labels: {invalid_labels.sum():,}")

if invalid_labels.sum() > 0:
    raise ValueError(
        "Invalid labels detected. Fix these before continuing."
    )

report.append("\nLABEL")
report.append("  No missing/invalid labels found.")


# ============================================================
# 2. CHECK ORIGINAL VARIANT IDENTIFIERS
# ============================================================

print("\n" + "=" * 70)
print("2. VARIANT IDENTIFIER CHECK")
print("=" * 70)

identifier_candidates = [
    "VariationID",
    "AlleleID(s)",
    "Canonical SPDI",
    "Name",
    "dbSNP ID",
]

identifier_columns = [
    c for c in identifier_candidates
    if c in df.columns
]

print("\nIdentifier columns found:")

for col in identifier_columns:
    unique_count = df[col].nunique(dropna=False)
    duplicate_count = len(df) - unique_count

    print(
        f"  {col:<25} "
        f"unique={unique_count:,} "
        f"duplicates={duplicate_count:,}"
    )

report.append("\nVARIANT IDENTIFIERS")

for col in identifier_columns:
    unique_count = df[col].nunique(dropna=False)
    duplicate_count = len(df) - unique_count

    report.append(
        f"  {col}: unique={unique_count:,}, "
        f"duplicates={duplicate_count:,}"
    )


# ============================================================
# 3. FIND TRUE DUPLICATE SOURCE VARIANTS
# ============================================================

print("\n" + "=" * 70)
print("3. TRUE DUPLICATE SOURCE-VARIANT CHECK")
print("=" * 70)

# VariationID is our strongest source-level identifier.
primary_id = find_column(
    df,
    ["VariationID", "Canonical SPDI", "AlleleID(s)"]
)

if primary_id is not None:

    duplicate_mask = df[primary_id].duplicated(keep=False)

    duplicate_count = duplicate_mask.sum()

    print(
        f"\nPrimary identifier: {primary_id}"
    )

    print(
        f"Rows participating in duplicate "
        f"{primary_id}: {duplicate_count:,}"
    )

    if duplicate_count > 0:

        duplicate_variants = (
            df.loc[duplicate_mask]
            .sort_values(primary_id)
        )

        duplicate_variants.to_csv(
            "Data/source_duplicate_variants.csv",
            index=False
        )

        print(
            "\nWARNING: Actual duplicate source variants "
            "were found."
        )

        print(
            "Saved: Data/source_duplicate_variants.csv"
        )

        report.append(
            f"\nTRUE SOURCE DUPLICATES\n"
            f"  Primary identifier: {primary_id}\n"
            f"  Duplicate rows: {duplicate_count:,}\n"
            f"  Saved: Data/source_duplicate_variants.csv"
        )

    else:

        print(
            "\nNo duplicate source variants detected."
        )

        report.append(
            "\nTRUE SOURCE DUPLICATES\n"
            "  None detected."
        )

else:

    print(
        "\nWARNING: No reliable variant identifier found."
    )

    report.append(
        "\nTRUE SOURCE DUPLICATES\n"
        "  Could not determine — no reliable identifier."
    )


# ============================================================
# 4. PROTEIN POSITION ANOMALIES
# ============================================================

print("\n" + "=" * 70)
print("4. PROTEIN POSITION ANOMALIES")
print("=" * 70)

required_position_columns = [
    "vep_protein_position",
    "protein_length"
]

if all(c in df.columns for c in required_position_columns):

    position = pd.to_numeric(
        df["vep_protein_position"],
        errors="coerce"
    )

    length = pd.to_numeric(
        df["protein_length"],
        errors="coerce"
    )

    anomaly_mask = (
        position.notna()
        & length.notna()
        & (position > length)
    )

    anomalies = df.loc[anomaly_mask].copy()

    print(
        f"\nProtein position > protein length: "
        f"{len(anomalies):,}"
    )

    if len(anomalies) > 0:

        # Add useful diagnostic information
        anomalies["position_minus_length"] = (
            position.loc[anomaly_mask]
            - length.loc[anomaly_mask]
        )

        if "relative_protein_position" in anomalies.columns:
            anomalies["relative_position_recomputed"] = (
                anomalies["vep_protein_position"]
                / anomalies["protein_length"]
            )

        anomalies.to_csv(
            OUTPUT_ANOMALIES,
            index=False
        )

        print(
            f"Saved: {OUTPUT_ANOMALIES}"
        )

        print("\nLargest anomalies:")

        display_columns = [
            c for c in [
                "VariationID",
                "AlleleID(s)",
                "Canonical SPDI",
                "vep_protein_id",
                "uniprot_accession",
                "vep_protein_position",
                "protein_length",
                "relative_protein_position",
                "protein_position_valid",
                "label",
                "position_minus_length"
            ]
            if c in anomalies.columns
        ]

        print(
            anomalies
            .sort_values("position_minus_length", ascending=False)
            [display_columns]
            .head(20)
            .to_string(index=False)
        )

        report.append(
            f"\nPROTEIN POSITION ANOMALIES\n"
            f"  Found: {len(anomalies):,}\n"
            f"  Action: FLAG ONLY — NOT DELETED\n"
            f"  Saved: {OUTPUT_ANOMALIES}"
        )

    else:

        print("No protein-position anomalies found.")

        report.append(
            "\nPROTEIN POSITION ANOMALIES\n"
            "  None found."
        )

else:

    print(
        "\nRequired protein-position columns unavailable."
    )

    report.append(
        "\nPROTEIN POSITION ANOMALIES\n"
        "  Could not evaluate."
    )


# ============================================================
# 5. CREATE METADATA / TRACEABILITY FILE
# ============================================================

print("\n" + "=" * 70)
print("5. CREATING VARIANT METADATA")
print("=" * 70)

metadata_candidates = [
    "VariationID",
    "AlleleID(s)",
    "Name",
    "Canonical SPDI",
    "dbSNP ID",
    "Gene(s)",
    "vep_gene",
    "vep_gene_id",
    "vep_transcript",
    "vep_protein_id",
    "uniprot_accession",
    "GRCh38Chromosome",
    "GRCh38Location",
    "Protein change",
    "label",
]

metadata_columns = [
    c for c in metadata_candidates
    if c in df.columns
]

metadata = df[metadata_columns].copy()

metadata.to_csv(
    OUTPUT_METADATA,
    index=False
)

print(
    f"Saved: {OUTPUT_METADATA}"
)

print(
    f"Metadata columns: {len(metadata_columns)}"
)

report.append(
    f"\nMETADATA\n"
    f"  Columns saved: {len(metadata_columns)}\n"
    f"  File: {OUTPUT_METADATA}"
)


# ============================================================
# 6. DROP PURE IDENTIFIERS / MEMORIZATION FEATURES
# ============================================================

print("\n" + "=" * 70)
print("6. DROPPING IDENTIFIER / MEMORIZATION FEATURES")
print("=" * 70)

identifier_features = [
    "VariationID",
    "AlleleID(s)",
    "Name",
    "Canonical SPDI",
    "dbSNP ID",

    # Gene / transcript / protein identifiers
    "vep_gene_id",
    "vep_transcript",
    "vep_protein_id",
    "uniprot_accession",

    # Genomic coordinates
    "GRCh38Chromosome",
    "GRCh38Location",
]

dropped_identifiers = drop_if_exists(
    df,
    identifier_features,
    "IDENTIFIER / MEMORIZATION FEATURES",
    report
)

print(
    f"\nDropped: {len(dropped_identifiers)} columns"
)


# ============================================================
# 7. DROP EMPTY / CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("7. DROPPING EMPTY / CONSTANT FEATURES")
print("=" * 70)

constant_columns = []

for col in df.columns:

    if col == "label":
        continue

    nunique = df[col].nunique(dropna=False)

    if nunique <= 1:
        constant_columns.append(col)

print(
    f"\nConstant/empty features found: "
    f"{len(constant_columns)}"
)

for col in constant_columns:
    print(f"  DROP: {col}")

if constant_columns:
    df.drop(
        columns=constant_columns,
        inplace=True
    )

report.append(
    "\nCONSTANT / EMPTY FEATURES\n"
    + (
        "\n".join(
            f"  DROP: {c}"
            for c in constant_columns
        )
        if constant_columns
        else "  None."
    )
)


# ============================================================
# 8. DROP OPERATIONAL GNOMAD FEATURES
# ============================================================

print("\n" + "=" * 70)
print("8. DROPPING OPERATIONAL gnomAD FEATURES")
print("=" * 70)

gnomad_operational = [
    "gnomad_variant_id",
    "gnomad_id_status",
    "gnomad_found",
    "gnomad_status",
    "gnomad_variant_id_returned",
    "gnomad_rsid",
    "gnomad_chrom",
    "gnomad_pos",
    "gnomad_ref",
    "gnomad_alt",
]

dropped_gnomad_operational = drop_if_exists(
    df,
    gnomad_operational,
    "OPERATIONAL gnomAD FEATURES",
    report
)

print(
    f"\nDropped: {len(dropped_gnomad_operational)} columns"
)

report.append(
    "\nREASON:\n"
    "  These describe whether/how gnomAD annotation was retrieved,\n"
    "  rather than biological properties of the variant."
)


# ============================================================
# 9. DROP RAW / REDUNDANT GNOMAD MEASUREMENTS
# ============================================================

print("\n" + "=" * 70)
print("9. SIMPLIFYING gnomAD MEASUREMENTS")
print("=" * 70)

gnomad_redundant = [
    # Raw AF retained indirectly through log_AF
    "gnomad_genome_af",
    "gnomad_exome_af",

    # AC / AN are highly redundant with AF and sample size
    "gnomad_genome_ac",
    "gnomad_genome_an",
    "gnomad_exome_ac",
    "gnomad_exome_an",

    # max AF duplicates information from genome/exome AF
    "max_gnomad_af",
]

dropped_gnomad_measurements = drop_if_exists(
    df,
    gnomad_redundant,
    "REDUNDANT gnomAD MEASUREMENTS",
    report
)

print(
    f"\nDropped: {len(dropped_gnomad_measurements)} columns"
)

print(
    "\nRetaining:"
)

for c in [
    "log_gnomad_genome_af",
    "log_gnomad_exome_af",
    "log_max_gnomad_af",
    "gnomad_genome_hom_count",
    "gnomad_exome_hom_count",
]:
    if c in df.columns:
        print(f"  KEEP: {c}")


# ============================================================
# 10. REMOVE REDUNDANT SIGNAL PEPTIDE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("10. SIGNAL PEPTIDE REDUNDANCY")
print("=" * 70)

signal_redundant = [
    "distance_to_nearest_signal_peptide",
    "protein_has_signal_peptide",
]

dropped_signal = drop_if_exists(
    df,
    signal_redundant,
    "SIGNAL PEPTIDE REDUNDANCY",
    report
)

print(
    f"\nDropped: {len(dropped_signal)} columns"
)

if "n_signal_peptide" in df.columns:
    print(
        "  KEEP: n_signal_peptide"
    )


# ============================================================
# 11. COUNT VS PRESENCE REDUNDANCY
# ============================================================

print("\n" + "=" * 70)
print("11. COUNT/PRESENCE REDUNDANCY")
print("=" * 70)

count_presence_redundant = [
    "protein_has_domain",
    "protein_has_dna_binding",
    "protein_has_active_site",
    "protein_has_binding_site",
    "protein_has_transmembrane",
    "protein_has_signal_peptide",
    "protein_has_disulfide_bond",
]

dropped_count_presence = drop_if_exists(
    df,
    count_presence_redundant,
    "COUNT/PRESENCE REDUNDANCY",
    report
)

print(
    f"\nDropped: {len(dropped_count_presence)} columns"
)

print(
    "\nCount features are retained where available."
)


# ============================================================
# 12. REMOVE CLINVAR REVIEW STATUS
# ============================================================

print("\n" + "=" * 70)
print("12. CLINVAR REVIEW STATUS")
print("=" * 70)

review_columns = [
    "Germline review status",
    "ClinVar review status",
    "review_status",
]

dropped_review = drop_if_exists(
    df,
    review_columns,
    "CLINVAR REVIEW STATUS",
    report
)

if dropped_review:
    print(
        "\nDropped because review status is closely tied "
        "to the evidence/label-generation process."
    )
else:
    print(
        "\nNo matching review-status column found."
    )


# ============================================================
# 13. REMOVE HIGHLY REDUNDANT SIGNAL-DERIVED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("13. OTHER DETERMINISTIC REDUNDANCY")
print("=" * 70)

# Only drop features whose redundancy has already been established.
deterministic_redundant = [
    "distance_to_nearest_signal_peptide",
]

dropped_deterministic = drop_if_exists(
    df,
    deterministic_redundant,
    "DETERMINISTIC REDUNDANCY",
    report
)


# ============================================================
# 14. HANDLE INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("14. NUMERICAL SANITIZATION")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

inf_count_before = 0

for col in numeric_columns:

    inf_count_before += np.isinf(
        df[col].to_numpy(
            dtype=float,
            na_value=np.nan
        )
    ).sum()

print(
    f"\nInfinite values found: {inf_count_before:,}"
)

if inf_count_before > 0:

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    print(
        "Infinite values converted to NaN."
    )

    report.append(
        f"\nINFINITE VALUES\n"
        f"  Found: {inf_count_before:,}\n"
        f"  Action: converted to NaN."
    )

else:

    report.append(
        "\nINFINITE VALUES\n"
        "  None found."
    )


# ============================================================
# 15. DO NOT IMPUTE
# ============================================================

print("\n" + "=" * 70)
print("15. MISSING VALUES")
print("=" * 70)

missing_count = df.isna().sum()

missing_features = (
    missing_count[
        (missing_count > 0)
        & (missing_count.index != "label")
    ]
    .sort_values(ascending=False)
)

print(
    f"\nFeatures with missing values: "
    f"{len(missing_features)}"
)

print(
    "\nTop missing features:"
)

for col, count in missing_features.head(20).items():

    pct = 100 * count / len(df)

    print(
        f"  {col:<45} "
        f"{count:>8,} "
        f"({pct:>6.2f}%)"
    )

print(
    "\nIMPORTANT:"
)

print(
    "  Missing values are NOT imputed here."
)

print(
    "  Imputation will be learned from TRAIN ONLY "
    "after the train/validation/test split."
)

report.append(
    "\nMISSING VALUES\n"
    "  No imputation performed.\n"
    "  Missing values will be handled after train/test split."
)


# ============================================================
# 16. FINAL SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("16. FINAL DATASET")
print("=" * 70)

# Ensure label remains last
feature_columns = [
    c for c in df.columns
    if c != "label"
]

df = df[
    feature_columns + ["label"]
]

final_rows = len(df)
final_columns = len(df.columns)

print(
    f"\nFinal rows:    {final_rows:,}"
)

print(
    f"Final columns: {final_columns}"
)

print(
    f"Features:      {final_columns - 1}"
)

# Check duplicate rows in cleaned source-level dataset.
duplicate_rows = df.duplicated().sum()

print(
    f"\nExact duplicate rows AFTER cleaning: "
    f"{duplicate_rows:,}"
)

# IMPORTANT:
# We intentionally DO NOT remove these here.
# Different original variants can have identical feature vectors.

report.append(
    f"\nFINAL DATASET\n"
    f"  Rows: {final_rows:,}\n"
    f"  Columns: {final_columns}\n"
    f"  Features: {final_columns - 1}\n"
    f"  Exact duplicate feature rows: {duplicate_rows:,}\n"
    f"  Action: NOT automatically removed."
)


# ============================================================
# 17. SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("17. SAVING")
print("=" * 70)

df.to_csv(
    OUTPUT_CLEAN,
    index=False
)

print(
    f"\nSaved cleaned dataset:"
)

print(
    f"  {OUTPUT_CLEAN}"
)

report.append(
    f"\nOUTPUT\n"
    f"  Clean dataset: {OUTPUT_CLEAN}"
)


# ============================================================
# 18. FEATURE LIST
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE LIST")
print("=" * 70)

for i, col in enumerate(
    [c for c in df.columns if c != "label"],
    start=1
):
    print(
        f"{i:3d}. {col}"
    )

report.append(
    "\nFINAL FEATURES\n"
)

for i, col in enumerate(
    [c for c in df.columns if c != "label"],
    start=1
):
    report.append(
        f"  {i:3d}. {col}"
    )


# ============================================================
# 19. SAVE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )

print(
    f"\nSaved report:"
)

print(
    f"  {OUTPUT_REPORT}"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(
    "\nFiles created:"
)

print(
    f"  1. {OUTPUT_CLEAN}"
)

print(
    f"  2. {OUTPUT_METADATA}"
)

if os.path.exists(OUTPUT_ANOMALIES):
    print(
        f"  3. {OUTPUT_ANOMALIES}"
    )

print(
    f"  4. {OUTPUT_REPORT}"
)

print(
    "\nIMPORTANT:"
)

print(
    "  No imputation was performed."
)

print(
    "  No scaling was performed."
)

print(
    "  No feature selection based on the target was performed."
)

print(
    "  Protein-position anomalies were flagged, not deleted."
)

print(
    "  Exact duplicate feature vectors were NOT automatically deleted."
)

print(
    "\nNext step: inspect the output and then perform "
    "TRAIN / VALIDATION / TEST splitting."
)