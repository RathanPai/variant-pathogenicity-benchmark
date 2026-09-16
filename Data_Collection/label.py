import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CLINVAR LABEL AUDIT
# ============================================================

INPUT_FILE = Path("Data/clinvar_gnomad.csv")
REPORT_FILE = Path("Data/clinvar_label_audit.txt")
SUSPICIOUS_FILE = Path("Data/clinvar_label_audit_suspicious.csv")

print("=" * 70)
print("CLINVAR LABEL AUDIT")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Loaded: {INPUT_FILE}")
print(f"Rows  : {len(df):,}")
print(f"Cols  : {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# NORMALIZE LABEL REPRESENTATION FOR ANALYSIS
# ============================================================

df["_label_string"] = (
    df["label"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# ============================================================
# POSSIBLE LABEL CATEGORIES
# ============================================================

# These are the labels we ideally expect for binary
# pathogenicity classification.

PATHOGENIC_LABELS = {
    "1",
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
    "pathogenic - likely pathogenic",
    "likely_pathogenic"
}

BENIGN_LABELS = {
    "0",
    "benign",
    "likely benign",
    "benign/likely benign",
    "benign - likely benign",
    "likely_benign"
}

UNCERTAIN_LABELS = {
    "vus",
    "uncertain significance",
    "uncertain",
    "unknown",
    "ambiguous",
    "conflicting",
    "conflicting interpretations",
    "conflicting interpretations of pathogenicity"
}


def classify_label(value):

    if pd.isna(value):
        return "MISSING"

    value = str(value).strip().lower()

    if value in PATHOGENIC_LABELS:
        return "PATHOGENIC"

    if value in BENIGN_LABELS:
        return "BENIGN"

    if value in UNCERTAIN_LABELS:
        return "UNCERTAIN / CONFLICTING"

    return "UNEXPECTED"


df["_label_category"] = (
    df["_label_string"]
    .apply(classify_label)
)


# ============================================================
# REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    def section(title):
        report.write("\n")
        report.write("=" * 70 + "\n")
        report.write(title + "\n")
        report.write("=" * 70 + "\n")

    def write(text=""):
        print(text)
        report.write(str(text) + "\n")


    # ========================================================
    # 1. BASIC DATASET
    # ========================================================

    section("1. BASIC DATASET INFORMATION")

    write(f"Total rows   : {len(df):,}")
    write(f"Total columns: {len(df.columns):,}")

    write(
        f"Label column dtype: "
        f"{df['label'].dtype}"
    )


    # ========================================================
    # 2. ALL RAW LABEL VALUES
    # ========================================================

    section("2. ALL UNIQUE RAW LABEL VALUES")

    raw_counts = (
        df["label"]
        .value_counts(dropna=False)
    )

    for label_value, count in raw_counts.items():

        pct = count / len(df) * 100

        write(
            f"{str(label_value):40s} : "
            f"{count:8,} ({pct:6.2f}%)"
        )


    # ========================================================
    # 3. LABEL CATEGORY SUMMARY
    # ========================================================

    section("3. LABEL CATEGORY SUMMARY")

    category_counts = (
        df["_label_category"]
        .value_counts(dropna=False)
    )

    for category, count in category_counts.items():

        pct = count / len(df) * 100

        write(
            f"{category:30s} : "
            f"{count:8,} ({pct:6.2f}%)"
        )


    # ========================================================
    # 4. BINARY CLASS COUNTS
    # ========================================================

    section("4. BINARY CLASS DISTRIBUTION")

    pathogenic_count = (
        df["_label_category"]
        .eq("PATHOGENIC")
        .sum()
    )

    benign_count = (
        df["_label_category"]
        .eq("BENIGN")
        .sum()
    )

    binary_total = (
        pathogenic_count +
        benign_count
    )

    write(
        f"Pathogenic class : "
        f"{pathogenic_count:,}"
    )

    write(
        f"Benign class     : "
        f"{benign_count:,}"
    )

    write(
        f"Binary total     : "
        f"{binary_total:,}"
    )

    if binary_total > 0:

        pathogenic_pct = (
            pathogenic_count /
            binary_total *
            100
        )

        benign_pct = (
            benign_count /
            binary_total *
            100
        )

        write(
            f"\nWithin binary dataset:"
        )

        write(
            f"Pathogenic: "
            f"{pathogenic_pct:.2f}%"
        )

        write(
            f"Benign    : "
            f"{benign_pct:.2f}%"
        )

        if benign_count > 0:

            imbalance_ratio = (
                max(pathogenic_count, benign_count)
                /
                min(pathogenic_count, benign_count)
            )

            write(
                f"\nClass imbalance ratio: "
                f"{imbalance_ratio:.2f}:1"
            )

            if imbalance_ratio < 1.5:
                write(
                    "IMBALANCE: LOW"
                )
            elif imbalance_ratio < 3:
                write(
                    "IMBALANCE: MODERATE"
                )
            elif imbalance_ratio < 10:
                write(
                    "IMBALANCE: HIGH"
                )
            else:
                write(
                    "IMBALANCE: VERY HIGH"
                )


    # ========================================================
    # 5. MISSING LABELS
    # ========================================================

    section("5. MISSING LABELS")

    missing_labels = (
        df["label"]
        .isna()
        .sum()
    )

    empty_labels = (
        df["_label_string"]
        .eq("")
        .sum()
    )

    write(
        f"NaN labels  : {missing_labels:,}"
    )

    write(
        f"Empty labels: {empty_labels:,}"
    )


    # ========================================================
    # 6. UNEXPECTED LABELS
    # ========================================================

    section("6. UNEXPECTED LABELS")

    unexpected = df[
        df["_label_category"] == "UNEXPECTED"
    ].copy()

    write(
        f"Unexpected label rows: "
        f"{len(unexpected):,}"
    )

    if len(unexpected) > 0:

        unexpected_counts = (
            unexpected["label"]
            .value_counts(dropna=False)
        )

        write("\nUnexpected values:")

        for value, count in unexpected_counts.items():

            pct = count / len(df) * 100

            write(
                f"  {str(value):35s} : "
                f"{count:8,} ({pct:.2f}%)"
            )


    # ========================================================
    # 7. UNCERTAIN / CONFLICTING
    # ========================================================

    section("7. UNCERTAIN / CONFLICTING LABELS")

    uncertain = df[
        df["_label_category"]
        ==
        "UNCERTAIN / CONFLICTING"
    ]

    write(
        f"Uncertain/conflicting rows: "
        f"{len(uncertain):,}"
    )

    if len(uncertain) > 0:

        counts = (
            uncertain["label"]
            .value_counts(dropna=False)
        )

        for value, count in counts.items():

            pct = count / len(df) * 100

            write(
                f"{str(value):40s} : "
                f"{count:8,} ({pct:6.2f}%)"
            )


    # ========================================================
    # 8. LABEL VS GERMLINE REVIEW STATUS
    # ========================================================

    section("8. LABEL × GERMLINE REVIEW STATUS")

    if "Germline review status" in df.columns:

        cross = pd.crosstab(
            df["_label_category"],
            df["Germline review status"],
            dropna=False
        )

        write(
            cross.to_string()
        )

    else:

        write(
            "Germline review status column not found."
        )


    # ========================================================
    # 9. LABEL × VARIANT TYPE
    # ========================================================

    section("9. LABEL × VARIANT TYPE")

    if "Variant type" in df.columns:

        cross = pd.crosstab(
            df["Variant type"],
            df["_label_category"],
            dropna=False
        )

        write(
            cross.to_string()
        )

    else:

        write(
            "Variant type column not found."
        )


    # ========================================================
    # 10. LABEL × MOLECULAR CONSEQUENCE
    # ========================================================

    section("10. LABEL × MOLECULAR CONSEQUENCE")

    if "Molecular consequence" in df.columns:

        cross = pd.crosstab(
            df["Molecular consequence"],
            df["_label_category"],
            dropna=False
        )

        write(
            cross.to_string()
        )

    else:

        write(
            "Molecular consequence column not found."
        )


    # ========================================================
    # 11. LABEL × GNOMAD AVAILABILITY
    # ========================================================

    section("11. LABEL × GNOMAD AVAILABILITY")

    if "gnomad_found" in df.columns:

        cross = pd.crosstab(
            df["_label_category"],
            df["gnomad_found"],
            dropna=False
        )

        write(
            cross.to_string()
        )

    else:

        write(
            "gnomad_found column not found."
        )


    # ========================================================
    # 12. LABEL × GENE AVAILABILITY
    # ========================================================

    section("12. LABEL × GENE AVAILABILITY")

    if "gene_available" in df.columns:

        cross = pd.crosstab(
            df["_label_category"],
            df["gene_available"],
            dropna=False
        )

        write(
            cross.to_string()
        )

    else:

        write(
            "gene_available column not found."
        )


    # ========================================================
    # 13. LABEL × PROTEIN CHANGE AVAILABILITY
    # ========================================================

    section("13. LABEL × PROTEIN CHANGE AVAILABILITY")

    if "protein_change_available" in df.columns:

        cross = pd.crosstab(
            df["_label_category"],
            df["protein_change_available"],
            dropna=False
        )

        write(
            cross.to_string()
        )

    else:

        write(
            "protein_change_available column not found."
        )


    # ========================================================
    # 14. PATHOGENICITY-RELATED FEATURE DISTRIBUTION
    # ========================================================

    section("14. EXISTING BIOLOGICAL FEATURE DISTRIBUTION")

    feature_columns = [
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
        "is_genic_downstream"
    ]

    existing_features = [
        col for col in feature_columns
        if col in df.columns
    ]

    if existing_features:

        for feature in existing_features:

            total_positive = (
                pd.to_numeric(
                    df[feature],
                    errors="coerce"
                )
                .eq(1)
                .sum()
            )

            write(
                f"{feature:30s} : "
                f"{total_positive:,} rows"
            )

    else:

        write(
            "No existing biological feature columns found."
        )


    # ========================================================
    # 15. EXAMPLES OF EACH CATEGORY
    # ========================================================

    section("15. EXAMPLES OF EACH LABEL CATEGORY")

    for category in [
        "PATHOGENIC",
        "BENIGN",
        "UNCERTAIN / CONFLICTING",
        "UNEXPECTED",
        "MISSING"
    ]:

        subset = df[
            df["_label_category"] == category
        ]

        write(
            f"\n--- {category} ---"
        )

        if len(subset) == 0:

            write("None")

            continue

        columns = [
            col for col in [
                "Name",
                "VariationID",
                "label",
                "Germline review status",
                "Variant type",
                "Molecular consequence",
                "Protein change",
                "Gene(s)",
                "gnomad_found",
                "gnomad_genome_af",
                "gnomad_exome_af"
            ]
            if col in df.columns
        ]

        write(
            subset[columns]
            .head(10)
            .to_string(index=False)
        )


    # ========================================================
    # 16. RECOMMENDED TRAINING SET
    # ========================================================

    section("16. RECOMMENDED INITIAL TRAINING SET")

    training_mask = (
        df["_label_category"]
        .isin([
            "PATHOGENIC",
            "BENIGN"
        ])
    )

    training_df = df[
        training_mask
    ].copy()

    write(
        f"Rows suitable for initial binary training: "
        f"{len(training_df):,}"
    )

    write(
        f"Pathogenic: {(training_df['_label_category'] == 'PATHOGENIC').sum():,}"
    )

    write(
        f"Benign: {(training_df['_label_category'] == 'BENIGN').sum():,}"
    )

    excluded = len(df) - len(training_df)

    write(
        f"Excluded initially: "
        f"{excluded:,}"
    )


    # ========================================================
    # 17. FINAL AUDIT VERDICT
    # ========================================================

    section("17. FINAL AUDIT VERDICT")

    if (
        pathogenic_count > 0
        and
        benign_count > 0
        and
        len(unexpected) == 0
        and
        missing_labels == 0
    ):

        write(
            "LABEL STRUCTURE: CLEAN"
        )

    else:

        write(
            "LABEL STRUCTURE: REQUIRES REVIEW"
        )

    write(
        "\nThe report does NOT automatically modify "
        "or delete any original rows."
    )

    write(
        "It only identifies which rows are suitable "
        "for the initial binary model."
    )


# ============================================================
# SAVE SUSPICIOUS / EXCLUDED ROWS
# ============================================================

suspicious_mask = (
    ~df["_label_category"]
    .isin([
        "PATHOGENIC",
        "BENIGN"
    ])
)

suspicious_columns = [
    col for col in [
        "Name",
        "VariationID",
        "AlleleID(s)",
        "label",
        "Germline review status",
        "Variant type",
        "variant_type_simple",
        "Molecular consequence",
        "Gene(s)",
        "Protein change",
        "gnomad_found",
        "gnomad_status"
    ]
    if col in df.columns
]

df.loc[
    suspicious_mask,
    suspicious_columns
].to_csv(
    SUSPICIOUS_FILE,
    index=False
)


# ============================================================
# CLEAN UP
# ============================================================

df.drop(
    columns=[
        "_label_string",
        "_label_category"
    ],
    inplace=True
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("LABEL AUDIT COMPLETE")
print("=" * 70)

print(f"\nReport:")
print(f"  {REPORT_FILE}")

print(f"\nSuspicious/excluded variants:")
print(f"  {SUSPICIOUS_FILE}")

print("\nSend me BOTH files.")