"""
======================================================================
FEATURE DECISION AUDIT
======================================================================

Purpose:
    Investigate suspicious relationships identified during the initial
    feature audit before final preprocessing.

Investigations:
    1. gnomAD availability vs pathogenicity
    2. gnomAD availability by label
    3. Protein position vs signal-peptide distance
    4. Signal-peptide annotation behavior
    5. is_missense vs is_amino_acid_change
    6. Count vs presence features

Input:
    Data/variant_features_engineered.csv

Output:
    Data/feature_decision_audit.txt

IMPORTANT:
    This script DOES NOT modify the dataset.
======================================================================
"""

import os
import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_FILE = "Data/variant_features_engineered.csv"
OUTPUT_FILE = "Data/feature_decision_audit.txt"

TARGET = "label"


# ======================================================================
# LOAD
# ======================================================================

print("=" * 70)
print("FEATURE DECISION AUDIT")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"\nLoaded: {len(df):,} rows × "
    f"{len(df.columns):,} columns"
)


# ======================================================================
# OUTPUT CAPTURE
# ======================================================================

output_lines = []


def log(text=""):
    """
    Print to terminal and save to output file.
    """

    print(text)
    output_lines.append(str(text))


# ======================================================================
# BASIC INFORMATION
# ======================================================================

log("\n" + "=" * 70)
log("1. BASIC DATASET INFORMATION")
log("=" * 70)

log(f"Rows: {len(df):,}")
log(f"Columns: {len(df.columns):,}")

log("\nLabel distribution:")

label_counts = df[TARGET].value_counts().sort_index()

for label, count in label_counts.items():

    percentage = (
        count / len(df) * 100
    )

    log(
        f"  Label {label}: "
        f"{count:,} ({percentage:.2f}%)"
    )


# ======================================================================
# 2. gnomAD AVAILABILITY VS LABEL
# ======================================================================

log("\n" + "=" * 70)
log("2. GNOMAD AVAILABILITY VS PATHOGENICITY")
log("=" * 70)

gnomad_features = [

    "has_gnomad_genome_af",
    "has_gnomad_exome_af"
]


for feature in gnomad_features:

    if feature not in df.columns:
        log(
            f"\n{feature}: NOT FOUND"
        )
        continue

    log(
        f"\n--- {feature} ---"
    )

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    log(
        "Overall distribution:"
    )

    overall = (
        df[feature]
        .value_counts(normalize=True)
        .sort_index()
    )

    for value, proportion in overall.items():

        log(
            f"  {value}: "
            f"{proportion:.4%}"
        )

    # --------------------------------------------------------------
    # By label
    # --------------------------------------------------------------

    log(
        "\nAvailability by label:"
    )

    table = pd.crosstab(
        df[TARGET],
        df[feature],
        normalize="index"
    )

    log(
        table.to_string()
    )

    # --------------------------------------------------------------
    # Pathogenicity rate by availability
    # --------------------------------------------------------------

    log(
        "\nPathogenicity rate by availability:"
    )

    rates = (
        df.groupby(feature)[TARGET]
        .agg(
            [
                "count",
                "mean"
            ]
        )
    )

    rates["percentage"] = (
        rates["mean"] * 100
    )

    log(
        rates.to_string()
    )


# ======================================================================
# 3. gnomAD AF BY LABEL
# ======================================================================

log("\n" + "=" * 70)
log("3. GNOMAD ALLELE FREQUENCY BY LABEL")
log("=" * 70)


af_features = [

    "gnomad_genome_af",
    "gnomad_exome_af",
    "max_gnomad_af",
    "log_gnomad_genome_af",
    "log_gnomad_exome_af",
    "log_max_gnomad_af"
]


for feature in af_features:

    if feature not in df.columns:
        continue

    log(
        f"\n--- {feature} ---"
    )

    grouped = (
        df.groupby(TARGET)[feature]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
    )

    log(
        grouped.to_string()
    )


# ======================================================================
# 4. PROTEIN POSITION VS SIGNAL PEPTIDE DISTANCE
# ======================================================================

log("\n" + "=" * 70)
log("4. PROTEIN POSITION VS SIGNAL PEPTIDE DISTANCE")
log("=" * 70)


position_features = [

    "protein_position",
    "distance_to_nearest_signal_peptide",
    "protein_length",
    "relative_protein_position"
]


for feature in position_features:

    if feature in df.columns:

        log(
            f"\n{feature}:"
        )

        log(
            df[feature]
            .describe()
            .to_string()
        )


if (
    "protein_position" in df.columns
    and
    "distance_to_nearest_signal_peptide" in df.columns
):

    valid = df[
        [
            "protein_position",
            "distance_to_nearest_signal_peptide"
        ]
    ].dropna()

    if len(valid) > 0:

        correlation = (
            valid[
                "protein_position"
            ]
            .corr(
                valid[
                    "distance_to_nearest_signal_peptide"
                ]
            )
        )

        log(
            f"\nCorrelation: "
            f"{correlation:.8f}"
        )

        # Difference analysis

        difference = (
            valid[
                "distance_to_nearest_signal_peptide"
            ]
            -
            valid[
                "protein_position"
            ]
        )

        log(
            "\nDifference "
            "(signal distance - protein position):"
        )

        log(
            difference.describe()
            .to_string()
        )

        log(
            "\nMost common differences:"
        )

        log(
            difference
            .round(4)
            .value_counts()
            .head(15)
            .to_string()
        )


# ======================================================================
# 5. SIGNAL PEPTIDE ANNOTATION
# ======================================================================

log("\n" + "=" * 70)
log("5. SIGNAL PEPTIDE ANNOTATION BEHAVIOR")
log("=" * 70)


signal_features = [

    "n_signal_peptide",
    "protein_has_signal_peptide",
    "distance_to_nearest_signal_peptide"
]


for feature in signal_features:

    if feature not in df.columns:
        continue

    log(
        f"\n--- {feature} ---"
    )

    log(
        "Missing:",
        )

    log(
        f"  {df[feature].isna().sum():,} "
        f"({df[feature].isna().mean():.2%})"
    )

    log(
        "\nValue counts:"
    )

    log(
        df[feature]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )


# ----------------------------------------------------------------------
# Check whether n_signal_peptide predicts protein_has_signal_peptide
# ----------------------------------------------------------------------

if (
    "n_signal_peptide" in df.columns
    and
    "protein_has_signal_peptide" in df.columns
):

    log(
        "\nCross-tabulation:"
    )

    signal_table = pd.crosstab(
        df["n_signal_peptide"],
        df["protein_has_signal_peptide"],
        dropna=False
    )

    log(
        signal_table.to_string()
    )


# ======================================================================
# 6. MISSENSE VS AMINO-ACID CHANGE
# ======================================================================

log("\n" + "=" * 70)
log("6. MISSENSE VS AMINO-ACID CHANGE")
log("=" * 70)


if (
    "is_missense" in df.columns
    and
    "is_amino_acid_change" in df.columns
):

    table = pd.crosstab(
        df["is_missense"],
        df["is_amino_acid_change"],
        margins=True
    )

    log(
        "\nCross-tabulation:"
    )

    log(
        table.to_string()
    )

    # Agreement

    agreement = (
        df["is_missense"]
        ==
        df["is_amino_acid_change"]
    ).mean()

    log(
        f"\nAgreement: "
        f"{agreement:.6%}"
    )

    # Disagreement examples

    disagreements = df[
        df["is_missense"]
        !=
        df["is_amino_acid_change"]
    ]

    log(
        f"\nDisagreements: "
        f"{len(disagreements):,}"
    )

    if len(disagreements) > 0:

        log(
            "\nDisagreement combinations:"
        )

        log(
            disagreements[
                [
                    "is_missense",
                    "is_amino_acid_change"
                ]
            ]
            .value_counts()
            .to_string()
        )


# ======================================================================
# 7. COUNT VS PRESENCE FEATURES
# ======================================================================

log("\n" + "=" * 70)
log("7. COUNT VS PRESENCE FEATURES")
log("=" * 70)


count_presence_pairs = [

    (
        "n_domain",
        "protein_has_domain"
    ),

    (
        "n_region",
        "protein_has_region"
    ),

    (
        "n_repeat",
        "protein_has_repeat"
    ),

    (
        "n_coiled_coil",
        "protein_has_coiled_coil"
    ),

    (
        "n_motif",
        "protein_has_motif"
    ),

    (
        "n_zinc_finger",
        "protein_has_zinc_finger"
    ),

    (
        "n_dna_binding",
        "protein_has_dna_binding"
    ),

    (
        "n_active_site",
        "protein_has_active_site"
    ),

    (
        "n_binding_site",
        "protein_has_binding_site"
    ),

    (
        "n_site",
        "protein_has_site"
    ),

    (
        "n_transmembrane",
        "protein_has_transmembrane"
    ),

    (
        "n_signal_peptide",
        "protein_has_signal_peptide"
    ),

    (
        "n_topological_domain",
        "protein_has_topological_domain"
    ),

    (
        "n_disulfide_bond",
        "protein_has_disulfide_bond"
    )
]


for count_feature, presence_feature in count_presence_pairs:

    if (
        count_feature not in df.columns
        or
        presence_feature not in df.columns
    ):
        continue

    log(
        f"\n--- {count_feature} vs "
        f"{presence_feature} ---"
    )

    # Create expected presence from count

    expected_presence = (
        pd.to_numeric(
            df[count_feature],
            errors="coerce"
        )
        .fillna(0)
        > 0
    ).astype(int)

    actual_presence = (
        pd.to_numeric(
            df[presence_feature],
            errors="coerce"
        )
        .fillna(0)
        > 0
    ).astype(int)

    agreement = (
        expected_presence
        ==
        actual_presence
    ).mean()

    log(
        f"Agreement: "
        f"{agreement:.6%}"
    )

    disagreements = (
        expected_presence
        !=
        actual_presence
    ).sum()

    log(
        f"Disagreements: "
        f"{disagreements:,}"
    )


# ======================================================================
# 8. DISTANCE FEATURE MISSINGNESS
# ======================================================================

log("\n" + "=" * 70)
log("8. DISTANCE FEATURE MISSINGNESS")
log("=" * 70)


distance_columns = [

    column
    for column in df.columns
    if column.startswith(
        "distance_to_nearest_"
    )
]


for column in distance_columns:

    missing = (
        df[column]
        .isna()
        .mean()
    )

    non_missing = (
        df[column]
        .dropna()
    )

    log(
        f"\n{column}"
    )

    log(
        f"  Missing: "
        f"{missing:.2%}"
    )

    if len(non_missing) > 0:

        log(
            f"  Min: "
            f"{non_missing.min()}"
        )

        log(
            f"  Median: "
            f"{non_missing.median()}"
        )

        log(
            f"  Max: "
            f"{non_missing.max()}"
        )


# ======================================================================
# 9. DISTANCE MISSINGNESS VS LABEL
# ======================================================================

log("\n" + "=" * 70)
log("9. DISTANCE MISSINGNESS VS LABEL")
log("=" * 70)


for column in distance_columns:

    missing_indicator = (
        df[column]
        .isna()
        .astype(int)
    )

    rates = (
        df.assign(
            missing=missing_indicator
        )
        .groupby("missing")[TARGET]
        .agg(
            [
                "count",
                "mean"
            ]
        )
    )

    log(
        f"\n--- {column} ---"
    )

    log(
        rates.to_string()
    )


# ======================================================================
# 10. SUMMARY / DECISIONS
# ======================================================================

log("\n" + "=" * 70)
log("10. PRELIMINARY FEATURE DECISIONS")
log("=" * 70)


log(
    """
The following are preliminary decisions only.
Final preprocessing decisions should be made after reviewing
the numerical results above.
"""
)


log(
    """
A. gnomAD:
   - Retain log-transformed allele frequencies.
   - Investigate whether gnomAD availability represents dataset
     construction bias.
   - Retain availability indicators initially.
   - Remove redundant raw AC/AN features during preprocessing.
"""
)


log(
    """
B. Protein position:
   - Retain protein_position.
   - Retain relative_protein_position.
   - Investigate whether signal-peptide distance is mathematically
     derived from protein position.
"""
)


log(
    """
C. Signal peptide:
   - If n_signal_peptide and protein_has_signal_peptide are
     effectively identical, retain the presence feature and
     remove the redundant count.
"""
)


log(
    """
D. Amino-acid change:
   - If is_amino_acid_change is effectively identical to is_missense,
     remove the redundant feature.
   - Otherwise retain both if they provide distinct information.
"""
)


log(
    """
E. Distance features:
   - Do NOT drop high-missingness distance features automatically.
   - Convert missingness into explicit annotation-presence indicators.
   - Impute the numerical distance values during preprocessing.
"""
)


# ======================================================================
# SAVE REPORT
# ======================================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(output_lines)
    )


# ======================================================================
# DONE
# ======================================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    f"\nReport saved to:"
    f"\n  {OUTPUT_FILE}"
)

print(
    "\nNo dataset columns were modified."
)

print("=" * 70)