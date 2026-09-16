"""
======================================================================
FEATURE ENGINEERING
======================================================================

Project:
Quantum-Enhanced Genetic Variant Pathogenicity Prediction

Input:
    variant_features(1).csv

Output:
    variant_features_engineered.csv
    feature_dictionary.csv

Purpose:
    Convert the integrated biological variant dataset into a clean,
    ML-ready feature dataset while avoiding obvious data leakage.

Pipeline:
    Raw variant dataset
            ↓
    Remove identifiers / leakage
            ↓
    Engineer population features
            ↓
    Engineer protein-position features
            ↓
    Engineer amino-acid properties
            ↓
    Clean biological features
            ↓
    Save engineered dataset

IMPORTANT:
    Train/test split, scaling, feature selection, and dimensionality
    reduction are intentionally NOT performed here.
======================================================================
"""

import os
import re
import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_FILE = "Data/variant_features.csv"
OUTPUT_FILE = "Data/variant_features_engineered.csv"
FEATURE_DICTIONARY_FILE = "Data/feature_dictionary.csv"

EPSILON = 1e-10


# ======================================================================
# AMINO ACID PROPERTIES
# ======================================================================
#
# Values are standard biochemical approximations.
# We use them to derive differences between reference and alternate
# amino acids for missense variants.
#
# Properties:
#   charge
#   hydrophobicity
#   molecular weight
#   polarity
#
# Charge:
#   +1 = positively charged
#    0 = neutral
#   -1 = negatively charged
# ======================================================================

AA_PROPERTIES = {

    "A": {
        "charge": 0,
        "hydrophobicity": 1.8,
        "molecular_weight": 89.09,
        "polarity": 8.1
    },

    "R": {
        "charge": 1,
        "hydrophobicity": -4.5,
        "molecular_weight": 174.20,
        "polarity": 10.5
    },

    "N": {
        "charge": 0,
        "hydrophobicity": -3.5,
        "molecular_weight": 132.12,
        "polarity": 11.6
    },

    "D": {
        "charge": -1,
        "hydrophobicity": -3.5,
        "molecular_weight": 133.10,
        "polarity": 13.0
    },

    "C": {
        "charge": 0,
        "hydrophobicity": 2.5,
        "molecular_weight": 121.16,
        "polarity": 5.5
    },

    "Q": {
        "charge": 0,
        "hydrophobicity": -3.5,
        "molecular_weight": 146.15,
        "polarity": 10.5
    },

    "E": {
        "charge": -1,
        "hydrophobicity": -3.5,
        "molecular_weight": 147.13,
        "polarity": 12.3
    },

    "G": {
        "charge": 0,
        "hydrophobicity": -0.4,
        "molecular_weight": 75.07,
        "polarity": 9.0
    },

    "H": {
        "charge": 0.5,
        "hydrophobicity": -3.2,
        "molecular_weight": 155.16,
        "polarity": 10.4
    },

    "I": {
        "charge": 0,
        "hydrophobicity": 4.5,
        "molecular_weight": 131.18,
        "polarity": 5.2
    },

    "L": {
        "charge": 0,
        "hydrophobicity": 3.8,
        "molecular_weight": 131.18,
        "polarity": 4.9
    },

    "K": {
        "charge": 1,
        "hydrophobicity": -3.9,
        "molecular_weight": 146.19,
        "polarity": 11.3
    },

    "M": {
        "charge": 0,
        "hydrophobicity": 1.9,
        "molecular_weight": 149.21,
        "polarity": 5.7
    },

    "F": {
        "charge": 0,
        "hydrophobicity": 2.8,
        "molecular_weight": 165.19,
        "polarity": 5.2
    },

    "P": {
        "charge": 0,
        "hydrophobicity": -1.6,
        "molecular_weight": 115.13,
        "polarity": 8.0
    },

    "S": {
        "charge": 0,
        "hydrophobicity": -0.8,
        "molecular_weight": 105.09,
        "polarity": 9.2
    },

    "T": {
        "charge": 0,
        "hydrophobicity": -0.7,
        "molecular_weight": 119.12,
        "polarity": 8.6
    },

    "W": {
        "charge": 0,
        "hydrophobicity": -0.9,
        "molecular_weight": 204.23,
        "polarity": 5.4
    },

    "Y": {
        "charge": 0,
        "hydrophobicity": -1.3,
        "molecular_weight": 181.19,
        "polarity": 6.2
    },

    "V": {
        "charge": 0,
        "hydrophobicity": 4.2,
        "molecular_weight": 117.15,
        "polarity": 5.9
    }
}


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def clean_numeric(series):
    """
    Convert a column to numeric.
    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def safe_log10(series):
    """
    Log10 transformation for allele frequencies.

    EPSILON prevents log10(0).
    """
    series = clean_numeric(series)

    return np.log10(
        series.clip(lower=0) + EPSILON
    )


def find_column(df, candidates):
    """
    Find the first matching column from a list of candidates.

    Matching is case-insensitive.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def get_column_or_nan(df, candidates):
    """
    Return a column if it exists.
    Otherwise return a NaN series.
    """

    column = find_column(df, candidates)

    if column is None:
        return pd.Series(
            np.nan,
            index=df.index
        )

    return df[column]


# ======================================================================
# AMINO ACID EXTRACTION
# ======================================================================

def extract_amino_acids(value):
    """
    Attempt to extract reference and alternate amino acids.

    Supports common representations such as:

        R/Q
        R220Q
        Arg/Gln
        p.Arg220Gln
        p.R220Q

    Returns:
        reference_aa, alternate_aa
    """

    if pd.isna(value):
        return np.nan, np.nan

    value = str(value).strip()

    # --------------------------------------------------------------
    # Three-letter amino acid names
    # --------------------------------------------------------------

    three_letter = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V"
    }

    # Example:
    # p.Arg220Gln

    match = re.search(
        r"(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)"
        r"\d+"
        r"(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)",
        value
    )

    if match:

        ref = three_letter.get(match.group(1))
        alt = three_letter.get(match.group(2))

        return ref, alt

    # --------------------------------------------------------------
    # One-letter notation
    #
    # Examples:
    #   R220Q
    #   p.R220Q
    # --------------------------------------------------------------

    match = re.search(
        r"([ACDEFGHIKLMNPQRSTVWY])\d+([ACDEFGHIKLMNPQRSTVWY])",
        value.upper()
    )

    if match:

        return match.group(1), match.group(2)

    # --------------------------------------------------------------
    # R/Q notation
    # --------------------------------------------------------------

    match = re.search(
        r"([ACDEFGHIKLMNPQRSTVWY])\s*/\s*"
        r"([ACDEFGHIKLMNPQRSTVWY])",
        value.upper()
    )

    if match:

        return match.group(1), match.group(2)

    return np.nan, np.nan


# ======================================================================
# MAIN
# ======================================================================

print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ======================================================================
# 1. LOAD DATA
# ======================================================================

print("\nLoading dataset...")

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ======================================================================
# 2. VERIFY TARGET
# ======================================================================

print("\nChecking target...")

if "label" not in df.columns:

    raise ValueError(
        "ERROR: 'label' column not found."
    )

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nLabel proportions:")

print(
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
)


# ======================================================================
# 3. CREATE FEATURE DATAFRAME
# ======================================================================

features = pd.DataFrame(index=df.index)

# Keep target separately
features["label"] = df["label"]


# ======================================================================
# 4. VARIANT / VEP FEATURES
# ======================================================================

print("\nEngineering variant features...")

variant_type = get_column_or_nan(
    df,
    [
        "variant_type_simple",
        "variant_type"
    ]
)

features["variant_type_simple"] = variant_type.astype("string")

impact = get_column_or_nan(
    df,
    [
        "vep_impact"
    ]
)

features["vep_impact"] = impact.astype("string")

consequence = get_column_or_nan(
    df,
    [
        "vep_primary_consequence"
    ]
)

features["vep_primary_consequence"] = (
    consequence.astype("string")
)


# Transcript count

transcript_count = get_column_or_nan(
    df,
    [
        "vep_transcript_count"
    ]
)

features["vep_transcript_count"] = clean_numeric(
    transcript_count
)


# ======================================================================
# 5. BINARY BIOLOGICAL FEATURES
# ======================================================================

binary_features = [

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

for column in binary_features:

    if column in df.columns:

        values = clean_numeric(df[column])

        features[column] = values.fillna(0).astype(int)


# ======================================================================
# 6. gnomAD FEATURES
# ======================================================================

print("Engineering population-genetics features...")

genome_af = get_column_or_nan(
    df,
    [
        "gnomad_genome_af"
    ]
)

exome_af = get_column_or_nan(
    df,
    [
        "gnomad_exome_af"
    ]
)

genome_ac = get_column_or_nan(
    df,
    [
        "gnomad_genome_ac"
    ]
)

genome_an = get_column_or_nan(
    df,
    [
        "gnomad_genome_an"
    ]
)

genome_hom = get_column_or_nan(
    df,
    [
        "gnomad_genome_hom_count"
    ]
)

exome_ac = get_column_or_nan(
    df,
    [
        "gnomad_exome_ac"
    ]
)

exome_an = get_column_or_nan(
    df,
    [
        "gnomad_exome_an"
    ]
)

exome_hom = get_column_or_nan(
    df,
    [
        "gnomad_exome_hom_count"
    ]
)


# Raw population features

features["gnomad_genome_af"] = clean_numeric(
    genome_af
)

features["gnomad_exome_af"] = clean_numeric(
    exome_af
)

features["gnomad_genome_ac"] = clean_numeric(
    genome_ac
)

features["gnomad_genome_an"] = clean_numeric(
    genome_an
)

features["gnomad_genome_hom_count"] = clean_numeric(
    genome_hom
)

features["gnomad_exome_ac"] = clean_numeric(
    exome_ac
)

features["gnomad_exome_an"] = clean_numeric(
    exome_an
)

features["gnomad_exome_hom_count"] = clean_numeric(
    exome_hom
)


# Log AF

features["log_gnomad_genome_af"] = safe_log10(
    genome_af
)

features["log_gnomad_exome_af"] = safe_log10(
    exome_af
)


# AF availability

features["has_gnomad_genome_af"] = (
    clean_numeric(genome_af)
    .notna()
    .astype(int)
)

features["has_gnomad_exome_af"] = (
    clean_numeric(exome_af)
    .notna()
    .astype(int)
)


# Maximum observed AF

genome_af_numeric = clean_numeric(genome_af)
exome_af_numeric = clean_numeric(exome_af)

features["max_gnomad_af"] = pd.concat(
    [
        genome_af_numeric,
        exome_af_numeric
    ],
    axis=1
).max(axis=1, skipna=True)


features["log_max_gnomad_af"] = safe_log10(
    features["max_gnomad_af"]
)


# ======================================================================
# 7. PROTEIN POSITION FEATURES
# ======================================================================

print("Engineering protein-position features...")

protein_position = get_column_or_nan(
    df,
    [
        "vep_protein_position",
        "protein_position"
    ]
)

protein_length = get_column_or_nan(
    df,
    [
        "protein_length"
    ]
)

protein_position = clean_numeric(
    protein_position
)

protein_length = clean_numeric(
    protein_length
)

features["protein_position"] = protein_position

features["protein_length"] = protein_length


# Relative position within protein

features["relative_protein_position"] = (
    protein_position / protein_length.replace(0, np.nan)
)


# Distance from beginning/end

features["distance_from_protein_start"] = (
    protein_position
)

features["distance_from_protein_end"] = (
    protein_length - protein_position
)


# ======================================================================
# 8. STRUCTURAL / FUNCTIONAL ANNOTATIONS
# ======================================================================

print("Engineering protein structural features...")

structural_binary_features = [

    "variant_in_domain",
    "variant_in_region",
    "variant_in_repeat",
    "variant_in_coiled_coil",
    "variant_in_motif",
    "variant_in_zinc_finger",
    "variant_in_dna_binding",
    "variant_in_active_site",
    "variant_in_binding_site",
    "variant_in_site",
    "variant_in_transmembrane",
    "variant_in_signal_peptide",
    "variant_in_topological_domain",
    "variant_in_disulfide_bond"
]

for column in structural_binary_features:

    if column in df.columns:

        features[column] = (
            clean_numeric(df[column])
            .fillna(0)
            .astype(int)
        )


# Protein-level annotation counts

count_features = [

    "n_domain",
    "n_region",
    "n_repeat",
    "n_coiled_coil",
    "n_motif",
    "n_zinc_finger",
    "n_dna_binding",
    "n_active_site",
    "n_binding_site",
    "n_site",
    "n_transmembrane",
    "n_signal_peptide",
    "n_topological_domain",
    "n_disulfide_bond"
]

for column in count_features:

    if column in df.columns:

        features[column] = clean_numeric(
            df[column]
        )


# Protein annotation availability

protein_presence_features = [

    "protein_has_domain",
    "protein_has_active_site",
    "protein_has_binding_site",
    "protein_has_transmembrane",
    "protein_has_signal_peptide",
    "protein_has_disulfide_bond",
    "protein_has_dna_binding"
]

for column in protein_presence_features:

    if column in df.columns:

        features[column] = (
            clean_numeric(df[column])
            .fillna(0)
            .astype(int)
        )


# ======================================================================
# 9. DISTANCE-TO-FUNCTIONAL-REGION FEATURES
# ======================================================================

print("Engineering functional-region distances...")

distance_columns = [

    column
    for column in df.columns
    if column.startswith("distance_to_nearest_")
]

for column in distance_columns:

    features[column] = clean_numeric(
        df[column]
    )


# ======================================================================
# 10. AMINO ACID FEATURES
# ======================================================================

print("Engineering amino-acid features...")

aa_source_column = find_column(
    df,
    [
        "vep_amino_acids",
        "Protein change",
        "protein_change",
        "vep_hgvsp"
    ]
)

if aa_source_column is not None:

    aa_pairs = df[aa_source_column].apply(
        extract_amino_acids
    )

    features["reference_aa"] = aa_pairs.apply(
        lambda x: x[0]
    )

    features["alternate_aa"] = aa_pairs.apply(
        lambda x: x[1]
    )

else:

    features["reference_aa"] = pd.Series(
        np.nan,
        index=df.index
    )

    features["alternate_aa"] = pd.Series(
        np.nan,
        index=df.index
    )


# ----------------------------------------------------------------------
# Amino-acid properties
# ----------------------------------------------------------------------

for property_name in [
    "charge",
    "hydrophobicity",
    "molecular_weight",
    "polarity"
]:

    features[
        f"reference_aa_{property_name}"
    ] = features["reference_aa"].map(
        lambda aa:
            AA_PROPERTIES.get(aa, {}).get(
                property_name,
                np.nan
            )
    )

    features[
        f"alternate_aa_{property_name}"
    ] = features["alternate_aa"].map(
        lambda aa:
            AA_PROPERTIES.get(aa, {}).get(
                property_name,
                np.nan
            )
    )


# ----------------------------------------------------------------------
# Amino-acid changes
# ----------------------------------------------------------------------

features["aa_charge_change"] = (
    features["alternate_aa_charge"]
    -
    features["reference_aa_charge"]
)

features["aa_hydrophobicity_change"] = (
    features["alternate_aa_hydrophobicity"]
    -
    features["reference_aa_hydrophobicity"]
)

features["aa_molecular_weight_change"] = (
    features["alternate_aa_molecular_weight"]
    -
    features["reference_aa_molecular_weight"]
)

features["aa_polarity_change"] = (
    features["alternate_aa_polarity"]
    -
    features["reference_aa_polarity"]
)


# Absolute changes

features["abs_aa_charge_change"] = (
    features["aa_charge_change"].abs()
)

features["abs_aa_hydrophobicity_change"] = (
    features["aa_hydrophobicity_change"].abs()
)

features["abs_aa_molecular_weight_change"] = (
    features["aa_molecular_weight_change"].abs()
)

features["abs_aa_polarity_change"] = (
    features["aa_polarity_change"].abs()
)


# Is amino acid actually changed?

features["is_amino_acid_change"] = (
    (
        features["reference_aa"].notna()
        &
        features["alternate_aa"].notna()
        &
        (
            features["reference_aa"]
            !=
            features["alternate_aa"]
        )
    )
    .astype(int)
)


# ======================================================================
# 11. CLEAN STRING FEATURES
# ======================================================================

print("Cleaning categorical features...")

# Replace missing categorical values with explicit category

categorical_columns = [

    "variant_type_simple",
    "vep_impact",
    "vep_primary_consequence",
    "reference_aa",
    "alternate_aa"
]

for column in categorical_columns:

    if column in features.columns:

        features[column] = (
            features[column]
            .astype("string")
            .fillna("UNKNOWN")
        )


# ======================================================================
# 12. REMOVE INF VALUES
# ======================================================================

print("Cleaning infinite values...")

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)


# ======================================================================
# 13. REMOVE COMPLETELY EMPTY FEATURES
# ======================================================================

print("Removing completely empty features...")

empty_columns = [

    column
    for column in features.columns
    if column != "label"
    and features[column].isna().all()
]

if empty_columns:

    print("\nRemoved completely empty columns:")

    for column in empty_columns:

        print(f"  - {column}")

    features = features.drop(
        columns=empty_columns
    )


# ======================================================================
# 14. REMOVE DUPLICATE FEATURES
# ======================================================================

print("\nChecking duplicate columns...")

duplicate_columns = []

columns = features.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        col_a = columns[i]
        col_b = columns[j]

        if col_a == "label" or col_b == "label":
            continue

        try:

            if features[col_a].equals(
                features[col_b]
            ):

                duplicate_columns.append(col_b)

        except Exception:

            pass


duplicate_columns = list(
    dict.fromkeys(duplicate_columns)
)

if duplicate_columns:

    print("Duplicate columns removed:")

    for column in duplicate_columns:

        print(f"  - {column}")

    features = features.drop(
        columns=duplicate_columns
    )


# ======================================================================
# 15. SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(
    f"Original rows       : {len(df):,}"
)

print(
    f"Original columns    : {len(df.columns):,}"
)

print(
    f"Engineered columns  : {len(features.columns):,}"
)

print(
    f"Feature count       : {len(features.columns) - 1:,}"
)

print(
    f"Rows retained       : {len(features):,}"
)


# ======================================================================
# 16. MISSING VALUE SUMMARY
# ======================================================================

print("\nTop missing-value features:")

missing_summary = (
    features
    .isna()
    .mean()
    .sort_values(ascending=False)
)

print(
    missing_summary
    .head(20)
    .to_string()
)


# ======================================================================
# 17. FEATURE DICTIONARY
# ======================================================================

print("\nCreating feature dictionary...")

feature_descriptions = {

    "label":
        "Target label: 1 = pathogenic, 0 = benign",

    "log_gnomad_genome_af":
        "Log10-transformed genome allele frequency",

    "log_gnomad_exome_af":
        "Log10-transformed exome allele frequency",

    "max_gnomad_af":
        "Maximum allele frequency observed across genome and exome",

    "log_max_gnomad_af":
        "Log10-transformed maximum gnomAD allele frequency",

    "has_gnomad_genome_af":
        "Whether genome allele frequency is available",

    "has_gnomad_exome_af":
        "Whether exome allele frequency is available",

    "relative_protein_position":
        "Protein position divided by protein length",

    "distance_from_protein_start":
        "Distance of variant from beginning of protein",

    "distance_from_protein_end":
        "Distance of variant from end of protein",

    "aa_charge_change":
        "Change in amino-acid charge caused by the variant",

    "aa_hydrophobicity_change":
        "Change in amino-acid hydrophobicity",

    "aa_molecular_weight_change":
        "Change in amino-acid molecular weight",

    "aa_polarity_change":
        "Change in amino-acid polarity",

    "abs_aa_charge_change":
        "Absolute amino-acid charge change",

    "abs_aa_hydrophobicity_change":
        "Absolute amino-acid hydrophobicity change",

    "abs_aa_molecular_weight_change":
        "Absolute amino-acid molecular-weight change",

    "abs_aa_polarity_change":
        "Absolute amino-acid polarity change",

    "is_amino_acid_change":
        "Whether a reference amino acid differs from the alternate amino acid"
}


dictionary_rows = []

for column in features.columns:

    if column in feature_descriptions:

        description = feature_descriptions[column]

    elif column.startswith(
        "distance_to_nearest_"
    ):

        description = (
            "Distance from variant to nearest "
            + column.replace(
                "distance_to_nearest_",
                ""
            )
            + " functional region"
        )

    elif column.startswith("is_"):

        description = (
            "Binary indicator for "
            + column[3:].replace("_", " ")
        )

    elif column.startswith("n_"):

        description = (
            "Number of annotated "
            + column[2:].replace("_", " ")
            + " regions in the protein"
        )

    elif column.startswith("protein_has_"):

        description = (
            "Whether the protein contains "
            + column.replace(
                "protein_has_",
                ""
            ).replace("_", " ")
        )

    else:

        description = "Engineered biological feature"

    dictionary_rows.append({

        "feature": column,

        "dtype":
            str(features[column].dtype),

        "description":
            description,

        "missing_count":
            int(features[column].isna().sum()),

        "missing_fraction":
            float(features[column].isna().mean())

    })


feature_dictionary = pd.DataFrame(
    dictionary_rows
)


# ======================================================================
# 18. SAVE
# ======================================================================

print("\nSaving engineered dataset...")

features.to_csv(
    OUTPUT_FILE,
    index=False
)

feature_dictionary.to_csv(
    FEATURE_DICTIONARY_FILE,
    index=False
)


# ======================================================================
# 19. FINAL CHECK
# ======================================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nEngineered dataset:"
    f"\n  {OUTPUT_FILE}"
)

print(
    f"\nFeature dictionary:"
    f"\n  {FEATURE_DICTIONARY_FILE}"
)

print(
    "\nFinal shape:"
)

print(
    f"  {features.shape[0]:,} rows × "
    f"{features.shape[1]:,} columns"
)

print("\nTarget distribution:")

print(
    features["label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nNext stage:")
print(
    "  1. Inspect engineered features"
)
print(
    "  2. Handle missing values"
)
print(
    "  3. Encode categorical variables"
)
print(
    "  4. Train/test split"
)
print(
    "  5. Scale numerical features"
)
print(
    "  6. Feature selection / dimensionality reduction"
)
print(
    "  7. Classical ML baseline"
)
print(
    "  8. QNN"
)

print("\n" + "=" * 70)