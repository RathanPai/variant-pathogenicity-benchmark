
"""
======================================================================
BUILD VARIANT-LEVEL BIOLOGICAL FEATURE DATASET
======================================================================

Inputs:
    Data/clinvar_gnomad_v2.csv
    Data/clinvar_vep_features.csv
    Data/ensembl_to_uniprot.csv
    Data/uniprot_features_raw.tsv

Outputs:
    Data/variant_features.csv
    Data/variant_features_audit.txt

Purpose:
    Combine ClinVar + gnomAD + VEP + UniProt into a single
    variant-level biological feature dataset.

Important:
    UniProt disease/pathogenicity/variant/mutagenesis annotations
    are NOT used as model features.

======================================================================
"""

import os
import re
import sys
import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

CLINVAR_FILE = "Data/clinvar_gnomad_v2.csv"
VEP_FILE = "Data/clinvar_vep_features.csv"
MAPPING_FILE = "Data/ensembl_to_uniprot.csv"
UNIPROT_RAW_FILE = "Data/uniprot_features_raw.tsv"

OUTPUT_FILE = "Data/variant_features.csv"
AUDIT_FILE = "Data/variant_features_audit.txt"


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("BUILDING VARIANT-LEVEL BIOLOGICAL FEATURE DATASET")
print("=" * 70)


# ======================================================================
# FILE CHECK
# ======================================================================

def require_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


print()
print("Checking input files...")

for path in [
    CLINVAR_FILE,
    VEP_FILE,
    MAPPING_FILE,
    UNIPROT_RAW_FILE
]:

    require_file(path)

    print(f"  OK: {path}")


# ======================================================================
# POSITION PARSER
# ======================================================================

def parse_position_ranges(value):
    """
    Extract UniProt positional ranges.

    Examples:

        175
        175-180
        175..180

    Returns:
        [(start, end), ...]
    """

    if pd.isna(value):

        return []

    text = str(value).strip()

    if not text:

        return []

    ranges = []

    # Ranges such as:
    # 100-200
    # 100..200

    range_pattern = re.compile(
        r"(?<!\d)(\d+)\s*(?:-|\.{2})\s*(\d+)(?!\d)"
    )

    for match in range_pattern.finditer(text):

        start = int(match.group(1))
        end = int(match.group(2))

        if start > end:

            start, end = end, start

        ranges.append(
            (start, end)
        )

    # Standalone positions

    if not ranges:

        singles = re.findall(
            r"(?<!\d)(\d+)(?!\d)",
            text
        )

        for value in singles:

            position = int(value)

            if position > 0:

                ranges.append(
                    (position, position)
                )

    return ranges


# ======================================================================
# POSITIONAL FEATURE CALCULATION
# ======================================================================

def get_feature_position_info(
    position,
    annotation
):

    if pd.isna(position):

        return 0, np.nan

    try:

        position = int(position)

    except (TypeError, ValueError):

        return 0, np.nan

    ranges = parse_position_ranges(
        annotation
    )

    if not ranges:

        return 0, np.nan

    minimum_distance = None

    for start, end in ranges:

        # Variant lies inside annotation

        if start <= position <= end:

            return 1, 0

        # Variant before annotation

        if position < start:

            distance = start - position

        # Variant after annotation

        else:

            distance = position - end

        if (
            minimum_distance is None
            or distance < minimum_distance
        ):

            minimum_distance = distance

    return 0, minimum_distance


# ======================================================================
# ANNOTATION COMBINER
# ======================================================================

def combine_annotations(values):

    cleaned = []

    for value in values:

        if pd.isna(value):

            continue

        value = str(value).strip()

        if not value:

            continue

        if value.lower() == "nan":

            continue

        cleaned.append(value)

    if not cleaned:

        return np.nan

    # Preserve order and remove exact duplicates

    return "; ".join(
        dict.fromkeys(cleaned)
    )


# ======================================================================
# ANNOTATION COUNTER
# ======================================================================

def count_annotations(value):

    if pd.isna(value):

        return 0

    text = str(value).strip()

    if not text:

        return 0

    parts = re.split(
        r";\s*|\n+",
        text
    )

    parts = [
        p.strip()
        for p in parts
        if p.strip()
    ]

    return len(parts)


# ======================================================================
# LOAD CLINVAR + GNOMAD
# ======================================================================

print()
print("Loading ClinVar + gnomAD dataset...")

clinvar = pd.read_csv(
    CLINVAR_FILE,
    low_memory=False
)

print(
    f"ClinVar/gnomAD rows: "
    f"{len(clinvar):,}"
)


if "VariationID" not in clinvar.columns:

    raise ValueError(
        "VariationID missing from ClinVar dataset."
    )


clinvar["VariationID"] = (
    clinvar["VariationID"]
    .astype(str)
    .str.strip()
)


# ======================================================================
# LOAD VEP
# ======================================================================

print()
print("Loading VEP features...")

vep = pd.read_csv(
    VEP_FILE,
    low_memory=False
)

print(
    f"VEP rows: "
    f"{len(vep):,}"
)


if "VariationID" not in vep.columns:

    raise ValueError(
        "VariationID missing from VEP dataset."
    )


vep["VariationID"] = (
    vep["VariationID"]
    .astype(str)
    .str.strip()
)


# ======================================================================
# VEP DUPLICATE CHECK
# ======================================================================

duplicate_vep = (
    vep["VariationID"]
    .duplicated()
    .sum()
)


if duplicate_vep:

    raise ValueError(
        f"VEP contains {duplicate_vep:,} duplicate VariationIDs."
    )


# ======================================================================
# REMOVE POTENTIAL OLD VEP COLUMNS FROM CLINVAR
# ======================================================================
#
# This prevents accidental _x / _y column collisions.
# ======================================================================

vep_columns_to_add = [
    column
    for column in vep.columns
    if column != "VariationID"
]


overlap = [
    column
    for column in vep_columns_to_add
    if column in clinvar.columns
]


if overlap:

    print()
    print(
        "Removing overlapping columns from ClinVar before VEP merge:"
    )

    for column in overlap:

        print(
            f"  {column}"
        )

    clinvar = clinvar.drop(
        columns=overlap
    )


# ======================================================================
# MERGE VEP
# ======================================================================

print()
print("Merging VEP features...")

master = clinvar.merge(
    vep,
    on="VariationID",
    how="left",
    validate="one_to_one"
)


print(
    f"Rows after VEP merge: "
    f"{len(master):,}"
)


if len(master) != len(clinvar):

    raise RuntimeError(
        "VEP merge changed variant count."
    )


# ======================================================================
# CHECK VEP PROTEIN INFORMATION
# ======================================================================

required_vep = [
    "vep_protein_id",
    "vep_protein_position"
]


for column in required_vep:

    if column not in master.columns:

        raise ValueError(
            f"Missing VEP column: {column}"
        )


master["vep_protein_position"] = pd.to_numeric(
    master["vep_protein_position"],
    errors="coerce"
)


# ======================================================================
# LOAD ENSEMBL → UNIPROT MAPPING
# ======================================================================

print()
print("Loading Ensembl → UniProt mapping...")

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype=str
)


required_mapping = [
    "ensembl_protein_id",
    "uniprot_accession"
]


for column in required_mapping:

    if column not in mapping.columns:

        raise ValueError(
            f"Missing mapping column: {column}"
        )


mapping = mapping[
    required_mapping
].dropna()


mapping["ensembl_protein_id"] = (
    mapping["ensembl_protein_id"]
    .astype(str)
    .str.strip()
)


mapping["uniprot_accession"] = (
    mapping["uniprot_accession"]
    .astype(str)
    .str.strip()
)


mapping = mapping.drop_duplicates()


print(
    f"Mapping rows: "
    f"{len(mapping):,}"
)

print(
    f"Ensembl proteins: "
    f"{mapping['ensembl_protein_id'].nunique():,}"
)

print(
    f"UniProt accessions: "
    f"{mapping['uniprot_accession'].nunique():,}"
)


# ======================================================================
# LOAD RAW UNIPROT
# ======================================================================

print()
print("Loading raw UniProt annotations...")

uniprot = pd.read_csv(
    UNIPROT_RAW_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)


print(
    f"UniProt rows: "
    f"{len(uniprot):,}"
)


print()
print("UniProt columns:")

for column in uniprot.columns:

    print(
        f"  {column}"
    )


# ======================================================================
# NORMALIZE UNIPROT COLUMN NAMES
# ======================================================================

rename_map = {

    "Entry":
        "uniprot_accession",

    "Length":
        "protein_length",

    "Reviewed":
        "reviewed",

    "Domain [FT]":
        "domain",

    "Region":
        "region",

    "Repeat":
        "repeat",

    "Coiled coil":
        "coiled_coil",

    "Coiled-coil":
        "coiled_coil",

    "Motif":
        "motif",

    "Zinc finger":
        "zinc_finger",

    "DNA binding":
        "dna_binding",

    "Active site":
        "active_site",

    "Binding site":
        "binding_site",

    "Site":
        "site",

    "Transmembrane":
        "transmembrane",

    "Signal peptide":
        "signal_peptide",

    "Topological domain":
        "topological_domain",

    "Disulfide bond":
        "disulfide_bond",

    "Chain":
        "chain",
}


uniprot = uniprot.rename(
    columns=rename_map
)


if "uniprot_accession" not in uniprot.columns:

    raise ValueError(
        "UniProt accession column missing after normalization."
    )


# ======================================================================
# NORMALIZE UNIPROT DATA
# ======================================================================

uniprot["uniprot_accession"] = (
    uniprot["uniprot_accession"]
    .astype(str)
    .str.strip()
)


uniprot["protein_length"] = pd.to_numeric(
    uniprot["protein_length"],
    errors="coerce"
)


# Reviewed

uniprot["reviewed"] = (
    uniprot["reviewed"]
    .map({
        "reviewed": 1,
        "unreviewed": 0,
        "Reviewed": 1,
        "Unreviewed": 0
    })
)


# ======================================================================
# UNIPROT DUPLICATE CHECK
# ======================================================================

duplicate_uniprot = (
    uniprot["uniprot_accession"]
    .duplicated()
    .sum()
)


if duplicate_uniprot:

    print()
    print(
        f"WARNING: {duplicate_uniprot:,} duplicate UniProt accessions."
    )

    uniprot = uniprot.drop_duplicates(
        subset=["uniprot_accession"],
        keep="first"
    )


# ======================================================================
# POSITIONAL ANNOTATION COLUMNS
# ======================================================================

annotation_columns = [

    "domain",
    "region",
    "repeat",
    "coiled_coil",
    "motif",
    "zinc_finger",
    "dna_binding",

    "active_site",
    "binding_site",
    "site",

    "transmembrane",
    "signal_peptide",
    "topological_domain",

    "disulfide_bond",

]


annotation_columns = [
    column
    for column in annotation_columns
    if column in uniprot.columns
]


print()
print("Positional UniProt annotations used:")

for column in annotation_columns:

    print(
        f"  {column}"
    )


# ======================================================================
# KEEP ONLY NEEDED UNIPROT COLUMNS
# ======================================================================

uniprot_keep = [
    "uniprot_accession",
    "protein_length",
    "reviewed",
]


uniprot_keep += annotation_columns


uniprot = uniprot[
    uniprot_keep
]


# ======================================================================
# MERGE MAPPING WITH UNIPROT
# ======================================================================

print()
print("Joining Ensembl → UniProt → protein annotations...")

protein_table = mapping.merge(
    uniprot,
    on="uniprot_accession",
    how="left",
    validate="many_to_one"
)


print(
    f"Protein mapping rows: "
    f"{len(protein_table):,}"
)


# ======================================================================
# RESOLVE ONE-TO-MANY MAPPINGS
# ======================================================================

print()
print("Resolving one-to-many Ensembl → UniProt mappings...")


aggregation = {

    "protein_length":
        "median",

    "reviewed":
        "max",
}


for column in annotation_columns:

    aggregation[column] = combine_annotations


protein_by_ensembl = (
    protein_table
    .groupby(
        "ensembl_protein_id",
        as_index=False
    )
    .agg(aggregation)
)


print(
    f"Unique Ensembl protein feature rows: "
    f"{len(protein_by_ensembl):,}"
)


# ======================================================================
# CRITICAL MERGE FIX
# ======================================================================
#
# IMPORTANT:
#
# We do NOT merge uniprot_accession into master here.
#
# Instead, we first create a single clean protein feature table keyed
# by Ensembl protein ID.
#
# Then we explicitly rename any possible existing columns before merge.
# ======================================================================

if "uniprot_accession" in master.columns:

    print()
    print(
        "Removing pre-existing uniprot_accession column "
        "before protein merge."
    )

    master = master.drop(
        columns=["uniprot_accession"]
    )


# protein_by_ensembl does not currently contain uniprot_accession.
#
# We deliberately create a representative accession for traceability.
#
# For one-to-many mappings, multiple accessions are joined together.

accession_by_ensembl = (
    mapping
    .groupby("ensembl_protein_id")
    ["uniprot_accession"]
    .apply(
        lambda x: ";".join(
            sorted(
                set(x)
            )
        )
    )
    .reset_index()
)


protein_by_ensembl = protein_by_ensembl.merge(
    accession_by_ensembl,
    on="ensembl_protein_id",
    how="left",
    validate="one_to_one"
)


# ======================================================================
# MERGE PROTEIN FEATURES INTO VARIANTS
# ======================================================================

print()
print("Merging protein annotations into variants...")


before_rows = len(master)


master = master.merge(
    protein_by_ensembl,
    left_on="vep_protein_id",
    right_on="ensembl_protein_id",
    how="left",
    validate="many_to_one"
)


if len(master) != before_rows:

    raise RuntimeError(
        "Protein feature merge changed variant count."
    )


print(
    f"Rows after protein merge: "
    f"{len(master):,}"
)


# ======================================================================
# CHECK EXPECTED COLUMNS
# ======================================================================

if "uniprot_accession" not in master.columns:

    raise RuntimeError(
        "uniprot_accession was not created by protein merge.\n"
        f"Available columns:\n{list(master.columns)}"
    )


# ======================================================================
# PROTEIN COVERAGE
# ======================================================================

print()
print("Protein annotation coverage:")


with_protein_id = (
    master["vep_protein_id"]
    .notna()
)


with_uniprot = (
    master["uniprot_accession"]
    .notna()
)


with_position = (
    master["vep_protein_position"]
    .notna()
)


print(
    f"Variants with VEP protein ID: "
    f"{with_protein_id.sum():,} "
    f"({with_protein_id.mean() * 100:.2f}%)"
)


print(
    f"Variants with UniProt mapping: "
    f"{with_uniprot.sum():,} "
    f"({with_uniprot.mean() * 100:.2f}%)"
)


print(
    f"Variants with protein position: "
    f"{with_position.sum():,} "
    f"({with_position.mean() * 100:.2f}%)"
)


# ======================================================================
# VARIANT-LEVEL POSITIONAL FEATURES
# ======================================================================

print()
print("Creating variant-level positional features...")


for annotation in annotation_columns:

    print(
        f"  Processing {annotation}..."
    )


    inside_values = []
    distance_values = []


    for position, annotation_text in zip(
        master["vep_protein_position"],
        master[annotation]
    ):

        inside, distance = (
            get_feature_position_info(
                position,
                annotation_text
            )
        )


        inside_values.append(
            inside
        )


        distance_values.append(
            distance
        )


    master[
        f"variant_in_{annotation}"
    ] = inside_values


    master[
        f"distance_to_nearest_{annotation}"
    ] = distance_values


# ======================================================================
# RELATIVE PROTEIN POSITION
# ======================================================================

print()
print("Calculating relative protein position...")


master["relative_protein_position"] = (
    master["vep_protein_position"]
    /
    master["protein_length"]
)


master.loc[
    (
        master["vep_protein_position"] <= 0
    )
    |
    (
        master["protein_length"] <= 0
    ),
    "relative_protein_position"
] = np.nan


# ======================================================================
# VALID PROTEIN POSITION
# ======================================================================

master["protein_position_valid"] = (
    master["vep_protein_position"].notna()
    &
    master["protein_length"].notna()
    &
    (master["vep_protein_position"] >= 1)
    &
    (
        master["vep_protein_position"]
        <=
        master["protein_length"]
    )
).astype(int)


# ======================================================================
# PROTEIN-LEVEL COUNTS
# ======================================================================

print()
print("Creating protein-level feature counts...")


count_mapping = {

    "n_domain":
        "domain",

    "n_region":
        "region",

    "n_repeat":
        "repeat",

    "n_coiled_coil":
        "coiled_coil",

    "n_motif":
        "motif",

    "n_zinc_finger":
        "zinc_finger",

    "n_dna_binding":
        "dna_binding",

    "n_active_site":
        "active_site",

    "n_binding_site":
        "binding_site",

    "n_site":
        "site",

    "n_transmembrane":
        "transmembrane",

    "n_signal_peptide":
        "signal_peptide",

    "n_topological_domain":
        "topological_domain",

    "n_disulfide_bond":
        "disulfide_bond",

}


for output_column, source_column in count_mapping.items():

    if source_column in master.columns:

        master[output_column] = (
            master[source_column]
            .apply(count_annotations)
        )


# ======================================================================
# PROTEIN BOOLEAN FEATURES
# ======================================================================

boolean_sources = [

    "domain",
    "active_site",
    "binding_site",
    "transmembrane",
    "signal_peptide",
    "disulfide_bond",
    "dna_binding",

]


for source in boolean_sources:

    if source not in master.columns:

        continue


    master[
        f"protein_has_{source}"
    ] = (
        master[source]
        .apply(count_annotations)
        .gt(0)
        .astype(int)
    )


# ======================================================================
# CLEAN INTERNAL COLUMN
# ======================================================================

if "ensembl_protein_id" in master.columns:

    master = master.drop(
        columns=["ensembl_protein_id"]
    )


# ======================================================================
# LEAKAGE AUDIT
# ======================================================================

print()
print("=" * 70)
print("LEAKAGE AUDIT")
print("=" * 70)


# Explicitly forbidden biological annotation types

forbidden_keywords = [

    "pathogenic",
    "benign",
    "disease",
    "mutagen",
    "clinvar",

]


suspicious = []


for column in master.columns:

    lower = column.lower()


    if any(
        keyword in lower
        for keyword in forbidden_keywords
    ):

        suspicious.append(
            column
        )


if suspicious:

    print(
        "WARNING: These columns require manual review:"
    )

    for column in suspicious:

        print(
            f"  {column}"
        )

else:

    print(
        "No obvious disease/pathogenicity columns detected."
    )


# ======================================================================
# REMOVE RAW UNIPROT TEXT
# ======================================================================
#
# Raw positional annotations are retained in:
#
#     Data/uniprot_features_raw.tsv
#
# The final ML dataset should contain derived numeric features instead.
# ======================================================================

raw_columns = [
    column
    for column in annotation_columns
    if column in master.columns
]


master_model = master.drop(
    columns=raw_columns
)


# ======================================================================
# FINAL DATASET ORDER
# ======================================================================

priority_columns = [

    "VariationID",

    "label",

    "Gene(s)",

    "vep_gene",
    "vep_gene_id",
    "vep_transcript",

    "vep_primary_consequence",
    "vep_impact",
    "vep_biotype",

    "vep_protein_id",
    "uniprot_accession",

    "vep_protein_position",
    "protein_length",

    "relative_protein_position",
    "protein_position_valid",

]


priority_columns = [
    column
    for column in priority_columns
    if column in master_model.columns
]


remaining_columns = [
    column
    for column in master_model.columns
    if column not in priority_columns
]


master_model = master_model[
    priority_columns +
    remaining_columns
]


# ======================================================================
# FINAL VALIDATION
# ======================================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    f"Final rows: "
    f"{len(master_model):,}"
)


print(
    f"Final columns: "
    f"{len(master_model.columns):,}"
)


duplicate_variations = (
    master_model["VariationID"]
    .duplicated()
    .sum()
)


print(
    f"Duplicate VariationIDs: "
    f"{duplicate_variations:,}"
)


if len(master_model) != 117111:

    print()
    print(
        "WARNING: Expected 117,111 rows."
    )


if duplicate_variations != 0:

    raise RuntimeError(
        "Duplicate VariationIDs detected."
    )


# ======================================================================
# TARGET DISTRIBUTION
# ======================================================================

if "label" in master_model.columns:

    print()
    print("Target distribution:")


    counts = (
        master_model["label"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


    for label, count in counts.items():

        print(
            f"  label={label}: "
            f"{count:,} "
            f"({count / len(master_model) * 100:.2f}%)"
        )


# ======================================================================
# POSITIONAL FEATURE SUMMARY
# ======================================================================

print()
print("Variant-level positional feature coverage:")


positional_features = [
    column
    for column in master_model.columns
    if column.startswith("variant_in_")
]


for column in positional_features:

    count = (
        master_model[column]
        .sum()
    )


    percentage = (
        count
        /
        len(master_model)
        *
        100
    )


    print(
        f"  {column:<42}"
        f"{count:>7,} "
        f"({percentage:>6.2f}%)"
    )


# ======================================================================
# SAVE DATASET
# ======================================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


master_model.to_csv(
    OUTPUT_FILE,
    index=False
)


# ======================================================================
# SAVE AUDIT REPORT
# ======================================================================

with open(
    AUDIT_FILE,
    "w"
) as audit:

    audit.write(
        "VARIANT FEATURE DATASET AUDIT\n"
    )

    audit.write(
        "=" * 70 +
        "\n\n"
    )


    audit.write(
        f"Rows: {len(master_model):,}\n"
    )


    audit.write(
        f"Columns: {len(master_model.columns):,}\n"
    )


    audit.write(
        f"Duplicate VariationIDs: "
        f"{duplicate_variations:,}\n"
    )


    if "label" in master_model.columns:

        audit.write(
            "\nTARGET DISTRIBUTION\n"
        )

        audit.write(
            "-" * 40 +
            "\n"
        )


        for label, count in (
            master_model["label"]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .items()
        ):

            audit.write(
                f"label={label}: "
                f"{count:,} "
                f"({count / len(master_model) * 100:.2f}%)\n"
            )


    audit.write(
        "\nPROTEIN COVERAGE\n"
    )

    audit.write(
        "-" * 40 +
        "\n"
    )


    for column in [
        "vep_protein_id",
        "uniprot_accession",
        "vep_protein_position",
        "protein_length",
    ]:

        if column in master_model.columns:

            count = (
                master_model[column]
                .notna()
                .sum()
            )


            audit.write(
                f"{column}: "
                f"{count:,} "
                f"({count / len(master_model) * 100:.2f}%)\n"
            )


    audit.write(
        "\nPOSITIONAL FEATURES\n"
    )

    audit.write(
        "-" * 40 +
        "\n"
    )


    for column in positional_features:

        count = (
            master_model[column]
            .sum()
        )


        audit.write(
            f"{column}: "
            f"{count:,} "
            f"({count / len(master_model) * 100:.2f}%)\n"
        )


# ======================================================================
# FINISH
# ======================================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)


print(
    f"Final dataset:\n"
    f"  {OUTPUT_FILE}"
)


print(
    f"Audit report:\n"
    f"  {AUDIT_FILE}"
)


print()
print("=" * 70)
print("VARIANT FEATURE CONSTRUCTION COMPLETE")
print("=" * 70)
