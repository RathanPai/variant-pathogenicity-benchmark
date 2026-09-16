
"""
======================================================================
UNIPROTKB PROTEIN FEATURE EXTRACTION
======================================================================

Project:
    Quantum-Enhanced Genetic Variant Pathogenicity Prediction

Input:
    Data/ensembl_to_uniprot.csv

Output:
    Data/uniprot_protein_features.csv
    Data/uniprot_features_raw.tsv

Purpose:
    Retrieve biological protein-level annotations from UniProtKB.

Features retrieved:
    - Protein length
    - Reviewed / Swiss-Prot status
    - Domain annotations
    - Active sites
    - Binding sites
    - Transmembrane regions
    - Signal peptides
    - Disulfide bonds
    - Functional regions
    - Coiled-coils
    - Repeats
    - Zinc fingers
    - DNA-binding regions
    - Motifs
    - Generic sites
    - Protein chains
    - Topological domains

IMPORTANT:
    Disease annotations, ClinVar annotations, pathogenicity annotations,
    natural variants, and mutagenesis annotations are intentionally NOT
    retrieved to avoid target leakage.

The raw feature annotations are preserved so that variant-level
position overlap can be calculated later.
======================================================================
"""

import os
import sys
import time
import re
from io import StringIO

import pandas as pd
import requests


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_FILE = "Data/ensembl_to_uniprot.csv"

OUTPUT_FILE = "Data/uniprot_protein_features.csv"

RAW_OUTPUT_FILE = "Data/uniprot_features_raw.tsv"

API_BASE = "https://rest.uniprot.org"

# Number of UniProt accessions per search request.
# Keeping this moderate avoids excessively long URLs.
BATCH_SIZE = 100

REQUEST_TIMEOUT = 120

MAX_RETRIES = 5

RETRY_DELAY = 3


# ======================================================================
# UNIProt RETURN FIELDS
# ======================================================================
#
# These correspond to UniProtKB REST API fields.
#
# We deliberately exclude:
#   cc_disease
#   ft_variant
#   ft_mutagen
#   ClinVar cross-references
#   pathogenicity-related annotations
#
# See UniProt return-fields documentation.
# ======================================================================

FIELDS = [
    "accession",
    "length",
    "reviewed",

    # Family / domains
    "ft_domain",
    "ft_region",
    "ft_repeat",
    "ft_coiled",
    "ft_motif",
    "ft_zn_fing",
    "ft_dna_bind",

    # Functional sites
    "ft_act_site",
    "ft_binding",
    "ft_site",

    # Membrane / localization
    "ft_transmem",
    "ft_signal",
    "ft_topo_dom",

    # Processing / structure
    "ft_disulfid",
    "ft_chain",
]


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("UNIPROTKB PROTEIN FEATURE EXTRACTION")
print("=" * 70)


# ======================================================================
# HTTP SESSION
# ======================================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
        "Bioinformatics-Variant-Pathogenicity-Pipeline/1.0"
})


# ======================================================================
# REQUEST FUNCTION
# ======================================================================

def request_with_retry(
    method,
    url,
    *,
    params=None,
    timeout=REQUEST_TIMEOUT
):
    """
    Perform HTTP request with retries for temporary failures.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.request(
                method,
                url,
                params=params,
                timeout=timeout
            )

            if response.status_code in (
                429,
                500,
                502,
                503,
                504
            ):

                print(
                    f"    HTTP {response.status_code}; "
                    f"retry {attempt}/{MAX_RETRIES}"
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        RETRY_DELAY * attempt
                    )

                    continue

            response.raise_for_status()

            return response

        except requests.RequestException as e:

            print(
                f"    Request error "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY * attempt
                )

            else:

                raise


# ======================================================================
# LOAD MAPPING
# ======================================================================

print()
print("Loading Ensembl → UniProt mapping...")


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


mapping = pd.read_csv(
    INPUT_FILE,
    dtype=str
)


required_columns = {
    "ensembl_protein_id",
    "uniprot_accession"
}


missing_columns = (
    required_columns -
    set(mapping.columns)
)


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing_columns))
    )


# Clean
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


# Remove empty values
mapping = mapping[
    (mapping["ensembl_protein_id"] != "") &
    (mapping["uniprot_accession"] != "") &
    (mapping["uniprot_accession"] != "nan")
]


# Remove duplicate exact mappings
mapping = mapping.drop_duplicates()


print(
    f"Mapping rows:                 "
    f"{len(mapping):,}"
)


print(
    f"Ensembl proteins:              "
    f"{mapping['ensembl_protein_id'].nunique():,}"
)


print(
    f"Unique UniProt accessions:      "
    f"{mapping['uniprot_accession'].nunique():,}"
)


# ======================================================================
# UNIQUE UNIPROT ACCESSIONS
# ======================================================================

accessions = sorted(
    mapping["uniprot_accession"]
    .dropna()
    .unique()
)


print()
print(
    f"UniProt accessions to retrieve: "
    f"{len(accessions):,}"
)


if len(accessions) == 0:

    raise ValueError(
        "No UniProt accessions found."
    )


# ======================================================================
# UNIProt SEARCH
# ======================================================================

search_url = (
    f"{API_BASE}/uniprotkb/search"
)


all_results = []


total_batches = (
    (len(accessions) + BATCH_SIZE - 1)
    // BATCH_SIZE
)


print()
print(
    f"Retrieving UniProt annotations in "
    f"{total_batches:,} batches..."
)


for batch_number, start in enumerate(
    range(
        0,
        len(accessions),
        BATCH_SIZE
    ),
    start=1
):

    batch = accessions[
        start:start + BATCH_SIZE
    ]


    # --------------------------------------------------------------
    # Build UniProt accession query
    # --------------------------------------------------------------

    accession_query = " OR ".join(
        f"accession:{acc}"
        for acc in batch
    )


    params = {
        "query": f"({accession_query})",
        "format": "tsv",
        "fields": ",".join(FIELDS),
        "size": len(batch),
    }


    print(
        f"  Batch {batch_number:>3}/{total_batches}: "
        f"{len(batch):>3} accessions...",
        end=" ",
        flush=True
    )


    response = request_with_retry(
        "GET",
        search_url,
        params=params
    )


    text = response.text


    if not text.strip():

        print("EMPTY")

        continue


    batch_df = pd.read_csv(
        StringIO(text),
        sep="\t",
        dtype=str
    )


    if len(batch_df) == 0:

        print("0 results")

        continue


    all_results.append(
        batch_df
    )


    print(
        f"{len(batch_df):,} results"
    )


    # Be polite to the API.
    time.sleep(0.2)


# ======================================================================
# COMBINE RESULTS
# ======================================================================

print()
print("Combining UniProt results...")


if not all_results:

    raise RuntimeError(
        "No UniProt results were returned."
    )


features_raw = pd.concat(
    all_results,
    ignore_index=True
)


# ======================================================================
# NORMALIZE COLUMN NAMES
# ======================================================================

print(
    f"Raw UniProt rows: "
    f"{len(features_raw):,}"
)


print()
print("Returned columns:")

for column in features_raw.columns:

    print(f"  {column}")


# UniProt TSV normally uses human-readable labels.
#
# We normalize them into stable machine-readable names.

rename_map = {
    "Entry": "uniprot_accession",
    "Length": "protein_length",
    "Reviewed": "reviewed",

    "Domain [FT]": "domain",
    "Region": "region",
    "Repeat": "repeat",
    "Coiled-coil": "coiled_coil",
    "Motif": "motif",
    "Zinc finger": "zinc_finger",
    "DNA binding": "dna_binding",

    "Active site": "active_site",
    "Binding site": "binding_site",
    "Site": "site",

    "Transmembrane": "transmembrane",
    "Signal peptide": "signal_peptide",
    "Topological domain": "topological_domain",

    "Disulfide bond": "disulfide_bond",
    "Chain": "chain",
}


features_raw = features_raw.rename(
    columns=rename_map
)


# ======================================================================
# VALIDATE ACCESSION COLUMN
# ======================================================================

if "uniprot_accession" not in features_raw.columns:

    raise ValueError(
        "Could not find UniProt accession column.\n"
        f"Columns returned:\n"
        f"{list(features_raw.columns)}"
    )


# ======================================================================
# SAVE RAW RESULTS
# ======================================================================

os.makedirs(
    os.path.dirname(RAW_OUTPUT_FILE),
    exist_ok=True
)


features_raw.to_csv(
    RAW_OUTPUT_FILE,
    sep="\t",
    index=False
)


print()
print(
    f"Raw annotations saved to:\n"
    f"  {RAW_OUTPUT_FILE}"
)


# ======================================================================
# NUMERIC PROTEIN LENGTH
# ======================================================================

if "protein_length" in features_raw.columns:

    features_raw["protein_length"] = pd.to_numeric(
        features_raw["protein_length"],
        errors="coerce"
    )


# ======================================================================
# FEATURE COUNT FUNCTION
# ======================================================================

def count_features(value):
    """
    Count the number of annotated feature records.

    UniProt may return multiple feature annotations in one cell.
    The exact textual representation can vary, so this function
    uses semicolon/newline separation as the primary delimiter.
    """

    if pd.isna(value):

        return 0


    value = str(value).strip()


    if not value:

        return 0


    # UniProt TSV feature annotations generally separate
    # individual annotations with semicolons.

    parts = re.split(
        r";\s*|\n+",
        value
    )


    parts = [
        p.strip()
        for p in parts
        if p.strip()
    ]


    return len(parts)


# ======================================================================
# FEATURE COLUMNS
# ======================================================================

feature_columns = [
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
    "chain",
]


# ======================================================================
# CREATE NUMERIC COUNTS
# ======================================================================

print()
print("Calculating protein-level feature counts...")


for column in feature_columns:

    if column in features_raw.columns:

        output_column = (
            "n_"
            + column
        )


        features_raw[output_column] = (
            features_raw[column]
            .apply(count_features)
        )


# ======================================================================
# BOOLEAN FEATURE PRESENCE
# ======================================================================

boolean_features = [
    "domain",
    "active_site",
    "binding_site",
    "transmembrane",
    "signal_peptide",
    "disulfide_bond",
    "dna_binding",
]


for column in boolean_features:

    if column in features_raw.columns:

        output_column = (
            "has_"
            + column
        )


        features_raw[output_column] = (
            features_raw[column]
            .apply(count_features)
            .gt(0)
            .astype(int)
        )


# ======================================================================
# BUILD CLEAN PROTEIN FEATURE TABLE
# ======================================================================

keep_columns = [
    "uniprot_accession",
    "protein_length",
    "reviewed",
]


# Add all generated numeric features
for column in features_raw.columns:

    if column.startswith("n_"):

        keep_columns.append(column)


# Add all generated boolean features
for column in features_raw.columns:

    if column.startswith("has_"):

        keep_columns.append(column)


# Remove duplicate column names while preserving order
keep_columns = list(
    dict.fromkeys(keep_columns)
)


features = features_raw[
    [
        column
        for column in keep_columns
        if column in features_raw.columns
    ]
].copy()


# ======================================================================
# NORMALIZE REVIEWED
# ======================================================================

if "reviewed" in features.columns:

    features["reviewed"] = (
        features["reviewed"]
        .map({
            "reviewed": 1,
            "unreviewed": 0,
            "Reviewed": 1,
            "Unreviewed": 0
        })
    )


# ======================================================================
# REMOVE DUPLICATE ACCESSIONS
# ======================================================================

features = features.drop_duplicates(
    subset=["uniprot_accession"]
)


# ======================================================================
# SAVE CLEAN FEATURE TABLE
# ======================================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ======================================================================
# COVERAGE AUDIT
# ======================================================================

requested_accessions = set(
    accessions
)

returned_accessions = set(
    features["uniprot_accession"]
)


missing_accessions = (
    requested_accessions -
    returned_accessions
)


coverage = (
    len(returned_accessions)
    /
    len(requested_accessions)
    *
    100
)


# ======================================================================
# SUMMARY
# ======================================================================

print()
print("=" * 70)
print("UNIPROT FEATURE EXTRACTION SUMMARY")
print("=" * 70)


print(
    f"Requested UniProt accessions:   "
    f"{len(requested_accessions):,}"
)


print(
    f"Returned UniProt accessions:    "
    f"{len(returned_accessions):,}"
)


print(
    f"Missing UniProt accessions:     "
    f"{len(missing_accessions):,}"
)


print(
    f"Coverage:                       "
    f"{coverage:.2f}%"
)


print(
    f"Feature table rows:             "
    f"{len(features):,}"
)


# ======================================================================
# FEATURE COVERAGE
# ======================================================================

print()
print("Feature availability:")


availability_columns = [
    "protein_length",
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
    "n_disulfide_bond",
    "n_chain",
]


for column in availability_columns:

    if column not in features.columns:

        continue


    if column == "protein_length":

        available = features[column].notna().sum()

    else:

        available = (
            features[column]
            .fillna(0)
            .gt(0)
            .sum()
        )


    percentage = (
        available
        /
        len(features)
        *
        100
    )


    print(
        f"  {column:<25} "
        f"{available:>6,} "
        f"({percentage:>6.2f}%)"
    )


# ======================================================================
# MISSING ACCESSIONS
# ======================================================================

if missing_accessions:

    print()
    print(
        "Example missing UniProt accessions:"
    )

    for accession in sorted(
        missing_accessions
    )[:20]:

        print(
            f"  {accession}"
        )


# ======================================================================
# EXAMPLE DATA
# ======================================================================

print()
print("Example protein features:")

print(
    features
    .head(10)
    .to_string(index=False)
)


# ======================================================================
# FINAL FILES
# ======================================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"Clean feature table:\n"
    f"  {OUTPUT_FILE}"
)

print(
    f"Raw UniProt annotations:\n"
    f"  {RAW_OUTPUT_FILE}"
)

print("=" * 70)

print()
print(
    "UniProt protein feature extraction completed successfully."
)
