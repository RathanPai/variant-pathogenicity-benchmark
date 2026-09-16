


"""
======================================================================
ENSEMBL PROTEIN → UNIPROTKB MAPPING
======================================================================

Input:
    Data/vep_unique_protein_ids.txt

Output:
    Data/ensembl_to_uniprot.csv
    Data/ensembl_proteins_unmapped.txt

Purpose:
    Map VEP Ensembl Protein IDs (ENSP...) to UniProtKB accessions.

Notes:
    - Uses the official UniProt ID Mapping API.
    - Supports one-to-many Ensembl → UniProt mappings.
    - Does NOT retrieve disease/pathogenicity annotations.
    - This mapping is only used to connect VEP protein IDs to
      UniProt protein-level functional features later.
======================================================================
"""

import os
import sys
import time
from io import StringIO

import pandas as pd
import requests


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT_FILE = "Data/vep_unique_protein_ids.txt"

OUTPUT_FILE = "Data/ensembl_to_uniprot.csv"
UNMAPPED_FILE = "Data/ensembl_proteins_unmapped.txt"

API_BASE = "https://rest.uniprot.org"

FROM_DB = "Ensembl_Protein"
TO_DB = "UniProtKB"

POLL_INTERVAL = 5
REQUEST_TIMEOUT = 60
RESULT_TIMEOUT = 180

MAX_RETRIES = 5


# ======================================================================
# PRINT HEADER
# ======================================================================

print("=" * 70)
print("ENSEMBL PROTEIN → UNIPROTKB MAPPING")
print("=" * 70)


# ======================================================================
# HTTP SESSION
# ======================================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "Bioinformatics-Variant-Pathogenicity-Pipeline/1.0"
})


def request_with_retry(
    method,
    url,
    *,
    timeout=REQUEST_TIMEOUT,
    **kwargs
):
    """
    Make an HTTP request with retry handling for temporary failures.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.request(
                method,
                url,
                timeout=timeout,
                **kwargs
            )

            # Retry temporary server/rate-limit errors
            if response.status_code in (429, 500, 502, 503, 504):

                print(
                    f"  HTTP {response.status_code}; "
                    f"retry {attempt}/{MAX_RETRIES}..."
                )

                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                    continue

            response.raise_for_status()

            return response

        except requests.RequestException as e:

            print(
                f"  Request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                raise

    raise RuntimeError("Request failed after maximum retries.")


# ======================================================================
# LOAD ENSEMBL PROTEIN IDS
# ======================================================================

print()
print("Loading protein IDs...")

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


with open(INPUT_FILE, "r") as f:

    ids = [
        line.strip()
        for line in f
        if line.strip()
    ]


# Remove duplicates while preserving order
ids = list(dict.fromkeys(ids))


print(f"Unique Ensembl protein IDs: {len(ids):,}")


if len(ids) == 0:

    raise ValueError(
        "No Ensembl protein IDs were found in the input file."
    )


# ======================================================================
# BASIC VALIDATION
# ======================================================================

invalid_ids = [
    x for x in ids
    if not x.startswith("ENSP")
]

if invalid_ids:

    print()
    print(
        f"WARNING: {len(invalid_ids):,} IDs do not start with 'ENSP'."
    )

    print("First few unexpected IDs:")

    for x in invalid_ids[:10]:
        print(f"  {x}")


# ======================================================================
# SUBMIT UNIPROT ID MAPPING JOB
# ======================================================================

print()
print("Submitting UniProt ID mapping job...")


mapping_url = f"{API_BASE}/idmapping/run"


payload = {
    "from": FROM_DB,
    "to": TO_DB,
    "ids": ",".join(ids)
}


response = request_with_retry(
    "POST",
    mapping_url,
    data=payload
)


print(f"HTTP status: {response.status_code}")


response_json = response.json()


job_id = response_json.get("jobId")


if not job_id:

    raise RuntimeError(
        "UniProt did not return a job ID.\n"
        f"Response:\n{response.text[:2000]}"
    )


print(f"Job ID: {job_id}")


# ======================================================================
# WAIT FOR JOB TO FINISH
# ======================================================================

print()
print("Waiting for UniProt job...")


status_url = f"{API_BASE}/idmapping/status/{job_id}"


while True:

    response = request_with_retry(
        "GET",
        status_url
    )

    status = response.json()


    # --------------------------------------------------------------
    # UniProt may report completion using different fields.
    # --------------------------------------------------------------

    job_status = status.get("jobStatus")

    if job_status:

        if job_status == "FINISHED":
            break

        if job_status in (
            "ERROR",
            "FAILED"
        ):

            raise RuntimeError(
                f"UniProt mapping job failed:\n{status}"
            )


    # Some API responses indicate results directly
    if "results" in status:

        break


    print("  Still processing...")

    time.sleep(POLL_INTERVAL)


print("Job finished.")


# ======================================================================
# DOWNLOAD RESULTS
# ======================================================================

print()
print("Downloading mapping results...")


# The stream endpoint returns the actual mapping table.
results_url = f"{API_BASE}/idmapping/stream/{job_id}"


response = request_with_retry(
    "GET",
    results_url,
    timeout=RESULT_TIMEOUT,
    params={
        "format": "tsv"
    }
)


print(f"HTTP status: {response.status_code}")


# ======================================================================
# PARSE TSV
# ======================================================================

if not response.text.strip():

    raise RuntimeError(
        "UniProt returned an empty mapping result."
    )


mapping = pd.read_csv(
    StringIO(response.text),
    sep="\t",
    dtype=str
)


print()
print(f"Mapping rows returned: {len(mapping):,}")


print()
print("Columns:")

for column in mapping.columns:

    print(f"  {column}")


# ======================================================================
# VALIDATE RESPONSE
# ======================================================================

if "From" not in mapping.columns:

    raise ValueError(
        "UniProt response does not contain the expected "
        "'From' column.\n"
        f"Columns received: {list(mapping.columns)}"
    )


if "To" not in mapping.columns:

    raise ValueError(
        "UniProt response does not contain the expected "
        "'To' column.\n"
        f"Columns received: {list(mapping.columns)}"
    )


# ======================================================================
# RENAME COLUMNS
# ======================================================================

mapping = mapping.rename(
    columns={
        "From": "ensembl_protein_id",
        "To": "uniprot_accession"
    }
)


# Keep only the two columns we need
mapping = mapping[
    [
        "ensembl_protein_id",
        "uniprot_accession"
    ]
]


# Remove completely empty rows
mapping = mapping.dropna(
    subset=[
        "ensembl_protein_id",
        "uniprot_accession"
    ]
)


# Remove duplicate mappings
mapping = mapping.drop_duplicates()


# Clean whitespace
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


# ======================================================================
# SAVE MAPPING
# ======================================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


mapping.to_csv(
    OUTPUT_FILE,
    index=False
)


# ======================================================================
# DETERMINE MAPPED / UNMAPPED IDS
# ======================================================================

input_id_set = set(ids)

mapped_id_set = set(
    mapping["ensembl_protein_id"]
)


mapped_ids = input_id_set.intersection(
    mapped_id_set
)


unmapped_ids = input_id_set - mapped_id_set


# ======================================================================
# SAVE UNMAPPED IDS
# ======================================================================

with open(
    UNMAPPED_FILE,
    "w"
) as f:

    for protein_id in sorted(unmapped_ids):

        f.write(
            protein_id + "\n"
        )


# ======================================================================
# COVERAGE
# ======================================================================

total_ids = len(input_id_set)

mapped_count = len(mapped_ids)

unmapped_count = len(unmapped_ids)


if total_ids > 0:

    coverage = (
        mapped_count /
        total_ids
    ) * 100

else:

    coverage = 0.0


# ======================================================================
# ONE-TO-MANY MAPPINGS
# ======================================================================

mapping_counts = (
    mapping
    .groupby("ensembl_protein_id")
    ["uniprot_accession"]
    .nunique()
)


one_to_many = mapping_counts[
    mapping_counts > 1
]


# ======================================================================
# PRINT SUMMARY
# ======================================================================

print()
print("=" * 70)
print("MAPPING SUMMARY")
print("=" * 70)

print(
    f"Input Ensembl protein IDs:       "
    f"{total_ids:,}"
)

print(
    f"Mapped Ensembl protein IDs:      "
    f"{mapped_count:,}"
)

print(
    f"Unmapped Ensembl protein IDs:    "
    f"{unmapped_count:,}"
)

print(
    f"Mapping coverage:                "
    f"{coverage:.2f}%"
)

print()

print(
    f"Unique UniProt accessions:       "
    f"{mapping['uniprot_accession'].nunique():,}"
)

print(
    f"Total mapping rows:              "
    f"{len(mapping):,}"
)

print(
    f"One-to-many Ensembl mappings:    "
    f"{len(one_to_many):,}"
)


# ======================================================================
# SHOW EXAMPLES
# ======================================================================

print()
print("Example mappings:")

print(
    mapping
    .head(10)
    .to_string(index=False)
)


if unmapped_count > 0:

    print()
    print("Example unmapped Ensembl proteins:")

    for protein_id in sorted(unmapped_ids)[:10]:

        print(f"  {protein_id}")


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"Mapping file:       {OUTPUT_FILE}"
)

print(
    f"Unmapped IDs:       {UNMAPPED_FILE}"
)

print("=" * 70)

print()
print("UniProt mapping completed successfully.")
