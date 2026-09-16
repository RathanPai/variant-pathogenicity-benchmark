
import pandas as pd
import requests
import time
import json
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("Data/clinvar_clean.csv")
OUTPUT_FILE = Path("Data/clinvar_gnomad.csv")
CHECKPOINT_FILE = Path("Data/gnomad_checkpoint.csv")

API_URL = "https://gnomad.broadinstitute.org/api?raw=true"
DATASET_ID = "gnomad_r4"

# REAL batching now
BATCH_SIZE = 5

# Delay BETWEEN batch requests
SLEEP_SECONDS = 0.5

CHECKPOINT_EVERY = 100
MAX_RETRIES = 6
REQUEST_TIMEOUT = 60


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "ClinVar-gNOMAD-Annotation-Pipeline/1.0"
})


# ============================================================
# LOAD CLINVAR
# ============================================================

print("=" * 70)
print("GNOMAD BULK ANNOTATION - BATCH MODE")
print("=" * 70)

print("\nLoading ClinVar dataset...")

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    low_memory=False
)

print(f"ClinVar variants: {len(df):,}")


# ============================================================
# SPDI → GNOMAD ID
# ============================================================

chromosome_map = {
    "NC_000001.11": "1",
    "NC_000002.12": "2",
    "NC_000003.12": "3",
    "NC_000004.12": "4",
    "NC_000005.10": "5",
    "NC_000006.12": "6",
    "NC_000007.14": "7",
    "NC_000008.11": "8",
    "NC_000009.12": "9",
    "NC_000010.11": "10",
    "NC_000011.10": "11",
    "NC_000012.12": "12",
    "NC_000013.11": "13",
    "NC_000014.9": "14",
    "NC_000015.10": "15",
    "NC_000016.10": "16",
    "NC_000017.11": "17",
    "NC_000018.10": "18",
    "NC_000019.10": "19",
    "NC_000020.11": "20",
    "NC_000021.9": "21",
    "NC_000022.11": "22",
    "NC_000023.11": "X",
    "NC_000024.10": "Y"
}


def spdi_to_gnomad(spdi):

    if pd.isna(spdi):
        return None, "invalid_spdi"

    spdi = str(spdi).strip()

    if not spdi:
        return None, "invalid_spdi"

    parts = spdi.split(":")

    if len(parts) != 4:
        return None, "invalid_spdi"

    sequence, position, ref, alt = parts

    chromosome = chromosome_map.get(sequence)

    if chromosome is None:
        return None, "unsupported_sequence"

    try:
        position = int(position) + 1
    except (ValueError, TypeError):
        return None, "invalid_position"

    if alt == "":
        return None, "empty_alt"

    if ref == "":
        return None, "empty_ref"

    variant_id = f"{chromosome}-{position}-{ref}-{alt}"

    return variant_id, "valid"


print("\nCreating gnomAD variant IDs...")

conversion_results = df["Canonical SPDI"].apply(
    spdi_to_gnomad
)

df["gnomad_variant_id"] = conversion_results.apply(
    lambda x: x[0]
)

df["gnomad_id_status"] = conversion_results.apply(
    lambda x: x[1]
)


valid_count = (
    df["gnomad_id_status"] == "valid"
).sum()

invalid_count = len(df) - valid_count

print(f"Valid gnomAD IDs: {valid_count:,}")
print(f"Invalid/skipped:  {invalid_count:,}")


# ============================================================
# EMPTY ANNOTATION
# ============================================================

def empty_annotation(status):

    return {
        "gnomad_found": 0,
        "gnomad_status": status,

        "gnomad_variant_id_returned": None,
        "gnomad_rsid": None,

        "gnomad_chrom": None,
        "gnomad_pos": None,
        "gnomad_ref": None,
        "gnomad_alt": None,

        "gnomad_genome_ac": None,
        "gnomad_genome_an": None,
        "gnomad_genome_af": None,
        "gnomad_genome_hom_count": None,

        "gnomad_exome_ac": None,
        "gnomad_exome_an": None,
        "gnomad_exome_af": None,
        "gnomad_exome_hom_count": None
    }


# ============================================================
# EXTRACT ONE VARIANT
# ============================================================

def extract_variant(result, requested_variant_id):

    if result is None:
        return empty_annotation("not_found")

    returned_id = result.get("variant_id")

    # CRITICAL SAFETY CHECK
    if returned_id != requested_variant_id:

        raise RuntimeError(
            "VARIANT ID MISMATCH\n"
            f"Requested: {requested_variant_id}\n"
            f"Returned:  {returned_id}"
        )

    genome = result.get("genome") or {}
    exome = result.get("exome") or {}

    genome_ac = genome.get("ac")
    genome_an = genome.get("an")

    exome_ac = exome.get("ac")
    exome_an = exome.get("an")

    genome_af = None

    if genome_ac is not None and genome_an not in (None, 0):
        genome_af = float(genome_ac) / float(genome_an)

    exome_af = None

    if exome_ac is not None and exome_an not in (None, 0):
        exome_af = float(exome_ac) / float(exome_an)

    return {

        "gnomad_found": 1,
        "gnomad_status": "found",

        "gnomad_variant_id_returned":
            returned_id,

        "gnomad_rsid":
            "|".join(result.get("rsids") or []),

        "gnomad_chrom":
            result.get("chrom"),

        "gnomad_pos":
            result.get("pos"),

        "gnomad_ref":
            result.get("ref"),

        "gnomad_alt":
            result.get("alt"),

        "gnomad_genome_ac":
            genome_ac,

        "gnomad_genome_an":
            genome_an,

        "gnomad_genome_af":
            genome_af,

        "gnomad_genome_hom_count":
            genome.get("ac_hom"),

        "gnomad_exome_ac":
            exome_ac,

        "gnomad_exome_an":
            exome_an,

        "gnomad_exome_af":
            exome_af,

        "gnomad_exome_hom_count":
            exome.get("ac_hom")
    }


# ============================================================
# BUILD REAL BATCH GRAPHQL QUERY
# ============================================================

def build_batch_query(variant_ids):

    fields = []

    for i, variant_id in enumerate(variant_ids):

        # JSON escaping protects IDs containing unusual characters
        escaped_id = json.dumps(variant_id)

        fields.append(
            f"""
            v{i}: variant(
                variantId: {escaped_id}
                dataset: {DATASET_ID}
            ) {{
                variant_id
                rsids
                chrom
                pos
                ref
                alt

                genome {{
                    ac
                    an
                    ac_hom
                }}

                exome {{
                    ac
                    an
                    ac_hom
                }}
            }}
            """
        )

    return """
    query GnomadBatch {
    """ + "\n".join(fields) + """
    }
    """


# ============================================================
# QUERY ONE BATCH
# ============================================================

def query_batch(variant_ids):

    query = build_batch_query(variant_ids)

    payload = {
        "query": query
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.post(
                API_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                try:
                    wait_time = int(retry_after)
                except (TypeError, ValueError):
                    wait_time = 30

                print(
                    f"\nRate limited. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            # ------------------------------------------------
            # SERVER ERROR
            # ------------------------------------------------

            if response.status_code >= 500:

                if attempt >= MAX_RETRIES:
                    raise RuntimeError(
                        f"gnomAD server error "
                        f"HTTP {response.status_code}"
                    )

                wait_time = min(
                    10 * attempt,
                    60
                )

                print(
                    f"\nServer error "
                    f"HTTP {response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            content_type = response.headers.get(
                "content-type",
                ""
            ).lower()

            if "application/json" not in content_type:

                raise RuntimeError(
                    "gnomAD returned non-JSON response.\n"
                    f"Content-Type: {content_type}\n"
                    f"Response: {response.text[:500]}"
                )

            result = response.json()

            data = result.get("data") or {}

            errors = result.get("errors") or []

            # ------------------------------------------------
            # IMPORTANT:
            #
            # GraphQL can return both data AND errors.
            #
            # A "Variant not found" error for one alias should
            # NOT invalidate the other variants in the batch.
            # ------------------------------------------------

            error_by_alias = {}

            for error in errors:

                message = str(
                    error.get("message", "")
                )

                path = error.get("path") or []

                alias = None

                if path:
                    alias = path[0]

                if alias is not None:

                    error_by_alias[str(alias)] = message

            # ------------------------------------------------
            # PROCESS EACH REQUESTED VARIANT
            # ------------------------------------------------

            annotations = {}

            for i, requested_id in enumerate(variant_ids):

                alias = f"v{i}"

                # --------------------------------------------
                # GraphQL error for this particular variant
                # --------------------------------------------

                if alias in error_by_alias:

                    message = error_by_alias[alias]

                    if "variant not found" in message.lower():

                        annotations[requested_id] = (
                            empty_annotation("not_found")
                        )

                        continue

                    if "invalid variant id" in message.lower():

                        annotations[requested_id] = (
                            empty_annotation(
                                "invalid_variant_id"
                            )
                        )

                        continue

                    # Unknown permanent error
                    raise RuntimeError(
                        f"GraphQL error for "
                        f"{requested_id}:\n{message}"
                    )

                # --------------------------------------------
                # Get data
                # --------------------------------------------

                variant = data.get(alias)

                if variant is None:

                    annotations[requested_id] = (
                        empty_annotation("not_found")
                    )

                    continue

                # --------------------------------------------
                # STRICT ID VALIDATION
                # --------------------------------------------

                returned_id = variant.get(
                    "variant_id"
                )

                if returned_id != requested_id:

                    raise RuntimeError(
                        "\n"
                        "!!! VARIANT ID MISMATCH !!!\n"
                        f"Requested: {requested_id}\n"
                        f"Returned:  {returned_id}\n"
                        "\n"
                        "The batch will NOT be accepted."
                    )

                annotations[requested_id] = (
                    extract_variant(
                        variant,
                        requested_id
                    )
                )

            return annotations

        except requests.exceptions.RequestException as e:

            if attempt >= MAX_RETRIES:

                raise RuntimeError(
                    f"Network error:\n{e}"
                )

            wait_time = min(
                5 * attempt,
                60
            )

            print(
                f"\nNetwork error: {e}"
            )

            print(
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Batch query failed."
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(results):

    rows = []

    for variant_id, annotation in results.items():

        row = {
            "gnomad_variant_id": variant_id
        }

        row.update(annotation)

        rows.append(row)

    checkpoint_df = pd.DataFrame(rows)

    temp_file = CHECKPOINT_FILE.with_suffix(
        ".tmp.csv"
    )

    checkpoint_df.to_csv(
        temp_file,
        index=False
    )

    temp_file.replace(
        CHECKPOINT_FILE
    )

    print(
        f"\nCheckpoint saved: "
        f"{len(results):,} unique variants"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

results = {}

if CHECKPOINT_FILE.exists():

    print("\nCheckpoint found!")
    print("Loading previous results...")

    try:

        checkpoint_df = pd.read_csv(
            CHECKPOINT_FILE,
            dtype=str,
            low_memory=False
        )

    except pd.errors.EmptyDataError:

        checkpoint_df = pd.DataFrame()

    if not checkpoint_df.empty:

        checkpoint_df = checkpoint_df.drop_duplicates(
            subset=["gnomad_variant_id"],
            keep="last"
        )

        for _, row in checkpoint_df.iterrows():

            variant_id = row[
                "gnomad_variant_id"
            ]

            results[variant_id] = {
                key: row[key]
                for key in checkpoint_df.columns
                if key != "gnomad_variant_id"
            }

    print(
        f"Previously processed: "
        f"{len(results):,}"
    )

else:

    print("\nNo checkpoint found.")


# ============================================================
# VALID UNIQUE IDs
# ============================================================

valid_variant_ids = (
    df.loc[
        df["gnomad_id_status"] == "valid",
        "gnomad_variant_id"
    ]
    .dropna()
    .drop_duplicates()
    .tolist()
)


remaining = [
    variant_id
    for variant_id in valid_variant_ids
    if variant_id not in results
]


# ============================================================
# QUERY PLAN
# ============================================================

print("\n" + "=" * 70)
print("QUERY PLAN")
print("=" * 70)

print(
    f"ClinVar rows:        {len(df):,}"
)

print(
    f"Valid unique IDs:    {len(valid_variant_ids):,}"
)

print(
    f"Already processed:   {len(results):,}"
)

print(
    f"Remaining variants:  {len(remaining):,}"
)

print(
    f"Batch size:           {BATCH_SIZE}"
)

print(
    f"Approx API requests: "
    f"{(len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE:,}"
)


# ============================================================
# BATCH PROCESSING
# ============================================================

if remaining:

    total_batches = (
        len(remaining) +
        BATCH_SIZE -
        1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(
            0,
            len(remaining),
            BATCH_SIZE
        ),
        start=1
    ):

        batch = remaining[
            start:start + BATCH_SIZE
        ]

        print(
            f"\nBatch "
            f"{batch_number:,}/"
            f"{total_batches:,}"
        )

        print(
            f"Variants: {len(batch)}"
        )

        print(
            f"Range: "
            f"{batch[0]} → {batch[-1]}"
        )

        try:

            batch_results = query_batch(
                batch
            )

            # --------------------------------------------
            # Every requested variant MUST receive a result.
            # --------------------------------------------

            for variant_id in batch:

                if variant_id not in batch_results:

                    raise RuntimeError(
                        "Missing result for "
                        f"{variant_id}"
                    )

                results[variant_id] = (
                    batch_results[variant_id]
                )

            # --------------------------------------------
            # Report
            # --------------------------------------------

            found = sum(
                1
                for variant_id in batch
                if batch_results[variant_id].get(
                    "gnomad_status"
                ) == "found"
            )

            not_found = sum(
                1
                for variant_id in batch
                if batch_results[variant_id].get(
                    "gnomad_status"
                ) == "not_found"
            )

            invalid = sum(
                1
                for variant_id in batch
                if batch_results[variant_id].get(
                    "gnomad_status"
                ) == "invalid_variant_id"
            )

            print(
                f"Found: {found} | "
                f"Not found: {not_found} | "
                f"Invalid: {invalid}"
            )

        except Exception as e:

            print("\n" + "=" * 70)
            print("FATAL ERROR")
            print("=" * 70)

            print(
                f"\nBatch: "
                f"{batch_number}"
            )

            print(
                f"Variants: "
                f"{batch}"
            )

            print(
                f"\nError:\n{e}"
            )

            print(
                "\nSaving successfully "
                "processed variants..."
            )

            save_checkpoint(results)

            print(
                "\nThe failed batch was NOT "
                "added to the checkpoint."
            )

            print(
                "You can rerun the script to resume."
            )

            raise

        # --------------------------------------------
        # CHECKPOINT
        # --------------------------------------------

        if (
            len(results) % CHECKPOINT_EVERY
            < BATCH_SIZE
        ):

            save_checkpoint(results)

        # --------------------------------------------
        # Delay between requests
        # --------------------------------------------

        time.sleep(
            SLEEP_SECONDS
        )


# ============================================================
# FINAL CHECKPOINT
# ============================================================

print("\nSaving final checkpoint...")

save_checkpoint(results)


# ============================================================
# BUILD GNOMAD DATAFRAME
# ============================================================

rows = []

for variant_id, annotation in results.items():

    row = {
        "gnomad_variant_id": variant_id
    }

    row.update(annotation)

    rows.append(row)


gnomad_df = pd.DataFrame(rows)


# ============================================================
# MERGE
# ============================================================

print("\n" + "=" * 70)
print("MERGING GNOMAD + CLINVAR")
print("=" * 70)

original_row_count = len(df)

if not gnomad_df.empty:

    gnomad_df = gnomad_df.drop_duplicates(
        subset=["gnomad_variant_id"],
        keep="last"
    )

    df = df.merge(
        gnomad_df,
        on="gnomad_variant_id",
        how="left",
        validate="many_to_one"
    )


# ============================================================
# INVALID SPDI / EMPTY ALT
# ============================================================

invalid_mask = (
    df["gnomad_id_status"] != "valid"
)

df.loc[
    invalid_mask,
    "gnomad_found"
] = 0

df.loc[
    invalid_mask,
    "gnomad_status"
] = (
    "skipped_" +
    df.loc[
        invalid_mask,
        "gnomad_id_status"
    ].astype(str)
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "gnomad_genome_ac",
    "gnomad_genome_an",
    "gnomad_genome_af",
    "gnomad_genome_hom_count",

    "gnomad_exome_ac",
    "gnomad_exome_an",
    "gnomad_exome_af",
    "gnomad_exome_hom_count"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    else:

        df[col] = pd.NA


# ============================================================
# VALIDATE ROW COUNT
# ============================================================

if len(df) != original_row_count:

    raise RuntimeError(
        "MERGE ERROR!\n"
        f"Before: {original_row_count:,}\n"
        f"After:  {len(df):,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"\nClinVar rows: "
    f"{len(df):,}"
)

print(
    f"Unique valid gnomAD IDs: "
    f"{len(valid_variant_ids):,}"
)

print(
    f"Processed gnomAD IDs: "
    f"{len(results):,}"
)

found_count = (
    df["gnomad_status"] == "found"
).sum()

not_found_count = (
    df["gnomad_status"] == "not_found"
).sum()

skipped_count = (
    df["gnomad_status"]
    .astype(str)
    .str.startswith("skipped_")
    .sum()
)

print(
    f"\nFound:       {found_count:,}"
)

print(
    f"Not found:   {not_found_count:,}"
)

print(
    f"Skipped:     {skipped_count:,}"
)


print("\ngnomAD status breakdown:")

print(
    df["gnomad_status"]
    .fillna("missing")
    .value_counts()
    .to_string()
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    f"\nOriginal rows: "
    f"{original_row_count:,}"
)

print(
    f"Final rows:    "
    f"{len(df):,}"
)

if len(df) == original_row_count:

    print(
        "\n✓ Row count preserved."
    )

else:

    print(
        "\n✗ WARNING: Row count changed!"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nOutput:"
    f"\n{OUTPUT_FILE}"
)

print(
    f"\nCheckpoint:"
    f"\n{CHECKPOINT_FILE}"
)

print(
    "\nSafe to rerun — checkpoint resume is enabled."
)
