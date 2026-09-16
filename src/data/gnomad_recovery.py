
import os
import time
import random
import requests
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Data/gnomad_skipped_recovery.csv"

CHECKPOINT_FILE = (
    "Data/gnomad_http_retry_checkpoint.csv"
)

OUTPUT_FILE = (
    "Data/gnomad_http_retry_results.csv"
)

GNOMAD_API = (
    "https://gnomad.broadinstitute.org/api?raw=true"
)

DATASET = "gnomad_r4"

# Keep this small because we're recovering unreliable requests
BATCH_SIZE = 5

# Network retry settings
MAX_RETRIES = 6
INITIAL_BACKOFF = 2
MAX_BACKOFF = 60

# Delay between batches
REQUEST_DELAY = 0.5

# Save every 100 variants
CHECKPOINT_EVERY = 100

# HTTP codes that are genuinely worth retrying
RETRYABLE_HTTP_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504
}


# ============================================================
# GRAPHQL QUERY
# ============================================================

def make_query(variant_ids):

    query_parts = []

    for i, variant_id in enumerate(variant_ids):

        safe_id = (
            str(variant_id)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        query_parts.append(
            f"""
            v{i}: variant(
                variantId: "{safe_id}",
                dataset: {DATASET}
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
    query {
    """ + "\n".join(query_parts) + """
    }
    """


# ============================================================
# QUERY ONE BATCH
# ============================================================

def query_batch(variant_ids):

    query = make_query(variant_ids)

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = requests.post(
                GNOMAD_API,
                json={"query": query},
                timeout=90,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent":
                        "Bioinformatics-Variant-Pathogenicity-Pipeline"
                }
            )

            # =================================================
            # HTTP STATUS MUST BE CHECKED FIRST
            # =================================================

            if response.status_code != 200:

                status_code = response.status_code

                # Permanent HTTP failure
                if status_code not in RETRYABLE_HTTP_CODES:

                    return {
                        "status": "permanent_http_error",
                        "http_status": status_code,
                        "text": response.text[:1000]
                    }

                # Retry exhausted
                if attempt >= MAX_RETRIES:

                    return {
                        "status": "retry_exhausted",
                        "http_status": status_code,
                        "text": response.text[:1000]
                    }

                backoff = min(
                    INITIAL_BACKOFF * (2 ** attempt),
                    MAX_BACKOFF
                )

                backoff += random.uniform(0, 1)

                print(
                    f"    HTTP {status_code} -> "
                    f"retry {attempt + 1}/{MAX_RETRIES} "
                    f"in {backoff:.1f}s"
                )

                time.sleep(backoff)

                continue

            # =================================================
            # HTTP 200
            # =================================================

            try:

                data = response.json()

            except Exception as e:

                if attempt >= MAX_RETRIES:

                    return {
                        "status": "invalid_json",
                        "error": str(e)
                    }

                backoff = min(
                    INITIAL_BACKOFF * (2 ** attempt),
                    MAX_BACKOFF
                )

                print(
                    f"    Invalid JSON -> "
                    f"retry {attempt + 1}/{MAX_RETRIES} "
                    f"in {backoff:.1f}s"
                )

                time.sleep(backoff)

                continue

            # =================================================
            # IMPORTANT:
            #
            # HTTP 200 + GraphQL errors is STILL a valid
            # response. Individual variants can be found
            # or not found.
            # =================================================

            return {
                "status": "success",
                "data": data
            }

        # =====================================================
        # NETWORK TIMEOUT
        # =====================================================

        except requests.exceptions.Timeout:

            if attempt >= MAX_RETRIES:

                return {
                    "status": "timeout"
                }

            backoff = min(
                INITIAL_BACKOFF * (2 ** attempt),
                MAX_BACKOFF
            )

            backoff += random.uniform(0, 1)

            print(
                f"    Timeout -> "
                f"retry {attempt + 1}/{MAX_RETRIES} "
                f"in {backoff:.1f}s"
            )

            time.sleep(backoff)

        # =====================================================
        # OTHER REQUEST ERROR
        # =====================================================

        except requests.exceptions.RequestException as e:

            if attempt >= MAX_RETRIES:

                return {
                    "status": "request_error",
                    "error": str(e)
                }

            backoff = min(
                INITIAL_BACKOFF * (2 ** attempt),
                MAX_BACKOFF
            )

            backoff += random.uniform(0, 1)

            print(
                f"    Request error -> "
                f"retry {attempt + 1}/{MAX_RETRIES} "
                f"in {backoff:.1f}s"
            )

            time.sleep(backoff)

    return {
        "status": "retry_exhausted"
    }


# ============================================================
# EXTRACT INDIVIDUAL RESULTS
# ============================================================

def extract_results(data, variant_ids):

    results = []

    response_data = data.get("data", {})

    graphql_errors = data.get("errors", [])

    # Map GraphQL errors to aliases.
    error_by_alias = {}

    for error in graphql_errors:

        path = error.get("path", [])

        if path:

            alias = path[0]

            error_by_alias[alias] = (
                error.get("message", "")
            )

    # ========================================================
    # Process every requested variant individually
    # ========================================================

    for i, variant_id in enumerate(variant_ids):

        alias = f"v{i}"

        variant = response_data.get(alias)

        # ----------------------------------------------------
        # FOUND
        # ----------------------------------------------------

        if variant is not None:

            genome = (
                variant.get("genome")
                or {}
            )

            exome = (
                variant.get("exome")
                or {}
            )

            genome_ac = genome.get("ac")
            genome_an = genome.get("an")
            genome_hom = genome.get("ac_hom")

            exome_ac = exome.get("ac")
            exome_an = exome.get("an")
            exome_hom = exome.get("ac_hom")

            # Calculate AF ourselves
            genome_af = None

            if (
                genome_ac is not None
                and genome_an is not None
                and genome_an > 0
            ):

                genome_af = (
                    genome_ac /
                    genome_an
                )

            exome_af = None

            if (
                exome_ac is not None
                and exome_an is not None
                and exome_an > 0
            ):

                exome_af = (
                    exome_ac /
                    exome_an
                )

            results.append({

                "recovery_gnomad_id":
                    variant_id,

                "retry_status":
                    "found",

                "returned_variant_id":
                    variant.get("variant_id"),

                "returned_chrom":
                    variant.get("chrom"),

                "returned_pos":
                    variant.get("pos"),

                "returned_ref":
                    variant.get("ref"),

                "returned_alt":
                    variant.get("alt"),

                "rsids":
                    ",".join(
                        variant.get("rsids") or []
                    ),

                "genome_ac":
                    genome_ac,

                "genome_an":
                    genome_an,

                "genome_af":
                    genome_af,

                "genome_hom":
                    genome_hom,

                "exome_ac":
                    exome_ac,

                "exome_an":
                    exome_an,

                "exome_af":
                    exome_af,

                "exome_hom":
                    exome_hom,

                "api_message":
                    None

            })

        # ----------------------------------------------------
        # NOT FOUND
        # ----------------------------------------------------

        else:

            message = error_by_alias.get(
                alias,
                "Variant not found"
            )

            results.append({

                "recovery_gnomad_id":
                    variant_id,

                "retry_status":
                    "not_found",

                "returned_variant_id":
                    None,

                "returned_chrom":
                    None,

                "returned_pos":
                    None,

                "returned_ref":
                    None,

                "returned_alt":
                    None,

                "rsids":
                    None,

                "genome_ac":
                    None,

                "genome_an":
                    None,

                "genome_af":
                    None,

                "genome_hom":
                    None,

                "exome_ac":
                    None,

                "exome_an":
                    None,

                "exome_af":
                    None,

                "exome_hom":
                    None,

                "api_message":
                    message

            })

    return results


# ============================================================
# LOAD ORIGINAL RECOVERY FILE
# ============================================================

print("=" * 70)
print("GNOMAD CORRECTED HTTP/GRAPHQL RECOVERY")
print("=" * 70)

print("\nLoading:")
print(f"  {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(
    f"Total rows: {len(df):,}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "VariationID",
    "variant_type_simple",
    "label",
    "recovery_gnomad_id",
    "recovery_status"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# SELECT ONLY ORIGINAL HTTP ERRORS
# ============================================================

retry_df = df[
    df["recovery_status"]
    .astype(str)
    .str.upper()
    .eq("HTTP_ERROR")
].copy()

print(
    f"\nOriginal HTTP_ERROR rows: "
    f"{len(retry_df):,}"
)


# ============================================================
# CLEAN IDs
# ============================================================

retry_df["recovery_gnomad_id"] = (
    retry_df["recovery_gnomad_id"]
    .astype(str)
    .str.strip()
)

retry_df = retry_df[
    retry_df["recovery_gnomad_id"]
    != ""
].copy()

print(
    f"Valid IDs: {len(retry_df):,}"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if os.path.exists(CHECKPOINT_FILE):

    print(
        f"\nCheckpoint found:"
        f"\n  {CHECKPOINT_FILE}"
    )

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )

    completed_ids = set(
        checkpoint[
            "recovery_gnomad_id"
        ].astype(str)
    )

    print(
        f"Already processed: "
        f"{len(completed_ids):,}"
    )

else:

    checkpoint = pd.DataFrame()

    completed_ids = set()

    print(
        "\nNo checkpoint found."
    )


# ============================================================
# REMOVE COMPLETED
# ============================================================

retry_df = retry_df[
    ~retry_df[
        "recovery_gnomad_id"
    ].isin(completed_ids)
].copy()

retry_df.reset_index(
    drop=True,
    inplace=True
)

print(
    f"Remaining: {len(retry_df):,}"
)

if len(retry_df) == 0:

    print(
        "\nNothing left to process."
    )

    raise SystemExit


# ============================================================
# PROCESS
# ============================================================

results = []

total = len(retry_df)

start_time = time.time()

for start in range(
    0,
    total,
    BATCH_SIZE
):

    batch = retry_df.iloc[
        start:start + BATCH_SIZE
    ]

    variant_ids = (
        batch[
            "recovery_gnomad_id"
        ]
        .astype(str)
        .tolist()
    )

    end = min(
        start + BATCH_SIZE,
        total
    )

    print(
        f"\n[{start + 1:,}-{end:,}/{total:,}]"
    )

    print(
        "  Querying:"
    )

    for vid in variant_ids:
        print(
            f"    {vid}"
        )

    response = query_batch(
        variant_ids
    )

    # ========================================================
    # SUCCESSFUL HTTP REQUEST
    # ========================================================

    if response["status"] == "success":

        batch_results = extract_results(
            response["data"],
            variant_ids
        )

        results.extend(
            batch_results
        )

        found = sum(
            r["retry_status"] == "found"
            for r in batch_results
        )

        not_found = sum(
            r["retry_status"] == "not_found"
            for r in batch_results
        )

        print(
            f"  Found: {found}"
        )

        print(
            f"  Not found: {not_found}"
        )

    # ========================================================
    # GENUINE FAILURE
    # ========================================================

    else:

        print(
            f"  FAILED: "
            f"{response['status']}"
        )

        for vid in variant_ids:

            result = {

                "recovery_gnomad_id":
                    vid,

                "retry_status":
                    response["status"],

                "returned_variant_id":
                    None,

                "returned_chrom":
                    None,

                "returned_pos":
                    None,

                "returned_ref":
                    None,

                "returned_alt":
                    None,

                "rsids":
                    None,

                "genome_ac":
                    None,

                "genome_an":
                    None,

                "genome_af":
                    None,

                "genome_hom":
                    None,

                "exome_ac":
                    None,

                "exome_an":
                    None,

                "exome_af":
                    None,

                "exome_hom":
                    None,

                "api_message":
                    response.get(
                        "error",
                        response.get(
                            "text",
                            None
                        )
                    )
            }

            if "http_status" in response:

                result[
                    "http_status"
                ] = response[
                    "http_status"
                ]

            results.append(result)

    # ========================================================
    # DELAY
    # ========================================================

    time.sleep(
        REQUEST_DELAY +
        random.uniform(0, 0.25)
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    processed = start + len(batch)

    if (
        processed % CHECKPOINT_EVERY == 0
        or processed >= total
    ):

        current = pd.DataFrame(
            results
        )

        if len(checkpoint) > 0:

            combined = pd.concat(
                [
                    checkpoint,
                    current
                ],
                ignore_index=True
            )

        else:

            combined = current

        combined = combined.drop_duplicates(
            subset=[
                "recovery_gnomad_id"
            ],
            keep="last"
        )

        combined.to_csv(
            CHECKPOINT_FILE,
            index=False
        )

        elapsed = (
            time.time() -
            start_time
        )

        rate = (
            processed /
            elapsed *
            60
        )

        print(
            "\n  CHECKPOINT SAVED"
        )

        print(
            f"  Progress: "
            f"{processed:,}/{total:,}"
        )

        print(
            f"  Rate: "
            f"{rate:.1f} variants/min"
        )


# ============================================================
# FINAL SAVE
# ============================================================

current = pd.DataFrame(
    results
)

if len(checkpoint) > 0:

    final = pd.concat(
        [
            checkpoint,
            current
        ],
        ignore_index=True
    )

else:

    final = current

final = final.drop_duplicates(
    subset=[
        "recovery_gnomad_id"
    ],
    keep="last"
)

final.to_csv(
    OUTPUT_FILE,
    index=False
)

final.to_csv(
    CHECKPOINT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

elapsed = (
    time.time() -
    start_time
)

print("\n")
print("=" * 70)
print("RECOVERY COMPLETE")
print("=" * 70)

print(
    f"\nProcessed: "
    f"{len(final):,}"
)

print(
    "\nRetry status:"
)

print(
    final[
        "retry_status"
    ].value_counts()
)

if (
    "variant_type_simple"
    in final.columns
):

    # Merge type information for summary
    type_map = (
        retry_df[
            [
                "recovery_gnomad_id",
                "variant_type_simple"
            ]
        ]
        .drop_duplicates(
            "recovery_gnomad_id"
        )
    )

    summary = final.merge(
        type_map,
        on="recovery_gnomad_id",
        how="left"
    )

    print(
        "\nBy variant type:"
    )

    print(
        pd.crosstab(
            summary[
                "variant_type_simple"
            ],
            summary[
                "retry_status"
            ]
        )
    )

print(
    f"\nRuntime: "
    f"{elapsed / 60:.1f} minutes"
)

print(
    f"\nOutput:"
    f"\n  {OUTPUT_FILE}"
)

print(
    f"\nCheckpoint:"
    f"\n  {CHECKPOINT_FILE}"
)

print("\nIMPORTANT:")
print(
    "Do NOT merge into clinvar_gnomad.csv yet."
)

print("=" * 70)