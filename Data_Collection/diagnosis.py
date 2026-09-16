import pandas as pd
import requests
import time
import random

INPUT = "Data/clinvar_gnomad.csv"
OUTPUT = "Data/gnomad_recovery_test.csv"

API = "https://gnomad.broadinstitute.org/api?raw=true"
DATASET = "gnomad_r4"

# ------------------------------------------------------------
# GraphQL query
# ------------------------------------------------------------

QUERY = """
query VariantQuery($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
    variant_id
    rsids
    chrom
    pos
    ref
    alt
    genome {
      ac
      an
      ac_hom
    }
    exome {
      ac
      an
      ac_hom
    }
  }
}
"""

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT, low_memory=False)

# We deliberately test different failure categories.
#
# 1. 30 SNVs with rsIDs
# 2. 30 SNVs without gnomAD result
# 3. 30 deletions
# 4. 20 duplications
# 5. 20 indels
# 6. 20 insertions with valid IDs
#
# Total <= 150

samples = []

# SNV failures with rsID
x = df[
    (df["variant_type_simple"] == "SNV") &
    (df["gnomad_status"] == "not_found") &
    (df["dbSNP ID"].notna())
].sample(
    n=min(30, len(df[
        (df["variant_type_simple"] == "SNV") &
        (df["gnomad_status"] == "not_found") &
        (df["dbSNP ID"].notna())
    ])),
    random_state=42
)

samples.append(x)

# SNV failures regardless of rsID
x = df[
    (df["variant_type_simple"] == "SNV") &
    (df["gnomad_status"] == "not_found")
].sample(
    n=min(30, len(df[
        (df["variant_type_simple"] == "SNV") &
        (df["gnomad_status"] == "not_found")
    ])),
    random_state=43
)

samples.append(x)

# Non-SNV valid IDs
for vt, n, seed in [
    ("Deletion", 30, 44),
    ("Duplication", 20, 45),
    ("Indel", 20, 46),
    ("Insertion", 20, 47),
]:
    x = df[
        (df["variant_type_simple"] == vt) &
        (df["gnomad_id_status"] == "valid") &
        (df["gnomad_status"] == "not_found")
    ]

    if len(x):
        samples.append(
            x.sample(
                n=min(n, len(x)),
                random_state=seed
            )
        )

test = pd.concat(samples).drop_duplicates(
    subset=["VariationID"]
).reset_index(drop=True)

print("=" * 70)
print("GNOMAD RECOVERY TEST")
print("=" * 70)

print(f"Test variants: {len(test)}")

print("\nTest composition:")
print(
    test["variant_type_simple"]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# Query function
# ------------------------------------------------------------

session = requests.Session()

def query_gnomad(variant_id):

    payload = {
        "query": QUERY,
        "variables": {
            "variantId": variant_id,
            "dataset": DATASET
        }
    }

    try:
        r = session.post(
            API,
            json=payload,
            timeout=30
        )

        if r.status_code != 200:
            return {
                "result": "HTTP_ERROR",
                "http_status": r.status_code,
                "raw": r.text[:500]
            }

        data = r.json()

        if "errors" in data:
            return {
                "result": "GRAPHQL_ERROR",
                "http_status": r.status_code,
                "raw": str(data["errors"])[:500]
            }

        variant = data.get("data", {}).get("variant")

        if variant is None:
            return {
                "result": "NOT_FOUND",
                "http_status": r.status_code,
                "raw": ""
            }

        return {
            "result": "FOUND",
            "http_status": r.status_code,
            "returned_variant_id": variant.get("variant_id"),
            "rsids": ";".join(variant.get("rsids") or []),
            "chrom": variant.get("chrom"),
            "pos": variant.get("pos"),
            "ref": variant.get("ref"),
            "alt": variant.get("alt"),
            "genome_ac": (
                variant.get("genome") or {}
            ).get("ac"),
            "genome_an": (
                variant.get("genome") or {}
            ).get("an"),
            "genome_hom": (
                variant.get("genome") or {}
            ).get("ac_hom"),
            "exome_ac": (
                variant.get("exome") or {}
            ).get("ac"),
            "exome_an": (
                variant.get("exome") or {}
            ).get("an"),
            "exome_hom": (
                variant.get("exome") or {}
            ).get("ac_hom"),
        }

    except Exception as e:
        return {
            "result": "REQUEST_ERROR",
            "http_status": None,
            "raw": str(e)
        }


# ------------------------------------------------------------
# Run test
# ------------------------------------------------------------

results = []

for i, (_, row) in enumerate(test.iterrows(), 1):

    variant_id = row["gnomad_variant_id"]

    result = query_gnomad(variant_id)

    output = {
        "VariationID": row["VariationID"],
        "variant_type_simple": row["variant_type_simple"],
        "label": row["label"],
        "dbSNP_ID": row["dbSNP ID"],
        "Canonical_SPDI": row["Canonical SPDI"],
        "original_gnomad_variant_id": variant_id,
        **result
    }

    results.append(output)

    print(
        f"[{i:3d}/{len(test)}] "
        f"{row['variant_type_simple']:12s} "
        f"{variant_id} -> "
        f"{result['result']}"
    )

    # Avoid hammering API
    time.sleep(0.15 + random.random() * 0.1)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = pd.DataFrame(results)

out.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    out.groupby(
        ["variant_type_simple", "result"]
    ).size().to_string()
)

print(f"\nSaved: {OUTPUT}")