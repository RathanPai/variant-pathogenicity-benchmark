import pandas as pd
import requests
import pysam
import time
import random

INPUT = "Data/clinvar_gnomad.csv"
OUTPUT = "Data/gnomad_normalization_test.csv"

FASTA = "/home/xern/.vep/homo_sapiens/116_GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

API = "https://gnomad.broadinstitute.org/api?raw=true"
DATASET = "gnomad_r4"

# ------------------------------------------------------------
# gnomAD query
# ------------------------------------------------------------

QUERY = """
query VariantDetails($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
    variant_id
    chrom
    pos
    ref
    alt
    rsids
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
# Load data + FASTA
# ------------------------------------------------------------

df = pd.read_csv(INPUT, low_memory=False)

fasta = pysam.FastaFile(FASTA)

# ------------------------------------------------------------
# SPDI parser
# ------------------------------------------------------------

def parse_spdi(spdi):
    if pd.isna(spdi):
        return None

    parts = str(spdi).split(":")

    if len(parts) != 4:
        return None

    seq_id, pos, ref, alt = parts

    try:
        pos = int(pos)
    except ValueError:
        return None

    return seq_id, pos, ref, alt


# ------------------------------------------------------------
# Chromosome conversion
# ------------------------------------------------------------

def normalize_chrom(seq_id):

    mapping = {
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
        "NC_000024.10": "Y",
        "NC_012920.1": "MT",
    }

    return mapping.get(seq_id)


# ------------------------------------------------------------
# SPDI -> VCF-style allele
#
# SPDI position is 0-based.
# VCF position is 1-based.
#
# Empty REF/ALT require an anchor.
# ------------------------------------------------------------

def spdi_to_vcf(seq_id, pos, ref, alt):

    chrom = normalize_chrom(seq_id)

    if chrom is None:
        return None

    ref = ref or ""
    alt = alt or ""

    # -------------------------
    # SNV / substitution
    # -------------------------
    if ref and alt:

        vcf_pos = pos + 1

        return chrom, vcf_pos, ref, alt

    # -------------------------
    # Deletion
    # SPDI: position + REF + empty ALT
    # -------------------------
    if ref and not alt:

        if pos <= 0:
            return None

        anchor_pos = pos
        anchor_base = fasta.fetch(
            chrom,
            anchor_pos - 1,
            anchor_pos
        ).upper()

        vcf_pos = anchor_pos

        return (
            chrom,
            vcf_pos,
            anchor_base + ref,
            anchor_base
        )

    # -------------------------
    # Insertion
    # SPDI: position + empty REF + ALT
    # -------------------------
    if not ref and alt:

        anchor_pos = pos + 1

        anchor_base = fasta.fetch(
            chrom,
            pos,
            pos + 1
        ).upper()

        return (
            chrom,
            anchor_pos,
            anchor_base,
            anchor_base + alt
        )

    return None


# ------------------------------------------------------------
# Left-align simple indels
#
# This is intentionally conservative.
# We only left-align alleles where the
# representation is unambiguous.
# ------------------------------------------------------------

def left_align(chrom, pos, ref, alt):

    # SNV / same-length substitution
    if len(ref) == len(alt):
        return chrom, pos, ref, alt

    # Trim common suffix/prefix first
    while (
        len(ref) > 1 and
        len(alt) > 1 and
        ref[-1] == alt[-1]
    ):
        ref = ref[:-1]
        alt = alt[:-1]

    while (
        len(ref) > 1 and
        len(alt) > 1 and
        ref[0] == alt[0]
    ):
        ref = ref[1:]
        alt = alt[1:]
        pos += 1

    # Conservative left-shifting.
    #
    # We retain the VCF anchor base while shifting
    # through repeated sequence contexts.

    while pos > 1:

        prev_base = fasta.fetch(
            chrom,
            pos - 2,
            pos - 1
        ).upper()

        if not prev_base:
            break

        # deletion
        if len(ref) > len(alt):

            if ref[-1] != prev_base:
                break

            ref = prev_base + ref[:-1]
            alt = prev_base + alt
            pos -= 1

        # insertion
        elif len(alt) > len(ref):

            if alt[-1] != prev_base:
                break

            ref = prev_base + ref
            alt = prev_base + alt[:-1]
            pos -= 1

        else:
            break

    return chrom, pos, ref, alt


# ------------------------------------------------------------
# Convert SPDI -> original and normalized IDs
# ------------------------------------------------------------

def make_ids(spdi):

    parsed = parse_spdi(spdi)

    if parsed is None:
        return None

    seq_id, pos, ref, alt = parsed

    vcf = spdi_to_vcf(
        seq_id,
        pos,
        ref,
        alt
    )

    if vcf is None:
        return None

    chrom, vcf_pos, vcf_ref, vcf_alt = vcf

    original_id = (
        f"{chrom}-{vcf_pos}-{vcf_ref}-{vcf_alt}"
    )

    normalized = left_align(
        chrom,
        vcf_pos,
        vcf_ref,
        vcf_alt
    )

    n_chrom, n_pos, n_ref, n_alt = normalized

    normalized_id = (
        f"{n_chrom}-{n_pos}-{n_ref}-{n_alt}"
    )

    return {
        "original_id": original_id,
        "normalized_id": normalized_id,
        "original_pos": vcf_pos,
        "original_ref": vcf_ref,
        "original_alt": vcf_alt,
        "normalized_pos": n_pos,
        "normalized_ref": n_ref,
        "normalized_alt": n_alt,
    }


# ------------------------------------------------------------
# Select failures
# ------------------------------------------------------------

# We only test variants where we have a valid gnomAD ID
# but gnomAD returned "not_found".

failures = df[
    (df["gnomad_status"] == "not_found") &
    (df["gnomad_id_status"] == "valid")
].copy()

print("=" * 80)
print("GNOMAD NORMALIZATION RECOVERY TEST")
print("=" * 80)

print(f"Total valid-ID failures: {len(failures):,}")

# Balanced sample
samples = []

for vt, n, seed in [
    ("SNV", 50, 100),
    ("Deletion", 50, 101),
    ("Duplication", 50, 102),
    ("Indel", 30, 103),
    ("Insertion", 30, 104),
]:

    x = failures[
        failures["variant_type_simple"] == vt
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

print(f"Test sample: {len(test)}")

print("\nComposition:")
print(
    test["variant_type_simple"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Build normalized IDs
# ------------------------------------------------------------

rows = []

for _, row in test.iterrows():

    try:
        ids = make_ids(row["Canonical SPDI"])
    except Exception as e:
        ids = None

    if ids is None:

        rows.append({
            "VariationID": row["VariationID"],
            "variant_type_simple": row["variant_type_simple"],
            "label": row["label"],
            "dbSNP_ID": row["dbSNP ID"],
            "Canonical_SPDI": row["Canonical SPDI"],
            "old_gnomad_id": row["gnomad_variant_id"],
            "original_id": None,
            "normalized_id": None,
            "normalization_changed": False,
            "conversion_status": "FAILED"
        })

        continue

    rows.append({
        "VariationID": row["VariationID"],
        "variant_type_simple": row["variant_type_simple"],
        "label": row["label"],
        "dbSNP_ID": row["dbSNP ID"],
        "Canonical_SPDI": row["Canonical SPDI"],
        "old_gnomad_id": row["gnomad_variant_id"],
        **ids,
        "normalization_changed": (
            ids["original_id"] != ids["normalized_id"]
        ),
        "conversion_status": "OK"
    })


result = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("NORMALIZATION RESULTS")
print("=" * 80)

print(
    result["conversion_status"]
    .value_counts()
    .to_string()
)

print("\nNormalization changes:")
print(
    result["normalization_changed"]
    .value_counts()
    .to_string()
)

print("\nChanges by variant type:")
print(
    pd.crosstab(
        result["variant_type_simple"],
        result["normalization_changed"]
    ).to_string()
)

# ------------------------------------------------------------
# Query normalized IDs
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

        response = session.post(
            API,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:

            return {
                "result": "HTTP_ERROR",
                "raw": response.text[:500]
            }

        data = response.json()

        if "errors" in data:

            messages = [
                e.get("message", "")
                for e in data["errors"]
            ]

            # Important:
            # "Variant not found" is NOT an API failure.
            if any(
                "Variant not found" in m
                for m in messages
            ):

                return {
                    "result": "NOT_FOUND",
                    "raw": str(messages)
                }

            return {
                "result": "GRAPHQL_ERROR",
                "raw": str(messages)
            }

        variant = (
            data
            .get("data", {})
            .get("variant")
        )

        if variant is None:

            return {
                "result": "NOT_FOUND",
                "raw": ""
            }

        genome = variant.get("genome") or {}
        exome = variant.get("exome") or {}

        return {
            "result": "FOUND",
            "returned_variant_id": variant.get(
                "variant_id"
            ),
            "rsids": ";".join(
                variant.get("rsids") or []
            ),
            "chrom": variant.get("chrom"),
            "pos": variant.get("pos"),
            "ref": variant.get("ref"),
            "alt": variant.get("alt"),
            "genome_ac": genome.get("ac"),
            "genome_an": genome.get("an"),
            "genome_hom": genome.get("ac_hom"),
            "exome_ac": exome.get("ac"),
            "exome_an": exome.get("an"),
            "exome_hom": exome.get("ac_hom"),
            "raw": ""
        }

    except Exception as e:

        return {
            "result": "REQUEST_ERROR",
            "raw": str(e)
        }


# ------------------------------------------------------------
# Query both original + normalized
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("QUERYING GNOMAD")
print("=" * 80)

for i, row in result.iterrows():

    if row["conversion_status"] != "OK":
        continue

    print(
        f"[{i+1:3d}/{len(result)}] "
        f"{row['variant_type_simple']:12s} "
        f"{row['normalized_id']}"
    )

    # Query normalized representation
    normalized = query_gnomad(
        row["normalized_id"]
    )

    result.loc[i, "normalized_result"] = (
        normalized["result"]
    )

    result.loc[i, "normalized_returned_id"] = (
        normalized.get("returned_variant_id")
    )

    result.loc[i, "normalized_rsid"] = (
        normalized.get("rsids")
    )

    result.loc[i, "normalized_genome_ac"] = (
        normalized.get("genome_ac")
    )

    result.loc[i, "normalized_genome_an"] = (
        normalized.get("genome_an")
    )

    result.loc[i, "normalized_genome_hom"] = (
        normalized.get("genome_hom")
    )

    result.loc[i, "normalized_exome_ac"] = (
        normalized.get("exome_ac")
    )

    result.loc[i, "normalized_exome_an"] = (
        normalized.get("exome_an")
    )

    result.loc[i, "normalized_exome_hom"] = (
        normalized.get("exome_hom")
    )

    result.loc[i, "normalized_raw"] = (
        normalized.get("raw", "")
    )

    time.sleep(
        0.15 + random.random() * 0.1
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RECOVERY SUMMARY")
print("=" * 80)

print("\nNormalized query result:")
print(
    result["normalized_result"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nBy variant type:")
print(
    pd.crosstab(
        result["variant_type_simple"],
        result["normalized_result"]
    ).to_string()
)

print("\nRecovered variants:")
recovered = result[
    result["normalized_result"] == "FOUND"
]

print(f"Recovered: {len(recovered)}")

if len(recovered):

    print(
        recovered[
            [
                "VariationID",
                "variant_type_simple",
                "Canonical_SPDI",
                "old_gnomad_id",
                "original_id",
                "normalized_id",
                "normalized_returned_id",
                "normalized_rsid"
            ]
        ]
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

result.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)

fasta.close()