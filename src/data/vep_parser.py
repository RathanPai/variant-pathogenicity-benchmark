import pandas as pd
from pathlib import Path

INPUT = Path("Data/clinvar_vep_output.txt")
OUTPUT = Path("Data/clinvar_vep_features.csv")


# ----------------------------------------------------------------------
# VEP consequence severity
#
# Lower number = more severe.
# ----------------------------------------------------------------------

CONSEQUENCE_SEVERITY = {
    "transcript_ablation": 1,
    "splice_acceptor_variant": 2,
    "splice_donor_variant": 3,
    "stop_gained": 4,
    "frameshift_variant": 5,
    "stop_lost": 6,
    "start_lost": 7,
    "missense_variant": 8,
    "inframe_insertion": 9,
    "inframe_deletion": 10,
    "protein_altering_variant": 11,
    "splice_region_variant": 12,
    "regulatory_region_variant": 13,
    "5_prime_UTR_variant": 14,
    "3_prime_UTR_variant": 15,
    "non_coding_transcript_exon_variant": 16,
    "synonymous_variant": 17,
    "intron_variant": 18,
    "NMD_transcript_variant": 19,
    "coding_sequence_variant": 20,
    "upstream_gene_variant": 21,
    "downstream_gene_variant": 22,
    "non_coding_transcript_variant": 23,
    "intergenic_variant": 24,
}


# ----------------------------------------------------------------------
# Parse VEP Extra field
# ----------------------------------------------------------------------

def parse_extra(extra):

    result = {}

    if pd.isna(extra):
        return result

    for item in str(extra).split(";"):

        if "=" in item:

            key, value = item.split("=", 1)

            result[key] = value

        else:

            result[item] = True

    return result


# ----------------------------------------------------------------------
# Determine severity of a consequence string
# ----------------------------------------------------------------------

def consequence_rank(consequence):

    if pd.isna(consequence):
        return 999

    consequences = str(consequence).split(",")

    ranks = [
        CONSEQUENCE_SEVERITY.get(c, 999)
        for c in consequences
    ]

    return min(ranks)


def primary_consequence(consequence):

    if pd.isna(consequence):
        return None

    consequences = str(consequence).split(",")

    consequences = sorted(
        consequences,
        key=lambda x: CONSEQUENCE_SEVERITY.get(x, 999)
    )

    return consequences[0]


# ----------------------------------------------------------------------
# Read VEP output
#
# VEP output has metadata lines beginning with ##.
# The actual header begins with #Uploaded_variation.
# ----------------------------------------------------------------------

print("=" * 70)
print("VEP FEATURE EXTRACTION")
print("=" * 70)

print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print()


if not INPUT.exists():
    raise FileNotFoundError(
        f"VEP output not found: {INPUT}"
    )


# Find the actual header line

header = None

with open(INPUT, "r") as f:

    for line in f:

        if line.startswith("#Uploaded_variation"):

            header = line.rstrip("\n").split("\t")

            break


if header is None:
    raise RuntimeError(
        "Could not find VEP header."
    )


# Remove leading # from first column

header[0] = header[0].lstrip("#")


print("VEP columns detected:")
print(header)
print()


# ----------------------------------------------------------------------
# Read data
# ----------------------------------------------------------------------

df = pd.read_csv(
    INPUT,
    sep="\t",
    comment="#",
    names=header,
    dtype=str,
)


print(
    f"VEP annotation rows loaded: {len(df):,}"
)


# ----------------------------------------------------------------------
# Parse Extra fields
# ----------------------------------------------------------------------

extra_records = []

for extra in df["Extra"]:

    extra_records.append(
        parse_extra(extra)
    )


extra_df = pd.DataFrame(extra_records)


# ----------------------------------------------------------------------
# Add selected Extra fields
# ----------------------------------------------------------------------

for column in [
    "SYMBOL",
    "BIOTYPE",
    "CANONICAL",
    "MANE",
    "MANE_SELECT",
    "HGVSc",
    "HGVSp",
    "ENSP",
    "VARIANT_CLASS",
    "STRAND",
]:

    if column in extra_df.columns:

        df[column] = extra_df[column]

    else:

        df[column] = pd.NA


# ----------------------------------------------------------------------
# Transcript count
# ----------------------------------------------------------------------

transcript_counts = (
    df.groupby("Uploaded_variation")
    .size()
    .rename("transcript_count")
)

df = df.merge(
    transcript_counts,
    on="Uploaded_variation",
    how="left",
)


# ----------------------------------------------------------------------
# MANE / canonical flags
# ----------------------------------------------------------------------

df["has_mane"] = (
    df["MANE"]
    .fillna("")
    .eq("MANE_Select")
)

df["is_canonical"] = (
    df["CANONICAL"]
    .fillna("")
    .eq("YES")
)


# ----------------------------------------------------------------------
# Protein-coding flag
#
# BIOTYPE may not always be present depending on the VEP output.
# ----------------------------------------------------------------------

df["is_protein_coding"] = (
    df["BIOTYPE"]
    .fillna("")
    .eq("protein_coding")
)


# ----------------------------------------------------------------------
# Consequence severity
# ----------------------------------------------------------------------

df["consequence_rank"] = (
    df["Consequence"]
    .apply(consequence_rank)
)


df["primary_consequence"] = (
    df["Consequence"]
    .apply(primary_consequence)
)


# ----------------------------------------------------------------------
# Transcript selection
#
# Priority:
#
# 1. MANE Select
# 2. Canonical
# 3. Protein-coding
# 4. Most severe consequence
#
# The final sorting is deterministic.
# ----------------------------------------------------------------------

df["_mane_rank"] = (
    ~df["has_mane"]
).astype(int)

df["_canonical_rank"] = (
    ~df["is_canonical"]
).astype(int)

df["_protein_rank"] = (
    ~df["is_protein_coding"]
).astype(int)


df = df.sort_values(
    by=[
        "Uploaded_variation",
        "_mane_rank",
        "_canonical_rank",
        "_protein_rank",
        "consequence_rank",
        "Feature",
    ],
    ascending=[
        True,
        True,
        True,
        True,
        True,
        True,
    ],
)


# ----------------------------------------------------------------------
# Select one transcript per variant
# ----------------------------------------------------------------------

selected = (
    df
    .drop_duplicates(
        subset="Uploaded_variation",
        keep="first",
    )
    .copy()
)


# ----------------------------------------------------------------------
# Rename / extract features
# ----------------------------------------------------------------------

selected["VariationID"] = (
    selected["Uploaded_variation"]
    .str.replace(
        "ClinVar_",
        "",
        regex=False,
    )
)


features = pd.DataFrame({

    "VariationID":
        selected["VariationID"],

    "vep_gene":
        selected["SYMBOL"],

    "vep_gene_id":
        selected["Gene"],

    "vep_transcript":
        selected["Feature"],

    "vep_consequence":
        selected["Consequence"],

    "vep_primary_consequence":
        selected["primary_consequence"],

    "vep_impact":
        selected["Extra"].apply(
            lambda x:
            parse_extra(x).get("IMPACT")
        ),

    "vep_biotype":
        selected["BIOTYPE"],

    "vep_canonical":
        selected["is_canonical"].astype(int),

    "vep_mane":
        selected["has_mane"].astype(int),

    "vep_mane_select":
        selected["MANE_SELECT"],

    "vep_hgvsc":
        selected["HGVSc"],

    "vep_hgvsp":
        selected["HGVSp"],

    "vep_protein_id":
        selected["ENSP"],

    "vep_variant_class":
        selected["VARIANT_CLASS"],

    "vep_transcript_count":
        selected["transcript_count"],

    "vep_protein_position":
        selected["Protein_position"],

    "vep_amino_acids":
        selected["Amino_acids"],

    "vep_cds_position":
        selected["CDS_position"],

    "vep_cdna_position":
        selected["cDNA_position"],

    "vep_strand":
        selected["STRAND"],

})


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    f"Unique VEP variants:      "
    f"{df['Uploaded_variation'].nunique():,}"
)

print(
    f"Selected feature rows:    "
    f"{len(features):,}"
)

print(
    f"Duplicate VariationIDs:   "
    f"{features['VariationID'].duplicated().sum():,}"
)

print(
    f"Missing VariationIDs:     "
    f"{features['VariationID'].isna().sum():,}"
)


# ----------------------------------------------------------------------
# Consequence distribution after transcript selection
# ----------------------------------------------------------------------

print()
print("Selected primary consequences:")

print(
    features["vep_primary_consequence"]
    .value_counts()
    .head(30)
    .to_string()
)


# ----------------------------------------------------------------------
# MANE / canonical coverage
# ----------------------------------------------------------------------

print()
print("Transcript selection coverage:")

print(
    f"MANE Select: "
    f"{features['vep_mane'].sum():,} "
    f"({features['vep_mane'].mean() * 100:.2f}%)"
)

print(
    f"Canonical:   "
    f"{features['vep_canonical'].sum():,} "
    f"({features['vep_canonical'].mean() * 100:.2f}%)"
)


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

features.to_csv(
    OUTPUT,
    index=False,
)


print()
print("=" * 70)
print("VEP FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Output: {OUTPUT}"
)

print(
    f"Rows:   {len(features):,}"
)

print(
    f"Cols:   {len(features.columns):,}"
)