import pandas as pd
import re

INPUT = "Data/clinvar_gnomad.csv"

df = pd.read_csv(INPUT, low_memory=False)

# ------------------------------------------------------------
# Parse SPDI
# NC_000001.11:123:G:T
# NC_000001.11:123:ATG:
# NC_000001.11:123::ATG
# ------------------------------------------------------------

def parse_spdi(spdi):
    if pd.isna(spdi):
        return None, None, None, None

    parts = str(spdi).split(":")

    if len(parts) != 4:
        return None, None, None, None

    seq_id, pos, ref, alt = parts

    try:
        pos = int(pos)
    except:
        return seq_id, None, ref, alt

    return seq_id, pos, ref, alt


parsed = df["Canonical SPDI"].apply(parse_spdi)

df["spdi_seq"] = parsed.apply(lambda x: x[0])
df["spdi_pos"] = parsed.apply(lambda x: x[1])
df["spdi_ref"] = parsed.apply(lambda x: x[2])
df["spdi_alt"] = parsed.apply(lambda x: x[3])

df["ref_len"] = df["spdi_ref"].fillna("").str.len()
df["alt_len"] = df["spdi_alt"].fillna("").str.len()

df["size_change"] = (
    df["alt_len"] - df["ref_len"]
).abs()

# ------------------------------------------------------------
# Classify
# ------------------------------------------------------------

def classify(row):

    ref = row["ref_len"]
    alt = row["alt_len"]

    if ref == 0 and alt > 0:
        return "Insertion"

    if alt == 0 and ref > 0:
        return "Deletion"

    if ref == alt == 1:
        return "SNV"

    if ref != alt:
        return "Indel"

    return "Other"


df["spdi_type"] = df.apply(classify, axis=1)

# ------------------------------------------------------------
# Overall size distribution
# ------------------------------------------------------------

print("=" * 80)
print("SPDI VARIANT LENGTH ANALYSIS")
print("=" * 80)

print("\nOverall SPDI types:")
print(
    df["spdi_type"]
    .value_counts(dropna=False)
    .to_string()
)

# ------------------------------------------------------------
# Size bins
# ------------------------------------------------------------

bins = [
    -1,
    0,
    1,
    5,
    10,
    20,
    50,
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    100000,
    float("inf")
]

labels = [
    "0",
    "1",
    "2-5",
    "6-10",
    "11-20",
    "21-50",
    "51-100",
    "101-500",
    "501-1,000",
    "1,001-5,000",
    "5,001-10,000",
    "10,001-50,000",
    "50,001-100,000",
    ">100,000"
]

df["size_bin"] = pd.cut(
    df["size_change"],
    bins=bins,
    labels=labels
)

print("\n" + "=" * 80)
print("SIZE DISTRIBUTION BY VARIANT TYPE")
print("=" * 80)

table = pd.crosstab(
    df["variant_type_simple"],
    df["size_bin"]
)

print(table.to_string())

# ------------------------------------------------------------
# Detailed summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SIZE SUMMARY")
print("=" * 80)

summary = (
    df.groupby("variant_type_simple")
      ["size_change"]
      .agg(["count", "min", "median", "mean", "max"])
)

print(summary.to_string())

# ------------------------------------------------------------
# Non-SNV examples
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("LARGEST DELETIONS")
print("=" * 80)

cols = [
    "VariationID",
    "Canonical SPDI",
    "variant_type_simple",
    "ref_len",
    "alt_len",
    "size_change",
    "dbSNP ID",
    "gnomad_status"
]

print(
    df[df["variant_type_simple"] == "Deletion"]
    .sort_values("size_change", ascending=False)
    [cols]
    .head(30)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("LARGEST DUPLICATIONS")
print("=" * 80)

print(
    df[df["variant_type_simple"] == "Duplication"]
    .sort_values("size_change", ascending=False)
    [cols]
    .head(30)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("LARGEST INSERTIONS")
print("=" * 80)

print(
    df[df["variant_type_simple"] == "Insertion"]
    .sort_values("size_change", ascending=False)
    [cols]
    .head(30)
    .to_string(index=False)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT = "Data/clinvar_variant_lengths.csv"

df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)