import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# GNOMAD STEPS A + B
# FINAL VALIDATION + NOT-FOUND INVESTIGATION
# ============================================================

INPUT_FILE = Path("Data/clinvar_gnomad.csv")
OUTPUT_REPORT = Path("Data/gnomad_validation_final.txt")
NOT_FOUND_CSV = Path("Data/gnomad_not_found_analysis.csv")

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

print("=" * 70)
print("LOADING GNOMAD OUTPUT")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"File : {INPUT_FILE}")
print(f"Rows : {len(df):,}")
print(f"Cols : {len(df.columns)}")


# ============================================================
# VARIANT TYPE CLASSIFICATION
# ============================================================

def classify_variant(row):
    """
    Classify variant using ClinVar's existing Variant type
    and available GRCh38 REF/ALT information.
    """

    # First use ClinVar's Variant type if available
    if "Variant type" in row.index:
        value = row["Variant type"]

        if pd.notna(value):
            value = str(value).strip().lower()

            if value:
                if "single nucleotide" in value or value == "snv":
                    return "SNP"

                if "deletion" in value:
                    return "Deletion"

                if "insertion" in value:
                    return "Insertion"

                if "indel" in value:
                    return "Indel"

                if "duplication" in value:
                    return "Duplication"

                if "microsatellite" in value:
                    return "Microsatellite"

                if "variation" in value:
                    return "Other"

    # --------------------------------------------------------
    # Fall back to GRCh38 coordinates
    # --------------------------------------------------------

    if (
        "GRCh38Chromosome" in row.index and
        "GRCh38Location" in row.index
    ):
        location = row["GRCh38Location"]

        if pd.notna(location):

            location = str(location)

            # Simple coordinate parsing
            if ":" in location:
                try:
                    coords = location.split(":")[-1]

                    if "-" in coords:
                        start, end = coords.split("-")[:2]

                        if start.isdigit() and end.isdigit():

                            if start == end:
                                return "SNP"

                            return "Indel"

                except Exception:
                    pass

    return "Unknown"


print("\nClassifying variants...")

df["_variant_class"] = df.apply(
    classify_variant,
    axis=1
)


# ============================================================
# GNOMAD STATUS
# ============================================================

if "gnomad_found" not in df.columns:
    raise ValueError("gnomad_found column is missing.")

if "gnomad_status" not in df.columns:
    raise ValueError("gnomad_status column is missing.")


# ============================================================
# REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as report:

    def section(title):
        report.write("\n")
        report.write("=" * 70 + "\n")
        report.write(title + "\n")
        report.write("=" * 70 + "\n")

    def write(text=""):
        print(text)
        report.write(str(text) + "\n")


    # ========================================================
    # 1. BASIC SUMMARY
    # ========================================================

    section("1. BASIC SUMMARY")

    total = len(df)

    found = (
        df["gnomad_found"]
        .eq(1)
        .sum()
    )

    not_found = (
        df["gnomad_found"]
        .eq(0)
        .sum()
    )

    write(f"Total variants : {total:,}")
    write(f"Found          : {found:,} ({found / total * 100:.2f}%)")
    write(f"Not found      : {not_found:,} ({not_found / total * 100:.2f}%)")


    # ========================================================
    # 2. STATUS BREAKDOWN
    # ========================================================

    section("2. GNOMAD STATUS BREAKDOWN")

    status_counts = (
        df["gnomad_status"]
        .value_counts(dropna=False)
    )

    for status, count in status_counts.items():

        pct = count / total * 100

        write(
            f"{str(status):30s} : "
            f"{count:8,} ({pct:6.2f}%)"
        )


    # ========================================================
    # 3. ACTUAL GNOMAD COLUMN CHECK
    # ========================================================

    section("3. GNOMAD FREQUENCY COLUMN CHECK")

    expected = [
        "gnomad_genome_ac",
        "gnomad_genome_an",
        "gnomad_genome_af",
        "gnomad_genome_hom_count",
        "gnomad_exome_ac",
        "gnomad_exome_an",
        "gnomad_exome_af",
        "gnomad_exome_hom_count"
    ]

    for col in expected:

        if col not in df.columns:

            write(f"{col:30s} : MISSING COLUMN")
            continue

        present = df[col].notna().sum()
        missing = df[col].isna().sum()

        write(
            f"{col:30s} : "
            f"present={present:8,} | "
            f"missing={missing:8,} "
            f"({missing / total * 100:6.2f}%)"
        )


    # ========================================================
    # 4. GENOME AF VALIDATION
    # ========================================================

    section("4. GENOME AF VALIDATION")

    required = [
        "gnomad_genome_ac",
        "gnomad_genome_an",
        "gnomad_genome_af"
    ]

    if all(col in df.columns for col in required):

        mask = (
            df["gnomad_genome_ac"].notna()
            &
            df["gnomad_genome_an"].notna()
            &
            df["gnomad_genome_af"].notna()
            &
            (df["gnomad_genome_an"] > 0)
        )

        subset = df.loc[mask].copy()

        calculated = (
            subset["gnomad_genome_ac"]
            /
            subset["gnomad_genome_an"]
        )

        reported = subset["gnomad_genome_af"]

        difference = (
            calculated - reported
        ).abs()

        tolerance = 1e-6

        mismatches = (
            difference > tolerance
        ).sum()

        write(f"Rows checked       : {len(subset):,}")
        write(f"AF mismatches      : {mismatches:,}")
        write(f"Maximum difference : {difference.max():.12f}")

        if mismatches == 0:
            write("STATUS: PASS")
        else:
            write("STATUS: WARNING")


    # ========================================================
    # 5. EXOME AF VALIDATION
    # ========================================================

    section("5. EXOME AF VALIDATION")

    required = [
        "gnomad_exome_ac",
        "gnomad_exome_an",
        "gnomad_exome_af"
    ]

    if all(col in df.columns for col in required):

        mask = (
            df["gnomad_exome_ac"].notna()
            &
            df["gnomad_exome_an"].notna()
            &
            df["gnomad_exome_af"].notna()
            &
            (df["gnomad_exome_an"] > 0)
        )

        subset = df.loc[mask].copy()

        calculated = (
            subset["gnomad_exome_ac"]
            /
            subset["gnomad_exome_an"]
        )

        reported = subset["gnomad_exome_af"]

        difference = (
            calculated - reported
        ).abs()

        tolerance = 1e-6

        mismatches = (
            difference > tolerance
        ).sum()

        write(f"Rows checked       : {len(subset):,}")
        write(f"AF mismatches      : {mismatches:,}")
        write(f"Maximum difference : {difference.max():.12f}")

        if mismatches == 0:
            write("STATUS: PASS")
        else:
            write("STATUS: WARNING")


    # ========================================================
    # 6. INVALID FREQUENCY VALUES
    # ========================================================

    section("6. INVALID FREQUENCY VALUES")

    af_columns = [
        "gnomad_genome_af",
        "gnomad_exome_af"
    ]

    for col in af_columns:

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        negative = (
            values < 0
        ).sum()

        greater_than_one = (
            values > 1
        ).sum()

        write(
            f"{col}: "
            f"negative={negative:,}, "
            f">1={greater_than_one:,}"
        )


    # ========================================================
    # 7. AC / AN VALIDATION
    # ========================================================

    section("7. AC / AN VALIDATION")

    for prefix in ["gnomad_genome", "gnomad_exome"]:

        ac = prefix + "_ac"
        an = prefix + "_an"

        if ac not in df.columns or an not in df.columns:
            continue

        subset = df[
            df[ac].notna()
            &
            df[an].notna()
        ]

        invalid = (
            subset[ac] > subset[an]
        ).sum()

        negative_ac = (
            subset[ac] < 0
        ).sum()

        negative_an = (
            subset[an] < 0
        ).sum()

        write(f"\n{prefix}:")
        write(f"  AC > AN      : {invalid:,}")
        write(f"  Negative AC  : {negative_ac:,}")
        write(f"  Negative AN  : {negative_an:,}")


    # ========================================================
    # 8. HOMOZYGOTE CONSISTENCY
    # ========================================================

    section("8. HOMOZYGOTE COUNT CONSISTENCY")

    checks = [
        (
            "gnomad_genome_ac",
            "gnomad_genome_hom_count",
            "Genome"
        ),
        (
            "gnomad_exome_ac",
            "gnomad_exome_hom_count",
            "Exome"
        )
    ]

    for ac_col, hom_col, label in checks:

        if ac_col not in df.columns or hom_col not in df.columns:
            continue

        subset = df[
            df[ac_col].notna()
            &
            df[hom_col].notna()
        ]

        violations = (
            subset[ac_col]
            <
            2 * subset[hom_col]
        ).sum()

        write(
            f"{label}: AC < 2 × homozygote count = "
            f"{violations:,}"
        )


    # ========================================================
    # 9. FOUND VARIANTS BY TYPE
    # ========================================================

    section("9. FOUND VARIANTS BY VARIANT TYPE")

    found_df = df[
        df["gnomad_found"] == 1
    ]

    found_types = (
        found_df["_variant_class"]
        .value_counts(dropna=False)
    )

    for variant_type, count in found_types.items():

        pct = count / len(found_df) * 100

        write(
            f"{str(variant_type):20s} : "
            f"{count:8,} ({pct:6.2f}%)"
        )


    # ========================================================
    # 10. NOT-FOUND BY VARIANT TYPE
    # ========================================================

    section("10. NOT-FOUND BY VARIANT TYPE")

    missing_df = df[
        df["gnomad_found"] == 0
    ]

    missing_types = (
        missing_df["_variant_class"]
        .value_counts(dropna=False)
    )

    for variant_type, count in missing_types.items():

        pct = count / len(missing_df) * 100

        write(
            f"{str(variant_type):20s} : "
            f"{count:8,} ({pct:6.2f}%)"
        )


    # ========================================================
    # 11. CROSS-TABULATION
    # ========================================================

    section("11. VARIANT TYPE × GNOMAD STATUS")

    cross = pd.crosstab(
        df["_variant_class"],
        df["gnomad_status"]
    )

    write(cross.to_string())


    # ========================================================
    # 12. NOT-FOUND RATE BY TYPE
    # ========================================================

    section("12. NOT-FOUND RATE BY VARIANT TYPE")

    type_summary = (
        df.groupby("_variant_class")
        .agg(
            total=("gnomad_found", "size"),
            found=("gnomad_found", lambda x: (x == 1).sum()),
            not_found=("gnomad_found", lambda x: (x == 0).sum())
        )
    )

    type_summary["found_pct"] = (
        type_summary["found"]
        /
        type_summary["total"]
        * 100
    )

    type_summary["not_found_pct"] = (
        type_summary["not_found"]
        /
        type_summary["total"]
        * 100
    )

    write(
        type_summary
        .round(2)
        .to_string()
    )


    # ========================================================
    # 13. NOT-FOUND EXAMPLES BY CATEGORY
    # ========================================================

    section("13. NOT-FOUND EXAMPLES")

    for variant_type in missing_types.index:

        subset = missing_df[
            missing_df["_variant_class"]
            ==
            variant_type
        ]

        write(f"\n--- {variant_type} ---")

        columns = [
            col for col in [
                "Name",
                "VariationID",
                "Canonical SPDI",
                "GRCh38Chromosome",
                "GRCh38Location",
                "Variant type",
                "variant_type_simple",
                "gnomad_variant_id",
                "gnomad_id_status",
                "gnomad_status"
            ]
            if col in df.columns
        ]

        write(
            subset[columns]
            .head(10)
            .to_string(index=False)
        )


    # ========================================================
    # 14. SUCCESSFUL ANNOTATION EXAMPLES
    # ========================================================

    section("14. SUCCESSFUL GNOMAD EXAMPLES")

    columns = [
        col for col in [
            "Name",
            "VariationID",
            "Canonical SPDI",
            "gnomad_variant_id",
            "gnomad_variant_id_returned",
            "gnomad_rsid",
            "gnomad_genome_ac",
            "gnomad_genome_an",
            "gnomad_genome_af",
            "gnomad_genome_hom_count",
            "gnomad_exome_ac",
            "gnomad_exome_an",
            "gnomad_exome_af",
            "gnomad_exome_hom_count"
        ]
        if col in df.columns
    ]

    write(
        found_df[columns]
        .head(10)
        .to_string(index=False)
    )


    # ========================================================
    # 15. FINAL VERDICT
    # ========================================================

    section("15. FINAL VERDICT")

    row_count_ok = len(df) == 117111

    id_mismatch_count = 0

    if (
        "gnomad_variant_id" in df.columns
        and
        "gnomad_variant_id_returned" in df.columns
    ):

        comparable = df[
            df["gnomad_variant_id"].notna()
            &
            df["gnomad_variant_id_returned"].notna()
        ]

        id_mismatch_count = (
            comparable["gnomad_variant_id"].astype(str)
            !=
            comparable["gnomad_variant_id_returned"].astype(str)
        ).sum()

    write(
        f"Row count correct       : "
        f"{'YES' if row_count_ok else 'NO'}"
    )

    write(
        f"Returned ID mismatches : "
        f"{id_mismatch_count:,}"
    )

    if row_count_ok and id_mismatch_count == 0:
        write("\nCORE GNOMAD INTEGRITY: PASS")
    else:
        write("\nCORE GNOMAD INTEGRITY: WARNING")

    write(
        "\nIMPORTANT:"
    )

    write(
        "A gnomAD 'not_found' result does NOT mean "
        "the variant has AF = 0."
    )

    write(
        "Missing gnomAD annotations should remain missing "
        "and must not be converted to zero."


    )


# ============================================================
# SAVE NOT-FOUND ANALYSIS CSV
# ============================================================

analysis_columns = [
    col for col in [
        "Name",
        "VariationID",
        "AlleleID(s)",
        "Canonical SPDI",
        "Gene(s)",
        "Protein change",
        "GRCh38Chromosome",
        "GRCh38Location",
        "Variant type",
        "variant_type_simple",
        "Molecular consequence",
        "label",
        "gnomad_variant_id",
        "gnomad_id_status",
        "gnomad_found",
        "gnomad_status"
    ]
    if col in df.columns
]

not_found_output = df[
    df["gnomad_found"] == 0
][analysis_columns].copy()

not_found_output.to_csv(
    NOT_FOUND_CSV,
    index=False
)


# ============================================================
# CLEAN UP
# ============================================================

df.drop(
    columns=["_variant_class"],
    inplace=True
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FINISHED")
print("=" * 70)

print(f"\nFull report:")
print(f"  {OUTPUT_REPORT}")

print(f"\nNot-found analysis:")
print(f"  {NOT_FOUND_CSV}")

print("\nSend me:")
print("  1. gnomad_validation_final.txt")
print("  2. gnomad_not_found_analysis.csv")