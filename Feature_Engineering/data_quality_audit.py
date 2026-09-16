#!/usr/bin/env python3

"""
======================================================================
COMPREHENSIVE DATA-QUALITY AUDIT — VERSION 2
======================================================================

Input:
    Data/variant_features_engineered.csv

Outputs:
    Data/data_quality_audit.txt
    Data/duplicate_variants_summary.csv
    Data/conflicting_labels_summary.csv
    Data/invalid_values.csv
    Data/missingness_report.csv
    Data/feature_provenance.csv
    Data/feature_redundancy.csv
    Data/biological_consistency.csv
    Data/protein_position_anomalies.csv
    Data/distribution_summary.csv
    Data/exact_duplicate_summary.csv

IMPORTANT:
    This is an AUDIT.

    It does NOT automatically:
        - delete rows
        - delete features
        - impute values
        - scale values
        - change labels

    We inspect the results first and make cleaning decisions afterward.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

INPUT = Path("Data/variant_features_engineered.csv")
OUT = Path("Data_Share")

LABEL = "label"

OUT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# HELPERS
# ======================================================================

def pct(x):
    return round(float(x) * 100, 4)


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )

def save(df, filename):
    path = OUT / filename
    df.to_csv(path, index=False)
    return path


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 78)
    print("COMPREHENSIVE DATA-QUALITY AUDIT — VERSION 2")
    print("=" * 78)

    if not INPUT.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{INPUT}\n"
            "\nRun this script from your project root."
        )

    print(f"\nLoading: {INPUT}")

    df = pd.read_csv(
        INPUT,
        low_memory=False
    )

    n_rows, n_cols = df.shape

    print(f"Rows:    {n_rows:,}")
    print(f"Columns: {n_cols:,}")

    report = []

    report.append("=" * 78)
    report.append(
        "COMPREHENSIVE DATA-QUALITY AUDIT — VERSION 2"
    )
    report.append("=" * 78)
    report.append("")
    report.append(f"Input: {INPUT}")
    report.append(f"Rows: {n_rows:,}")
    report.append(f"Columns: {n_cols:,}")
    report.append("")


    # ==================================================================
    # 1. BASIC STRUCTURE
    # ==================================================================

    print("\n[1/11] Basic structure...")

    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    exact_duplicate_mask = df.duplicated(
        keep=False
    )

    exact_duplicate_rows = int(
        df.duplicated().sum()
    )

    unique_rows = int(
        (~exact_duplicate_mask).sum()
    )

    report.append("1. BASIC STRUCTURE")
    report.append("-" * 60)

    report.append(
        f"Duplicate column names: "
        f"{len(duplicate_columns)}"
    )

    report.append(
        f"Rows duplicated against an identical row: "
        f"{int(exact_duplicate_mask.sum()):,}"
    )

    report.append(
        f"Duplicate rows excluding first occurrence: "
        f"{exact_duplicate_rows:,}"
    )

    report.append(
        f"Rows unique across all columns: "
        f"{unique_rows:,}"
    )

    if duplicate_columns:

        report.append(
            f"Duplicate column names: "
            f"{duplicate_columns}"
        )

    report.append("")


    # ==================================================================
    # 2. LABEL QUALITY
    # ==================================================================

    print("[2/11] Label quality...")

    if LABEL not in df.columns:

        raise ValueError(
            f"Required target column '{LABEL}' not found."
        )

    label = numeric(
        df[LABEL]
    )

    invalid_label_mask = (
        label.isna()
        | ~label.isin([0, 1])
    )

    invalid_label_count = int(
        invalid_label_mask.sum()
    )

    label_counts = (
        label
        .value_counts(dropna=False)
        .sort_index()
    )

    report.append("2. LABEL QUALITY")
    report.append("-" * 60)

    report.append(
        f"Missing/invalid labels: "
        f"{invalid_label_count:,}"
    )

    report.append("Label distribution:")

    for value, count in label_counts.items():

        report.append(
            f"    {value}: {int(count):,}"
        )

    report.append("")


    # ==================================================================
    # 3. REAL IDENTIFIER DETECTION
    # ==================================================================

    print("[3/11] Checking real variant identifiers...")

    # IMPORTANT:
    #
    # Do NOT use substring matching such as:
    #     if "variant" in column_name
    #
    # because this incorrectly identifies:
    #     variant_in_domain
    #     variant_in_repeat
    #     variant_type_simple
    #
    # as variant IDs.
    #
    # Instead we use explicit names.

    exact_identifier_names = {

        "variant_id",
        "clinvar_id",
        "clinvar_variant_id",
        "variation_id",
        "variationid",
        "rsid",
        "rs_id",
        "allele_id",
        "accession",
        "variant_accession",
        "clinvar_accession",
        "vcv",
        "rcv"
    }

    identifier_columns = [
        col
        for col in df.columns
        if col.lower().strip()
        in exact_identifier_names
    ]

    report.append("3. VARIANT IDENTIFIERS")
    report.append("-" * 60)

    report.append(
        f"Identifier columns found: "
        f"{identifier_columns}"
    )

    if not identifier_columns:

        report.append(
            "No explicit variant identifier column "
            "was found in the engineered dataset."
        )

        report.append(
            "This means duplicate variant detection "
            "requires reconstructing a variant key "
            "from the available columns."
        )

    report.append("")


    # ==================================================================
    # 4. DUPLICATE / CONFLICTING IDENTIFIERS
    # ==================================================================

    print("[4/11] Duplicate and conflicting identifiers...")

    duplicate_summary_rows = []
    conflict_summary_rows = []

    for col in identifier_columns:

        s = df[col]

        valid = (
            s.notna()
            & s.astype(str)
            .str.strip()
            .ne("")
        )

        valid_count = int(
            valid.sum()
        )

        if valid_count == 0:
            continue

        duplicated = (
            valid
            & s.duplicated(
                keep=False
            )
        )

        duplicate_row_count = int(
            duplicated.sum()
        )

        duplicate_id_count = int(
            s.loc[duplicated]
            .nunique()
        )

        # --------------------------------------------------------------
        # Label conflicts
        # --------------------------------------------------------------

        tmp = pd.DataFrame({

            "identifier": s,

            "label": label
        })

        tmp = tmp[
            valid
        ]

        label_counts_by_id = (
            tmp.groupby(
                "identifier"
            )["label"]
            .nunique(
                dropna=True
            )
        )

        conflicting_ids = (
            label_counts_by_id[
                label_counts_by_id > 1
            ]
        )

        duplicate_summary_rows.append({

            "identifier_column": col,

            "valid_identifier_rows":
                valid_count,

            "duplicate_identifier_values":
                duplicate_id_count,

            "rows_with_duplicate_identifier":
                duplicate_row_count,

            "duplicate_row_pct":
                pct(
                    duplicate_row_count
                    / n_rows
                )
        })

        conflict_summary_rows.append({

            "identifier_column": col,

            "conflicting_identifier_values":
                len(conflicting_ids),

            "rows_in_conflicting_groups":
                int(
                    tmp[
                        tmp["identifier"]
                        .isin(
                            conflicting_ids.index
                        )
                    ].shape[0]
                )
        })


    duplicate_summary = pd.DataFrame(
        duplicate_summary_rows
    )

    conflicting_summary = pd.DataFrame(
        conflict_summary_rows
    )

    save(
        duplicate_summary,
        "duplicate_variants_summary.csv"
    )

    save(
        conflicting_summary,
        "conflicting_labels_summary.csv"
    )

    report.append("4. DUPLICATE / CONFLICTING IDENTIFIERS")
    report.append("-" * 60)

    if duplicate_summary.empty:

        report.append(
            "No explicit identifier columns available."
        )

    else:

        for _, row in duplicate_summary.iterrows():

            report.append(
                f"{row['identifier_column']}: "
                f"{int(row['duplicate_identifier_values'])} "
                f"duplicate IDs | "
                f"{int(row['rows_with_duplicate_identifier'])} rows"
            )

    report.append("")


    # ==================================================================
    # 5. NUMERICAL VALIDITY
    # ==================================================================

    print("[5/11] Numerical validity...")

    numeric_cols = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    invalid_records = []

    for col in numeric_cols:

        s = numeric(
            df[col]
        )

        # --------------------------------------------------------------
        # Infinite values
        # --------------------------------------------------------------

        inf_mask = df[col].isin([
            np.inf,
            -np.inf
        ])

        if inf_mask.any():

            invalid_records.append(
                pd.DataFrame({

                    "feature": col,

                    "row_index":
                        df.index[inf_mask],

                    "issue":
                        "infinite_value",

                    "value":
                        df.loc[
                            inf_mask,
                            col
                        ].values
                })
            )

        # --------------------------------------------------------------
        # AF range
        # --------------------------------------------------------------

        if col.lower().endswith("_af"):

            bad = (
                s.notna()
                & ~s.between(0, 1)
            )

            # IMPORTANT:
            # log_*_af is NOT caught here because it does
            # not end with "_af".

            if bad.any():

                invalid_records.append(
                    pd.DataFrame({

                        "feature": col,

                        "row_index":
                            df.index[bad],

                        "issue":
                            "allele_frequency_outside_0_1",

                        "value":
                            s[bad].values
                    })
                )

        # --------------------------------------------------------------
        # Negative positions / lengths
        # --------------------------------------------------------------

        if any(
            keyword in col.lower()
            for keyword in [
                "position",
                "length"
            ]
        ):

            bad = (
                s.notna()
                & (s < 0)
            )

            if bad.any():

                invalid_records.append(
                    pd.DataFrame({

                        "feature": col,

                        "row_index":
                            df.index[bad],

                        "issue":
                            "negative_position_or_length",

                        "value":
                            s[bad].values
                    })
                )

        # --------------------------------------------------------------
        # Negative counts
        # --------------------------------------------------------------

        if (
            col.lower().startswith("n_")
            or "count" in col.lower()
        ):

            bad = (
                s.notna()
                & (s < 0)
            )

            if bad.any():

                invalid_records.append(
                    pd.DataFrame({

                        "feature": col,

                        "row_index":
                            df.index[bad],

                        "issue":
                            "negative_count",

                        "value":
                            s[bad].values
                    })
                )

        # --------------------------------------------------------------
        # Negative distances
        # --------------------------------------------------------------

        if "distance" in col.lower():

            bad = (
                s.notna()
                & (s < 0)
            )

            if bad.any():

                invalid_records.append(
                    pd.DataFrame({

                        "feature": col,

                        "row_index":
                            df.index[bad],

                        "issue":
                            "negative_distance",

                        "value":
                            s[bad].values
                    })
                )

    # ==================================================================
    # AF / AC / AN CONSISTENCY
    # ==================================================================

    for af_col in [
        c
        for c in df.columns
        if c.lower().endswith("_af")
    ]:

        prefix = af_col[:-3]

        ac_col = prefix + "_ac"
        an_col = prefix + "_an"

        if (
            ac_col not in df.columns
            or an_col not in df.columns
        ):
            continue

        af = numeric(
            df[af_col]
        )

        ac = numeric(
            df[ac_col]
        )

        an = numeric(
            df[an_col]
        )

        # AC <= AN

        bad_ac_an = (
            ac.notna()
            & an.notna()
            & (ac > an)
        )

        if bad_ac_an.any():

            invalid_records.append(
                pd.DataFrame({

                    "feature": ac_col,

                    "row_index":
                        df.index[bad_ac_an],

                    "issue":
                        "AC_greater_than_AN",

                    "value":
                        ac[bad_ac_an].values
                })
            )

        # AF ~= AC / AN

        valid = (
            af.notna()
            & ac.notna()
            & an.notna()
            & (an > 0)
        )

        expected_af = (
            ac / an
        )

        bad_af = (
            valid
            & ~np.isclose(
                af,
                expected_af,
                rtol=1e-3,
                atol=1e-8
            )
        )

        if bad_af.any():

            invalid_records.append(
                pd.DataFrame({

                    "feature": af_col,

                    "row_index":
                        df.index[bad_af],

                    "issue":
                        "AF_inconsistent_with_AC_div_AN",

                    "value":
                        af[bad_af].values
                })
            )


    # ------------------------------------------------------------------
    # Save numerical issues
    # ------------------------------------------------------------------

    if invalid_records:

        invalid_values = pd.concat(
            invalid_records,
            ignore_index=True
        )

    else:

        invalid_values = pd.DataFrame(
            columns=[
                "feature",
                "row_index",
                "issue",
                "value"
            ]
        )

    save(
        invalid_values,
        "invalid_values.csv"
    )

    report.append("5. NUMERICAL VALIDITY")
    report.append("-" * 60)

    report.append(
        f"Numeric columns: "
        f"{len(numeric_cols)}"
    )

    report.append(
        f"Invalid-value records: "
        f"{len(invalid_values):,}"
    )

    report.append("")


    # ==================================================================
    # 6. SPECIAL CHECK: LOG GNOMAD FEATURES
    # ==================================================================

    print("[6/11] Checking log-transformed gnomAD features...")

    log_af_pairs = [

        (
            "gnomad_genome_af",
            "log_gnomad_genome_af"
        ),

        (
            "gnomad_exome_af",
            "log_gnomad_exome_af"
        ),

        (
            "max_gnomad_af",
            "log_max_gnomad_af"
        )
    ]

    log_rows = []

    for af_col, log_col in log_af_pairs:

        if (
            af_col not in df.columns
            or log_col not in df.columns
        ):
            continue

        af = numeric(
            df[af_col]
        )

        log_af = numeric(
            df[log_col]
        )

        # Expected transformation assuming log1p(AF)
        expected = np.log1p(
            af.clip(lower=0)
        )

        valid = (
            af.notna()
            & log_af.notna()
        )

        # Exact/near exact agreement
        matches = (
            valid
            & np.isclose(
                log_af,
                expected,
                rtol=1e-5,
                atol=1e-8
            )
        )

        mismatches = (
            valid
            & ~matches
        )

        negative_log = (
            log_af.notna()
            & (log_af < 0)
        )

        log_rows.append({

            "af_feature":
                af_col,

            "log_feature":
                log_col,

            "af_missing":
                int(af.isna().sum()),

            "log_missing":
                int(log_af.isna().sum()),

            "negative_log_values":
                int(negative_log.sum()),

            "valid_pairs":
                int(valid.sum()),

            "matching_log1p_values":
                int(matches.sum()),

            "mismatching_values":
                int(mismatches.sum()),

            "match_pct":
                pct(
                    matches.sum()
                    / valid.sum()
                )
                if valid.sum()
                else 0
        })

    log_audit = pd.DataFrame(
        log_rows
    )

    save(
        log_audit,
        "gnomad_log_transform_audit.csv"
    )

    report.append("6. gnomAD LOG TRANSFORM")
    report.append("-" * 60)

    for _, row in log_audit.iterrows():

        report.append(
            f"{row['log_feature']}: "
            f"{int(row['negative_log_values'])} "
            f"negative values | "
            f"{row['match_pct']:.2f}% match log1p(AF)"
        )

    report.append("")


    # ==================================================================
    # 7. MISSINGNESS
    # ==================================================================

    print("[7/11] Missingness...")

    missing_rows = []

    for col in df.columns:

        missing = df[col].isna()

        row = {

            "feature":
                col,

            "dtype":
                str(df[col].dtype),

            "missing_count":
                int(missing.sum()),

            "missing_pct":
                pct(missing.mean()),

            "nonmissing_count":
                int((~missing).sum()),

            "unique_nonmissing":
                int(
                    df[col]
                    .nunique(
                        dropna=True
                    )
                )
        }

        if (label == 0).any():

            row[
                "label0_missing_pct"
            ] = pct(
                df.loc[
                    label == 0,
                    col
                ].isna().mean()
            )

        else:

            row[
                "label0_missing_pct"
            ] = np.nan

        if (label == 1).any():

            row[
                "label1_missing_pct"
            ] = pct(
                df.loc[
                    label == 1,
                    col
                ].isna().mean()
            )

        else:

            row[
                "label1_missing_pct"
            ] = np.nan

        missing_rows.append(row)

    missingness = pd.DataFrame(
        missing_rows
    )

    missingness[
        "label_missingness_gap"
    ] = (
        missingness[
            "label1_missing_pct"
        ]
        -
        missingness[
            "label0_missing_pct"
        ]
    )

    missingness = missingness.sort_values(
        "missing_pct",
        ascending=False
    )

    save(
        missingness,
        "missingness_report.csv"
    )

    report.append("7. MISSINGNESS")
    report.append("-" * 60)

    for _, row in missingness.head(20).iterrows():

        report.append(
            f"{row['feature']}: "
            f"{row['missing_pct']:.2f}% missing | "
            f"label0={row['label0_missing_pct']:.2f}% | "
            f"label1={row['label1_missing_pct']:.2f}%"
        )

    report.append("")


    # ==================================================================
    # 8. DISTRIBUTIONS
    # ==================================================================

    print("[8/11] Numerical distributions...")

    distribution_rows = []

    for col in numeric_cols:

        s = numeric(
            df[col]
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(s) == 0:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if s.nunique() > 1:

            skew = s.skew()

        else:

            skew = 0

        distribution_rows.append({

            "feature":
                col,

            "count":
                len(s),

            "missing_pct":
                pct(df[col].isna().mean()),

            "min":
                s.min(),

            "q01":
                s.quantile(0.01),

            "q25":
                q1,

            "median":
                s.median(),

            "q75":
                q3,

            "q99":
                s.quantile(0.99),

            "max":
                s.max(),

            "mean":
                s.mean(),

            "std":
                s.std(),

            "skew":
                skew,

            "zero_pct":
                pct(
                    (s == 0).mean()
                ),

            "negative_pct":
                pct(
                    (s < 0).mean()
                ),

            "iqr_outlier_pct":
                pct(
                    (
                        (s < lower)
                        | (s > upper)
                    ).mean()
                )
        })

    distribution_summary = pd.DataFrame(
        distribution_rows
    )

    save(
        distribution_summary,
        "distribution_summary.csv"
    )

    report.append("8. DISTRIBUTIONS")
    report.append("-" * 60)

    highly_skewed = (
        distribution_summary[
            distribution_summary["skew"].abs() > 5
        ]
        if not distribution_summary.empty
        else pd.DataFrame()
    )

    zero_heavy = (
        distribution_summary[
            distribution_summary["zero_pct"] > 95
        ]
        if not distribution_summary.empty
        else pd.DataFrame()
    )

    report.append(
        f"Highly skewed features "
        f"(|skew| > 5): "
        f"{len(highly_skewed)}"
    )

    report.append(
        f"Zero-heavy features "
        f"(>95% zeros): "
        f"{len(zero_heavy)}"
    )

    report.append("")


    # ==================================================================
    # 9. CATEGORICAL QUALITY
    # ==================================================================

    print("[9/11] Categorical quality...")

    categorical_cols = (
        df.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    categorical_rows = []

    for col in categorical_cols:

        s = (
            df[col]
            .dropna()
            .astype(str)
        )

        stripped = (
            s.str.strip()
        )

        normalized = (
            stripped.str.lower()
        )

        raw_unique = (
            s.nunique()
        )

        normalized_unique = (
            normalized.nunique()
        )

        categorical_rows.append({

            "feature":
                col,

            "raw_unique":
                raw_unique,

            "normalized_unique":
                normalized_unique,

            "case_whitespace_variants":
                raw_unique
                - normalized_unique,

            "missing_pct":
                pct(
                    df[col].isna().mean()
                ),

            "top_values":
                " | ".join(
                    map(
                        str,
                        s.value_counts()
                        .head(15)
                        .index
                        .tolist()
                    )
                )
        })

    categorical_quality = pd.DataFrame(
        categorical_rows
    )

    save(
        categorical_quality,
        "categorical_quality.csv"
    )

    report.append("9. CATEGORICAL FEATURES")
    report.append("-" * 60)

    report.append(
        f"Categorical columns: "
        f"{len(categorical_cols)}"
    )

    for _, row in categorical_quality.iterrows():

        report.append(
            f"{row['feature']}: "
            f"{int(row['raw_unique'])} raw unique, "
            f"{int(row['normalized_unique'])} normalized unique"
        )

    report.append("")


    # ==================================================================
    # 10. FEATURE REDUNDANCY
    # ==================================================================

    print("[10/11] Feature redundancy...")

    redundancy_rows = []

    if len(numeric_cols) > 1:

        numeric_df = df[
            numeric_cols
        ].copy()

        pearson = numeric_df.corr(
            method="pearson"
        )

        spearman = numeric_df.corr(
            method="spearman"
        )

        seen = set()

        for col1 in pearson.columns:

            for col2 in pearson.columns:

                if col1 == col2:
                    continue

                pair = tuple(
                    sorted(
                        [
                            col1,
                            col2
                        ]
                    )
                )

                if pair in seen:
                    continue

                seen.add(pair)

                p = pearson.loc[
                    col1,
                    col2
                ]

                s = spearman.loc[
                    col1,
                    col2
                ]

                if (
                    abs(p) >= 0.90
                    or abs(s) >= 0.90
                ):

                    redundancy_rows.append({

                        "feature_1":
                            col1,

                        "feature_2":
                            col2,

                        "pearson":
                            p,

                        "abs_pearson":
                            abs(p),

                        "spearman":
                            s,

                        "abs_spearman":
                            abs(s),

                        "flag":
                            (
                                "VERY_HIGH"
                                if (
                                    abs(p) >= 0.95
                                    or abs(s) >= 0.95
                                )
                                else "HIGH"
                            )
                    })

    redundancy = pd.DataFrame(
        redundancy_rows
    )

    if not redundancy.empty:

        redundancy = redundancy.sort_values(
            [
                "abs_pearson",
                "abs_spearman"
            ],
            ascending=False
        )

    save(
        redundancy,
        "feature_redundancy.csv"
    )

    report.append("10. FEATURE REDUNDANCY")
    report.append("-" * 60)

    report.append(
        f"Highly correlated numeric pairs "
        f"(>=0.90): "
        f"{len(redundancy):,}"
    )

    for _, row in redundancy.head(30).iterrows():

        report.append(
            f"{row['feature_1']} <-> "
            f"{row['feature_2']} | "
            f"Pearson={row['pearson']:.4f} | "
            f"Spearman={row['spearman']:.4f}"
        )

    report.append("")


    # ==================================================================
    # 11. BIOLOGICAL CONSISTENCY
    # ==================================================================

    print("[11/11] Biological consistency...")

    consistency_rows = []

    def add_check(
        name,
        mask,
        description
    ):

        count = int(mask.sum())

        consistency_rows.append({

            "check":
                name,

            "description":
                description,

            "violations":
                count,

            "violation_pct":
                pct(
                    count / n_rows
                )
                if n_rows
                else 0
        })


    # ------------------------------------------------------------------
    # Missense vs AA change
    # ------------------------------------------------------------------

    if {
        "is_missense",
        "is_amino_acid_change"
    }.issubset(df.columns):

        missense = numeric(
            df["is_missense"]
        )

        aa_change = numeric(
            df["is_amino_acid_change"]
        )

        disagreement = (
            missense.notna()
            & aa_change.notna()
            & (missense != aa_change)
        )

        add_check(
            "missense_vs_amino_acid_change",
            disagreement,
            "is_missense differs from is_amino_acid_change"
        )


    # ------------------------------------------------------------------
    # Variant annotation requires protein annotation
    # ------------------------------------------------------------------

    feature_pairs = [

        (
            "variant_in_domain",
            "protein_has_domain"
        ),

        (
            "variant_in_active_site",
            "protein_has_active_site"
        ),

        (
            "variant_in_binding_site",
            "protein_has_binding_site"
        ),

        (
            "variant_in_transmembrane",
            "protein_has_transmembrane"
        ),

        (
            "variant_in_signal_peptide",
            "protein_has_signal_peptide"
        ),

        (
            "variant_in_dna_binding",
            "protein_has_dna_binding"
        ),

        (
            "variant_in_disulfide_bond",
            "protein_has_disulfide_bond"
        )
    ]

    for variant_col, protein_col in feature_pairs:

        if {
            variant_col,
            protein_col
        }.issubset(df.columns):

            variant_value = numeric(
                df[variant_col]
            )

            protein_value = numeric(
                df[protein_col]
            )

            violation = (
                variant_value.eq(1)
                & protein_value.eq(0)
            )

            add_check(
                f"{variant_col}_requires_{protein_col}",
                violation,
                f"{variant_col}=1 while "
                f"{protein_col}=0"
            )


    # ------------------------------------------------------------------
    # Count vs presence
    # ------------------------------------------------------------------

    count_presence_pairs = [

        (
            "n_domain",
            "protein_has_domain"
        ),

        (
            "n_dna_binding",
            "protein_has_dna_binding"
        ),

        (
            "n_active_site",
            "protein_has_active_site"
        ),

        (
            "n_binding_site",
            "protein_has_binding_site"
        ),

        (
            "n_transmembrane",
            "protein_has_transmembrane"
        ),

        (
            "n_signal_peptide",
            "protein_has_signal_peptide"
        ),

        (
            "n_disulfide_bond",
            "protein_has_disulfide_bond"
        )
    ]

    for count_col, presence_col in count_presence_pairs:

        if {
            count_col,
            presence_col
        }.issubset(df.columns):

            count = numeric(
                df[count_col]
            )

            presence = numeric(
                df[presence_col]
            )

            expected = (
                count > 0
            ).astype(int)

            violation = (
                count.notna()
                & presence.notna()
                & (expected != presence)
            )

            add_check(
                f"{count_col}_vs_{presence_col}",
                violation,
                f"{presence_col} should equal "
                f"1 when {count_col} > 0"
            )


    # ------------------------------------------------------------------
    # Protein position <= protein length
    # ------------------------------------------------------------------

    position_anomalies = pd.DataFrame()

    if {
        "protein_position",
        "protein_length"
    }.issubset(df.columns):

        position = pd.to_numeric(
            df["protein_position"],
            errors="coerce"
        )

        length = pd.to_numeric(
            df["protein_length"],
            errors="coerce"
        )

        violation = (
            position.notna()
            & length.notna()
            & (length > 0)
            & (position > length)
        )

        add_check(
            "protein_position_exceeds_length",
            violation,
            "protein_position > protein_length"
        )

        if violation.any():

            cols_to_save = [
                "protein_position",
                "protein_length",
                "relative_protein_position",
                "label",
                "variant_type_simple",
                "vep_impact",
                "vep_primary_consequence"
            ]

            cols_to_save = [
                c for c in cols_to_save
                if c in df.columns
            ]

            position_anomalies = (
                df.loc[
                    violation,
                    cols_to_save
                ]
                .copy()
            )

            position_anomalies[
                "position_minus_length"
            ] = (
                pd.to_numeric(
                    position_anomalies[
                        "protein_position"
                    ],
                    errors="coerce"
                )
                -
                pd.to_numeric(
                    position_anomalies[
                        "protein_length"
                    ],
                    errors="coerce"
                )
            )

    save(
        position_anomalies,
        "protein_position_anomalies.csv"
    )


    # ------------------------------------------------------------------
    # Relative position 0..1
    # ------------------------------------------------------------------

    if "relative_protein_position" in df.columns:

        rel = numeric(
            df[
                "relative_protein_position"
            ]
        )

        violation = (
            rel.notna()
            & ~rel.between(
                0,
                1
            )
        )

        add_check(
            "relative_position_outside_0_1",
            violation,
            "relative_protein_position outside [0,1]"
        )


    # ------------------------------------------------------------------
    # Log AF consistency
    # ------------------------------------------------------------------

    for af_col, log_col in log_af_pairs:

        if {
            af_col,
            log_col
        }.issubset(df.columns):

            af = numeric(
                df[af_col]
            )

            log_af = numeric(
                df[log_col]
            )

            valid = (
                af.notna()
                & log_af.notna()
            )

            expected = np.log1p(
                af.clip(
                    lower=0
                )
            )

            violation = (
                valid
                & ~np.isclose(
                    log_af,
                    expected,
                    rtol=1e-5,
                    atol=1e-8
                )
            )

            add_check(
                f"{log_col}_vs_log1p_source_AF",
                violation,
                f"{log_col} does not match log1p({af_col})"
            )


    biological_consistency = pd.DataFrame(
        consistency_rows
    )

    save(
        biological_consistency,
        "biological_consistency.csv"
    )

    report.append("11. BIOLOGICAL CONSISTENCY")
    report.append("-" * 60)

    for _, row in biological_consistency.iterrows():

        report.append(
            f"{row['check']}: "
            f"{int(row['violations']):,} violations "
            f"({row['violation_pct']:.4f}%)"
        )

    report.append("")


    # ==================================================================
    # FEATURE PROVENANCE
    # ==================================================================

    print("[+] Building feature provenance...")

    provenance_rows = []

    for col in df.columns:

        name = col.lower()

        if col == LABEL:

            category = "TARGET"
            source = "ClinVar-derived"
            meaning = (
                "Binary pathogenicity target: "
                "0=benign, 1=pathogenic"
            )

        elif name.startswith("variant_in_"):

            category = "VARIANT_LEVEL"
            source = "UniProt / engineered"
            meaning = (
                "Whether variant lies within "
                "a protein annotation"
            )

        elif name.startswith("distance_to_nearest_"):

            category = "VARIANT_LEVEL"
            source = "UniProt / engineered"
            meaning = (
                "Distance from variant position "
                "to nearest annotated feature"
            )

        elif name.startswith("is_"):

            category = "VARIANT_LEVEL"
            source = "VEP / engineered"
            meaning = (
                "Binary consequence/location indicator"
            )

        elif name.startswith("n_"):

            category = "PROTEIN_LEVEL"
            source = "UniProt / engineered"
            meaning = (
                "Number of annotated protein features"
            )

        elif name.startswith("protein_has_"):

            category = "PROTEIN_LEVEL"
            source = "UniProt / engineered"
            meaning = (
                "Whether protein contains "
                "a particular annotation"
            )

        elif "gnomad" in name:

            category = "POPULATION_LEVEL"
            source = "gnomAD"
            meaning = (
                "Population frequency/count information"
            )

        elif (
            "protein_position" in name
            or "protein_length" in name
            or "relative_protein_position" in name
            or name.endswith("_aa")
            or "hydrophobicity" in name
            or "polarity" in name
            or "molecular_weight" in name
            or "charge" in name
        ):

            category = "PROTEIN_VARIANT_LEVEL"
            source = "VEP / UniProt / engineered"
            meaning = (
                "Protein or amino-acid-level "
                "variant information"
            )

        elif name.startswith("vep_"):

            category = "VARIANT_LEVEL"
            source = "VEP"
            meaning = (
                "Variant Effect Predictor annotation"
            )

        elif (
            df[col].dtype == "object"
            or str(df[col].dtype) == "category"
        ):

            category = "CATEGORICAL"
            source = "Mixed / engineered"
            meaning = (
                "Categorical biological annotation"
            )

        else:

            category = "OTHER"
            source = "Unknown"
            meaning = "Requires manual review"

        provenance_rows.append({

            "feature":
                col,

            "category":
                category,

            "source":
                source,

            "meaning":
                meaning,

            "dtype":
                str(df[col].dtype),

            "missing_pct":
                pct(
                    df[col].isna().mean()
                ),

            "unique_values":
                int(
                    df[col].nunique(
                        dropna=True
                    )
                )
        })

    feature_provenance = pd.DataFrame(
        provenance_rows
    )

    save(
        feature_provenance,
        "feature_provenance.csv"
    )


    # ==================================================================
    # EXACT DUPLICATE SUMMARY
    # ==================================================================

    duplicate_groups = (
        df.groupby(
            list(df.columns),
            dropna=False,
            sort=False
        )
        .size()
        .reset_index(
            name="duplicate_count"
        )
    )

    duplicate_groups = duplicate_groups[
        duplicate_groups["duplicate_count"] > 1
    ]

    exact_duplicate_summary = pd.DataFrame({

        "metric": [
            "total_rows",
            "unique_exact_rows",
            "duplicate_groups",
            "rows_in_duplicate_groups",
            "extra_duplicate_rows"
        ],

        "value": [
            n_rows,
            len(duplicate_groups)
            + unique_rows,
            len(duplicate_groups),
            int(
                exact_duplicate_mask.sum()
            ),
            exact_duplicate_rows
        ]
    })

    save(
        exact_duplicate_summary,
        "exact_duplicate_summary.csv"
    )


    # ==================================================================
    # FINAL REPORT
    # ==================================================================

    report.append("=" * 78)
    report.append("AUDIT SUMMARY")
    report.append("=" * 78)

    report.append(
        f"Rows: {n_rows:,}"
    )

    report.append(
        f"Columns: {n_cols:,}"
    )

    report.append(
        f"Exact duplicate rows: "
        f"{exact_duplicate_rows:,}"
    )

    report.append(
        f"Invalid labels: "
        f"{invalid_label_count:,}"
    )

    report.append(
        f"Invalid numerical records: "
        f"{len(invalid_values):,}"
    )

    report.append(
        f"Highly correlated feature pairs: "
        f"{len(redundancy):,}"
    )

    report.append("")

    report.append(
        "IMPORTANT:"
    )

    report.append(
        "No rows or features were automatically removed."
    )

    report.append(
        "No imputation, scaling, encoding or feature "
        "selection was performed."
    )

    report.append("")

    report.append(
        "NEXT CLEANING DECISION:"
    )

    report.append(
        "Review the generated audit files and then decide:"
    )

    report.append(
        "  1. Which rows are genuinely duplicated."
    )

    report.append(
        "  2. Whether any variant labels conflict."
    )

    report.append(
        "  3. Whether invalid numerical values are "
        "real errors or sentinel values."
    )

    report.append(
        "  4. Whether missing annotations have biological meaning."
    )

    report.append(
        "  5. Which redundant features should be removed."
    )

    report.append(
        "  6. Which highly skewed features need transformation."
    )

    report.append(
        "  7. Which biological consistency issues need correction."
    )

    report.append(
        "  8. Only THEN perform train/validation/test splitting."
    )

    report.append(
        "  9. Learned preprocessing must be fit on TRAIN ONLY."
    )

    report.append("")


    report_path = (
        OUT /
        "data_quality_audit.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )


    # ==================================================================
    # OUTPUT LIST
    # ==================================================================

    print("\n" + "=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    output_files = [

        "data_quality_audit.txt",

        "duplicate_variants_summary.csv",

        "conflicting_labels_summary.csv",

        "invalid_values.csv",

        "gnomad_log_transform_audit.csv",

        "missingness_report.csv",

        "categorical_quality.csv",

        "feature_provenance.csv",

        "feature_redundancy.csv",

        "biological_consistency.csv",

        "protein_position_anomalies.csv",

        "distribution_summary.csv",

        "exact_duplicate_summary.csv"
    ]

    print("\nGenerated:")

    for filename in output_files:

        path = OUT / filename

        if path.exists():

            size_kb = (
                path.stat().st_size
                / 1024
            )

            print(
                f"  OK  {filename:<40} "
                f"{size_kb:,.1f} KB"
            )

    print("\nIMPORTANT:")
    print(
        "Do NOT delete anything yet."
    )

    print(
        "Send the new data_quality_audit.txt "
        "and the small CSV outputs."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()