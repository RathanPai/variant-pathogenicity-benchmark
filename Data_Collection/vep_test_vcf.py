import gzip
import pandas as pd
from pathlib import Path

INPUT = Path("Data/clinvar_gnomad.csv")
OUTPUT = Path("Data/vep_test_all_types.vcf")

FASTA = Path(
    "~/.vep/homo_sapiens/116_GRCh38/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
).expanduser()

TARGETS = {
    "single nucleotide variant": 10,
    "Deletion": 5,
    "Duplication": 5,
    "Insertion": 5,
    "Indel": 5,
}


# ----------------------------------------------------------------------
# LOAD GRCh38 FASTA
# ----------------------------------------------------------------------

def load_fasta():

    print("=" * 70)
    print("LOADING GRCh38 REFERENCE")
    print("=" * 70)

    if not FASTA.exists():
        raise FileNotFoundError(f"FASTA not found: {FASTA}")

    sequences = {}

    current_chrom = None
    chunks = []

    with gzip.open(FASTA, "rt") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                # Save previous chromosome
                if current_chrom is not None:
                    sequences[current_chrom] = "".join(chunks).upper()

                header = line[1:].split()

                # Ensembl GRCh38 FASTA uses headers such as:
                #
                # >1 dna:chromosome chromosome:GRCh38:1:1:248956422:1
                #
                # The first token is the chromosome name.
                current_chrom = header[0]

                chunks = []

            else:

                if current_chrom is not None:
                    chunks.append(line)

        # Save final chromosome
        if current_chrom is not None:
            sequences[current_chrom] = "".join(chunks).upper()

    print(f"Chromosomes/contigs loaded: {len(sequences)}")
    print(
        "Chromosome 1 available:",
        "1" in sequences
    )

    print()

    return sequences


def get_base(sequences, chrom, position):

    """
    Get reference base at a 1-based genomic position.
    """

    if chrom not in sequences:
        raise ValueError(
            f"Chromosome {chrom} not found in FASTA"
        )

    seq = sequences[chrom]

    if position < 1 or position > len(seq):
        raise ValueError(
            f"Position {position} outside chromosome {chrom}"
        )

    return seq[position - 1]


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

df = pd.read_csv(INPUT)

selected = []

for variant_type, n in TARGETS.items():

    subset = df[df["Variant type"] == variant_type].copy()

    if len(subset) < n:
        raise RuntimeError(
            f"Only {len(subset)} variants available for "
            f"'{variant_type}', but {n} required."
        )

    # Reproducible selection
    selected.append(subset.head(n))


test_df = pd.concat(selected, ignore_index=True)


# ----------------------------------------------------------------------
# LOAD REFERENCE
# ----------------------------------------------------------------------

reference = load_fasta()


# ----------------------------------------------------------------------
# SPDI → VCF
# ----------------------------------------------------------------------

records = []
errors = []


for idx, row in test_df.iterrows():

    spdi = str(row["Canonical SPDI"]).strip()

    variant_type = str(row["Variant type"])

    variation_id = str(row["VariationID"])

    parts = spdi.split(":")

    if len(parts) != 4:

        errors.append(
            (
                idx,
                variation_id,
                variant_type,
                spdi,
                "Invalid SPDI format"
            )
        )

        continue


    accession, pos, ref, alt = parts


    # --------------------------------------------------------------
    # Chromosome mapping
    # --------------------------------------------------------------

    accession_to_chrom = {
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
        "NC_000018.9": "18",
        "NC_000019.10": "19",
        "NC_000020.11": "20",
        "NC_000021.9": "21",
        "NC_000022.11": "22",
        "NC_000023.11": "X",
        "NC_000024.10": "Y",
    }


    if accession not in accession_to_chrom:

        errors.append(
            (
                idx,
                variation_id,
                variant_type,
                spdi,
                f"Unknown accession: {accession}"
            )
        )

        continue


    chrom = accession_to_chrom[accession]


    # --------------------------------------------------------------
    # Position
    # --------------------------------------------------------------

    try:

        pos = int(pos)

    except ValueError:

        errors.append(
            (
                idx,
                variation_id,
                variant_type,
                spdi,
                "Invalid position"
            )
        )

        continue


    # --------------------------------------------------------------
    # CASE 1: INSERTION
    #
    # SPDI:
    #
    # NC_000001.11:1439576::A
    #
    # REF = ""
    # ALT = "A"
    #
    # VCF needs an anchor base.
    # --------------------------------------------------------------

    if ref == "" and alt != "":

        try:

            anchor = get_base(
                reference,
                chrom,
                pos
            )

        except Exception as e:

            errors.append(
                (
                    idx,
                    variation_id,
                    variant_type,
                    spdi,
                    f"Insertion anchor error: {e}"
                )
            )

            continue


        vcf_pos = pos

        vcf_ref = anchor

        vcf_alt = anchor + alt


    # --------------------------------------------------------------
    # CASE 2: DELETION
    #
    # SPDI:
    #
    # NC_000001.11:1048977:AG:
    #
    # REF = "AG"
    # ALT = ""
    #
    # We anchor using the preceding reference base.
    # --------------------------------------------------------------

    elif ref != "" and alt == "":

        try:

            anchor = get_base(
                reference,
                chrom,
                pos
            )

        except Exception as e:

            errors.append(
                (
                    idx,
                    variation_id,
                    variant_type,
                    spdi,
                    f"Deletion anchor error: {e}"
                )
            )

            continue


        vcf_pos = pos

        vcf_ref = anchor + ref

        vcf_alt = anchor


    # --------------------------------------------------------------
    # CASE 3: NORMAL SNV / INDEL
    # --------------------------------------------------------------

    elif ref != "" and alt != "":

        vcf_pos = pos + 1

        vcf_ref = ref

        vcf_alt = alt


    # --------------------------------------------------------------
    # CASE 4: BOTH EMPTY
    # --------------------------------------------------------------

    else:

        errors.append(
            (
                idx,
                variation_id,
                variant_type,
                spdi,
                "Both REF and ALT are empty"
            )
        )

        continue


    # --------------------------------------------------------------
    # Store record
    # --------------------------------------------------------------

    records.append(
        {
            "CHROM": chrom,
            "POS": vcf_pos,
            "ID": f"ClinVar_{variation_id}",
            "REF": vcf_ref,
            "ALT": vcf_alt,
            "SPDI": spdi,
            "VariationID": variation_id,
            "VariantType": variant_type,
            "Name": row["Name"],
        }
    )


# ----------------------------------------------------------------------
# WRITE VCF
# ----------------------------------------------------------------------

out = pd.DataFrame(records)


with open(OUTPUT, "w") as f:

    f.write("##fileformat=VCFv4.2\n")
    f.write("##reference=GRCh38\n")
    f.write("##source=ClinVar_VEP_all_types_test\n")

    f.write(
        '##INFO=<ID=VariationID,Number=1,Type=String,'
        'Description="ClinVar VariationID">\n'
    )

    f.write(
        '##INFO=<ID=SPDI,Number=1,Type=String,'
        'Description="Original ClinVar SPDI">\n'
    )

    f.write(
        '##INFO=<ID=VariantType,Number=1,Type=String,'
        'Description="ClinVar variant type">\n'
    )

    f.write(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )


    for _, r in out.iterrows():

        variant_type = (
            str(r["VariantType"])
            .replace(" ", "_")
        )

        info = (
            f"VariationID={r['VariationID']};"
            f"SPDI={r['SPDI']};"
            f"VariantType={variant_type}"
        )

        f.write(
            f"{r['CHROM']}\t"
            f"{r['POS']}\t"
            f"{r['ID']}\t"
            f"{r['REF']}\t"
            f"{r['ALT']}\t"
            f".\t"
            f"PASS\t"
            f"{info}\n"
        )


# ----------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------

print("=" * 70)
print("VEP ALL-TYPES TEST VCF")
print("=" * 70)

print(f"Input variants: {len(test_df):,}")
print(f"VCF records:    {len(out):,}")
print(f"Errors:         {len(errors):,}")

print(f"Output:         {OUTPUT}")

print("\nSelected variant types:")

print(
    test_df["Variant type"]
    .value_counts()
)


print("\nVCF variant types:")

print(
    out["VariantType"]
    .value_counts()
)


if errors:

    print("\nERRORS:")

    for error in errors:

        print(error)


print("\nVCF records:")

print(
    out[
        [
            "CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "VariantType"
        ]
    ].to_string(index=False)
)