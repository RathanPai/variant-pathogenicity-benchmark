import gzip
import pandas as pd
from pathlib import Path

INPUT = Path("Data/clinvar_gnomad.csv")
OUTPUT = Path("Data/clinvar_all.vcf")
FASTA = Path(
    "~/.vep/homo_sapiens/116_GRCh38/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
).expanduser()
FAI = Path(str(FASTA) + ".fai")


# ----------------------------------------------------------------------
# GRCh38 RefSeq accession -> chromosome
# ----------------------------------------------------------------------

CHROM_MAP = {
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
    "NC_000018.10": "18",
    "NC_000019.10": "19",
    "NC_000020.11": "20",
    "NC_000021.9": "21",
    "NC_000022.11": "22",
    "NC_000023.11": "X",
    "NC_000024.10": "Y",
    "NC_012920.1": "MT",
}


# ----------------------------------------------------------------------
# Load FASTA index
#
# .fai columns:
# name, length, offset, line_bases, line_width
# ----------------------------------------------------------------------

def load_fai():

    if not FAI.exists():
        raise FileNotFoundError(
            f"FASTA index not found:\n{FAI}"
        )

    index = {}

    with open(FAI, "r") as f:

        for line in f:

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 5:
                continue

            name = fields[0]
            length = int(fields[1])
            offset = int(fields[2])
            line_bases = int(fields[3])
            line_width = int(fields[4])

            index[name] = {
                "length": length,
                "offset": offset,
                "line_bases": line_bases,
                "line_width": line_width,
            }

    return index


# ----------------------------------------------------------------------
# FASTA random-access reader
# ----------------------------------------------------------------------

class IndexedFasta:

    def __init__(self, fasta_path, fai_path):

        self.fasta_path = fasta_path
        self.index = load_fai()

        self.handle = gzip.open(
            fasta_path,
            "rb"
        )

    def get_base(self, chrom, position):

        """
        Return one reference base.

        position is 1-based.
        """

        if chrom not in self.index:
            raise ValueError(
                f"Chromosome {chrom} not found in FASTA index"
            )

        info = self.index[chrom]

        if position < 1 or position > info["length"]:
            raise ValueError(
                f"Position {position} outside chromosome {chrom}"
            )

        zero_based = position - 1

        line_number = (
            zero_based // info["line_bases"]
        )

        position_in_line = (
            zero_based % info["line_bases"]
        )

        byte_offset = (
            info["offset"]
            + line_number * info["line_width"]
            + position_in_line
        )

        self.handle.seek(byte_offset)

        base = self.handle.read(1).decode().upper()

        if not base:
            raise ValueError(
                f"Could not read base at {chrom}:{position}"
            )

        return base

    def close(self):
        self.handle.close()


# ----------------------------------------------------------------------
# Load input
# ----------------------------------------------------------------------

print("=" * 70)
print("CLINVAR SPDI → VCF PRODUCTION CONVERSION")
print("=" * 70)

print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print()

if not INPUT.exists():
    raise FileNotFoundError(f"Input not found: {INPUT}")

if not FASTA.exists():
    raise FileNotFoundError(f"FASTA not found: {FASTA}")

if not FAI.exists():
    raise FileNotFoundError(f"FASTA index not found: {FAI}")


df = pd.read_csv(INPUT)

print(f"ClinVar variants loaded: {len(df):,}")


# ----------------------------------------------------------------------
# Check required columns
# ----------------------------------------------------------------------

required_columns = [
    "VariationID",
    "Canonical SPDI",
    "Variant type",
    "Name",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ----------------------------------------------------------------------
# Initialize indexed FASTA
# ----------------------------------------------------------------------

fasta = IndexedFasta(
    FASTA,
    FAI
)

print(
    f"FASTA sequences indexed: {len(fasta.index):,}"
)

print()


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------

records = []
errors = []

status_counts = {
    "normal": 0,
    "insertion_anchored": 0,
    "deletion_anchored": 0,
}


for idx, row in df.iterrows():

    variation_id = str(row["VariationID"])
    variant_type = str(row["Variant type"])
    spdi = str(row["Canonical SPDI"]).strip()

    # --------------------------------------------------------------
    # Validate SPDI
    # --------------------------------------------------------------

    if not spdi or spdi.lower() == "nan":

        errors.append({
            "row": idx,
            "VariationID": variation_id,
            "VariantType": variant_type,
            "SPDI": spdi,
            "error": "Missing SPDI",
        })

        continue


    parts = spdi.split(":")

    if len(parts) != 4:

        errors.append({
            "row": idx,
            "VariationID": variation_id,
            "VariantType": variant_type,
            "SPDI": spdi,
            "error": "Invalid SPDI format",
        })

        continue


    accession, pos_string, ref, alt = parts


    # --------------------------------------------------------------
    # Chromosome
    # --------------------------------------------------------------

    if accession not in CHROM_MAP:

        errors.append({
            "row": idx,
            "VariationID": variation_id,
            "VariantType": variant_type,
            "SPDI": spdi,
            "error": f"Unsupported accession: {accession}",
        })

        continue


    chrom = CHROM_MAP[accession]


    # --------------------------------------------------------------
    # Position
    # --------------------------------------------------------------

    try:

        pos = int(pos_string)

    except ValueError:

        errors.append({
            "row": idx,
            "VariationID": variation_id,
            "VariantType": variant_type,
            "SPDI": spdi,
            "error": "Invalid SPDI position",
        })

        continue


    # --------------------------------------------------------------
    # CASE 1 — INSERTION
    #
    # Example:
    #
    # NC_000001.11:1439576::A
    #
    # SPDI:
    #     position = 1439576
    #     REF      = ""
    #     ALT      = "A"
    #
    # VCF:
    #     POS = 1439576
    #     REF = anchor
    #     ALT = anchor + inserted sequence
    # --------------------------------------------------------------

    if ref == "" and alt != "":

        try:

            anchor = fasta.get_base(
                chrom,
                pos
            )

        except Exception as e:

            errors.append({
                "row": idx,
                "VariationID": variation_id,
                "VariantType": variant_type,
                "SPDI": spdi,
                "error": f"Insertion anchor failed: {e}",
            })

            continue


        vcf_pos = pos
        vcf_ref = anchor
        vcf_alt = anchor + alt

        status_counts["insertion_anchored"] += 1


    # --------------------------------------------------------------
    # CASE 2 — DELETION
    #
    # Example:
    #
    # NC_000001.11:1048977:AG:
    #
    # SPDI:
    #     position = 1048977
    #     REF      = AG
    #     ALT      = ""
    #
    # VCF:
    #     POS = 1048977
    #     REF = anchor + AG
    #     ALT = anchor
    # --------------------------------------------------------------

    elif ref != "" and alt == "":

        try:

            anchor = fasta.get_base(
                chrom,
                pos
            )

        except Exception as e:

            errors.append({
                "row": idx,
                "VariationID": variation_id,
                "VariantType": variant_type,
                "SPDI": spdi,
                "error": f"Deletion anchor failed: {e}",
            })

            continue


        vcf_pos = pos
        vcf_ref = anchor + ref
        vcf_alt = anchor

        status_counts["deletion_anchored"] += 1


    # --------------------------------------------------------------
    # CASE 3 — NORMAL SUBSTITUTION / INDEL
    # --------------------------------------------------------------

    elif ref != "" and alt != "":

        vcf_pos = pos + 1
        vcf_ref = ref
        vcf_alt = alt

        status_counts["normal"] += 1


    # --------------------------------------------------------------
    # CASE 4 — BOTH EMPTY
    # --------------------------------------------------------------

    else:

        errors.append({
            "row": idx,
            "VariationID": variation_id,
            "VariantType": variant_type,
            "SPDI": spdi,
            "error": "Both REF and ALT are empty",
        })

        continue


    # --------------------------------------------------------------
    # Store
    # --------------------------------------------------------------

    records.append({
        "CHROM": chrom,
        "POS": vcf_pos,
        "ID": f"ClinVar_{variation_id}",
        "REF": vcf_ref,
        "ALT": vcf_alt,
        "VariationID": variation_id,
        "SPDI": spdi,
        "VariantType": variant_type,
        "Name": row["Name"],
    })


# ----------------------------------------------------------------------
# Close FASTA
# ----------------------------------------------------------------------

fasta.close()


# ----------------------------------------------------------------------
# Create output dataframe
# ----------------------------------------------------------------------

out = pd.DataFrame(records)


# ----------------------------------------------------------------------
# Write VCF
# ----------------------------------------------------------------------

with open(OUTPUT, "w") as f:

    f.write("##fileformat=VCFv4.2\n")
    f.write("##reference=GRCh38\n")
    f.write("##source=ClinVar_SPDI_Production\n")

    f.write(
        '##INFO=<ID=VariationID,Number=1,Type=String,'
        'Description="ClinVar VariationID">\n'
    )

    f.write(
        '##INFO=<ID=SPDI,Number=1,Type=String,'
        'Description="Original ClinVar Canonical SPDI">\n'
    )

    f.write(
        '##INFO=<ID=VariantType,Number=1,Type=String,'
        'Description="Original ClinVar variant type">\n'
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
# Save errors separately
# ----------------------------------------------------------------------

if errors:

    error_df = pd.DataFrame(errors)

    error_file = Path(
        "Data/clinvar_spdi_to_vcf_errors.csv"
    )

    error_df.to_csv(
        error_file,
        index=False
    )

else:

    error_file = None


# ----------------------------------------------------------------------
# FINAL REPORT
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print(
    f"Input variants:       {len(df):,}"
)

print(
    f"Successful VCF rows:  {len(out):,}"
)

print(
    f"Errors:               {len(errors):,}"
)

print()

print("Conversion types:")

for key, value in status_counts.items():

    print(
        f"  {key:22s}: {value:,}"
    )

print()

print("Original variant types:")

print(
    df["Variant type"]
    .value_counts()
    .to_string()
)

print()

print("VCF variant types:")

if len(out) > 0:

    print(
        out["VariantType"]
        .value_counts()
        .to_string()
    )

print()

if error_file:

    print(
        f"ERROR FILE: {error_file}"
    )

print(
    f"VCF OUTPUT: {OUTPUT}"
)

print("=" * 70)