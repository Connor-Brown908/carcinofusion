"""Generate synthetic CPDB-shaped fixture data.

This is NOT carcinogenicity data. Every value is invented. Its only purpose is
to exercise the pipeline end to end with the structural awkwardness the real
files contain: multiple experiments per chemical, conflicting results across
species and sex, salts and stereoisomers, malformed SMILES, chemicals with no
structure, and placeholder null tokens.

Usage:
    python tests/make_fixture.py --outdir data/raw/cpdb_test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# (chemcode, cas, name, smiles) - structures are real molecules, the
# carcinogenicity assessments attached to them below are invented.
CHEMICALS = [
    ("C00001", "71-43-2", "benzene", "c1ccccc1"),
    ("C00002", "91-20-3", "naphthalene", "c1ccc2ccccc2c1"),
    ("C00003", "108-88-3", "toluene", "Cc1ccccc1"),
    ("C00004", "100-41-4", "ethylbenzene", "CCc1ccccc1"),
    ("C00005", "106-42-3", "p-xylene", "Cc1ccc(C)cc1"),
    ("C00006", "95-47-6", "o-xylene", "Cc1ccccc1C"),
    ("C00007", "64-17-5", "ethanol", "CCO"),
    ("C00008", "67-56-1", "methanol", "CO"),
    ("C00009", "71-23-8", "propanol", "CCCO"),
    ("C00010", "75-65-0", "tert-butanol", "CC(C)(C)O"),
    ("C00011", "110-54-3", "hexane", "CCCCCC"),
    ("C00012", "109-66-0", "pentane", "CCCCC"),
    ("C00013", "50-00-0", "formaldehyde", "C=O"),
    ("C00014", "75-07-0", "acetaldehyde", "CC=O"),
    ("C00015", "67-64-1", "acetone", "CC(C)=O"),
    ("C00016", "98-86-2", "acetophenone", "CC(=O)c1ccccc1"),
    ("C00017", "103-90-2", "paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("C00018", "50-78-2", "aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("C00019", "58-08-2", "caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("C00020", "83-67-0", "theobromine", "Cn1cnc2c1c(=O)[nH]c(=O)n2C"),
    ("C00021", "62-53-3", "aniline", "Nc1ccccc1"),
    ("C00022", "106-50-3", "p-phenylenediamine", "Nc1ccc(N)cc1"),
    ("C00023", "95-53-4", "o-toluidine", "Cc1ccccc1N"),
    ("C00024", "92-67-1", "4-aminobiphenyl", "Nc1ccc(-c2ccccc2)cc1"),
    ("C00025", "92-87-5", "benzidine", "Nc1ccc(-c2ccc(N)cc2)cc1"),
    ("C00026", "108-95-2", "phenol", "Oc1ccccc1"),
    ("C00027", "120-80-9", "catechol", "Oc1ccccc1O"),
    ("C00028", "123-31-9", "hydroquinone", "Oc1ccc(O)cc1"),
    ("C00029", "108-46-3", "resorcinol", "Oc1cccc(O)c1"),
    ("C00030", "87-66-1", "pyrogallol", "Oc1cccc(O)c1O"),
    # Stereoisomer pair: must remain two distinct compounds.
    ("C00031", "56-41-7", "L-alanine", "C[C@H](N)C(=O)O"),
    ("C00032", "338-69-2", "D-alanine", "C[C@@H](N)C(=O)O"),
    # Salt: standardises to the acetate parent.
    ("C00033", "127-09-3", "sodium acetate", "CC(=O)[O-].[Na+]"),
    # Two source chemicals that standardise to the SAME structure.
    ("C00034", "64-19-7", "acetic acid", "CC(=O)O"),
    ("C00035", "64-19-7b", "acetic acid (duplicate record)", "OC(C)=O"),
    # Malformed SMILES: must land in the review table.
    ("C00036", "999-99-9", "unparseable substance", "this_is_not_a_smiles"),
    # Inorganic: no organic fragment.
    ("C00037", "7647-14-5", "sodium chloride", "[Na+].[Cl-]"),
    ("C00038", "121-14-2", "2,4-dinitrotoluene", "Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),
    ("C00039", "118-96-7", "TNT", "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"),
    ("C00040", "98-95-3", "nitrobenzene", "O=[N+]([O-])c1ccccc1"),
]

# Chemical with no structure in the resolution table at all.
UNRESOLVED = ("C00041", "000-00-0", "substance of unknown structure")

# (chemcode, species, sex, strain, route, assessment)
# Invented assessments covering every branch of the label policy.
EXPERIMENTS = [
    # Consistent positives
    ("C00001", "r", "m", "F344", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00001", "r", "f", "F344", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00001", "m", "m", "B6C3F1", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00024", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00024", "m", "f", "B6C3F1", "diet", "Clear evidence of carcinogenic activity"),
    ("C00025", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00025", "r", "f", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00025", "m", "m", "B6C3F1", "diet", "Clear evidence of carcinogenic activity"),
    ("C00038", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00038", "r", "f", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00002", "m", "m", "B6C3F1", "inhalation", "Clear evidence of carcinogenic activity"),
    ("C00002", "m", "f", "B6C3F1", "inhalation", "Clear evidence of carcinogenic activity"),
    ("C00021", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00021", "r", "f", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00023", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00023", "m", "m", "B6C3F1", "diet", "Clear evidence of carcinogenic activity"),
    ("C00028", "r", "m", "F344", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00028", "r", "f", "F344", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00039", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00040", "r", "m", "F344", "inhalation", "Clear evidence of carcinogenic activity"),
    ("C00040", "m", "m", "B6C3F1", "inhalation", "Clear evidence of carcinogenic activity"),
    ("C00022", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    ("C00027", "r", "m", "F344", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00027", "m", "f", "B6C3F1", "gavage", "Clear evidence of carcinogenic activity"),
    # Consistent negatives
    ("C00007", "r", "m", "F344", "drinking water", "No evidence of carcinogenic activity"),
    ("C00007", "r", "f", "F344", "drinking water", "No evidence of carcinogenic activity"),
    ("C00007", "m", "m", "B6C3F1", "drinking water", "No evidence of carcinogenic activity"),
    ("C00008", "r", "m", "F344", "drinking water", "No evidence of carcinogenic activity"),
    ("C00008", "m", "f", "B6C3F1", "drinking water", "No evidence of carcinogenic activity"),
    ("C00009", "r", "m", "F344", "gavage", "No evidence of carcinogenic activity"),
    ("C00009", "r", "f", "F344", "gavage", "No evidence of carcinogenic activity"),
    ("C00015", "r", "m", "F344", "drinking water", "No evidence of carcinogenic activity"),
    ("C00015", "m", "m", "B6C3F1", "drinking water", "No evidence of carcinogenic activity"),
    ("C00018", "r", "m", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00018", "r", "f", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00018", "m", "m", "B6C3F1", "diet", "No evidence of carcinogenic activity"),
    ("C00019", "r", "m", "F344", "drinking water", "No evidence of carcinogenic activity"),
    ("C00019", "m", "f", "B6C3F1", "drinking water", "No evidence of carcinogenic activity"),
    ("C00020", "r", "m", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00020", "r", "f", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00005", "r", "m", "F344", "gavage", "No evidence of carcinogenic activity"),
    ("C00005", "m", "m", "B6C3F1", "gavage", "No evidence of carcinogenic activity"),
    ("C00006", "r", "f", "F344", "gavage", "No evidence of carcinogenic activity"),
    ("C00006", "m", "f", "B6C3F1", "gavage", "No evidence of carcinogenic activity"),
    ("C00010", "r", "m", "F344", "drinking water", "No evidence of carcinogenic activity"),
    ("C00012", "r", "m", "F344", "inhalation", "No evidence of carcinogenic activity"),
    ("C00012", "m", "m", "B6C3F1", "inhalation", "No evidence of carcinogenic activity"),
    ("C00016", "r", "f", "F344", "gavage", "No evidence of carcinogenic activity"),
    ("C00029", "r", "m", "F344", "gavage", "No evidence of carcinogenic activity"),
    ("C00029", "m", "m", "B6C3F1", "gavage", "No evidence of carcinogenic activity"),
    ("C00031", "r", "m", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00031", "r", "f", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00032", "r", "m", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00033", "r", "m", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00033", "m", "m", "B6C3F1", "diet", "No evidence of carcinogenic activity"),
    ("C00034", "r", "m", "F344", "diet", "No evidence of carcinogenic activity"),
    ("C00035", "m", "f", "B6C3F1", "diet", "No evidence of carcinogenic activity"),
    # CONFLICT: positive in rat, negative in mouse (across species)
    ("C00003", "r", "m", "F344", "inhalation", "Clear evidence of carcinogenic activity"),
    ("C00003", "m", "m", "B6C3F1", "inhalation", "No evidence of carcinogenic activity"),
    # CONFLICT: same species, different sex
    ("C00004", "r", "m", "F344", "inhalation", "Clear evidence of carcinogenic activity"),
    ("C00004", "r", "f", "F344", "inhalation", "No evidence of carcinogenic activity"),
    # CONFLICT: same species AND same sex, different route
    ("C00026", "r", "m", "F344", "gavage", "Clear evidence of carcinogenic activity"),
    ("C00026", "r", "m", "F344", "drinking water", "No evidence of carcinogenic activity"),
    # Some-evidence only: null under primary, positive under sensitivity
    ("C00013", "r", "m", "F344", "inhalation", "Some evidence of carcinogenic activity"),
    ("C00013", "r", "f", "F344", "inhalation", "Some evidence of carcinogenic activity"),
    ("C00014", "r", "m", "F344", "drinking water", "Some evidence of carcinogenic activity"),
    # Equivocal only: null under both endpoints
    ("C00011", "r", "m", "F344", "inhalation", "Equivocal evidence of carcinogenic activity"),
    ("C00030", "m", "m", "B6C3F1", "gavage", "Equivocal evidence of carcinogenic activity"),
    # Missing assessment: null token, must NOT become negative
    ("C00017", "r", "m", "F344", "diet", "."),
    ("C00017", "m", "m", "B6C3F1", "diet", "NA"),
    # Non-rodent only: excluded from the endpoint's eligible species
    ("C00037", "h", "m", "Syrian", "diet", "No evidence of carcinogenic activity"),
    # Unparseable structure, but a real assessment
    ("C00036", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    # No structure at all
    ("C00041", "r", "m", "F344", "diet", "Clear evidence of carcinogenic activity"),
    # Unknown assessment string: must become null and be reported
    ("C00010", "m", "f", "B6C3F1", "diet", "Probably fine honestly"),
]


def write_tab(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="latin-1") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/raw/cpdb_test")
    ap.add_argument("--resolution", default="data/external/structure_resolution_test.tsv")
    args = ap.parse_args()

    out = Path(args.outdir)

    # Split experiments between the NCI/NTP and literature files, so the
    # concat path and per-source reporting are both exercised.
    # Headers match the real CPDB distribution, verified against the live
    # files with scripts/check_columns.py. The earlier fixture used invented
    # names (and a chemcode in the dose tables, which CPDB does not have), so
    # the tests passed against a schema that did not exist.
    ncintp_rows, lit_rows = [], []
    for i, (code, sp, sex, strain, route, assessment) in enumerate(EXPERIMENTS):
        clear = assessment.lower().startswith("clear")
        td50 = f"{(i % 17) * 3.5 + 1.2:.2f}" if clear else "."
        idnum = f"{1000 + i}"
        row = [
            idnum, code, sp, sex, strain, route,
            "104", "24",                                   # xprtime, xpotime
            "liver" if i % 3 == 0 else "lung",             # tissue
            "adenoma" if i % 2 == 0 else "carcinoma",      # tumor
            td50,
            f"{(i % 17) * 2.1 + 0.6:.2f}" if clear else ".",   # lc
            f"{(i % 17) * 5.4 + 2.8:.2f}" if clear else ".",   # uc
            "0.001" if clear else ".",                     # pval
            "50", "2",                                     # ctotal, ctumors
            "3",                                           # ndoses
            "",                                            # inad
            assessment,
            f"{2000 + i}",                                 # papernum
            "",
        ]
        (ncintp_rows if i % 2 == 0 else lit_rows).append(row)

    exp_header = [
        "idnum", "chemcode", "species", "sex", "strain", "route",
        "xprtime", "xpotime", "tissue", "tumor", "td50", "lc", "uc", "pval",
        "ctotal", "ctumors", "ndoses", "inad", "opinion", "papernum", "notes",
    ]
    write_tab(out / "cpdb.ncintp.tab", exp_header, ncintp_rows)
    write_tab(out / "cpdb.lit.tab", exp_header, lit_rows)

    chem_rows = [[c, cas, name, str(i)] for i, (c, cas, name, _) in enumerate(CHEMICALS)]
    chem_rows.append([UNRESOLVED[0], UNRESOLVED[1], UNRESOLVED[2], str(len(CHEMICALS))])
    write_tab(out / "cpdb.chemname.tab", ["chemcode", "cas", "name", "sortordr"], chem_rows)

    # Dose tables carry NO chemcode. One row per dose group, linked to the
    # experiment by idnum, with `order` giving its place in the dose series.
    # Written out of order on purpose, so the parser is shown to sort by
    # `order` rather than trusting file order.
    dose_rows = []
    for i in range(len(EXPERIMENTS)):
        idnum = f"{1000 + i}"
        groups = [(1, 25, 50, 2), (2, 50, 50, 8), (3, 100, 50, 19)]
        for order, dose, total, tumors in reversed(groups):
            dose_rows.append([idnum, str(order), str(dose), str(total), str(tumors)])
    write_tab(
        out / "cpdb.ncntdose.tab",
        ["idnum", "order", "dose", "total", "tumors"],
        dose_rows,
    )

    # Structure resolution table. C00041 is deliberately absent.
    res = Path(args.resolution)
    res.parent.mkdir(parents=True, exist_ok=True)
    with res.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["chemcode", "cas_number", "chemical_name", "smiles", "source", "retrieved_date"])
        for code, cas, name, smi in CHEMICALS:
            w.writerow([code, cas, name, smi, "PubChem", "2026-10-29"])

    print(f"fixture written: {out}")
    print(f"  chemicals: {len(CHEMICALS) + 1} (1 with no structure)")
    print(f"  experiments: {len(EXPERIMENTS)}")
    print(f"  dose rows: {len(dose_rows)}")
    print(f"  resolution table: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
