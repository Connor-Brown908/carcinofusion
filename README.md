# carcinofusion

Phase 1 data pipeline for a structure-based carcinogenicity prediction
project: acquire CPDB, resolve chemical identity, harmonise labels, audit the
result, and generate leakage-resistant splits that can be frozen and verified.

No model is trained here. The purpose of this phase is to establish whether the
core dataset is usable, and to make every curation decision explicit enough
that a result derived from it can be reproduced and defended.

---

## Quick start

```bash
# 1. environment
conda env create -f environment.yml && conda activate carcinofusion
# or: pip install rdkit pandas pyarrow pyyaml pytest

# 2. prove the pipeline works, on synthetic data
bash tests/run_pipeline.sh

# 3. run it for real
#    place the CPDB files in data/raw/cpdb/ first (see docs/dataset_protocol.md)
python -m src.ingestion.download_cpdb
python -m src.ingestion.parse_cpdb
python -m src.chemistry.resolve_identity
python -m src.curation.harmonize_labels
python -m src.curation.audit_conflicts
python -m src.datasets.build_dataset
python -m src.datasets.split_dataset
python -m src.curation.build_ledger
python -m src.datasets.freeze_dataset --note "initial Phase 1 freeze"
```

---

## Before the first real run

Two things need your attention. Both fail loudly rather than silently, but
both are worth doing deliberately.

1. **Verify the CPDB column mapping.** `configs/cpdb_columns.yaml` declares
   which source column carries which field. The defaults reflect field names
   in common use but have not been checked against your download. Read the
   CPDB documentation and correct the file. The parser validates the real
   header and fails, naming the column, rather than guessing.

2. **Supply a structure resolution table.** The pipeline performs no network
   lookup, so that the structure a compound had when the dataset was frozen is
   recorded rather than re-fetched later. Create
   `data/external/structure_resolution.tsv` with `chemcode` or `cas_number`,
   plus `smiles`, `source` and `retrieved_date`. See
   `docs/dataset_protocol.md`.

---

## Design commitments

These are the decisions the rest of the code exists to enforce.

**Missing evidence is never negative evidence.** A chemical never tested in a
sufficiently informative bioassay is `null`, not `0`. This holds from the
source file through parquet into PostgreSQL as SQL `NULL`.

**Conflicting evidence is never resolved by majority vote.** One positive
against nine negatives is conflicted, not negative. Conflicted compounds leave
the primary training set and stay in the database with their evidence intact.

**Originals are preserved.** Every standardised column has a `*_raw`
companion. Auditing a conflicting result is impossible once the source value
has been overwritten.

**Stereochemistry survives standardisation.** RDKit's tautomer canonicaliser
strips sp3 stereo by default, which would merge L- and D-alanine into one
compound and take both their labels with them. The pipeline overrides it, and
a test asserts it.

**Exact identity and grouping identity are different things.** Stereoisomers
get distinct `compound_id`s but share a `scaffold_group`, so they cannot be
separated across a train/test boundary — which matters because Morgan
fingerprints cannot tell them apart.

**Splits are verified, not assumed.** A shared InChIKey or a scaffold group
spanning partitions stops the run. Nearest-neighbour similarity between
partitions is measured and reported, because scaffold separation does not
imply chemical dissimilarity.

**Nothing disappears.** Unresolved structures, unmapped assessments and
conflicted compounds are all retained with their reason. The test suite
asserts that the set of compounds entering the splitter equals the set leaving
it.

**Every stage records what it did.** Inputs and outputs with checksums, config
versions, counts and warnings land in `manifests/`.

**Every exclusion and transformation is enumerated.** `curation_ledger.parquet`
holds one row per event with the entity it applied to and a closed-vocabulary
reason code, so "why is this compound not in the training set" is one filter
rather than four files.

---

## Layout

```
carcinofusion/
├── configs/
│   ├── data.yaml              sources, paths, structure resolution
│   ├── cpdb_columns.yaml      column mapping — verify before first run
│   ├── labels.yaml            THE label policy; code hard-codes nothing
│   └── splits.yaml            split strategy and leakage thresholds
├── data/
│   ├── raw/                   untouched source files, never written to
│   ├── interim/               Dataset A, Dataset B, labelled observations
│   ├── processed/             model-ready tables, splits, audit, frozen/
│   └── external/              resolution table, external validation set
├── docs/
│   ├── dataset_protocol.md    the curation protocol
│   ├── label_harmonization.md how an assessment becomes a label
│   └── data_dictionary.md     every column
├── sql/schema.sql             PostgreSQL DDL
├── src/
│   ├── common.py              config, manifests, column mapping
│   ├── ingestion/             download_cpdb, parse_cpdb
│   ├── chemistry/             resolve_identity, standardize
│   ├── curation/              harmonize_labels, audit_conflicts, build_ledger
│   ├── features/              fingerprints, descriptors
│   ├── datasets/              build, split, freeze, export_for_postgres
│   ├── models/                (Phase 2)
│   └── evaluation/            (Phase 2)
├── tests/
│   ├── make_fixture.py        synthetic CPDB-shaped data
│   ├── run_pipeline.sh        end-to-end run
│   ├── test_structures.py
│   ├── test_labels.py
│   ├── test_splits.py
│   └── test_acceptance.py     the eight acceptance criteria
└── manifests/                 per-stage provenance
```

The structure is modular on purpose: the source dataset, label policy or
molecular encoder can each be replaced without rewriting the rest.

---

## Stages

| # | Stage | Command | Output |
| --- | --- | --- | --- |
| 1 | acquire | `src.ingestion.download_cpdb` | checksums, retrieval date |
| 2 | parse | `src.ingestion.parse_cpdb` | Dataset A |
| 3 | identity | `src.chemistry.resolve_identity` | Dataset B, review table |
| 4 | labels | `src.curation.harmonize_labels` | record and compound labels |
| 5 | audit | `src.curation.audit_conflicts` | audit report, conflicts |
| 6 | build | `src.datasets.build_dataset` | model-ready table per endpoint |
| 7 | split | `src.datasets.split_dataset` | locked splits, split report |
| 8 | ledger | `src.curation.build_ledger` | curation ledger, summary |
| 9 | freeze | `src.datasets.freeze_dataset` | immutable Dataset v1 |
| 10 | export | `src.datasets.export_for_postgres` | CSVs + `load.sql` |

Exit codes: `2` missing input, `3` column mapping or freeze conflict,
`4` leakage or null group key, `5` frozen dataset drift.

---

## Two endpoints

Both are built every run, so you can measure how far a result depends on the
label definition rather than the chemistry.

| Assessment | `primary` | `sensitivity` |
| --- | --- | --- |
| Clear evidence | 1 | 1 |
| Some evidence | null | 1 |
| Equivocal | null | null |
| No evidence | 0 | 0 |

---

## Acceptance criteria

`tests/test_acceptance.py` encodes the Phase 1 acceptance criteria as
executable tests — one class per criterion, checked against the artefacts the
pipeline produced rather than asserted in prose.

| Criterion | Enforced by | How it is met |
| --- | --- | --- |
| Traceability | `TestTraceability` | every label → compound → experiment record → `source_file:line`, with the raw file's SHA-256 recorded at parse time |
| Structure integrity | `TestStructureIntegrity` | every included compound has a parseable structure, a recorded `input_smiles`, a named provenance and a standardisation version |
| Duplicate control | `TestDuplicateControl` | InChIKey is unique; merges are recorded with their source chemcodes and logged; stereoisomers stay distinct |
| Label consistency | `TestLabelConsistency` | labels are only 0/1/null, policy version stamped on every row, and re-applying the policy reproduces the stored labels |
| Conflict handling | `TestConflictHandling` | conflicted compounds are flagged, excluded from partitions, retained with evidence counts, and broken down by conflict axis |
| Leakage control | `TestLeakageControl` | checked at record level: every record of a compound shares one partition; no InChIKey or scaffold group spans partitions |
| Reproducibility | `TestReproducibility` | versioned configs, fixed seed, per-stage manifests, and a freeze lock re-verified against the files |
| Auditability | `TestAuditability` | the curation ledger enumerates every exclusion and transformation with a closed-vocabulary reason code |

Two runs of the pipeline from the same inputs produce byte-identical outputs.

## Tests

```bash
python -m pytest tests/ -v
```

137 tests: 77 unit tests covering stereochemistry preservation, salt and
mixture handling, rejection of malformed input, determinism of identity, the
label policy's three states, no-majority-vote aggregation, species
eligibility, group assignment and leakage detection; plus 60 acceptance tests
covering the criteria above.

The acceptance tests read real artefacts and skip if the pipeline has not been
run, so a clean `bash tests/run_pipeline.sh` is what makes them meaningful.

---

## PostgreSQL

```bash
python -m src.datasets.export_for_postgres
psql -d carcinofusion -f sql/schema.sql
psql -d carcinofusion -f data/processed/pg/load.sql
```

Five tables — `sources`, `chemicals`, `experiments`, `observations`,
`model_datasets` — plus a `v_training_set` view that excludes unresolved and
conflicted compounds. A local database is adequate. A `CHECK` constraint
permits only `0` and `1` for a label while allowing `NULL`, so "unresolved"
survives into the database as a distinct state.

---

## Not in this phase

Model training, feature selection beyond basic fingerprints and descriptors,
external dataset curation, and genotoxic versus non-genotoxic mechanism
labelling. The schema and the directory layout leave room for all four:
`observations` is already keyed by endpoint, and `model_datasets` is versioned
per endpoint and split strategy.
