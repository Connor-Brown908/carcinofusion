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
python3 scripts/check_env.py        # says exactly what is missing, and how to install it

conda env create -f environment.yml && conda activate carcinofusion
# or: pip install rdkit pandas pyarrow pyyaml numpy pytest scikit-learn lightgbm torch

# 2. prove the pipeline works, on synthetic data
bash tests/run_pipeline.sh

# 3. get the real CPDB files
bash scripts/fetch_cpdb.sh                        # downloads into data/raw/cpdb/
python -m src.ingestion.download_cpdb --print-urls  # or fetch them by hand

# 4. verify the column mapping against YOUR files
python3 scripts/check_columns.py

# 5. build the structure table (10-20 min)
python3 scripts/build_resolution_table.py

# 6. run the curation pipeline on real data
bash scripts/run_production.sh            # stops before the freeze, for review
bash scripts/run_production.sh --freeze --from 9   # freeze once reviewed
```

---

## Getting the CPDB data

CPDB is archival — NLM's catalogue entry states it is no longer updated — so
the files should be stable and the recorded checksums will prove whether they
drifted.

| Source | Location | Note |
| --- | --- | --- |
| NLM (authoritative) | `ftp.nlm.nih.gov/projects/SISFTP/CPDB/Data Files/` | plain filenames |
| ToxPlanet (mirror) | `files.toxplanet.com/cpdb/tab/` | appends `.txt`, stripped on save |
| Catalogue | [data.gov entry](https://catalog.data.gov/dataset/carcinogenic-potency-database-cpdb) | publisher and licensing |

```bash
bash scripts/fetch_cpdb.sh          # bash + curl only, no Python needed
MIRROR=toxplanet bash scripts/fetch_cpdb.sh
VERBOSE=1 bash scripts/fetch_cpdb.sh   # show transfer errors
```

Three files are required (`cpdb.ncintp.tab`, `cpdb.lit.tab`,
`cpdb.chemname.tab`); the dose tables, citations and the appendix lookup
tables are optional but worth having. The appendix tables decode CPDB's
species, strain, route, site and histopathology codes and supersede the small
hand-written vocabulary in `configs/cpdb_columns.yaml`.

Read the documentation before the first production run:
[Tab_Separated_CPDB.pdf](https://ftp.nlm.nih.gov/projects/SISFTP/CPDB/Documentation/Tab_Separated_CPDB.pdf)
and [Documentation_for_tab.pdf](https://files.toxplanet.com/cpdb/pdfs/Documentation_for_tab.pdf).

## Getting structures

CPDB's main tables carry `chemcode` and CAS but **no structures**, and nothing
can be modelled without them. This is the step that takes real time.

```bash
python scripts/build_resolution_table.py
```

It writes `data/external/structure_resolution.tsv` with a source and retrieval
date on every row, and `structure_unresolved.tsv` for everything it could not
resolve. Sources are tried in order per chemical:

1. **A local CPDB structure file**, via `--cpdb-structures`. CPDB shipped
   `chemical-structures.tab` (SMILES and InChI from EPA DSSTox), but its
   `toxnet.nlm.nih.gov` URLs died with TOXNET in 2019 and it was last updated
   in 2007 — a good starting point, not an authority.
2. **PubChem by CAS**, through PUG REST, rate-limited to PubChem's published
   5 requests per second.
3. **PubChem by name**, for chemicals whose CAS lookup misses.

Results are cached, so the run is resumable. Expect it to take 10–20 minutes
for CPDB's ~1,500 chemicals on a first pass.

Two things to do with the output before freezing, both of which the pipeline
cannot do for you:

- **Review `structure_unresolved.tsv`.** Those chemicals stay in
  `structure_review.parquet` and never get a label, so a gap here quietly
  shrinks the dataset rather than corrupting it.
- **Spot-check resolved rows.** A CAS number is not proof of identity, and
  the name-matched rows are the likeliest to be wrong.

Alternatives if PubChem coverage disappoints: the
[EPA CompTox Chemicals Dashboard](https://comptox.epa.gov/dashboard/) does
batch CAS to SMILES and exports a file you can reshape into the same columns.

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
   recorded rather than re-fetched later. `scripts/build_resolution_table.py`
   builds it (see *Getting structures* above); the format is `chemcode` or
   `cas_number`, plus `smiles`, `source` and `retrieved_date`.

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
│   ├── splits.yaml            split strategy and leakage thresholds
│   └── benchmark.yaml         THE Phase 2 protocol, frozen before training
├── data/
│   ├── raw/                   untouched source files, never written to
│   ├── interim/               Dataset A, Dataset B, labelled observations
│   ├── processed/             model-ready tables, splits, audit, frozen/
│   └── external/              resolution table, external validation set
├── docs/
│   ├── dataset_protocol.md    the curation protocol
│   ├── label_harmonization.md how an assessment becomes a label
│   ├── data_dictionary.md     every column
│   └── benchmark_protocol.md  the Phase 2 experimental protocol
├── scripts/
│   ├── fetch_cpdb.sh          standalone CPDB downloader
│   ├── build_resolution_table.py  CAS/name -> SMILES via PubChem
│   ├── check_env.py           report missing packages, stdlib-only
│   ├── diagnose_attrition.py  trace where compounds are lost
│   ├── build_mechanism_labels.py  genotoxic vs non-genotoxic tagging
│   ├── import_external_predictions.py  run incumbents on your test set
│   ├── make_figure.py         Figure 1, refuses on unsupportable data
│   ├── check_columns.py       verify cpdb_columns.yaml against real headers
│   └── run_production.sh      the pipeline on real data, with checkpoints
├── sql/schema.sql             PostgreSQL DDL
├── src/
│   ├── common.py              config, manifests, column mapping
│   ├── ingestion/             download_cpdb, parse_cpdb
│   ├── chemistry/             resolve_identity, standardize
│   ├── curation/              harmonize_labels, audit_conflicts, build_ledger
│   ├── features/              fingerprints, descriptors
│   ├── datasets/              build, split, freeze, export_for_postgres
│   ├── models/                base, baselines, mpnn, pretrained
│   └── evaluation/            metrics, runner, analysis
├── tests/
│   ├── make_fixture.py        synthetic CPDB-shaped data
│   ├── run_pipeline.sh        end-to-end run
│   ├── test_structures.py
│   ├── test_labels.py
│   ├── test_splits.py
│   ├── test_acceptance.py     the eight acceptance criteria
│   └── test_benchmark.py      Phase 2 metrics, models, protocol
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

202 tests: 80 Phase 1 unit tests covering stereochemistry preservation, salt and
mixture handling, rejection of malformed input, determinism of identity, the
label policy's three states, no-majority-vote aggregation, species
eligibility, group assignment and leakage detection; plus 60 acceptance tests
covering the criteria above, and 60 Phase 2 tests covering metric
definitions, uncertainty, threshold discipline, calibration, the model
interface and the frozen protocol.

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

---

# Phase 2 — benchmark suite

A six-model ladder behind one interface, run under both split strategies, with
enough provenance that any number ties back to the conditions that produced it.
Full protocol in `docs/benchmark_protocol.md`.

```bash
python -m src.evaluation.runner   --endpoint primary
python -m src.evaluation.analysis --endpoint primary
python -m src.datasets.supabase_sync
```

| ID | Model | Representation | Status here |
| --- | --- | --- | --- |
| B0 | Majority class | none | runs |
| B1 | Logistic regression | Morgan fingerprints | runs |
| B2 | Random forest | Morgan fingerprints | runs |
| B3 | Gradient boosting | fingerprints ± descriptors | runs (LightGBM / XGBoost) |
| B4 | Message-passing network | molecular graph | runs (PyTorch, CPU) |
| B5 | Pretrained encoder | SMILES | **unavailable** — weights unreachable |

**No benchmark has been run on real CPDB data.** Every result in this
repository is stamped `SYNTHETIC_FIXTURE` and is not a carcinogenicity finding.
Set the provenance marker before a real run:

```bash
echo "REAL_CPDB" > data/processed/DATA_PROVENANCE
```

### What the suite enforces

- **PR-AUC is read against prevalence**, never 0.5, and one definition
  (sklearn average precision) is used across every experiment.
- **Selection never touches test.** Hyperparameters *and* the operating
  threshold come from validation; the test set is scored once.
- **Intervals resample scaffold groups**, not compounds, because compounds
  sharing a scaffold are not independent.
- **Model comparison is paired** on identical held-out compounds; overlapping
  separate intervals are not a test of difference.
- **A model that cannot run is recorded, not dropped**, with its reason — a
  missing row reads as "tried and lost".
- **Ranking metrics return null on a single-class held-out set** rather than
  the 1.0 average precision would otherwise report as perfection.

### The experiment that matters

Not which model scores highest, but which keeps its score when the test
chemicals are structurally unlike the training ones. `generalization_gap`
reports random minus scaffold per model; `similarity_analysis` stratifies test
performance by nearest-neighbour Tanimoto to training, which is the
applicability-domain question.

---

# Supabase

The workspace is ephemeral; Supabase is where the dataset and results live
afterwards. Schema `carcinofusion` — deliberately not `public`, which Supabase
exposes through PostgREST where a policy-less research table would be readable
with the anon key. RLS is enabled on every table with no policies, so access is
deny-by-default.

```bash
python -m src.datasets.supabase_sync     # emits idempotent SQL
psql "$SUPABASE_DB_URL" -f data/processed/supabase/apply_all.sql
```

Nine tables: the six Phase 1 tables plus `benchmark_runs`,
`benchmark_metrics` and `benchmark_predictions`, with a
`v_benchmark_summary` view. Every statement is `ON CONFLICT DO UPDATE`, so a
re-sync updates in place.

---

## Not in this phase

Model training, feature selection beyond basic fingerprints and descriptors,
external dataset curation, and genotoxic versus non-genotoxic mechanism
labelling. The schema and the directory layout leave room for all four:
`observations` is already keyed by endpoint, and `model_datasets` is versioned
per endpoint and split strategy.
