# Dataset protocol — Phase 1

The curation protocol for the carcinogenicity dataset. It states what is
acquired, how chemical identity is established, and how the splits are made.
The label policy has its own document, `label_harmonization.md`.

This protocol is the artefact that makes the dataset defensible. A result that
cannot be reproduced from it is not a result.

---

## 1. Acquire CPDB

The Carcinogenic Potency Database is the primary experimental source. Download
the tab-separated files and their documentation, and place the **untouched**
originals in `data/raw/cpdb/`.

| File | Role | Required |
| --- | --- | --- |
| `cpdb.ncintp.tab` | NCI/NTP experimental records | yes |
| `cpdb.lit.tab` | literature experiments | yes |
| `cpdb.chemname.tab` | names, chemcodes, CAS numbers | yes |
| `cpdb.ncntdose.tab` | dose and tumour incidence, NCI/NTP | no |
| `cpdb.litdose.tab` | dose and tumour incidence, literature | no |

Then register them:

```bash
python -m src.ingestion.download_cpdb
```

This records each file's SHA-256, size and retrieval date in `manifests/`. It
writes nothing into `data/raw/`, which is read-only input for the whole
pipeline. If a required file is missing the step fails and names it.

### Verify the column mapping before the first production run

`configs/cpdb_columns.yaml` declares which source column carries which field.
The defaults reflect CPDB field names in common use, but they have not been
checked against your download. **Read the CPDB documentation and correct that
file.** The parser validates the real header against the mapping and fails,
naming the column, rather than guessing positionally or dropping a field
silently. A wrong mapping is a correctable config error; a silent one is a
corrupt dataset.

---

## 2. Inspect before committing

```bash
python -m src.ingestion.parse_cpdb
```

Produces `data/interim/carcinogenicity_experiments.parquet` — **Dataset A**,
one row per experimental record — and a manifest answering the Step 2
questions: how many records, how many unique chemicals, how many carry a
usable assessment, and the distribution of assessment values.

Do not assume the final dataset holds the number of chemicals the source
advertises. Filtering and structure resolution change the counts, and the
manifest is where the real numbers are.

### Originals are preserved

Every standardised column has a `*_raw` companion holding exactly what the
source said: `species_raw`, `sex_raw`, `author_assessment_raw`. Auditing a
conflicting result is impossible once the original value has been overwritten.

### Dose detail joins on the experiment identifier

CPDB's dose tables carry **no chemcode**. Each row is one dose group belonging
to the experiment named by `idnum`, with `order` giving its place in the dose
series. The join is therefore `dose.idnum -> experiment.idnum`: exact,
one-to-many, and needing no heuristics.

Groups are emitted in `order`, not file order, so the dose series and the
incidence series read low-to-high and line up with each other.

Verify the mapping against your own download before the first run:

```bash
python3 scripts/check_columns.py
```

It reports, per file, which canonical column matched which header, which
required ones are missing, and which columns exist in the file but are named
nowhere in the config — the last list is usually where a missing field turns
up under an unexpected name.

---

## 3. Build the chemical identity map

```bash
python -m src.chemistry.resolve_identity
```

Produces **Dataset B** (`compound_identity.parquet`), the review table
(`structure_review.parquet`) and the link table (`experiments_linked.parquet`).

### Structures come from a resolution table you supply

Create `data/external/structure_resolution.tsv`:

| Column | Required | Meaning |
| --- | --- | --- |
| `chemcode` or `cas_number` | at least one | join key back to CPDB |
| `smiles` | yes | the structure |
| `source` | yes | where it came from, e.g. PubChem |
| `retrieved_date` | recommended | when it was looked up |
| `chemical_name` | no | for human review |

The pipeline performs **no network lookup**. This is deliberate: the structure
a compound had on the day the dataset was frozen is recorded, not re-fetched
later and quietly changed. A CAS number or a name is never treated as proof of
identity; a structure without a named source is rejected.

### Standardisation

Applied in this order, version `std-v1.0.0`:

1. parse, rejecting malformed SMILES to the review table
2. metal disconnection
3. normalisation and cleanup
4. fragment handling — the largest **organic** fragment, chosen
   deterministically, with every discarded fragment recorded
5. charge normalisation
6. canonical tautomer
7. identifiers: canonical SMILES, isomeric SMILES, InChI, InChIKey, formula,
   weight, scaffold

Nothing is blindly stripped. Two specifics worth stating, because both are
easy to get wrong and expensive to discover later:

- **Stereochemistry is preserved.** RDKit's `TautomerEnumerator` strips sp3
  stereo by default (`tautomerRemoveSp3Stereo=True`). Left at the default,
  L- and D-alanine standardise to the same InChIKey and merge into one
  compound, taking both their labels with them. The pipeline overrides it.
- **Fragments are counted after metal disconnection**, so a covalently written
  salt such as `CC(=O)O[Na]` is recognised as the mixture it is.

A structure that fails any step goes to `structure_review.parquet` with its
reason. It is never discarded silently and never receives a label.

### Two levels of identity

They are not the same thing, and conflating them is how leakage gets in.

| Level | Key | Purpose |
| --- | --- | --- |
| Exact identity | full InChIKey | merge records describing the same defined structure |
| Grouping identity | `scaffold_group` | keep related compounds together across a split |

The InChIKey's second block carries the stereochemistry layer, so stereoisomers
receive **distinct** `compound_id`s. They nevertheless share a
`scaffold_group`, so they cannot be separated across a train/test boundary —
which matters because Morgan fingerprints cannot tell them apart, and a model
would simply recognise one from the other.

Compounds with no Murcko scaffold (acyclic) are grouped by their connectivity
skeleton rather than left null. A null group key is dropped by `groupby`
without a word, which would remove those compounds from the split entirely.

---

## 4. Apply the label policy

```bash
python -m src.curation.harmonize_labels
```

See `label_harmonization.md`. The short version: three states (1 / 0 / null),
null is not negative, and conflicting evidence is never resolved by majority
vote.

---

## 5. Audit

```bash
python -m src.curation.audit_conflicts
```

Writes `data/processed/audit_report.{json,md}` and
`conflicts_detail.parquet`, covering every Step 5 item: source records, unique
chemicals, resolved and failed structures, duplicates, label distribution by
source and species, and the per-compound conflict breakdown.

Conflicts are classified by the axis they disagree on:

| Type | Meaning |
| --- | --- |
| `across_species` | positive in one species, negative in another |
| `within_species_across_other_factors` | same species, differing by sex, route or organ |
| `within_species_and_sex` | the same species and sex gave both answers |

The third is the most serious: it cannot be explained by species or sex
differences, so either a study is unreliable or the curation is wrong. Read
these before freezing.

---

## 6. Build and split

```bash
python -m src.datasets.build_dataset
python -m src.datasets.split_dataset
```

### Splits are assigned over groups, never records

Every record for a compound stays with that compound. In the scaffold split,
every compound sharing a scaffold group stays together. The split is generated
once and frozen.

| Strategy | Group by | Role |
| --- | --- | --- |
| random | `compound_id` | diagnostic only, for comparison with baselines |
| scaffold | `scaffold_group` | **primary development evaluation** |
| external | `inchi_key` | final validation, held outside the process |

Target allocation is 70 / 15 / 15 **of compound groups**, not records. Groups
cannot be subdivided, so realised fractions differ; the split report states
both. Largest groups fill training first, which pushes rare chemistry into
validation and test and makes the evaluation harder on purpose.

### Leakage is verified, not assumed

Hard failures, which stop the run:

- an InChIKey appearing in more than one partition
- a scaffold group spanning partitions in the scaffold split
- a null group key, which `groupby` would drop silently

Measured and reported:

- nearest-neighbour Tanimoto similarity from each held-out compound to its
  nearest training compound — distribution, median, p90, max
- the count of held-out compounds whose fingerprint is **identical** to a
  training compound, reported separately because a low median hides them

Scaffold grouping reduces leakage. It does not guarantee chemical
dissimilarity, which is why the similarity is measured rather than assumed.

### Do not tune against the internal test set

Use validation for model selection. Reserve test for evaluation. A held-out set
consulted repeatedly during development stops being held out, because the
architecture and hyperparameter choices absorb its information.

---

## 7. Build the curation ledger

```bash
python -m src.curation.build_ledger
```

Writes `curation_ledger.parquet` and `curation_ledger_summary.json`: one row
per exclusion and per transformation, naming the entity it applied to, the
stage that applied it, and a reason code drawn from a closed vocabulary.

Runs after the split, because exclusions from the training set are only
decided at build and split time.

The information exists elsewhere — the review table, the
`training_eligibility` column, the standardisation notes, the unmapped
assessment list. Scattered, it cannot be queried: answering "why is this
compound absent from the training set" means opening four files and knowing
which to look in. The ledger makes that one filter.

| Event type | Example reason codes |
| --- | --- |
| exclusion | `structure_parse_failed`, `record_unlinked_no_compound`, `record_label_unresolved`, `compound_conflicting_evidence`, `compound_unresolved_label` |
| transformation | `fragment_discarded`, `stereocentre_unassigned`, `compounds_merged`, `value_normalized_species`, `structure_standardized` |

Value normalisations are logged once per distinct mapping with the count of
records affected, not once per record: the mapping is the transformation, and
seventy identical rows would bury the events that matter.

---

## 8. Freeze

```bash
python -m src.datasets.freeze_dataset --note "initial Phase 1 freeze"
python -m src.datasets.freeze_dataset --verify v1
```

Publishes an immutable copy under `data/processed/frozen/{version}/` with a
`DATASET_LOCK.json` recording the SHA-256 of every file and the config versions
in force. Re-freezing an existing version is refused.

Any later correction produces a **new version** with an entry in
`CHANGELOG.md`. Frozen files are never edited in place.

---

## 9. Load into PostgreSQL (optional)

```bash
python -m src.datasets.export_for_postgres
psql -d carcinofusion -f sql/schema.sql
psql -d carcinofusion -f data/processed/pg/load.sql
```

A local database is adequate; nothing here needs a hosted one. Labels are
exported as nullable integers so that an unresolved label arrives as SQL NULL
rather than as `0` or an empty string.

---

## Ordering

Each step depends on the one before. The order is not a preference.

| # | Stage | Command | Output |
| --- | --- | --- | --- |
| 1 | acquire | `src.ingestion.download_cpdb` | checksums in `manifests/` |
| 2 | parse | `src.ingestion.parse_cpdb` | Dataset A |
| 3 | identity | `src.chemistry.resolve_identity` | Dataset B, review table |
| 4 | labels | `src.curation.harmonize_labels` | record and compound labels |
| 5 | audit | `src.curation.audit_conflicts` | audit report, conflicts |
| 6 | build | `src.datasets.build_dataset` | model-ready table per endpoint |
| 7 | split | `src.datasets.split_dataset` | locked splits, split report |
| 8 | ledger | `src.curation.build_ledger` | curation ledger, summary |
| 9 | freeze | `src.datasets.freeze_dataset` | immutable Dataset v1 |

Do not start by downloading every toxicity database available. Establish that
the core dataset is usable first.

---

## Expanding beyond CPDB

Add sources in stages, keeping each source's evidence in its own table.

| Priority | Source | Use |
| --- | --- | --- |
| 1 | CPDB | primary experimental dataset |
| 2 | NTP Technical Reports | verify interpretations, study detail |
| 3 | PubChem | structures and identifiers |
| 4 | IARC Monographs | human hazard classification |
| 5 | EPA CompTox | identity resolution, supplementary data |

Two cautions that do not relax with time:

- **An IARC classification is not a rodent bioassay result.** It integrates a
  broader body of evidence and is not an interchangeable experimental label.
  Never merge it into the same label column.
- **An external dataset is not independent because it has a different URL.** It
  may contain the same chemicals, the same underlying studies, or copied
  classifications. Measure chemical overlap and, where possible, shared study
  references before calling it external validation.

Use additional sources to resolve identities and investigate disagreements.
Do not use them to overwrite CPDB labels automatically.
