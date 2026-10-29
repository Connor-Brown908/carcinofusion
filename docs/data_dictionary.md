# Data dictionary

Every column produced by the pipeline. Columns ending `_raw` hold the source's
original value and are never overwritten.

---

## Dataset A — `carcinogenicity_experiments.parquet`

One row per experimental record.

| Column | Type | Description |
| --- | --- | --- |
| `experiment_id` | str | internal identifier, `EXP0000001` |
| `source_database` | str | `CPDB:NCI/NTP` or `CPDB:literature` |
| `source_record_id` | str | `filename:rownumber`, traces to the exact source line |
| `chemcode` | str | CPDB chemical code, the documented join key |
| `chemical_name` | str | name from the chemname table |
| `cas_number` | str | CAS Registry Number where available |
| `species` | str | standardised: rat, mouse, hamster |
| `species_raw` | str | **original** source value |
| `sex` | str | standardised: male, female, both |
| `sex_raw` | str | **original** source value |
| `strain` | str | animal strain |
| `exposure_route` | str | gavage, diet, inhalation, drinking water |
| `duration` | str | study duration as recorded |
| `dose_duration` | str | exposure duration as recorded |
| `target_organ` | str | reported tumour site |
| `tumor_endpoint` | str | reported tumour type |
| `author_assessment` | str | the assessment used for labelling |
| `author_assessment_raw` | str | **original** source value |
| `td50` | float | potency estimate, null when absent or non-numeric |
| `tumor_incidence` | str | `tumour/treated` per dose group, `;`-separated |
| `dose_response` | str | dose levels, `;`-separated |
| `reference` | str | study citation |
| `notes` | str | source notes |
| `curation_status` | str | `parsed` until reviewed |
| `compound_id` | str | added by `resolve_identity`; null when unresolved |

---

## Dataset B — `compound_identity.parquet`

One row per standardised compound.

| Column | Type | Description |
| --- | --- | --- |
| `compound_id` | str | `CMP` + 12 hex chars, derived from the full InChIKey |
| `canonical_smiles` | str | canonical SMILES, **without** stereochemistry |
| `isomeric_smiles` | str | canonical SMILES **with** stereochemistry |
| `inchi_key` | str | full InChIKey; the second block carries the stereo layer |
| `connectivity_key` | str | first InChIKey block; shared by all stereoisomers |
| `scaffold_smiles` | str | Bemis–Murcko scaffold; null for acyclic compounds |
| `scaffold_group` | str | split grouping key; `acyclic:<connectivity>` when no scaffold. **Never null** |
| `molecular_formula` | str | |
| `molecular_weight` | float | |
| `n_stereocenters` | int | chiral centres found, assigned or not |
| `n_unassigned_stereocenters` | int | **> 0 means the source structure is underspecified** |
| `n_fragments` | int | fragments after metal disconnection |
| `structure_status` | str | see below |
| `standardization_version` | str | `std-v1.0.0` |
| `input_smiles` | str | the SMILES as supplied, before standardisation |
| `resolution_source` | str | where the structure came from |
| `resolution_retrieved_date` | str | when it was looked up |
| `resolution_matched_on` | str | `chemcode` or `cas_number` |
| `source_chemcodes` | str | `;`-separated source chemicals that merged here |
| `source_names` | str | `;`-separated source names |
| `n_source_chemicals` | int | **> 1 means a merge happened; review before freezing** |
| `standardization_notes` | str | discarded fragments, unassigned stereo, etc. |
| `transformations` | str | JSON list of `{type, detail}`, the structured form of the notes; consumed by the curation ledger |

### `structure_status`

| Value | Meaning | Usable |
| --- | --- | --- |
| `standardized` | clean single-fragment input | yes |
| `standardized_from_mixture` | salt or solvate resolved to its organic parent | yes |
| `parse_failed` | RDKit could not parse the SMILES | no |
| `no_structure` | no SMILES supplied | no |
| `no_organic_fragment` | purely inorganic | no |
| `standardize_failed` | a standardisation step raised | no |

Only the first two reach the dataset. The rest go to `structure_review.parquet`.

---

## `structure_review.parquet`

Source chemicals with no usable structure. Retained for manual review, never
labelled.

| Column | Description |
| --- | --- |
| `chemcode`, `cas_number`, `chemical_name` | source identity |
| `input_smiles` | what was attempted, if anything |
| `structure_status` | why it failed |
| `reason` | human-readable detail |
| `resolution_source` | where the structure came from |

---

## `observations_labeled.parquet`

Dataset A plus one label column per endpoint.

| Column | Type | Description |
| --- | --- | --- |
| `label_primary` | 1 / 0 / null | conservative endpoint |
| `label_sensitivity` | 1 / 0 / null | some-evidence-as-positive endpoint |
| `label_policy_version` | str | policy that produced these labels |

---

## `compound_labels.parquet`

Compound-level aggregation. Columns are suffixed by endpoint.

| Column | Type | Description |
| --- | --- | --- |
| `compound_id` | str | |
| `label_{endpoint}` | 1 / 0 / null | **null is unresolved, not negative** |
| `label_confidence_{endpoint}` | str | high / medium / low / none |
| `is_conflicted_{endpoint}` | bool | positive and negative evidence both present |
| `n_positive_{endpoint}` | int | eligible positive records |
| `n_negative_{endpoint}` | int | eligible negative records |
| `n_indeterminate_{endpoint}` | int | eligible null records |
| `aggregation_reason_{endpoint}` | str | why this label was assigned |
| `n_experiments` | int | all records for the compound |
| `n_eligible_experiments` | int | records in the endpoint's eligible species |
| `n_species` | int | distinct eligible species |

---

## `model_dataset_{endpoint}.parquet`

| Column | Description |
| --- | --- |
| structure columns | as Dataset B |
| `label`, `label_confidence`, `is_conflicted` | endpoint-specific, unsuffixed |
| `n_positive`, `n_negative`, `n_indeterminate` | evidence counts |
| `aggregation_reason` | why this label was assigned |
| `endpoint_name` | `primary` or `sensitivity` |
| `training_eligibility` | `eligible`, `excluded_conflicting_evidence`, `excluded_unresolved_label` |
| descriptors | with `--with-descriptors`: `logp`, `tpsa`, `h_donors`, … |

---

## `splits_{endpoint}.parquet`

| Column | Description |
| --- | --- |
| all model-dataset columns | |
| `split_strategy` | `random` or `scaffold` |
| `split_group` | the group that moved as a unit |
| `split_assignment` | `train`, `validation`, `test`, `external`, `excluded` |
| `split_version` | from `configs/splits.yaml` |

Every compound appears exactly once per strategy. Ineligible compounds carry
`split_assignment = excluded` rather than being dropped.

---

## `conflicts_detail.parquet`

One row per compound whose evidence disagrees with itself.

| Column | Description |
| --- | --- |
| `compound_id`, `endpoint` | |
| `n_positive`, `n_negative`, `n_indeterminate` | evidence counts |
| `positive_{axis}` / `negative_{axis}` | values on each side, for species, sex, exposure_route, target_organ |
| `{axis}_overlap` | values appearing on **both** sides — the sharpest signal |
| `conflict_type` | `across_species`, `within_species_across_other_factors`, `within_species_and_sex` |
| `conflicting_axes` | which axes overlap |

---

## `split_report_{endpoint}.json`

| Field | Description |
| --- | --- |
| `compounds_eligible` / `compounds_partitioned` | before and after removing external |
| `strategies.{s}.n_groups` | groups the split moved |
| `strategies.{s}.largest_group_size` | a very large group constrains the allocation |
| `strategies.{s}.realised_fractions` | actual, which differs from target since groups are indivisible |
| `strategies.{s}.partitions` | compound, positive and negative counts |
| `strategies.{s}.leakage_checks` | spanning InChIKeys and scaffold groups; both must be 0 |
| `strategies.{s}.nearest_neighbour_similarity_to_train` | median, mean, p90, max, `fraction_above_0.7`, `fraction_identical_fp` |

`fraction_identical_fp` above zero means held-out compounds are fingerprint-
identical to training compounds — memorisable regardless of what the median
says. Usually stereoisomers or tautomers the fingerprint cannot distinguish.

---

## `curation_ledger.parquet`

One row per exclusion or transformation. The answer to "why is this compound
not in the training set", in one filter.

| Column | Description |
| --- | --- |
| `event_id` | `EVT000001` |
| `event_type` | `exclusion` or `transformation` |
| `stage` | the stage that applied it, e.g. `03_resolve_identity` |
| `entity_type` | `source_chemical`, `compound`, `experiment_record`, `field` |
| `entity_id` | the chemcode, compound id, experiment id or field name |
| `reason_code` | closed vocabulary; see below |
| `detail` | human-readable specifics |
| `endpoint_name` | set when the event is endpoint-specific |
| `dataset_version`, `label_policy_version`, `standardization_version` | versions in force |

The ledger carries no per-row timestamp. When it was built is recorded once,
in the stage manifest and the summary; stamping a wall-clock value on every
row would make the file differ between two runs of identical inputs and break
byte-level reproducibility.

### Exclusion reason codes

| Code | Meaning |
| --- | --- |
| `structure_parse_failed` | RDKit could not parse the SMILES |
| `structure_no_organic_fragment` | purely inorganic |
| `structure_standardize_failed` | a standardisation step raised |
| `structure_unresolved_no_structure` | no structure in the resolution table |
| `structure_unresolved_empty_smiles` | the resolution table gave no SMILES |
| `record_unlinked_no_compound` | the record's chemical has no resolved structure |
| `record_label_unresolved` | the assessment is null under this endpoint |
| `compound_conflicting_evidence` | positive and negative evidence both present |
| `compound_unresolved_label` | no resolvable eligible evidence |
| `compound_in_external_set` | held out for final validation |

### Transformation reason codes

| Code | Meaning |
| --- | --- |
| `fragment_discarded` | a salt, solvate or counter-ion was removed |
| `stereocentre_unassigned` | the source structure is underspecified |
| `structure_standardized` | the stored structure differs from the input |
| `compounds_merged` | several source chemicals gave one structure |
| `value_normalized_species` / `value_normalized_sex` | one event per distinct mapping, with the count of records affected |
| `cas_filled_from_identity_table` | a missing CAS was filled from the chemname table |
| `td50_non_numeric_nulled` | a TD50 value was present but not numeric |
| `assessment_unmapped_to_null` | the assessment matched no policy rule |

## `curation_ledger_summary.json`

| Field | Description |
| --- | --- |
| `total_events` | rows in the ledger |
| `exclusions.by_reason` / `by_entity_type` | counts |
| `transformations.by_reason` | counts |
| `reason_code_vocabulary` | the closed vocabulary, both event types |

---

## Manifests — `manifests/{stage}_{timestamp}.json`

Every stage writes one, plus a `{stage}_latest.json` pointer.

| Field | Description |
| --- | --- |
| `stage`, `started_at`, `finished_at` | |
| `inputs` / `outputs` | paths with SHA-256, sizes, row counts |
| `config_versions` | dataset, label policy, split, seed |
| `counts` | the stage's own statistics |
| `warnings` | everything the stage flagged |
| `environment` | Python, platform, RDKit version |

---

## `DATASET_LOCK.json`

Written at freeze. Records `dataset_version`, `frozen_at`, the policy and split
versions, the random seed, the environment, and the SHA-256 of every published
file. `--verify` re-checks them and reports drift.
