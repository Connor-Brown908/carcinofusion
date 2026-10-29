# Label harmonisation

How an experimental assessment becomes a training label. This is the most
consequential decision in the project: everything downstream inherits it, and
a model can only be as good as the labels are honest.

The policy lives in `configs/labels.yaml`. No module hard-codes an
interpretation of any assessment string. Changing a mapping requires a version
bump, which produces a new dataset version with a change log.

Current policy version: **`v1.1.0`**

---

## The endpoint

> Evidence of carcinogenic activity in long-term rodent bioassays, determined
> from a predefined and reproducible interpretation of experimental records.

This is an **experimental-evidence** endpoint. It is not a claim about human
carcinogenicity, and the documentation should never let it drift into one.
Predicting rodent bioassay outcomes from structure is the stated task;
predicting human carcinogenicity is a different, harder claim that rodent
labels alone cannot support.

---

## Three label states

| State | Meaning | Treatment |
| --- | --- | --- |
| `1` | positive under the stated criteria | positive example |
| `0` | negative under the stated criteria | negative example |
| `null` | equivocal, insufficient, conflicting or unresolved | excluded from binary loss, retained for analysis |

**`null` is not `0`.** A chemical never tested in a sufficiently informative
long-term bioassay is unresolved, not safe. Recording it as negative would
teach the model that untested chemistry is benign, which is both false and the
most dangerous error this endpoint can make.

This holds at every layer. A missing assessment survives as `null` through
parquet, through aggregation, and into PostgreSQL as SQL `NULL` — never as `0`
and never as an empty string.

---

## Two endpoints, deliberately

Running both measures how far the results depend on the label definition
rather than the chemistry. Report both.

| Assessment | `primary` | `sensitivity` |
| --- | --- | --- |
| Clear evidence of carcinogenic activity | 1 | 1 |
| Some evidence of carcinogenic activity | null | 1 |
| Equivocal evidence | null | null |
| No evidence of carcinogenic activity | 0 | 0 |
| Inadequate study | null | null |
| anything unlisted | null | null |

`primary` is the conservative task: clearly positive against clearly negative.
`sensitivity` folds some-evidence results into the positive class. If a
headline result moves substantially between the two, the finding is about the
label policy, not about carcinogenicity, and should be reported that way.

### Unrecognised assessments

An assessment matching no rule becomes `null` and is **reported by name** in
the stage manifest under `unmapped_assessment_values`. It is never defaulted to
negative. Review that list after the first real run: it is how you discover
that the source uses wording the policy has not accounted for.

Missing values are distinguished from unrecognised ones. A value that
round-trips through parquet as float `NaN` would otherwise be reported as an
unmapped rule literally called `"nan"`, sending a curator to look for a policy
gap that does not exist.

---

## Compound-level aggregation

A chemical may be positive in one species, sex, route or organ and negative in
another. That is a scientific finding, not noise to be smoothed away.

### Conflicting evidence is never resolved by majority vote

| Evidence | Label | Conflicted |
| --- | --- | --- |
| all eligible records positive | 1 | no |
| all eligible records negative, at least `min_negative_records` of them | 0 | no |
| positive records, rest null | 1 | no |
| at least one positive **and** at least one negative | `null` | **yes** |
| only null records | `null` | no |
| no eligible record for the endpoint's species | `null` | no |

One positive against nine negatives is **conflicted**, not negative. A majority
vote would discard the single positive finding, which is exactly the finding a
carcinogenicity assessment most needs to see.

### Negatives need corroboration; positives do not

`min_negative_records` is **2** from policy v1.1.0. A compound with exactly one
negative record and no positives is labelled `null`, not `0`.

The asymmetry is deliberate. One positive bioassay establishes that a compound
*can* induce tumours. One negative establishes only that this particular study
did not detect them, which is a weaker claim — and replicate rodent bioassays
agree at roughly 81% balanced accuracy, with only about 69% of substances found
toxic being found toxic again on repeat. A lone negative is close to a
coin-flip's worth of evidence that a compound is clean.

It also follows from the context of use. Supporting an ICH S1B(R1) waiver means
being trusted to say a compound is **not** a carcinogen, which makes negative
predictive value the governing metric. A training set whose negatives rest on
single unreplicated studies cannot support that claim, whatever the headline
accuracy says.

The cost is real and worth watching. Compounds with exactly one negative record
move from `0` to `null`, which shrinks the negative class and raises apparent
prevalence — so PR-AUC must be read against the new prevalence, not compared
across policy versions. The audit report quantifies the shift, and the curation
ledger records each affected compound under `compound_unresolved_label`.

To go back to the looser rule, set `min_negative_records: 1` and bump the
policy version; do not edit a frozen dataset.

Conflicted compounds are excluded from the primary binary training set and
retained in the database in full, with their evidence counts intact, available
for multi-task and evidence-integration models later.

### Species eligibility

The headline endpoint uses rat and mouse records. Other species stay in the
database and take no part in the label: a hamster negative cannot create a
conflict with a rat positive, and cannot by itself produce a label.

### Confidence

| Band | Criterion |
| --- | --- |
| high | three or more agreeing eligible records |
| medium | two agreeing eligible records |
| low | a single eligible record (positives only; a lone negative is unresolved) |
| none | unresolved or conflicted |

Recorded per compound so downstream work can weight or filter without
re-deriving the evidence.

---

## What is retained

Nothing is dropped to make the dataset tidier.

- Records with no resolvable structure stay in `observations_labeled.parquet`
  and have no `compound_id`.
- Structures that failed standardisation stay in `structure_review.parquet`
  with their reason.
- Conflicted compounds stay in the model-ready table with
  `training_eligibility = excluded_conflicting_evidence`.
- Unresolved compounds stay with `training_eligibility =
  excluded_unresolved_label`.

Every compound is accounted for at every stage. The test suite asserts this:
the set of compounds entering the splitter equals the set leaving it.

---

## Changing the policy

1. Edit `configs/labels.yaml`.
2. Bump `label_policy_version`.
3. Bump `dataset_version` in `configs/data.yaml`.
4. Re-run from `harmonize_labels` onward.
5. Freeze, which appends to `CHANGELOG.md`.

Never edit a frozen dataset in place. The version is what makes a published
result reproducible, and a frozen file that quietly changed makes every result
derived from it unverifiable.
