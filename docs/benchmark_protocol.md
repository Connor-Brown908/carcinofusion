# Phase 2 — benchmark protocol

How the baseline suite is run and how its numbers may be read. The protocol
was frozen in `configs/benchmark.yaml` before any model was trained; the runner
reads that file and stamps its contents onto every result.

Protocol version: **`p2-v1.0.0`**

---

## Status of the results in this repository

**No benchmark has been run against real CPDB data.** The runs present here
were produced from the synthetic Phase 1 fixture to prove the harness
executes, and every one carries `data_provenance = SYNTHETIC_FIXTURE`.

They are not carcinogenicity findings and the table below is not a result. The
numbers come from 36 invented compounds with 4 in the test partition, which is
below the 50-compound floor the protocol sets for a headline number. The runner
emits that warning on every such run.

What has been established is that the pipeline runs end to end, that the
metrics behave correctly under adversarial conditions, and that results reach
a durable store with their provenance intact.

---

## Frozen decisions

| Decision | Specification |
| --- | --- |
| Task | Binary rodent carcinogenicity classification |
| Input | Standardised molecular structure from Phase 1 |
| Primary evaluation | Scaffold-based holdout |
| Secondary evaluation | Random holdout, diagnostic only |
| Final internal test | Scored once, after selection |
| External validation | Reserved, held outside the split process |
| Primary metric | PR-AUC, always read against prevalence |
| Secondary metrics | ROC-AUC, MCC, balanced accuracy, sensitivity, specificity, precision |
| Probability quality | Brier score, reliability curve, Murphy decomposition |
| Reproducibility | Fixed seeds, versioned configs, recorded dependency versions |

### Selection discipline

Hyperparameters are chosen on validation. The operating threshold is chosen on
validation too — a threshold is a model parameter, and tuning it on held-out
data makes the held-out score optimistic in a way that is easy to miss. The
test set is scored once, after selection is complete.

When the primary metric is undefined on validation, which happens whenever a
small or imbalanced fold holds a single class, selection falls through a chain
(PR-AUC → ROC-AUC → MCC → balanced accuracy). Which metric was actually used,
and on which partition, is recorded in `selection_metric`,
`selection_partition` and `selection_fallback`. If selection ever falls back to
the training fold, that is recorded too and the run's hyperparameters should be
treated as untrustworthy: selecting on train favours whichever model overfits
hardest.

---

## The benchmark ladder

| ID | Model | Representation | Purpose |
| --- | --- | --- | --- |
| B0 | Majority class | none | Sanity reference |
| B1 | Logistic regression | Morgan fingerprints | Linear baseline |
| B2 | Random forest | Morgan fingerprints | Nonlinear, same features as B1 |
| B3 | Gradient boosting | fingerprints ± descriptors | Strong tabular baseline |
| B4 | Message-passing network | molecular graph | Learned representation |
| B5 | Pretrained encoder | SMILES | Transfer-learning baseline |

B1 and B2 share featurizer settings so the comparison isolates the estimator.
B3 runs both with and without descriptors, because whether descriptors help is
a question worth answering rather than assuming.

Chirality is on in the fingerprints and in the graph atom features. Phase 1
goes to some trouble to preserve stereochemistry, and discarding it here would
make the comparison about what each model was shown rather than about the
representation.

### B5 is unavailable in this environment

Pretrained weights cannot be fetched: the HuggingFace hub returns
`403 Forbidden` through this workspace's proxy. The adapter is implemented in
full — frozen-embedding and fine-tuning paths both — and runs wherever the
weights are reachable, or from a local directory named by
`CARCINOFUSION_PRETRAINED_DIR`.

Where it cannot run, it raises `UnavailableModel` and the suite records
`status = unavailable` with the reason. It is not dropped from the table: a
silently missing row invites the reader to assume the model was tried and lost.

Pretraining is not assumed to help. On a few thousand compounds a pretrained
encoder frequently loses to gradient boosting on fingerprints, and reporting
that honestly is part of why it is run.

---

## Metrics

### PR-AUC is read against prevalence

Its baseline is the positive rate, not 0.5. A PR-AUC of 0.30 is poor at 25%
prevalence and good at 5%. Every table reports prevalence beside it, and B0
exists to make the floor visible: the majority classifier's PR-AUC *is* the
prevalence.

One definition is used everywhere: scikit-learn's `average_precision_score`.
That is average precision, which is related to but not numerically identical to
the trapezoidal area under a PR curve. Mixing the two across experiments would
make models look different when only the definition changed.

### Ranking metrics are null on a single-class set

`average_precision_score` returns 1.0 when every held-out compound is positive,
which reads as a perfect model rather than an unanswerable question. Both
PR-AUC and ROC-AUC are nulled in that case and the evaluation carries a note.

### Intervals resample scaffold groups

Bootstrap percentile intervals, 2000 resamples, resampling **whole scaffold
groups** rather than individual compounds. Compounds sharing a scaffold are not
independent; resampling them separately produces intervals that are too narrow.
Every interval records its resampling unit in `ci_method`.

### Comparison is paired

Two models are compared on the identical held-out compounds, bootstrapping the
difference. Overlapping separate intervals do not mean a difference is
insignificant, and unpaired comparison confounds the difference with which
compounds each model happened to be scored on. The test is whether the paired
difference interval excludes zero.

---

## The experiment that matters

Not which model scores highest, but which keeps its performance when the test
chemicals are structurally unlike the training ones.

`generalization_gap` reports random minus scaffold per model. A model that does
well on random and poorly on scaffold has learned the chemical distribution
rather than the structural determinants of the endpoint, and will not hold up
on new chemistry.

This is reinforced by `similarity_analysis`, which stratifies test performance
by nearest-neighbour Tanimoto to the training set. A model whose accuracy
collapses in the low-similarity bin has a narrow applicability domain whatever
its headline number — and under OECD guidance a prediction is only interpretable
inside a stated domain.

`fraction_identical_fp` in the Phase 1 split report is the companion check: a
held-out compound whose fingerprint is identical to a training compound is
memorisable even though no InChIKey is shared. Stereoisomers do this, because
Morgan fingerprints cannot distinguish them.

---

## Reproduction versus standardised comparison

A published number is only *reproduced* when the dataset, endpoint and
evaluation protocol are demonstrably comparable to the original. Where the
original code or split is unavailable, the honest description is a new,
standardised comparison, and the difference is stated rather than glossed.

There is a further distinction worth keeping: reproducing a published method is
not the same as evaluating *this implementation* of that method. Everything in
`src/models/` is a reimplementation. Where a published result is cited for
contrast, the citation says which of the two it is.

---

## Running it

```bash
# Phase 1 must have produced splits first
python -m src.evaluation.runner --endpoint primary
python -m src.evaluation.analysis --endpoint primary
python -m src.datasets.supabase_sync
```

Before a real run, write the provenance marker so results are not stamped
synthetic:

```bash
echo "REAL_CPDB" > data/processed/DATA_PROVENANCE
```

Useful flags: `--models B0 B1 B3` to restrict the ladder, `--splits scaffold`
to restrict the strategies, `--provenance REAL_CPDB` to set it explicitly.

---

## What would justify moving to Phase 3

1. A reproducible implementation of the baselines — **done**, 60 tests.
2. Random and scaffold results for each feasible model — harness done, awaiting
   real data.
3. Confidence intervals and class-specific metrics — **done**.
4. False positive and false negative analysis — **done** (`error_detail`,
   `error_summary`).
5. Probability calibration — **done** (reliability curves, Brier decomposition).
6. Chemical-similarity analysis of test compounds — **done**
   (`similarity_analysis`).
7. Documented comparison with published results where conditions are
   comparable — **outstanding**, and only possible once real numbers exist.

Items 2 and 7 need the CPDB data. The rest are built and tested.

A neural network is not required to win. If gradient boosting beats the graph
model, gradient boosting is the baseline CarcinoFusion must beat in Phase 3,
and that result should be reported rather than explained away.
