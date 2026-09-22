# Does wellness self-report earn its place? A nested-model decomposition of next-day fatigue prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22876422.svg)](https://doi.org/10.5281/zenodo.22876422)

Code and results for a study of what each class of athlete monitoring data
adds over a persistence baseline, using the open
[SoccerMon](https://doi.org/10.5281/zenodo.10033832) dataset from two elite
Norwegian women's first-division teams.

## The finding in one table

Four models, each a strict superset of the one before, fitted on identical
rows and identical folds. Values are the change in PR-AUC from adding that
block, with 95% confidence intervals bootstrapped over players.

| Step | Random forest | XGBoost |
|---|---|---|
| M0 → M1, add wellness self-report | +0.017 (+0.004, +0.031) | +0.009 (−0.007, +0.027) |
| **M1 → M2, add internal training load** | **+0.072 (+0.039, +0.111)** | **+0.068 (+0.044, +0.094)** |
| M2 → M3, add GPS external load | +0.016 (−0.005, +0.042) | +0.021 (+0.002, +0.041) |
| M0 → M3, total | +0.106 (+0.059, +0.160) | +0.098 (+0.058, +0.138) |

Three things follow.

**Persistence alone is close to worthless.** Carrying yesterday's Hooper
Index forward reaches a ROC-AUC of 0.519 and a PR-AUC of 0.205 against a
prevalence of 0.187.

**The wellness questionnaire adds little on top of it.** Every self-reported
item together moved PR-AUC by 0.009 to 0.017, and the two model families
disagree on whether that interval excludes zero.

**Internal training load is the block that carries the signal.** Session RPE
and its derived quantities moved PR-AUC by 0.068 to 0.072, with
P(Δ > 0) = 1.000 in both families. It is the cheapest of the three
instruments and it was worth four times the questionnaire.

The models also over-predict risk at every decile, and their Brier score
(0.157) is worse than that of a constant predictor returning the base rate
(0.139). They rank athletes. They do not measure them.

## Why a persistence baseline matters

Subjective fatigue is strongly autocorrelated, and the outcome here is
derived from the same questionnaire that supplies the inputs. Any model
receiving yesterday's wellness report inherits that autocorrelation for free.

Feature importance cannot separate the two. A ranking that places yesterday's
fatigue rating at the top tells you which input the model leans on, not what
the other inputs are worth once it is present. Only a nested comparison
against a persistence model answers that, and persistence baselines are
rarely reported in this literature.

The same applies to accuracy. At a prevalence of 0.187, predicting "not
fatigued" every single day scores 81.3% accuracy without any model at all.

## Repository layout

```
src/            the pipeline, 14 scripts, run in numeric order
data/real/      the derived analysis sample (see data/README.md)
results/        metrics and bootstrap output as JSON and CSV
figures/        the three figures reported below
```

A manuscript based on these results is in preparation. This repository will
be updated with the citation and the LaTeX source once it is published.

## Running it

```bash
pip install -r requirements.txt
cd src
python 11_incremental_value.py     # untuned nested blocks
python 13_nested_tuning.py         # nested CV with tuning inside each fold
```

Both write to `results/`. The random seed is fixed at 42 throughout, so runs
are reproducible.

`13_nested_tuning.py` caches each outer fold to `results/tuning_cache/`, so an
interrupted run resumes where it stopped rather than starting over. Delete
that directory to force a clean re-run.

Scripts `04` through `10` rebuild the analysis sample from the raw SoccerMon
archive. You only need them if you want to regenerate the data from source.
See `data/README.md`.

### The scripts

| Script | What it does |
|---|---|
| `01_data_generator.py` | Synthetic cohort used during development |
| `02_eda_visualization.py` | Exploratory plots, synthetic |
| `02b_eda_real.py` | Exploratory plots, real data |
| `03_model_training.py` | First-pass models, synthetic |
| `04_fetch_soccermon.py` | Download and unpack the raw archive |
| `05_real_data_audit.py` | Completeness and compliance audit |
| `06_build_real_dataset.py` | Reshape to one row per player-day |
| `07_labels.py` | Label construction and the sensitivity variants |
| `08_train_real.py` | Model pipelines and the three validation schemes |
| `09_transfer_analysis.py` | Synthetic-to-real transfer (not reported here) |
| `10_gps_aggregate.py` | GPS traces to per-session external load |
| `11_incremental_value.py` | Nested feature blocks, untuned, bootstrap CIs |
| `12_recovery_dynamics.py` | Recovery after high-load sessions (not reported here) |
| `13_nested_tuning.py` | **Nested CV with tuning inside each training fold** |

`13` is the one the headline table above comes from. `11` is its untuned
counterpart, and the difference between them matters: untuned, the M2 to M3
(GPS) step was +0.033 with an interval excluding zero. Tuning improved M1 and
M2 more than it improved M3, so the gap narrowed and the interval now
includes zero for random forest. The tuned estimate is the one reported.

## Method, briefly

**Outcome.** Elevated Hooper Index on day *t+1*, defined as exceeding that
player's own trailing 28-day mean by more than one standard deviation. The
window is trailing and shifted, so no same-day or future information enters
the baseline. Within-player thresholding is deliberate: a squad-wide cutoff
would mostly encode who reports pessimistically.

**Sample.** 4,998 player-days from 35 players, being the intersection on
which wellness, session load and GPS records all exist. The four models must
be compared on identical rows, or a difference in sample becomes
indistinguishable from a difference in feature set.

**Validation.** Nested cross-validation. The outer loop is a five-fold
stratified group split on player identity and produces the reported estimate.
Inside each outer training partition, an independent three-fold group split
selects hyperparameters. The outer test fold is never seen by the search.

**Uncertainty.** 1,000 paired bootstrap resamples over *players*, not rows.
A row-level bootstrap would treat 143 days from one athlete as 143
independent observations and produce intervals far too narrow.

## Citing this

If you use this code, please cite both the repository and the underlying
dataset.

**This repository:**

> Kokate, S. (2026). *Nested-model decomposition of next-day fatigue
> prediction in elite women's football* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.22876422

```bibtex
@software{kokate2026nested,
  author    = {Kokate, Siddharth},
  title     = {Nested-model decomposition of next-day fatigue prediction
               in elite women's football},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22876422},
  url       = {https://github.com/Retricato/football-fatigue-nested-models}
}
```

**The SoccerMon dataset** must be cited separately. The CC BY 4.0 licence
requires it. See `data/README.md` for both entries.

## Licence

Code is MIT, see `LICENSE`.

The SoccerMon dataset is © its authors and licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The derived files
in `data/real/` are covered by that licence and carry the required
attribution in `data/README.md`. They are not covered by the MIT licence.
