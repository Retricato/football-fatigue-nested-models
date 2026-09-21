# Data

## What is in this folder

Two derived files, both produced by the scripts in `src/`.

| File | Rows | What it is |
|---|---|---|
| `real/soccermon_gps_subset.csv` | 4,998 | The analysis sample. One row per player-day, 35 players, all features and all four label variants. This is the file the paper's results come from. |
| `real/gps_sessions.csv` | — | Per-session external load aggregated from the raw GPS traces by `src/10_gps_aggregate.py`. |

The intermediate files (`soccermon_daily.csv`, `soccermon_cohort.csv`,
`soccermon_model_ready*.csv`) are not committed. They are large and fully
reproducible from the scripts. Run `04` through `10` in order to regenerate
them.

The raw archive is not committed either. It is about 93 GB, most of which is
six billion GPS position measurements.

## Getting the source data

The underlying dataset is **SoccerMon**, openly available from Zenodo:

> https://doi.org/10.5281/zenodo.10033832

`src/04_fetch_soccermon.py` downloads and unpacks it into `data/soccermon/`.

## Required attribution

SoccerMon is licensed **CC BY 4.0**. Attribution is a condition of use, not a
courtesy. Cite both the data descriptor and the dataset record:

```bibtex
@article{midoglu2024soccermon,
  author  = {Midoglu, Cise and Kj{\ae}reng Winther, Andreas and Boeker,
             Matthias and Dahl Pettersen, Susann and Pedersen, Sigurd and
             Ragab, Nourhan and Kupka, Tomas and Hicks, Steven A. and
             Bredsgaard Randers, Morten and Jain, Ramesh and Dagenborg,
             H{\aa}vard J. and Pettersen, Svein Arne and Johansen, Dag and
             Riegler, Michael A. and Halvorsen, P{\aa}l},
  title   = {A large-scale multivariate soccer athlete health, performance,
             and position monitoring dataset},
  journal = {Scientific Data},
  volume  = {11}, pages = {558}, year = {2024},
  doi     = {10.1038/s41597-024-03386-x}
}

@misc{midoglu2022soccermondata,
  author    = {Midoglu, Cise and Boeker, Matthias and Kj{\ae}reng Winther,
               Andreas and Pettersen, Svein Arne and Johansen, Dag and
               Riegler, Michael and Halvorsen, P{\aa}l and Hicks, Steven},
  title     = {{SoccerMon}: A Large-Scale Multivariate Soccer Athlete Health,
               Performance, and Position Monitoring Dataset},
  year      = {2022},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.10033832}
}
```

## Consent and ethics

Redistributing the derived files here is permitted, and the basis is worth
stating plainly rather than assuming.

The dataset authors report that all players gave written informed consent
**including the open publication of the recorded data**. Player metadata were
removed before release and files renamed with randomly generated strings. The
study was approved by the Norwegian Data Protection Authority under reference
296155, and was exempted from further ethical approval because collection did
not involve a biobank, medical or health data related to illness, or
interference with normal operations.

The `player_id` values in these files are the anonymized identifiers assigned
by the dataset authors. No new data were collected for this work and no
further ethical approval was required.

## A note on the label

The `y` column is `label_A_hooper_1.0sd`, already shifted forward by one day.
Features on row *t* predict the outcome on day *t+1*.

The three other label columns are the sensitivity variants: a stricter
1.5 SD threshold, a looser 0.5 SD threshold, and a bottom-tertile readiness
label. The readiness label uses each player's full-season distribution, so it
is descriptive only and is not deployable prospectively. Do not present it as
an operational rule.
