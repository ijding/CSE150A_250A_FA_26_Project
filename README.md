# Predicting Song Popularity with a Bayesian Network

A CSE 150A/250A final project. We model how a song's audio features (energy, acousticness, tempo, valence, danceability, …) and release decade relate to its Spotify popularity, using a discrete Bayesian network built with [`pgmpy`](https://pgmpy.org).

**Data:** [Spotify Music Dataset](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset) (Kaggle, Solomon Ameh): about 4,500 songs after de-duplication.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1. Data | `data_processing.ipynb` | Cleans the data, discretizes the continuous features and makes the 80/20 train/test split |
| 2. Structure | `structure_learning.ipynb` | Compares a hand-designed DAG with structures learned by Hill Climb (BIC, AIC, K2, BDeu), GES, PC, Chow-Liu, Naive Bayes and TAN; covers cross-validated structure selection and bootstrap edge stability |
| 3. Parameters + exact inference | *in progress* | Dirichlet/BDeu CPT estimation with a pseudocount sweep; Variable Elimination queries |
| 4. Approximate inference | *in progress* | Gibbs sampling vs. Variable Elimination: accuracy and convergence |
| 5. Evaluation | *in progress* | Test-set accuracy, log-likelihood and confusion matrix against an always-majority baseline |

### Structure-learning findings so far

* Popularity is only weakly related to any audio feature. **Release decade** is its strongest neighbour. Beyond decade there is one small but consistent signal, carried equally well by **acousticness or energy**: the two are strongly correlated and interchangeable, and using both overfits. Danceability and duration add nothing significant.
* Hill Climb + BIC finds a 166-parameter graph with a much better BIC than our 1,031-parameter hand-designed DAG (−48,064 vs. −50,913). BIC supports only **one** parent for the 5-state popularity variable.
* Under 200 bootstrap resamples, most of our hand-drawn feature–feature edges hold up. Every v-structure we assumed is rejected, as is the energy → valence edge we proposed.

Details, figures and every configuration are in `structure_learning.ipynb`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONHASHSEED=0 jupyter lab
```

Python 3.12, with `pgmpy` pinned to **1.1.2**. Its API differs substantially from pgmpy 0.x, the version most online examples use.

## Repository layout

```
data/
  *_popularity_spotify_data.csv   raw Kaggle files (high- and low-popularity halves)
  spotify_processed.csv           cleaned, discretized dataset
  train.csv, test.csv             fixed 80/20 split
bn_structures.py                  loader for the data split, learned structures and state names
models/                           exported structures and result tables (JSON, CSV, LaTeX)
figures/<stage>/                  figures for each stage
*.ipynb                           one notebook per stage; each runs top to bottom
```

## Variables

All 10 variables are discrete. Their states are strings, exactly as they appear in the CSVs:

| Variable | States |
|---|---|
| `popularity` | `(-0.001, 20.0]`, `(20.0, 40.0]`, `(40.0, 60.0]`, `(60.0, 80.0]`, `(80.0, 100.0]` |
| `energy`, `valence`, `danceability`, `acousticness` | five equal-width bins on [0, 1], e.g. `(0.8, 1.0]` |
| `tempo` | `slow` (< 90 BPM), `mid` (90–120), `upbeat` (120–150), `fast` (> 150) |
| `duration` | `< 2 min`, `2-3 min`, `3-4 min`, `4-5 min`, `5+ min` |
| `decade` | `Before 1990`, `1990s`, `2000s`, `2010s`, `2020s` |
| `mode` | `0.0` (minor), `1.0` (major) |
| `key` | `0.0` … `11.0` (pitch class) |

## Using the learned structures

`models/structures.json` holds every structure compared in `structure_learning.ipynb`. Load one, fit it and query it like this:

```python
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination
from bn_structures import load_split, load_structure, list_structures, state_names, recommended

train, test = load_split()
model = load_structure("HC start=hand")           # list_structures() lists every option
model.fit(train, estimator=DiscreteBayesianEstimator(
    state_names=state_names(), prior_type="BDeu", equivalent_sample_size=5))

VariableElimination(model).query(["popularity"], evidence={"decade": "2020s", "acousticness": "(0.8, 1.0]"})
```

`recommended()` names the structure to use for each purpose:

| Role | Structure | Why |
|---|---|---|
| Main learned model | `HC + BIC` | Hill Climb + BIC from an empty graph |
| Highest BIC | `GES (BIC)` | Greedy equivalence search finds a slightly better optimum |
| Best predictive / feature queries | `HC start=hand` | Lowest cross-validated log-loss; popularity's Markov blanket is {decade, acousticness} |
| Prior-knowledge baseline | `Hand-designed` | The DAG from our proposal |

Pass `state_names=state_names()` when fitting. Then every CPT covers the full state space, and evidence on any state is valid.

## Reproducibility

* When two moves tie on score, pgmpy's structure search picks one based on Python's string hash order. Run with **`PYTHONHASHSEED=0`** to reproduce `models/structures.json` exactly. The notebook warns if the variable is not set.
* `data/train.csv` and `data/test.csv` are the canonical split. Regenerating them would change every downstream result.
* `structure_learning.ipynb` runs top to bottom in about 3 minutes on a laptop.
