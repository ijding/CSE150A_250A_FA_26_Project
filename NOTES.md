# Team notes

Internal working notes. The public overview, setup steps and usage examples are in `README.md`.

**Final report due Mon 11/23, 11:59 PM. There is no grace period.**

## Status

| Stage | Owner | Notebook (suggested names for ⏳ rows) | Report section | Status |
|---|---|---|---|---|
| 1. Data cleaning, discretization, split | Irving | `data_processing.ipynb` | §2 | ✅ done, split frozen |
| 2. Structure learning | Zachary | `structure_learning.ipynb` | §3 (structure) | ✅ done; LaTeX draft in `report/` |
| 3. Parameter learning + Variable Elimination | Ethan | `parameter_learning.ipynb` | §3 (CPTs, VE) | ⏳ |
| 4. Gibbs sampling + convergence | Andrew | `approximate_inference.ipynb` | §3 (Gibbs), §4 (convergence) | ⏳ |
| 5. Evaluation, results, report integration | Jeslyn | `evaluation.ipynb` | §1, §4, §5, §7 | ⏳ |

When you finish, update your row and add a handoff section below.

## Conventions

* **One notebook per stage.** It must run top to bottom with *Restart & Run All* and no manual steps. Launch Jupyter with `PYTHONHASHSEED=0 jupyter lab`.
* **Load data with `bn_structures.load_split()`.** Don't read the CSVs directly; the loader makes sure everyone sees identical string states with the ID columns dropped.
* **Train only on `train.csv`.** Tune with cross-validation inside train. `test.csv` is for final evaluation only.
* **Never rerun the split cell in `data_processing.ipynb`.** The split was made without a fixed seed, so rerunning it would change every result.
* **Fix seeds** (`seed=`, `random_state=`) everywhere.
* **Outputs:** put report figures in `figures/<your-stage>/` and anything another stage reads in `models/`. Make report figures at their printed size (5.5 in wide for the full text width), so fonts aren't shrunk to illegible sizes.

## pgmpy 1.1 gotchas

Most tutorials and AI answers use the pre-1.0 API. Everything below was checked against pgmpy 1.1.2.

1. **Fitting CPTs:** use `model.fit(data, estimator=DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5, state_names=...))` from `pgmpy.parameter_estimator`. Use `DiscreteMLE(state_names=...)` for plain MLE. The old form, `model.fit(data, estimator=BayesianEstimator, prior_type=...)`, raises `TypeError`.
2. **Always pass `state_names=state_names()`.** Without it, a CPT only covers states seen in its training data, and evidence on any other state raises a `KeyError`.
3. **`GibbsSampling` has no evidence argument.** `GibbsSampling(model).sample(size=n, seed=s)` samples from the *prior* joint only, and it returns **integer state indices**, not labels. Index `i` means `state_names()[var][i]`. On our network, *constructing* `GibbsSampling(model)` took about 4 minutes (sampling 2,000 draws then took about 2 s), so build it once and reuse it. For P(popularity | evidence), write a short sampler that clamps the evidence and resamples each other variable from its Markov-blanket conditional. `BayesianModelSampling(model).rejection_sample(evidence=[State(var, value)], size=n, seed=s)` (with `from pgmpy.factors.discrete import State`) does accept evidence and returns labels. It came within about 0.01 of VE on P(popularity | decade=2020s) with 3,000 samples.
4. **`HillClimbSearch.estimate(start_dag=g)` modifies `g` in place.** Pass `g.copy()`.
5. **Custom structure scores** must be instances from `pgmpy.estimators`, for example `BDeu(data, equivalent_sample_size=10, state_names=...)`. Instances from `pgmpy.structure_score` are rejected by `HillClimbSearch`.
6. pgmpy prints many `FutureWarning`s. Call `warnings.filterwarnings("ignore")` *before* importing pgmpy to silence them.

## Handoffs

### Structure learning (Zachary)

**Produced:**
* `models/structures.json`: 24 structures, the full state space (`_states`) and recommended structures for each role (`_recommended`). Load them with `bn_structures`.
* `models/structure_comparison.csv`: edges, parameter count, train BIC, 5-fold CV popularity log-loss and accuracy, and popularity's Markov blanket for every structure.
* `models/paired_cv_vs_decade_only.csv`, `models/second_parent_cv.csv` and `models/bootstrap_edge_stability.csv`.
* `figures/structure/*.png`. `report_dags.png` is sized for the report; the rest are for the notebook or an appendix.
* **Report material:** `report/section3_structure.tex` (draft text) and `models/structure_table.tex` (the table it includes).

* Popularity is only weakly predictable from audio features. **Decade** dominates. Beyond decade there is one small, consistent signal that **acousticness or energy** carry interchangeably: each beats decade-only on all 5 CV folds, and using both overfits. The structure searches happened to pick acousticness, but energy works just as well (`models/second_parent_cv.csv`). Danceability and duration add nothing significant.
* With fixed add-one smoothing, the learned structures reach about 46–47% CV accuracy on the training set, against 33% for always predicting the majority class. These are *structure-selection* numbers, not the official results.

**→ Ethan (parameter learning + VE):**
* Sweep the pseudocount on at least `HC + BIC`, `HC start=hand` and `Hand-designed`, and compare against `DiscreteMLE`.
* Run "which features shift popularity" queries on **`HC start=hand`** or **`Hand-designed`**. In `HC + BIC`, audio-feature evidence has no effect once decade is known. An acousticness effect should be read as "acousticness/energy", since the two are interchangeable.
* My notebook fits CPTs with fixed add-one smoothing only to compare structures. Your fitted CPTs replace it.
* Save your chosen pseudocount, the query list and the VE answers to `models/`, so Andrew can compare against them.

**→ Andrew (Gibbs + convergence):**
* Use the **same structure, pseudocount and queries as Ethan**, so the Gibbs vs. VE comparison is like-for-like.
* Read gotcha 3 before starting.
* VE is instant on this network, which makes it a reliable ground truth for error-vs-samples plots.

**→ Jeslyn (evaluation + report):**
* Evaluate the structures in `recommended()` plus `Naive Bayes`, using Ethan's tuned CPTs. Report accuracy, test log-likelihood and confusion matrix on `test.csv`.
* Nothing in my stage was computed on `test.csv`; every number in `models/` comes from cross-validation within train. So your test results are the first and only look at the test set.
* **Always-medium baseline:** popularity has 5 *equal-width* (not equal-count) bins. The most common bin, (40, 60], holds 34.4% of train, so always predicting it scores about a third. That figure comes from the bin counts. It only happens to match the 1/3 you'd get by construction with 3 equal-count bins. Say this in the report.
* The data was assembled from separate high-popularity and low-popularity files. That is worth listing as a limitation in §5.

### Zachary: drafts to edit for §6

* **Contribution (draft):** "Zachary designed the hand-built network structure and ran the structure-learning experiments: Hill Climb with BIC, AIC, K2 and BDeu, plus GES, PC, Chow-Liu, Naive Bayes and TAN. He compared these structures with cross-validation and bootstrap stability analysis, and wrote the structure-learning part of Section 3."
* **AI use (draft):** "Used Claude Code to help write and debug the structure-learning notebook, generate figures and draft text, and run an automated code review. All results were checked by rerunning the notebook, and the claims in the text were verified against its outputs."
