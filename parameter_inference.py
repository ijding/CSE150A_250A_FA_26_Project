"""Parameter fitting and exact popularity inference for the frozen project DAGs.

Run parameter_learning.ipynb to tune CPTs and regenerate the team artifacts.
No function in this module loads or selects parameters on the test split.
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

from bn_structures import load_structure, state_names

TARGET = "popularity"
SEED = 150
ESS_VALUES = (1, 5, 10, 20, 50, 100, 500, 1000)
REQUIRED_STRUCTURES = ("HC + BIC", "HC start=hand", "Hand-designed")


def validate_model(model):
    """Check all CPTs, including their state order and column normalization."""
    if not model.check_model():
        raise ValueError("pgmpy model check failed")
    states = state_names()
    rows = []
    for node in model.nodes():
        cpd = model.get_cpds(node)
        values = np.asarray(cpd.get_values(), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid probabilities in {node}")
        error = float(np.max(np.abs(values.sum(axis=0) - 1)))
        if error > 1e-10:
            raise ValueError(f"Unnormalized CPT for {node}")
        for variable in cpd.variables:
            if list(cpd.state_names[variable]) != states[variable]:
                raise ValueError(f"Unexpected state order for {variable}")
        rows.append({"variable": node, "parent_configurations": values.shape[1],
                     "zero_entries": int((values == 0).sum()),
                     "max_column_sum_error": error})
    return pd.DataFrame(rows)


def fit_parameters(data, structure, estimator="BDeu", equivalent_sample_size=5):
    """Fit a fresh exported DAG with pgmpy 1.1 estimators and fixed states.

    BDeu puts ESS / (r * q) pseudocounts in each cell, where r is the
    child's cardinality and q is the number of parent configurations.
    ESS is a total prior strength per node, not an add-alpha per cell.
    """
    states = state_names()
    model = load_structure(structure)
    if set(data.columns) != set(model.nodes()):
        raise ValueError("Training columns must match the exported model nodes")
    for variable in data:
        if not data[variable].isin(states[variable]).all():
            raise ValueError(f"Unknown or missing training state in {variable}")
    if estimator == "MLE":
        if equivalent_sample_size is not None:
            raise ValueError("Use equivalent_sample_size=None for MLE")
        fitter = DiscreteMLE(state_names=states, n_jobs=1)
    elif estimator == "BDeu":
        if equivalent_sample_size is None or equivalent_sample_size <= 0:
            raise ValueError("BDeu needs a positive equivalent sample size")
        fitter = DiscreteBayesianEstimator(
            state_names=states, prior_type="BDeu",
            equivalent_sample_size=equivalent_sample_size, n_jobs=1)
    else:
        raise ValueError("estimator must be 'MLE' or 'BDeu'")
    model.fit(data, estimator=fitter)
    validate_model(model)
    return model


def popularity_log_probabilities(model, features):
    """Exact log P(popularity | all other nodes), vectorized over songs.

    Enumerate the five target states and multiply all CPT factors in log
    space. This is exact when every other node is observed, and avoids
    thousands of separate VE calls during CV. Partial evidence uses VE.
    Including every factor also detects zero-probability MLE evidence.
    Undefined rows are returned as NaN, never silently replaced by a prior.
    """
    required = set(model.nodes()) - {TARGET}
    if set(features.columns) != required:
        raise ValueError("Provide every non-popularity node, without the target")
    states = model.get_cpds(TARGET).state_names[TARGET]
    scores = np.zeros((len(features), len(states)))
    for k, target_state in enumerate(states):
        for cpd in model.get_cpds():
            indices = []
            for variable in cpd.variables:
                labels = cpd.state_names[variable]
                if variable == TARGET:
                    indices.append(np.full(len(features), labels.index(target_state)))
                else:
                    codes = pd.Categorical(features[variable], categories=labels).codes
                    if (codes < 0).any():
                        raise ValueError(f"Unknown or missing state in {variable}")
                    indices.append(codes)
            with np.errstate(divide="ignore"):
                scores[:, k] += np.log(cpd.values[tuple(indices)])
    normalizer = logsumexp(scores, axis=1)
    result = np.full_like(scores, np.nan)
    valid = np.isfinite(normalizer)
    result[valid] = scores[valid] - normalizer[valid, None]
    return result


def validation_metrics(model, validation):
    """Unclipped log loss is the selection metric; zeros remain visible."""
    lp = popularity_log_probabilities(model, validation.drop(columns=TARGET))
    labels = model.get_cpds(TARGET).state_names[TARGET]
    y = pd.Categorical(validation[TARGET], categories=labels).codes
    if (y < 0).any():
        raise ValueError("Unknown validation target state")
    valid = ~np.isnan(lp).any(axis=1)
    true_lp = lp[np.arange(len(y)), y]
    zero_true = valid & np.isneginf(true_lp)
    # An impossible evidence row invalidates the candidate for selection.
    log_loss = float(-true_lp.mean()) if valid.all() else float("inf")
    accuracy = float((lp.argmax(axis=1) == y).mean()) if valid.all() else float("nan")
    clipped = float(-np.maximum(true_lp, np.log(1e-15)).mean()) if valid.all() else float("inf")
    return {"log_loss": log_loss, "log_loss_clipped_1e-15": clipped,
            "accuracy": accuracy, "zero_true_probability_rows": int(zero_true.sum()),
            "undefined_evidence_rows": int((~valid).sum())}


def parameter_sweep(train, structures=REQUIRED_STRUCTURES, ess_values=ESS_VALUES,
                    n_splits=5, seed=SEED):
    """Compare fixed exported DAGs using identical stratified training folds.

    This tunes parameters conditional on frozen graphs, some of which were
    learned on all of train. These scores are not a nested estimate of the
    complete structure-search pipeline's generalization performance.
    """
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(train, train[TARGET]))
    rows = []
    for structure in structures:
        for estimator, ess in [("MLE", None)] + [("BDeu", e) for e in ess_values]:
            for fold, (fit_index, validation_index) in enumerate(folds, start=1):
                model = fit_parameters(train.iloc[fit_index], structure, estimator, ess)
                rows.append({"structure": structure, "estimator": estimator,
                             "equivalent_sample_size": ess, "fold": fold,
                             "fit_rows": len(fit_index), "validation_rows": len(validation_index),
                             **validation_metrics(model, train.iloc[validation_index])})
    fold_scores = pd.DataFrame(rows)
    summary_rows = []
    keys = ["structure", "estimator", "equivalent_sample_size"]
    for key, group in fold_scores.groupby(keys, dropna=False, sort=False):
        loss = group.log_loss.to_numpy()
        summary_rows.append(dict(zip(keys, key)) | {
            "cv_log_loss": float(loss.mean()),
            "cv_log_loss_sd": float(loss.std(ddof=1)) if np.isfinite(loss).all() else np.nan,
            "cv_log_loss_clipped_1e-15": float(group["log_loss_clipped_1e-15"].mean()),
            "cv_accuracy": float(group.accuracy.mean()) if group.accuracy.notna().all() else np.nan,
            "cv_accuracy_sd": float(group.accuracy.std(ddof=1)) if group.accuracy.notna().all() else np.nan,
            "zero_true_probability_rows": int(group.zero_true_probability_rows.sum()),
            "undefined_evidence_rows": int(group.undefined_evidence_rows.sum()),
            "n_folds": n_splits})
    summary = pd.DataFrame(summary_rows).sort_values(
        ["cv_log_loss", "structure", "estimator", "equivalent_sample_size"], kind="stable")
    return summary.reset_index(drop=True), fold_scores


def configuration_from_row(row):
    """Make a JSON-safe, directly reusable fit configuration."""
    ess = row["equivalent_sample_size"]
    return {"structure": row["structure"], "estimator": row["estimator"],
            "equivalent_sample_size": None if pd.isna(ess) else float(ess),
            "cv_log_loss": float(row["cv_log_loss"]),
            "cv_accuracy": float(row["cv_accuracy"])}


def exact_popularity(model, evidence=None, inference=None):
    """Return a labeled Series P(popularity | evidence) using VE.

    Evidence may be empty or partial, must use the fitted string labels,
    and must not contain popularity. Reuse an inference engine for batches.
    """
    evidence = {} if evidence is None else dict(evidence)
    if TARGET in evidence:
        raise ValueError("Popularity is the query target, not an evidence feature")
    for variable, value in evidence.items():
        if variable not in model.nodes():
            raise ValueError(f"Unknown evidence variable {variable!r}")
        labels = model.get_cpds(variable).state_names[variable]
        if value not in labels:
            raise ValueError(f"Invalid state {value!r} for {variable}; choose from {labels}")
    engine = inference if inference is not None else VariableElimination(model)
    if engine.model is not model:
        raise ValueError("Inference engine belongs to a different model")
    # MLE may assign zero mass even to evidence pruned away by a query.
    # Reduce all factors and eliminate hidden nodes to check its support.
    if evidence and any((cpd.values == 0).any() for cpd in model.get_cpds()):
        from pgmpy.factors import factor_product

        factors = [cpd.to_factor() for cpd in model.get_cpds()]
        for factor in factors:
            factor.reduce([(v, evidence[v]) for v in factor.variables if v in evidence])
        hidden = set(model.nodes()) - set(evidence)
        while hidden:
            def cost(variable):
                scope = set().union(*(set(f.variables) for f in factors if variable in f.variables))
                return np.prod([len(model.get_cpds(v).state_names[v]) for v in scope])

            variable = min(sorted(hidden), key=cost)
            bucket = [f for f in factors if variable in f.variables]
            factors = [f for f in factors if variable not in f.variables]
            combined = factor_product(*bucket)
            combined.marginalize([variable])
            factors.append(combined)
            hidden.remove(variable)
        mass = float(factor_product(*factors).values)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("Evidence has zero probability under this fitted model")
    result = engine.query([TARGET], evidence=evidence, show_progress=False)
    posterior = pd.Series(result.values, index=result.state_names[TARGET], name="probability")
    posterior = posterior.reindex(model.get_cpds(TARGET).state_names[TARGET])
    if (not np.isfinite(posterior).all() or (posterior < 0).any()
            or not np.isclose(posterior.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("VE returned an invalid distribution; evidence may be impossible")
    return posterior


def make_queries(states=None):
    """Deterministic query list using the repository's actual state labels."""
    states = state_names() if states is None else states
    queries = [{"query_id": "prior", "experiment": "prior", "evidence": {}}]
    for i, decade in enumerate(states["decade"]):
        base = {"decade": decade}
        queries.append({"query_id": f"decade_{i}", "experiment": "decade", "evidence": base})
        for feature in ["energy", "acousticness"]:
            for j, category in enumerate(states[feature]):
                queries.append({"query_id": f"decade_{i}_{feature}_{j}",
                                "experiment": feature, "evidence": base | {feature: category}})
        for e, a in product([0, -1], repeat=2):
            queries.append({"query_id": f"decade_{i}_joint_{e % len(states['energy'])}_{a % len(states['acousticness'])}",
                            "experiment": "energy_acousticness",
                            "evidence": base | {"energy": states["energy"][e],
                                                 "acousticness": states["acousticness"][a]}})
    for feature in ["energy", "acousticness"]:
        for j in [0, len(states[feature]) - 1]:
            queries.append({"query_id": f"partial_{feature}_{j}", "experiment": "partial",
                            "evidence": {feature: states[feature][j]}})
    return queries


def run_queries(model, queries, model_id, training_data=None):
    """Long-format probabilities; JSON evidence can be passed to a sampler.

    Training support counts flag rare/unobserved evidence, not test support.
    Each (model_id, query_id) group contains one row per popularity state.
    """
    engine = VariableElimination(model)
    rows = []
    for query in queries:
        evidence = query["evidence"]
        posterior = exact_popularity(model, evidence, inference=engine)
        support = None
        if training_data is not None:
            mask = np.ones(len(training_data), dtype=bool)
            for variable, value in evidence.items():
                mask &= training_data[variable].eq(value).to_numpy()
            support = int(mask.sum())
        for state, probability in posterior.items():
            rows.append({"model_id": model_id, "query_id": query["query_id"],
                         "experiment": query["experiment"],
                         "evidence_json": json.dumps(evidence, sort_keys=True),
                         "train_evidence_count": support, "popularity_state": state,
                         "probability": float(probability)})
    return pd.DataFrame(rows)


def save_fitted_model(model, path):
    """Portable JSON with explicit factor axis order; no pickle required."""
    validate_model(model)
    payload = {"format": "spotify-discrete-bn-v1", "nodes": list(model.nodes()),
               "edges": [list(e) for e in sorted(model.edges())], "cpds": []}
    for cpd in model.get_cpds():
        payload["cpds"].append({"variables": list(cpd.variables),
                                "cardinality": [int(c) for c in cpd.cardinality],
                                "state_names": cpd.state_names,
                                "values": cpd.get_values().tolist()})
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_fitted_model(path):
    """Restore a trained pgmpy network, preserving every string state label."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["format"] != "spotify-discrete-bn-v1":
        raise ValueError("Unsupported fitted-model format")
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(payload["nodes"])
    model.add_edges_from(payload["edges"])
    for spec in payload["cpds"]:
        variables, cardinality = spec["variables"], spec["cardinality"]
        model.add_cpds(TabularCPD(
            variables[0], cardinality[0], spec["values"],
            evidence=variables[1:] or None, evidence_card=cardinality[1:] or None,
            state_names=spec["state_names"]))
    validate_model(model)
    return model
