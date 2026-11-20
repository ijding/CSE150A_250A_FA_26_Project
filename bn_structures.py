"""Shared loader for the Bayesian-network structures learned in structure_learning.ipynb.

Usage (pgmpy 1.1.x):
    from pgmpy.parameter_estimator import DiscreteBayesianEstimator
    from pgmpy.inference import VariableElimination
    from bn_structures import load_split, load_structure, list_structures, state_names, recommended

    train, test = load_split()
    model = load_structure("HC + BIC")   # DiscreteBayesianNetwork with no CPTs yet
    model.fit(train, estimator=DiscreteBayesianEstimator(state_names=state_names(),
                                                         prior_type="BDeu", equivalent_sample_size=5))
    VariableElimination(model).query(["popularity"], evidence={"decade": "2020s"})

Always pass state_names=state_names() when fitting. Then every CPT covers every state in the
dataset (train + test), and evidence on a state that never occurs in train still works.

Run the notebook with PYTHONHASHSEED=0 to reproduce models/structures.json exactly.
"""
import json
from pathlib import Path

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork

ROOT = Path(__file__).resolve().parent
STRUCTURES_PATH = ROOT / "models" / "structures.json"
ID_COLS = ["name", "artist"]


def _load():
    with open(STRUCTURES_PATH) as f:
        return json.load(f)


def list_structures():
    """Names of all exported structures."""
    return [k for k in _load() if not k.startswith("_")]


def recommended():
    """Which structures the structure-learning stage recommends for each role."""
    return _load()["_recommended"]


def state_names():
    """{variable: [states]} over the full dataset (train + test); pass to the CPT estimator."""
    return _load()["_states"]


def load_structure(name):
    """Return an unfitted pgmpy DiscreteBayesianNetwork with every variable as a node,
    including isolated ones, so that inference accepts evidence on any variable."""
    structures = _load()
    if name not in structures or name.startswith("_"):
        options = "\n  ".join(list_structures())
        raise ValueError(f"unknown structure {name!r}. Choose one of:\n  {options}")
    spec = structures[name]
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(spec["nodes"])
    model.add_edges_from([tuple(e) for e in spec["edges"]])
    return model


def load_split():
    """train/test DataFrames with identifier columns dropped and every value as a string state."""
    train = pd.read_csv(ROOT / "data" / "train.csv", index_col=0).drop(columns=ID_COLS).astype(str)
    test = pd.read_csv(ROOT / "data" / "test.csv", index_col=0).drop(columns=ID_COLS).astype(str)
    return train.reset_index(drop=True), test.reset_index(drop=True)
