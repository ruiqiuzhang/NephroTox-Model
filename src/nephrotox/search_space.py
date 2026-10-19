"""Reproducible search designs, generated without labels or model fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[2]


def load_search_design():
    return json.loads((ROOT / "configs/model_search_spaces.json").read_text(encoding="utf-8"))


def _plain(value):
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f"Unsupported search value: {type(value).__name__}")


def _key(params):
    return json.dumps(params,sort_keys=True,separators=(",",":"),allow_nan=False,default=_plain)


def graph_candidate_configurations(architecture):
    """Return reference plus random candidates, including training defaults.

    The reference is candidate 0, with no preferential score or tie privilege
    beyond the declared first-candidate tie rule. Sampling is independent of
    labels, validation scores and test data and uses the same seed each time.
    """
    search = load_search_design()
    original = json.loads((ROOT / "configs/original_parameter_design.json").read_text(encoding="utf-8"))
    if search["graph_method"] != "seeded_random_search" or not search["include_original_reference_candidate"]:
        raise ValueError("Graph search requires reference plus seeded random candidates")
    budget = search["graph_candidates_per_architecture"]
    if type(budget) is not int or budget < 2:
        raise ValueError("Graph hyperparameter search requires at least two candidates")
    space = search["graph_spaces"][architecture]
    reference = original["graph_reference_configurations"][architecture]
    fixed = space["fixed_parameters"]
    dimensions = space["parameters"]
    if set(reference) != set(fixed) | set(dimensions) or set(fixed) & set(dimensions):
        raise ValueError("Search dimensions must account for every original network/optimizer parameter exactly once")
    if any(reference[name] != value for name,value in fixed.items()):
        raise ValueError("Reference parameters differ from declared architectural constants")
    distributions = {}
    for name,spec in dimensions.items():
        if spec["type"] == "categorical":
            values = spec["values"]
            if not values or reference[name] not in values or len({_key(v) for v in values}) != len(values):
                raise ValueError(f"Invalid categorical graph search dimension: {name}")
            distributions[name] = values
        elif spec["type"] == "loguniform":
            low,high = float(spec["low"]),float(spec["high"])
            if not 0 < low < high or not low <= reference[name] <= high:
                raise ValueError(f"Invalid positive log-uniform graph search dimension: {name}")
            distributions[name] = loguniform(low,high)
        else:
            raise ValueError(f"Unsupported search distribution: {spec['type']}")
    defaults = original["implementation_training_defaults"]["graph"]
    candidates = [dict(defaults,**reference)]
    seen = {_key(candidates[0])}
    for sampled in ParameterSampler(distributions,n_iter=budget*4,random_state=int(search["random_state"])):
        params = json.loads(_key(dict(defaults,**fixed,**sampled)))
        key = _key(params)
        if key not in seen:
            candidates.append(params)
            seen.add(key)
        if len(candidates) == budget:
            return candidates
    raise RuntimeError("Insufficient distinct sampled configurations for the declared graph search budget")
