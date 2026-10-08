#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generates fuzz cases for the TypeScript port: random data and random settings, run through
LolliPop's real `deconvolute` command, with what the port should reproduce.

Each case is a scenario — a tally table and the configuration files LolliPop reads — drawn at
random from one of these sources:

- synthetic: a random panel of lineages along a random tree, random prevalence curves at a few
  locations, and binomial read counts with dropouts, outliers and deletions;
- preprint: a random slice (location, time window, positions, variants) of the tally table of the
  LolliPop preprint (`preprint/data/tallymut_line_full.tsv.zst`, from Git LFS);
- wasap: a random slice of a snapshot of W-ASAP data (Zürich, 2026);
- auto_no_loc: a random slice of LolliPop's own test tally (no location column);

and random settings: variant renaming, variants lists, `to_drop`, start and end dates, deletions,
`no_loc`, periods with their own variants (`var_dates`), mutation filters, and the deconvolution
configuration (a preset, or random kernel, bandwidth, `min_tol`, regressor, loss, `f_scale`,
Wald confidence intervals, bootstrap rounds).

For each case, this records (as gzipped JSON):

- the tally as LolliPop reads it (after `pd.read_csv`), and the configuration;
- `expected`: the output of `lollipop deconvolute` (or the error it raises), run in-process with
  NumPy's global generator seeded, so that the bootstrap is reproducible;
- `expectedTight`: the same with the robust regression solved to the minimum (SciPy's
  `least_squares` with tight tolerances and the exact Jacobian), for the robust regressor;
- `draws`: the indices every bootstrap round drew;
- `calls`: a sample of the regressions (design, observations, kernel values) with LolliPop's fit,
  the minimum, and the confidence band, to check each step on its own;
- `ties`: the dates where some fit couldn't tell some variants apart (see `record_ties`), and the
  condition of the Fisher information of Wald intervals;
- `precision`: per date, how precisely the minimum of the robust regression is determined (see
  `minimum_precision`);
- `losses`, `tightLosses`: the loss of every regression, in order (`tightLosses`: solved to the
  minimum; for NNLS, the actual residual norm where SciPy's is wrong, see `check_nnls`);
- `variantOrder`, `locationOrder`: the order of the variants in LolliPop's design matrix, and of
  the locations (both come from a Python set, unless the configuration gives them).

Usage (from `typescript/`, see `fuzz/run.mjs`):

    fuzz/.venv/bin/python fuzz/generate.py --seed 1 --count 50 --out fuzz/out/1
"""

import argparse
import contextlib
import copy
import gzip
import io
import json
import multiprocessing
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import ruamel.yaml
import scipy
import scipy.optimize

import lollipop as ll
from lollipop.cli.deconvolute import deconvolute as deconvolute_command

# (the package exports the command under the module's name)
deconvolute_module = sys.modules["lollipop.cli.deconvolute"]

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"

LINEAGES = [
    "B.1.1.7",
    "B.1.351",
    "P.1",
    "B.1.617.2",
    "BA.1",
    "BA.2",
    "BA.4",
    "BA.5",
    "BQ.1.1",
    "XBB.1.5",
    "EG.5",
    "JN.1",
    "KP.2",
    "KP.3",
    "LP.8.1",
    "XEC",
    "XFG",
    "NB.1.8.1",
]
LOCATIONS = [
    "Zürich (ZH)",
    "Genève (GE)",
    "Lugano (TI)",
    "Chur (GR)",
    "Laupen (BE)",
    "Basel (BS)",
]
LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]

PRESETS = {
    path.stem: ruamel.yaml.YAML(typ="safe").load(path.read_text())
    for path in sorted((REPO / "presets").glob("*.yaml"))
}

MAX_ROWS = 6000
MAX_RECORDED_CALLS = 6
MAX_RECORDED_ROWS = 2500


# --- scenarios ---------------------------------------------------------------------------------


def choice(rng, values):
    return values[rng.integers(len(values))]


def random_dates(rng, n_locations):
    start = np.datetime64("2021-01-01") + np.timedelta64(
        int(rng.integers(0, 365 * 5)), "D"
    )
    span = int(rng.choice([3, 10, 30, 60, 120]))
    step = int(rng.choice([1, 1, 2, 3, 7]))
    dates = {}
    for location in range(n_locations):
        days = np.arange(0, span + 1, step)
        days = days[rng.random(days.size) > 0.15]
        if days.size == 0:
            days = np.array([0])
        jitter = rng.integers(0, 2, days.size) if step > 1 else 0
        days = np.unique(days + jitter)
        dates[location] = [start + np.timedelta64(int(d), "D") for d in days]
    return dates


def synthetic(rng):
    """A random panel along a random tree, random prevalences, binomial counts."""
    # (one variant: every mutation is uninformative, and LolliPop has nothing to fit)
    n_variants = 1 if rng.random() < 0.03 else int(rng.integers(2, 7))
    names = list(rng.choice(LINEAGES, n_variants, replace=False))
    codes = names if rng.random() < 0.6 else [f"v{i}" for i in range(n_variants)]

    # Signatures along a random tree: mutations accumulate from the root to the tips.
    n_positions = int(rng.integers(4, 80))
    positions = np.sort(rng.choice(np.arange(100, 29800), n_positions, replace=False))
    bases = rng.choice(list("ACGT"), n_positions)
    bases[rng.random(n_positions) < 0.05] = "-"
    parents, carried = [-1], [
        set(rng.choice(n_positions, rng.integers(0, 3), replace=False).tolist())
    ]
    for i in range(1, n_variants):
        parent = int(rng.integers(0, i))
        new = set(
            rng.choice(
                n_positions, rng.integers(0, max(2, n_positions // 3)), replace=False
            ).tolist()
        )
        parents.append(parent)
        carried.append(carried[parent] | new)
    degenerate = None
    if n_variants > 1 and rng.random() < 0.08:
        # two variants that can't be told apart, or a variant with nothing of its own
        i, j = rng.choice(n_variants, 2, replace=False)
        carried[i] = set(carried[j]) if rng.random() < 0.5 else set()
        degenerate = "identical or empty signature"

    n_locations = int(rng.integers(1, 4))
    locations = list(rng.choice(LOCATIONS, n_locations, replace=False))
    dates = random_dates(rng, n_locations)

    rows = []
    for li, location in enumerate(locations):
        logits = rng.normal(0, 2, n_variants + 1)
        drift = rng.normal(0, 0.15, n_variants + 1)
        start = dates[li][0]
        samples_per_date = 2 if rng.random() < 0.15 else 1
        for date in dates[li]:
            t = (date - start).astype(int)
            weights = np.exp(logits + drift * t + rng.normal(0, 0.2, n_variants + 1))
            prevalence = weights / weights.sum()
            for s in range(samples_per_date if rng.random() < 0.5 else 1):
                sample = f"S{li}_{str(date)}_{s}"
                for p in range(n_positions):
                    true = sum(
                        prevalence[v] for v in range(n_variants) if p in carried[v]
                    )
                    cov = int(rng.lognormal(5, 1.5)) if rng.random() > 0.05 else 0
                    count = (
                        int(rng.binomial(cov, np.clip(true * 0.98 + 0.01, 0, 1)))
                        if cov > 0
                        else 0
                    )
                    frac = count / cov if cov > 0 else np.nan
                    if cov > 0 and rng.random() < 0.02:
                        frac = float(rng.random())  # an outlier
                    row = {
                        "sample": sample,
                        "batch": "b1",
                        "proto": "v3" if rng.random() < 0.9 else "v4",
                        "pos": int(positions[p]),
                        "base": str(bases[p]),
                        "date": str(date),
                        "location": location,
                        "cov": cov,
                        "var": count,
                        "frac": frac,
                    }
                    for v in range(n_variants):
                        if p in carried[v]:
                            n_carriers = sum(p in c for c in carried)
                            row[codes[v]] = (
                                "shared"
                                if n_carriers > 1 and rng.random() < 0.5
                                else choice(
                                    rng,
                                    ["mut", "mut", "mut", "extra", "subset", "revert"],
                                )
                            )
                        else:
                            row[codes[v]] = np.nan
                    rows.append(row)
    tally = pd.DataFrame(rows)
    if rng.random() < 0.15:
        tally["om1"] = np.where(rng.random(len(tally)) < 0.3, "mut", None)
    return {
        "source": "synthetic",
        "tally": tally,
        "names": names,
        "codes": codes,
        "notes": degenerate,
    }


_preprint = None


def preprint(rng, path):
    """A random slice of the preprint's tally table."""
    global _preprint
    if _preprint is None:
        _preprint = pd.read_csv(path, sep="\t", dtype={"location_code": "str"})
    df = _preprint
    location = choice(
        rng,
        [
            "Zürich (ZH)",
            "Genève (GE)",
            "Lugano (TI)",
            "Chur (GR)",
            "Laupen (BE)",
            "Altenrhein (SG)",
            "Lausanne (VD)",
            "Kanton Zürich",
        ],
    )
    df = df[df["location"] == location]
    dates = np.sort(df["date"].dropna().unique())
    start = int(rng.integers(0, max(1, len(dates) - 5)))
    span = int(rng.choice([4, 10, 25, 40]))
    window = set(dates[start : start + span])
    df = df[df["date"].isin(window) | (df["date"].isna() & (rng.random() < 0.5))]
    positions = df["pos"].unique()
    keep = rng.choice(
        positions, min(len(positions), int(rng.integers(10, 150))), replace=False
    )
    df = df[df["pos"].isin(keep)]
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=int(rng.integers(2**31))).sort_index()
    pangolin = {
        "al": "B.1.1.7",
        "be": "B.1.351",
        "ga": "P.1",
        "de": "B.1.617.2",
        "ka": "B.1.617.1",
        "om1": "BA.1",
        "om2": "BA.2",
        "om4": "BA.4",
        "om5": "BA.5",
    }
    present = [c for c in pangolin if c in df.columns and df[c].notna().any()]
    if len(present) < 1:
        present = ["al", "de"]
    used = list(
        rng.choice(
            present,
            int(rng.integers(min(2, len(present)), len(present) + 1)),
            replace=False,
        )
    )
    dropped = [pangolin[c] for c in pangolin if c not in used]
    return {
        "source": "preprint",
        "tally": df.reset_index(drop=True),
        "names": [pangolin[c] for c in used],
        "codes": used,
        "pangolin": pangolin,
        "not_reported": dropped + ["om2121", "om275"],
        "notes": f"{location}, {len(window)} dates, {len(keep)} positions",
    }


def wasap(rng):
    """A random slice of the W-ASAP snapshot."""
    data = json.loads((DATA / "wasap_zurich_2026.json").read_text())
    lineages = list(data["signatures"])
    names = list(
        rng.choice(lineages, int(rng.integers(2, len(lineages) + 1)), replace=False)
    )
    frequencies = pd.DataFrame(data["frequencies"])
    dates = np.sort(frequencies["date"].unique())
    start = int(rng.integers(0, max(1, len(dates) - 3)))
    window = dates[start : start + int(rng.choice([3, 8, 20, 60]))]
    frequencies = frequencies[frequencies["date"].isin(window)]
    mutations = frequencies["mutation"].unique()
    keep = rng.choice(
        mutations, min(len(mutations), int(rng.integers(5, 41))), replace=False
    )
    frequencies = frequencies[frequencies["mutation"].isin(keep)]
    rows = [
        {
            "sample": o.sampleId,
            "pos": int(o.mutation[1:-1]),
            "base": o.mutation[-1],
            "date": o.date,
            "location": "Zürich (ZH)",
            "frac": o.frac,
            **{
                name: "mut" if o.mutation in data["signatures"][name] else np.nan
                for name in names
            },
        }
        for o in frequencies.itertuples()
    ]
    return {
        "source": "wasap",
        "tally": pd.DataFrame(rows),
        "names": names,
        "codes": names,
        "notes": None,
    }


def auto_no_loc(rng):
    """A random slice of LolliPop's own test tally."""
    df = pd.read_csv(REPO / "tests/test_auto_no_loc/tallymut.tsv", sep="\t")
    positions = df["pos"].unique()
    keep = rng.choice(
        positions, min(len(positions), int(rng.integers(5, 60))), replace=False
    )
    df = df[df["pos"].isin(keep)]
    dates = np.sort(df["date"].unique())
    start = int(rng.integers(0, max(1, len(dates) - 3)))
    df = df[df["date"].isin(dates[start : start + int(rng.choice([3, 10, 40]))])]
    names = ["KP.2", "KP.3", "LP.8"]
    return {
        "source": "auto_no_loc",
        "tally": df.reset_index(drop=True),
        "names": names,
        "codes": names,
        "notes": None,
    }


# --- settings ----------------------------------------------------------------------------------


def random_filters(rng, tally):
    filters = {}
    for i in range(int(rng.integers(1, 4))):
        kind = choice(
            rng,
            ["amplicon", "mutations", "date", "proto", "sample", "base", "location"],
        )
        if kind == "amplicon" and "pos" in tally:
            low = int(rng.choice(tally["pos"]))
            filters[f"amplicon{i}"] = [
                "proto v3",
                f"pos >= {low}",
                f"pos <= {low + int(rng.integers(0, 400))}",
            ]
        elif kind == "mutations" and "pos" in tally and "base" in tally:
            picked = tally.sample(
                min(len(tally), 3), random_state=int(rng.integers(2**31))
            )
            names = [f"{p}{b}" for p, b in zip(picked["pos"], picked["base"])]
            names += ["-" + n for n in names if rng.random() < 0.5]
            filters[f"bad_mutations{i}"] = [f"mutations in [ {', '.join(names)} ]"]
        elif kind == "date" and "date" in tally and tally["date"].notna().any():
            date = choice(rng, list(tally["date"].dropna().unique()))
            filters[f"date{i}"] = [
                f"date {choice(rng, ['>', '<', '>=', '<=', '=='])} {date}"
            ]
        elif kind == "proto" and "proto" in tally:
            filters[f"proto{i}"] = [
                choice(
                    rng, ["proto v4", "proto == v4", "proto != v3", "proto in [v4, v5]"]
                )
            ]
        elif kind == "sample" and "sample" in tally:
            filters[f"sample{i}"] = [
                choice(
                    rng,
                    [
                        "sample ~ /_1$/",
                        "sample ~ /^S0/",
                        "sample !~ /^S1/",
                        "sample =~ @_0@",
                    ],
                )
            ]
        elif kind == "base" and "base" in tally:
            filters[f"base{i}"] = [
                f"base == {choice(rng, list('ACGT'))}",
                f"frac {choice(rng, ['<', '>'])} 0.{int(rng.integers(1, 10))}",
            ]
        elif (
            kind == "location"
            and "location" in tally
            and tally["location"].notna().any()
        ):
            filters[f"location{i}"] = [
                f"location == '{choice(rng, list(tally['location'].dropna().unique()))}'",
                "frac > 0.9",
            ]
    return filters or None


def random_deconv_config(rng, n_dates):
    if rng.random() < 0.3:
        name = choice(rng, sorted(PRESETS))
        config = copy.deepcopy(PRESETS[name])
        if config.get("bootstrap", 0) > 1:
            config["bootstrap"] = int(rng.integers(2, 12))
        if config.get("confint") == "wald":
            # LolliPop fails without an explicit scale (KeyError: 'scale'), see the README
            config.setdefault("confint_params", {}).setdefault("scale", "linear")
        return f"preset {name}", config
    config = {}
    if rng.random() < 0.7:
        config["kernel"] = choice(rng, ["gaussian", "box"])
    bandwidth = (
        float(np.exp(rng.uniform(np.log(0.5), np.log(80))))
        if rng.random() > 0.05
        else 1e-17
    )
    config["kernel_params"] = {
        "bandwidth": round(bandwidth, 3) if bandwidth > 1e-3 else bandwidth
    }
    regressor = choice(rng, ["nnls", "robust", "robust"])
    config["regressor"] = regressor
    if regressor == "robust":
        params = {"f_scale": float(choice(rng, [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]))}
        if rng.random() < 0.4:
            params["loss_type"] = choice(rng, LOSSES)
        config["regressor_params"] = params
    deconv_params = {}
    if rng.random() < 0.8:
        deconv_params["min_tol"] = float(choice(rng, [1e-10, 1e-3, 1e-3, 1e-2, 0.1]))
    if rng.random() < 0.1:
        deconv_params["renormalize"] = False
    if deconv_params:
        config["deconv_params"] = deconv_params
    roll = rng.random()
    if roll < 0.35:
        config["confint"] = "wald"
        config["confint_params"] = {
            "scale": choice(rng, ["linear", "logit"]),
            "level": float(choice(rng, [0.8, 0.9, 0.95, 0.99])),
            "pseudofrac": float(choice(rng, [0.001, 0.01, 0.05])),
            "quasi": bool(rng.random() < 0.5),
            "method": choice(rng, ["all", "strat"]),
        }
    elif roll < 0.75:
        # the bootstrap is the slow part: fewer rounds where there are many dates
        config["bootstrap"] = int(rng.integers(2, 4 if n_dates > 40 else 10))
    elif roll < 0.85:
        config["bootstrap"] = int(rng.choice([0, 1]))
        config["confint"] = "null"
    return "random", config


def make_case(case_seed, preprint_path):
    rng = np.random.default_rng(case_seed)
    sources = ["synthetic", "synthetic", "wasap", "auto_no_loc"] + (
        ["preprint", "preprint"] if preprint_path else []
    )
    source = choice(rng, sources)
    if source == "synthetic":
        scenario = synthetic(rng)
    elif source == "preprint":
        scenario = preprint(rng, preprint_path)
    elif source == "wasap":
        scenario = wasap(rng)
    else:
        scenario = auto_no_loc(rng)
    tally = scenario["tally"]
    names, codes = scenario["names"], scenario["codes"]

    variants_config = {
        "variants_pangolin": scenario.get("pangolin") or dict(zip(codes, names))
    }
    if scenario.get("not_reported"):
        variants_config["variants_not_reported"] = scenario["not_reported"]
    roll = rng.random()
    if source == "preprint" and roll < 0.85:
        # without it, LolliPop takes every variant of variants_pangolin, also the unreported ones,
        # and fails (KeyError)
        variants_config["variants_list"] = list(names)
    elif roll < 0.35:
        variants_config["variants_list"] = list(names)
    elif roll < 0.45 and len(names) > 1:
        variants_config["variants_list"] = list(
            rng.choice(names, len(names) - 1, replace=False)
        )
    if rng.random() < 0.6:
        variants_config["to_drop"] = [
            d
            for d, p in [("subset", 0.4), ("revert", 0.4), ("shared", 0.1)]
            if rng.random() < p
        ]
    dated = tally["date"].dropna().sort_values().unique() if "date" in tally else []
    if len(dated) > 3 and rng.random() < 0.2:
        variants_config["start_date"] = str(
            dated[int(rng.integers(0, len(dated) // 2))]
        )
    if len(dated) > 3 and rng.random() < 0.2:
        variants_config["end_date"] = str(
            dated[int(rng.integers(len(dated) // 2, len(dated)))]
        )
    if rng.random() < 0.3:
        variants_config["remove_deletions"] = bool(rng.random() < 0.5)

    # where the locations are, and dates
    locations = list(tally["location"].dropna().unique()) if "location" in tally else []
    loc = None
    roll = rng.random()
    if source == "synthetic" and roll < 0.08:
        tally = tally.rename(columns={"location": "location_code"})
    elif source == "synthetic" and roll < 0.12:
        tally = tally.drop(columns=["location"])
        variants_config["no_loc"] = True
    elif roll < 0.2 and len(locations) > 1:
        variants_config["no_loc"] = True
    elif len(locations) > 0 and rng.random() < 0.4:
        loc = list(
            rng.choice(
                locations, int(rng.integers(1, len(locations) + 1)), replace=False
            )
        )
    if source == "synthetic" and rng.random() < 0.06:
        # no dates at all: LolliPop deconvolutes every sample on its own
        tally = tally.drop(columns=["date"])
        variants_config["no_date"] = True

    var_dates = None
    if "date" in tally and len(dated) > 1 and rng.random() < 0.3:
        n_periods = int(rng.integers(1, min(4, len(dated) + 1)))
        starts = sorted(
            set(str(d) for d in rng.choice(dated, n_periods, replace=False))
        )
        if rng.random() < 0.5:
            starts[0] = str(dated[0])
        var_dates = {
            "var_dates": {
                start: list(
                    rng.choice(
                        names, int(rng.integers(1, len(names) + 1)), replace=False
                    )
                )
                for start in starts
            }
        }

    filters = None
    if rng.random() < 0.3:
        filters = random_filters(rng, tally)
    elif source == "preprint" and rng.random() < 0.3:
        filters = ruamel.yaml.YAML(typ="safe").load(
            (REPO / "filters_preprint.yaml").read_text()
        )

    n_dates = len(dated) if len(dated) else 1
    config_kind, deconv_config = random_deconv_config(rng, n_dates)
    if variants_config.get("no_date") and deconv_config.get("bootstrap", 0) > 1:
        deconv_config["bootstrap"] = 2

    # A multi-location bootstrap draws for the locations in the order of a Python set unless they
    # are given: give them, so that the draws are reproducible.
    if (
        deconv_config.get("bootstrap", 0) > 1
        and not loc
        and not variants_config.get("no_loc")
        and len(locations) > 1
    ):
        loc = list(rng.permutation(locations))

    return plain(
        {
            "source": source,
            "notes": scenario.get("notes"),
            "configKind": config_kind,
            "tally": tally,
            "variantsConfig": variants_config,
            "deconvConfig": deconv_config,
            "variantsDates": var_dates,
            "filters": filters,
            "loc": loc,
            "pythonSeed": int(rng.integers(0, 2**32 - 1)),
        }
    )


def plain(value):
    """Python values instead of NumPy's, for YAML and JSON (the tally stays a data frame)."""
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# --- running LolliPop --------------------------------------------------------------------------


def tight_fit(self, X, y, k, b0=None):
    """RobustReg.fit, solved to the minimum: tight tolerances and the exact Jacobian."""
    if b0 is None:
        b0 = np.ones(X.shape[1]) / X.shape[1]
    A = np.expand_dims(k, 1) * X
    ls = scipy.optimize.least_squares(
        lambda beta: A.dot(beta) - (k * y),
        b0,
        jac=lambda beta: A,
        bounds=(0, 1),
        loss=self.loss_type,
        f_scale=self.f_scale,
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        max_nfev=20000,
    )
    self.fitted, self.loss = ls.x, ls.cost
    self.solution = (A, k * y, ls.x, ls.cost)
    return self


def minimum_precision(reg, names):
    """
    How precisely a robust fit's minimum is determined, in the proportions: a change δ of the
    coefficients changes the cost by about ½ δᵀHδ, so where the cost is only known to a relative
    1e-14, the coefficients are only known to √(2e-14 cost / λ), λ the smallest eigenvalue of the
    Hessian H (over the coefficients off their bounds, and one per group of identical columns);
    and a solver that stops at a gradient of 1e-15 is off by 1e-15 / λ. None where H is singular
    to rounding.
    """
    A, b, x, cost = reg.solution
    r = A.dot(x) - b
    z = (r / reg.f_scale) ** 2
    curvature = {
        "linear": lambda z: np.ones_like(z),
        "soft_l1": lambda z: (1 + z) ** -1.5,
        "huber": lambda z: np.where(z <= 1, 1.0, 0.0),
        "cauchy": lambda z: (1 - z) / (1 + z) ** 2,
        "arctan": lambda z: (1 - 3 * z**2) / (1 + z**2) ** 2,
    }[reg.loss_type](z)
    columns = {}
    for j in range(A.shape[1]):
        if x[j] > 1e-10 and x[j] < 1 - 1e-10:
            columns.setdefault(A[:, j].tobytes(), j)
    free = sorted(columns.values())
    if not free:
        return 0.0
    H = A[:, free].T.dot(curvature[:, None] * A[:, free])
    eigenvalues = np.linalg.eigvalsh(H)
    smallest = eigenvalues.min()
    # below rounding, H has no information about the direction
    if not smallest > 1e-15 * eigenvalues.max():
        return None
    # SciPy also stops on an (absolute) gradient tolerance: the coefficients are then off by
    # about |g| / λ
    precision = max(np.sqrt(2e-14 * max(cost, 1e-300) / smallest), 1e-15 / smallest)
    return float(precision / max(x.sum(), 1e-300))


def robust_cost(X, y, k, beta, loss, f_scale):
    r = (np.expand_dims(k, 1) * X).dot(beta) - k * y
    z = (r / f_scale) ** 2
    rho = {
        "linear": lambda z: z,
        "soft_l1": lambda z: 2 * ((1 + z) ** 0.5 - 1),
        "huber": lambda z: np.where(z <= 1, z, 2 * z**0.5 - 1),
        "cauchy": np.log1p,
        "arctan": np.arctan,
    }[loss](z)
    return float(0.5 * f_scale**2 * rho.sum())


def floats(values):
    return [
        None if not np.isfinite(v) else float(v)
        for v in np.asarray(values, dtype=float).ravel()
    ]


def write_inputs(case, directory):
    yaml = ruamel.yaml.YAML(typ="safe")
    yaml.default_flow_style = False
    paths = {}
    paths["tally"] = directory / "tally.tsv"
    case["tally"].to_csv(paths["tally"], sep="\t", index=False)
    for name in ["variantsConfig", "deconvConfig", "variantsDates", "filters"]:
        if case[name] is not None:
            paths[name] = directory / f"{name}.yaml"
            with open(paths[name], "w") as file:
                yaml.dump(case[name], file)
    return paths


def read_tally(path, namefield="mutations"):
    """The tally as LolliPop's `deconvolute` reads it."""
    try:
        return pd.read_csv(
            path,
            sep="\t",
            parse_dates=["date"],
            dtype={"location_code": "str", namefield: "str"},
        )
    except ValueError:
        return pd.read_csv(path, sep="\t", dtype={"location_code": "str"})


def table_json(df):
    columns = list(df.columns)
    rows = []
    for values in df.itertuples(index=False, name=None):
        row = []
        for value in values:
            if isinstance(value, pd.Timestamp):
                row.append(value.strftime("%Y-%m-%d"))
            elif (
                value is None
                or (isinstance(value, float) and not np.isfinite(value))
                or value is pd.NaT
            ):
                row.append(None)
            elif isinstance(value, (np.integer,)):
                row.append(int(value))
            elif isinstance(value, (np.floating, float)):
                row.append(float(value))
            elif isinstance(value, (np.bool_, bool)):
                row.append(bool(value))
            else:
                row.append(value)
        rows.append(row)
    return {"columns": columns, "rows": rows}


def run_lollipop(case, paths, directory, tight, recorder):
    """Runs `lollipop deconvolute` in-process; returns its output, or the error it raised."""
    args = [
        "-c",
        str(paths["variantsConfig"]),
        "-k",
        str(paths["deconvConfig"]),
        "-o",
        str(directory / "out.tsv"),
    ]
    if "variantsDates" in paths:
        args += ["--vd", str(paths["variantsDates"])]
    if "filters" in paths:
        args += ["--filters", str(paths["filters"])]
    for location in case["loc"] or []:
        args += ["--loc", location]
    args.append(str(paths["tally"]))

    original_fit = ll.RobustReg.fit
    original_resample = ll.resample_mutations
    original_deconv = ll.KernelDeconv.deconv

    def resample(df, mutations, namefield="mutations"):
        result = original_resample(df, mutations, namefield)
        if recorder is not None and "draws" in recorder:
            recorder["draws"].append(
                {"half": int(len(mutations) / 2), "idcs": result[1].tolist()}
            )
        return result

    def deconv(self, date, min_tol=1e-10, renormalize=True):
        result = original_deconv(self, date, min_tol, renormalize)
        if recorder is not None:
            recorder["callCount"] += 1
            if "ties" in recorder:
                recorder["variantOrder"] = recorder["variantOrder"] or list(
                    self.variant_names[:-1]
                )
                record_ties(self, date, min_tol, recorder["ties"])
            if "losses" in recorder:
                loss = float(result.loss) if np.isfinite(result.loss) else None
                if isinstance(self.reg, ll.NnlsReg) and "ties" in recorder:
                    loss = check_nnls(self, date, min_tol, loss, recorder["ties"])
                recorder["losses"].append(loss)
            if "precision" in recorder and isinstance(self.reg, ll.RobustReg):
                key = pd.Timestamp(date).strftime("%Y-%m-%d")
                precision = minimum_precision(self.reg, list(self.variant_names))
                previous = recorder["precision"].get(key, 0.0)
                recorder["precision"][key] = (
                    None
                    if precision is None or previous is None
                    else max(previous, precision)
                )
            if (
                "calls" in recorder
                and len(recorder["calls"]) < MAX_RECORDED_CALLS
                and recorder["random"].random() < recorder["rate"]
            ):
                record_call(self, date, min_tol, renormalize, result, recorder)
        return result

    original_location_data = deconvolute_module._get_location_data

    def location_data(preproc, location, no_loc):
        if recorder is not None and "locationOrder" in recorder:
            recorder["locationOrder"].append(location)
        return original_location_data(preproc, location, no_loc)

    np.random.seed(case["pythonSeed"])
    deconvolute_module._get_location_data = location_data
    ll.resample_mutations = resample
    ll.KernelDeconv.deconv = deconv
    if tight:
        ll.RobustReg.fit = tight_fit
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            deconvolute_command.main(args, standalone_mode=False)
        out = pd.read_csv(
            directory / "out.tsv", sep="\t", dtype={"location": "str", "variant": "str"}
        )
        return {"output": table_json(out)}
    except Exception as error:  # noqa: BLE001 - every error is a result here
        return {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=3),
        }
    finally:
        ll.RobustReg.fit = original_fit
        ll.resample_mutations = original_resample
        ll.KernelDeconv.deconv = original_deconv
        deconvolute_module._get_location_data = original_location_data


def selected_rows(self, date, min_tol):
    kvals = (
        self.kernel.values(0, (date - self.dates) / pd.to_timedelta(1, unit="D"))
        * self.weights
    )
    mask = kvals.values >= min_tol
    return (
        self.X.values[mask, :].astype(float),
        self.y.values.flatten()[mask].astype(float),
        kvals.values[mask].astype(float),
    )


def check_nnls(self, date, min_tol, reported, ties):
    """
    SciPy's NNLS (1.18) sometimes returns a non-optimal solution with a residual norm that isn't
    its own: when it rejects a candidate column, it restores only the column's pivot, not the rest
    that LAPACK's dlarfgp has scaled. Where that happens, the date is marked (`scipyNnlsWrong`), and
    the actual residual norm of SciPy's solution is the loss.
    """
    X, y, k = selected_rows(self, date, min_tol)
    if X.shape[0] == 0 or reported is None:
        return reported
    A = np.expand_dims(k, 1) * X
    x, rnorm = scipy.optimize.nnls(A, k * y)
    actual = float(np.linalg.norm(A.dot(x) - k * y))
    if abs(actual - rnorm) > 1e-9 * max(1.0, rnorm):
        entry = ties.setdefault(
            pd.Timestamp(date).strftime("%Y-%m-%d"),
            {"groups": [], "all": False, "empty": False},
        )
        entry["scipyNnlsWrong"] = True
        return actual
    return reported


def record_ties(self, date, min_tol, ties):
    """
    Which variants a fit can't tell apart: there, the minimum isn't unique, and which of the
    minima a solver finds is down to rounding. Collects, per date, the groups of variants with
    identical columns in the design, or whether the design is degenerate otherwise ("all"), or
    empty (where SciPy's NNLS returns uninitialized memory).
    """
    X, _, k = selected_rows(self, date, min_tol)
    entry = ties.setdefault(
        pd.Timestamp(date).strftime("%Y-%m-%d"),
        {"groups": [], "all": False, "empty": False},
    )
    if X.shape[0] == 0:
        entry["empty"] = True
        return
    if isinstance(self.confint, ll.WaldConfint):
        coefs = self.reg.fitted
        condition = (
            fisher_condition(self.confint, X * np.expand_dims(k, 1), coefs)
            if np.all(np.isfinite(coefs))
            else None
        )
        previous = entry.get("fisherCondition", 0.0)
        entry["fisherCondition"] = (
            None if condition is None or previous is None else max(previous, condition)
        )
    with np.errstate(all="ignore"):
        condition = float(np.linalg.cond(X * np.expand_dims(k, 1)))
    previous = entry.get("designCondition", 0.0)
    entry["designCondition"] = (
        None
        if not np.isfinite(condition) or previous is None
        else max(previous, condition)
    )
    names = list(self.variant_names)
    by_column = {}
    for j in range(X.shape[1]):
        by_column.setdefault(X[:, j].tobytes(), []).append(names[j])
    groups = [group for group in by_column.values() if len(group) > 1]
    representatives = [names.index(group[0]) for group in by_column.values()]
    if np.linalg.matrix_rank(X[:, representatives]) < len(representatives):
        entry["all"] = True
    # merge with the groups of the other fits on the date
    for group in groups:
        overlapping = [g for g in entry["groups"] if set(g) & set(group)]
        merged = set(group).union(*map(set, overlapping))
        entry["groups"] = [g for g in entry["groups"] if g not in overlapping] + [
            sorted(merged)
        ]


def record_call(self, date, min_tol, renormalize, result, recorder):
    X, y, k = selected_rows(self, date, min_tol)
    if X.shape[0] > MAX_RECORDED_ROWS:
        return
    call = {
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
        "minTol": min_tol,
        "renormalize": renormalize,
        "variantNames": list(self.variant_names),
        "X": X.tolist(),
        "y": floats(y),
        "k": floats(k),
        "regressor": type(self.reg).__name__,
        "confint": type(self.confint).__name__,
        "fitted": floats(result.fitted),
        "loss": float(result.loss),
        "confBand": {
            "lower": floats(result.conf_band["lower"]),
            "upper": floats(result.conf_band["upper"]),
        },
    }
    if isinstance(self.reg, ll.RobustReg):
        call["regressorParams"] = {
            "loss_type": self.reg.loss_type,
            "f_scale": self.reg.f_scale,
        }
        raw = ll.RobustReg(self.reg.loss_type, self.reg.f_scale).fit(X, y, k)
        tight = tight_fit(ll.RobustReg(self.reg.loss_type, self.reg.f_scale), X, y, k)
        call["rawFitted"] = floats(raw.fitted)
        call["rawCost"] = robust_cost(
            X, y, k, raw.fitted, self.reg.loss_type, self.reg.f_scale
        )
        call["tightFitted"] = floats(tight.fitted)
        call["tightCost"] = robust_cost(
            X, y, k, tight.fitted, self.reg.loss_type, self.reg.f_scale
        )
    else:
        raw = ll.NnlsReg().fit(X, y, k)
        call["rawFitted"] = floats(raw.fitted)
        call["rawCost"] = float(raw.loss)
    if isinstance(self.confint, ll.WaldConfint):
        c = self.confint
        call["confintParams"] = {
            "level": c.level,
            "scale": c.scale,
            "pseudofrac": c.pseudofrac,
            "quasi": c.quasi,
            "method": c.method,
        }
        call["fisherCondition"] = fisher_condition(
            c, X * np.expand_dims(k, 1), result.fitted
        )
    recorder["calls"].append(call)


def fisher_condition(confint, X, coefs):
    """The condition number of the Fisher information that `WaldConfint` inverts."""
    pseudocoefs = coefs / np.sum(coefs)
    fitted_mut = X.dot(pseudocoefs) + confint.pseudofrac
    fitted_mut = fitted_mut / (fitted_mut + (1 - fitted_mut + 2 * confint.pseudofrac))
    info = X.T.dot(np.diag(1 / (fitted_mut * (1 - fitted_mut)))).dot(X)
    with np.errstate(all="ignore"):
        try:
            condition = float(np.linalg.cond(info))
        except np.linalg.LinAlgError:
            return None
    return condition if np.isfinite(condition) else None


def generate(case_seed, preprint_path):
    started = time.time()
    case = make_case(case_seed, preprint_path)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = write_inputs(case, directory)
        tally = read_tally(paths["tally"])
        recorder = {
            "draws": [],
            "calls": [],
            "callCount": 0,
            "ties": {},
            "variantOrder": None,
            "locationOrder": [],
            "losses": [],
            "random": random.Random(case_seed),
            "rate": 0.05,
        }
        expected = run_lollipop(case, paths, directory, tight=False, recorder=recorder)
        robust = case["deconvConfig"].get("regressor") == "robust"
        tight_recorder = {"precision": {}, "losses": [], "callCount": 0}
        expected_tight = (
            run_lollipop(case, paths, directory, tight=True, recorder=tight_recorder)
            if robust
            else None
        )
    return {
        "id": f"case-{case_seed}",
        "seed": case_seed,
        "source": case["source"],
        "notes": case["notes"],
        "configKind": case["configKind"],
        "tally": table_json(tally),
        "variantsConfig": case["variantsConfig"],
        "deconvConfig": case["deconvConfig"],
        "variantsDates": case["variantsDates"],
        "filters": case["filters"],
        "loc": case["loc"],
        "pythonSeed": case["pythonSeed"],
        "expected": expected,
        "expectedTight": expected_tight,
        "draws": recorder["draws"],
        "calls": recorder["calls"],
        "callCount": recorder["callCount"],
        "ties": {
            date: entry
            for date, entry in recorder["ties"].items()
            if entry["groups"]
            or entry["all"]
            or entry["empty"]
            or entry.get("fisherCondition", 0) != 0
            or entry.get("scipyNnlsWrong")
            or entry.get("designCondition") is None
            or entry["designCondition"] > 1e3
        },
        "precision": tight_recorder["precision"],
        "losses": recorder["losses"],
        "tightLosses": tight_recorder["losses"] if robust else None,
        "variantOrder": recorder["variantOrder"],
        "locationOrder": recorder["locationOrder"],
        "seconds": round(time.time() - started, 2),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }


def work(args):
    case_seed, preprint_path, out = args
    try:
        case = generate(case_seed, preprint_path)
    except Exception:  # noqa: BLE001 - a broken generator shouldn't stop the others
        return case_seed, None, traceback.format_exc()
    path = Path(out) / f"case-{case_seed}.json.gz"
    with gzip.open(path, "wt") as file:
        json.dump(case, file, allow_nan=False, separators=(",", ":"))
    summary = (
        "error"
        if "error" in case["expected"]
        else f"{len(case['expected']['output']['rows'])} rows"
    )
    return (
        case_seed,
        f"{case['source']:<11} {case['configKind']:<36} {summary:<10} {case['seconds']}s",
        None,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="the first case's seed; the others follow"
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--preprint",
        type=Path,
        default=REPO / "preprint/data/tallymut_line_full.tsv.zst",
        help="the preprint's tally table (skipped if it's only a Git LFS pointer)",
    )
    parser.add_argument(
        "--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1)
    )
    args = parser.parse_args()

    preprint_path = (
        args.preprint
        if args.preprint.exists() and args.preprint.stat().st_size > 1000
        else None
    )
    if preprint_path is None:
        print(
            f"note: no preprint data at {args.preprint} (Git LFS), skipping that source",
            file=sys.stderr,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(args.seed + i, preprint_path, str(args.out)) for i in range(args.count)]
    with multiprocessing.Pool(args.jobs) as pool:
        for case_seed, summary, error in pool.imap_unordered(work, tasks):
            if error:
                print(f"case-{case_seed}: generator failed\n{error}", file=sys.stderr)
            else:
                print(f"case-{case_seed}: {summary}", flush=True)


if __name__ == "__main__":
    main()
