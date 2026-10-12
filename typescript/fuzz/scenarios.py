# SPDX-License-Identifier: GPL-3.0-or-later

"""
Random scenarios for the fuzz test: a tally table from one of four sources, and random settings
for `lollipop deconvolute`. Every draw comes from `np.random.default_rng(case_seed)`, so a seed
always gives the same scenario. See README.md.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import ruamel.yaml

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"

LINEAGES = (
    "B.1.1.7 B.1.351 P.1 B.1.617.2 BA.1 BA.2 BA.4 BA.5 BQ.1.1 XBB.1.5 EG.5 JN.1 KP.2 KP.3 "
    "LP.8.1 XEC XFG NB.1.8.1"
).split()
LOCATIONS = (
    "Zürich (ZH)|Genève (GE)|Lugano (TI)|Chur (GR)|Laupen (BE)|Basel (BS)".split("|")
)
PREPRINT_LOCATIONS = (
    "Zürich (ZH)|Genève (GE)|Lugano (TI)|Chur (GR)|Laupen (BE)|Altenrhein (SG)|"
    "Lausanne (VD)|Kanton Zürich"
).split("|")
PREPRINT_PANGOLIN = dict(
    al="B.1.1.7",
    be="B.1.351",
    ga="P.1",
    de="B.1.617.2",
    ka="B.1.617.1",
    om1="BA.1",
    om2="BA.2",
    om4="BA.4",
    om5="BA.5",
)
LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
PRESETS = {
    path.stem: ruamel.yaml.YAML(typ="safe").load(path.read_text())
    for path in sorted((REPO / "presets").glob("*.yaml"))
}
MAX_ROWS = 6000

_preprint = None


def choice(rng, values):
    return values[rng.integers(len(values))]


def subset(rng, values, low, high):
    """Between `low` and `high` (exclusive) of `values`, at random."""
    return list(rng.choice(values, int(rng.integers(low, high)), replace=False))


# --- data --------------------------------------------------------------------------------------


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
    carried = [set(rng.choice(n_positions, rng.integers(0, 3), replace=False).tolist())]
    for i in range(1, n_variants):
        parent = int(rng.integers(0, i))
        new = rng.choice(
            n_positions, rng.integers(0, max(2, n_positions // 3)), replace=False
        )
        carried.append(carried[parent] | set(new.tolist()))
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
                        "sample": f"S{li}_{str(date)}_{s}",
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
                        row[codes[v]] = np.nan
                        if p in carried[v]:
                            shared = sum(p in c for c in carried) > 1
                            row[codes[v]] = (
                                "shared"
                                if shared and rng.random() < 0.5
                                else choice(
                                    rng,
                                    ["mut", "mut", "mut", "extra", "subset", "revert"],
                                )
                            )
                    rows.append(row)
    tally = pd.DataFrame(rows)
    if rng.random() < 0.15:
        tally["om1"] = np.where(rng.random(len(tally)) < 0.3, "mut", None)
    return dict(
        source="synthetic", tally=tally, names=names, codes=codes, notes=degenerate
    )


def preprint(rng, path):
    """A random slice (location, time window, positions, variants) of the preprint's tally."""
    global _preprint
    if _preprint is None:
        _preprint = pd.read_csv(path, sep="\t", dtype={"location_code": "str"})
    location = choice(rng, PREPRINT_LOCATIONS)
    df = _preprint[_preprint["location"] == location]
    dates = np.sort(df["date"].dropna().unique())
    start = int(rng.integers(0, max(1, len(dates) - 5)))
    window = set(dates[start : start + int(rng.choice([4, 10, 25, 40]))])
    df = df[df["date"].isin(window) | (df["date"].isna() & (rng.random() < 0.5))]
    positions = df["pos"].unique()
    keep = rng.choice(
        positions, min(len(positions), int(rng.integers(10, 150))), replace=False
    )
    df = df[df["pos"].isin(keep)]
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=int(rng.integers(2**31))).sort_index()
    pangolin = PREPRINT_PANGOLIN
    present = [c for c in pangolin if c in df.columns and df[c].notna().any()]
    present = present or ["al", "de"]
    used = subset(rng, present, min(2, len(present)), len(present) + 1)
    return dict(
        source="preprint",
        tally=df.reset_index(drop=True),
        names=[pangolin[c] for c in used],
        codes=used,
        pangolin=pangolin,
        not_reported=[pangolin[c] for c in pangolin if c not in used]
        + ["om2121", "om275"],
        notes=f"{location}, {len(window)} dates, {len(keep)} positions",
    )


def wasap(rng):
    """A random slice of the W-ASAP snapshot."""
    data = json.loads((DATA / "wasap_zurich_2026.json").read_text())
    names = subset(rng, list(data["signatures"]), 2, len(data["signatures"]) + 1)
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
    tally = pd.DataFrame(rows)
    return dict(source="wasap", tally=tally, names=names, codes=names, notes=None)


def auto_no_loc(rng):
    """A random slice of LolliPop's own test tally (no location column)."""
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
    tally = df.reset_index(drop=True)
    return dict(source="auto_no_loc", tally=tally, names=names, codes=names, notes=None)


# --- settings ----------------------------------------------------------------------------------


def random_filters(rng, tally):
    """Mutation filters (`--filters`), in LolliPop's filter syntax."""
    filters = {}
    for i in range(int(rng.integers(1, 4))):
        kind = choice(
            rng,
            ["amplicon", "mutations", "date", "proto", "sample", "base", "location"],
        )
        if kind == "amplicon" and "pos" in tally:
            low = int(rng.choice(tally["pos"]))
            high = low + int(rng.integers(0, 400))
            filters[f"amplicon{i}"] = ["proto v3", f"pos >= {low}", f"pos <= {high}"]
        elif kind == "mutations" and "pos" in tally and "base" in tally:
            picked = tally.sample(
                min(len(tally), 3), random_state=int(rng.integers(2**31))
            )
            names = [f"{p}{b}" for p, b in zip(picked["pos"], picked["base"])]
            names += ["-" + n for n in names if rng.random() < 0.5]
            filters[f"bad_mutations{i}"] = [f"mutations in [ {', '.join(names)} ]"]
        elif kind == "date" and "date" in tally and tally["date"].notna().any():
            date = choice(rng, list(tally["date"].dropna().unique()))
            operator = choice(rng, [">", "<", ">=", "<=", "=="])
            filters[f"date{i}"] = [f"date {operator} {date}"]
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
            location = choice(rng, list(tally["location"].dropna().unique()))
            filters[f"location{i}"] = [f"location == '{location}'", "frac > 0.9"]
    return filters or None


def random_deconv_config(rng, n_dates):
    """A preset (30 %), or a random deconvolution configuration (`--deconv-config`)."""
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
    config["regressor"] = choice(rng, ["nnls", "robust", "robust"])
    if config["regressor"] == "robust":
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


def make_scenario(case_seed, preprint_path):
    """The data and the settings of a case: what `lollipop deconvolute` gets."""
    rng = np.random.default_rng(case_seed)
    sources = ["synthetic", "synthetic", "wasap", "auto_no_loc"]
    source = choice(rng, sources + (["preprint", "preprint"] if preprint_path else []))
    if source == "synthetic":
        scenario = synthetic(rng)
    elif source == "preprint":
        scenario = preprint(rng, preprint_path)
    elif source == "wasap":
        scenario = wasap(rng)
    else:
        scenario = auto_no_loc(rng)
    tally, names, codes = scenario["tally"], scenario["names"], scenario["codes"]

    # the variants configuration (`--variants-config`)
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
        end = int(rng.integers(len(dated) // 2, len(dated)))
        variants_config["end_date"] = str(dated[end])
    if rng.random() < 0.3:
        variants_config["remove_deletions"] = bool(rng.random() < 0.5)

    # where the locations are (`--loc`, `no_loc`, `location_code`), and the dates (`no_date`)
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
        loc = subset(rng, locations, 1, len(locations) + 1)
    if source == "synthetic" and rng.random() < 0.06:
        # no dates at all: LolliPop deconvolutes every sample on its own
        tally = tally.drop(columns=["date"])
        variants_config["no_date"] = True

    # periods with their own variants (`--variants-dates`)
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
                start: subset(rng, names, 1, len(names) + 1) for start in starts
            }
        }

    filters = None
    if rng.random() < 0.3:
        filters = random_filters(rng, tally)
    elif source == "preprint" and rng.random() < 0.3:
        filters = ruamel.yaml.YAML(typ="safe").load(
            (REPO / "filters_preprint.yaml").read_text()
        )

    config_kind, deconv_config = random_deconv_config(rng, len(dated) or 1)
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
