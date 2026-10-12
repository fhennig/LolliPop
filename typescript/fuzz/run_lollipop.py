#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Runs LolliPop's `deconvolute` command on random scenarios (see scenarios.py), in-process and
instrumented, and writes each scenario with what happened as a fuzz case (`case-<seed>.json.gz`).
See README.md for what a case holds; `npm run fuzz` (cli.ts) calls this.

    fuzz/.venv/bin/python fuzz/run_lollipop.py --seed 1 --count 50 --out fuzz/out/1
"""

import argparse
import contextlib
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
from scenarios import REPO, make_scenario

# (the package exports the command under the module's name)
deconvolute_module = sys.modules["lollipop.cli.deconvolute"]

# Regressions recorded in full per case, picked at random among those with fewer rows than this.
RECORDED_CALLS = 6
RECORDED_CALL_RATE = 0.05
MAX_RECORDED_ROWS = 2500


def run_case(case_seed, preprint_path):
    """A fuzz case: a scenario, and what LolliPop did with it."""
    started = time.time()
    case = make_scenario(case_seed, preprint_path)
    recorder = Recorder(random.Random(case_seed))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = write_inputs(case, directory)
        tally = read_tally(paths["tally"])
        expected = run_lollipop(case, paths, directory, recorder)
    return {
        "id": f"case-{case_seed}",
        "seed": case_seed,
        **{key: value for key, value in case.items() if key != "tally"},
        "tally": table_json(tally),
        "expected": expected,
        "draws": recorder.draws,
        "calls": recorder.calls,
        "ties": {
            date: entry for date, entry in recorder.ties.items() if notable(entry)
        },
        "losses": recorder.losses,
        "tightLosses": recorder.tight_losses if recorder.robust else None,
        "variantOrder": recorder.variant_order,
        "locationOrder": recorder.location_order,
        "seconds": round(time.time() - started, 2),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }


# --- running LolliPop --------------------------------------------------------------------------


def write_inputs(case, directory):
    """The files `lollipop deconvolute` reads."""
    yaml = ruamel.yaml.YAML(typ="safe")
    yaml.default_flow_style = False
    paths = {"tally": directory / "tally.tsv"}
    case["tally"].to_csv(paths["tally"], sep="\t", index=False)
    for name in ["variantsConfig", "deconvConfig", "variantsDates", "filters"]:
        if case[name] is not None:
            paths[name] = directory / f"{name}.yaml"
            with open(paths[name], "w") as file:
                yaml.dump(case[name], file)
    return paths


def read_tally(path, namefield="mutations"):
    """The tally as LolliPop's `deconvolute` reads it (what the port gets)."""
    try:
        return pd.read_csv(
            path,
            sep="\t",
            parse_dates=["date"],
            dtype={"location_code": "str", namefield: "str"},
        )
    except ValueError:  # no date column
        return pd.read_csv(path, sep="\t", dtype={"location_code": "str"})


def run_lollipop(case, paths, directory, recorder):
    """Runs `lollipop deconvolute` in-process; returns its output, or the error it raised."""
    args = ["-c", paths["variantsConfig"], "-k", paths["deconvConfig"]]
    args += ["-o", directory / "out.tsv"]
    if "variantsDates" in paths:
        args += ["--vd", paths["variantsDates"]]
    if "filters" in paths:
        args += ["--filters", paths["filters"]]
    for location in case["loc"] or []:
        args += ["--loc", location]
    args.append(paths["tally"])

    # instrument LolliPop: the bootstrap draws, every regression, the order of the locations
    resample_mutations = ll.resample_mutations
    deconv = ll.KernelDeconv.deconv
    get_location_data = deconvolute_module._get_location_data

    def recorded_resample(df, mutations, namefield="mutations"):
        result = resample_mutations(df, mutations, namefield)
        recorder.draws.append({"half": len(mutations) // 2, "idcs": result[1].tolist()})
        return result

    def recorded_deconv(self, date, min_tol=1e-10, renormalize=True):
        result = deconv(self, date, min_tol, renormalize)
        try:
            recorder.record(self, date, min_tol, renormalize, result)
        except Exception as error:
            # (not LolliPop's error: it mustn't pass for one)
            recorder.failure = error
            raise
        return result

    def recorded_location_data(preproc, location, no_loc):
        recorder.location_order.append(location)
        return get_location_data(preproc, location, no_loc)

    ll.resample_mutations = recorded_resample
    ll.KernelDeconv.deconv = recorded_deconv
    deconvolute_module._get_location_data = recorded_location_data
    # (LolliPop's --seed doesn't seed the generator the bootstrap uses: seed that one)
    np.random.seed(case["pythonSeed"])
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            deconvolute_command.main([str(a) for a in args], standalone_mode=False)
        output = pd.read_csv(
            directory / "out.tsv", sep="\t", dtype={"location": "str", "variant": "str"}
        )
        return {"output": table_json(output)}
    except Exception as error:  # noqa: BLE001 - every error is a result here
        if recorder.failure is not None:
            raise RuntimeError("recording failed") from recorder.failure
        return {
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=3),
        }
    finally:
        ll.resample_mutations = resample_mutations
        ll.KernelDeconv.deconv = deconv
        deconvolute_module._get_location_data = get_location_data


# --- recording ---------------------------------------------------------------------------------


class Recorder:
    """What happens in every regression of a run (`KernelDeconv.deconv`)."""

    def __init__(self, rng):
        self.rng = rng
        self.draws = []
        self.calls = []
        self.ties = {}
        self.losses = []
        self.tight_losses = []
        self.robust = False
        self.variant_order = None
        self.location_order = []
        self.failure = None

    def record(self, kd, date, min_tol, renormalize, result):
        X, y, k = selected_rows(kd, date, min_tol)
        entry = self.ties.setdefault(
            pd.Timestamp(date).strftime("%Y-%m-%d"),
            {"groups": [], "all": False, "empty": False},
        )
        self.variant_order = self.variant_order or list(kd.variant_names[:-1])
        record_ties(kd, X, k, entry)
        loss = float(result.loss) if np.isfinite(result.loss) else None
        if isinstance(kd.reg, ll.NnlsReg):
            loss = check_nnls(X, y, k, loss, entry)
        self.losses.append(loss)
        if isinstance(kd.reg, ll.RobustReg):
            self.robust = True
            tight = solve_to_minimum(kd.reg, X, y, k)
            self.tight_losses.append(tight[1] if tight else None)
        if (
            len(self.calls) < RECORDED_CALLS
            and self.rng.random() < RECORDED_CALL_RATE
            and 0 < len(y) <= MAX_RECORDED_ROWS
        ):
            self.calls.append(
                record_call(kd, date, min_tol, renormalize, result, X, y, k)
            )


def selected_rows(kd, date, min_tol):
    """The rows a regression of `KernelDeconv.deconv` sees: design, observations, kernel."""
    kvals = (
        kd.kernel.values(0, (date - kd.dates) / pd.to_timedelta(1, unit="D"))
        * kd.weights
    )
    mask = kvals.values >= min_tol
    return (
        kd.X.values[mask, :].astype(float),
        kd.y.values.flatten()[mask].astype(float),
        kvals.values[mask].astype(float),
    )


def keep_max(entry, key, value):
    """The largest `value` over the fits on a date; None (singular) wins."""
    previous = entry.get(key, 0.0)
    singular = value is None or not np.isfinite(value) or previous is None
    entry[key] = None if singular else max(previous, value)


def record_ties(kd, X, k, entry):
    """
    Where a fit can't tell variants apart, the minimum isn't unique, and which of the minima a
    solver finds is down to rounding. Records, per date, the groups of variants with identical
    columns in the design, whether the design is degenerate otherwise ("all") or empty, and the
    conditions of the design and of the Fisher information.
    """
    if X.shape[0] == 0:
        entry["empty"] = True  # (SciPy's NNLS then returns uninitialized memory)
        return
    if isinstance(kd.confint, ll.WaldConfint):
        coefs = kd.reg.fitted
        finite = np.all(np.isfinite(coefs))
        keep_max(
            entry,
            "fisherCondition",
            fisher_condition(kd.confint, X * k[:, None], coefs) if finite else None,
        )
    names = list(kd.variant_names)
    by_column = {}
    for j in range(X.shape[1]):
        by_column.setdefault(X[:, j].tobytes(), []).append(names[j])
    representatives = [names.index(group[0]) for group in by_column.values()]
    # The condition of the design with each group of tied variants merged into one column: the
    # fit is unique there, and it determines the other variants and the sums over the groups.
    # (With the tied columns, the condition is infinite, and says nothing about the rest.)
    with np.errstate(all="ignore"):
        keep_max(
            entry,
            "designCondition",
            float(np.linalg.cond(X[:, representatives] * k[:, None])),
        )
    if np.linalg.matrix_rank(X[:, representatives]) < len(representatives):
        entry["all"] = True
    # merge with the groups of the other fits on the date
    for group in (group for group in by_column.values() if len(group) > 1):
        overlapping = [g for g in entry["groups"] if set(g) & set(group)]
        merged = sorted(set(group).union(*map(set, overlapping)))
        entry["groups"] = [g for g in entry["groups"] if g not in overlapping] + [
            merged
        ]


def notable(entry):
    """Whether a date's entry says anything the comparison needs."""
    condition = entry.get("designCondition", 0)
    return bool(
        entry["groups"]
        or entry["all"]
        or entry["empty"]
        or entry.get("fisherCondition", 0) != 0
        or entry.get("scipyNnlsWrong")
        or condition is None
        or condition > 1e3
    )


def check_nnls(X, y, k, reported, entry):
    """
    SciPy's NNLS (1.18) sometimes returns a non-optimal solution with a residual norm that isn't
    its own (scipy/scipy#25389). Where that happens, the date is marked (`scipyNnlsWrong`), and
    the actual residual norm of SciPy's solution is the loss.
    """
    if X.shape[0] == 0 or reported is None:
        return reported
    A = X * k[:, None]
    x, rnorm = scipy.optimize.nnls(A, k * y)
    actual = float(np.linalg.norm(A.dot(x) - k * y))
    if abs(actual - rnorm) > 1e-9 * max(1.0, rnorm):
        entry["scipyNnlsWrong"] = True
        return actual
    return reported


def solve_to_minimum(reg, X, y, k):
    """`RobustReg.fit` solved to the minimum (tight tolerances, exact Jacobian): (x, cost)."""
    if X.shape[0] == 0:
        return None
    A = X * k[:, None]
    ls = scipy.optimize.least_squares(
        lambda beta: A.dot(beta) - k * y,
        np.ones(X.shape[1]) / X.shape[1],
        jac=lambda beta: A,
        bounds=(0, 1),
        loss=reg.loss_type,
        f_scale=reg.f_scale,
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        max_nfev=20000,
    )
    return ls.x, robust_cost(X, y, k, ls.x, reg.loss_type, reg.f_scale)


def robust_cost(X, y, k, beta, loss, f_scale):
    """SciPy's cost of the robust regression at `beta`."""
    z = (((X * k[:, None]).dot(beta) - k * y) / f_scale) ** 2
    rho = {
        "linear": lambda z: z,
        "soft_l1": lambda z: 2 * ((1 + z) ** 0.5 - 1),
        "huber": lambda z: np.where(z <= 1, z, 2 * z**0.5 - 1),
        "cauchy": np.log1p,
        "arctan": np.arctan,
    }[loss](z)
    return float(0.5 * f_scale**2 * rho.sum())


def fisher_condition(confint, X, coefs):
    """The condition number of the Fisher information that `WaldConfint` inverts."""
    pseudocoefs = coefs / np.sum(coefs)
    fitted = X.dot(pseudocoefs) + confint.pseudofrac
    fitted = fitted / (fitted + (1 - fitted + 2 * confint.pseudofrac))
    info = X.T.dot(np.diag(1 / (fitted * (1 - fitted)))).dot(X)
    with np.errstate(all="ignore"):
        try:
            condition = float(np.linalg.cond(info))
        except np.linalg.LinAlgError:
            return None
    return condition if np.isfinite(condition) else None


def record_call(kd, date, min_tol, renormalize, result, X, y, k):
    """A regression in full: its inputs, LolliPop's fit, and its confidence band."""
    reg, confint = kd.reg, kd.confint
    call = {
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
        "minTol": min_tol,
        "renormalize": renormalize,
        "variantNames": list(kd.variant_names),
        "X": X.tolist(),
        "y": floats(y),
        "k": floats(k),
        "regressor": type(reg).__name__,
        "confint": type(confint).__name__,
        "fitted": floats(result.fitted),
        "loss": float(result.loss),
        "confBand": {
            "lower": floats(result.conf_band["lower"]),
            "upper": floats(result.conf_band["upper"]),
        },
    }
    if isinstance(reg, ll.RobustReg):
        call["regressorParams"] = {"loss_type": reg.loss_type, "f_scale": reg.f_scale}
        tight_x, tight_cost = solve_to_minimum(reg, X, y, k)
        call["tightFitted"] = floats(tight_x)
        call["tightCost"] = tight_cost
    else:
        raw = ll.NnlsReg().fit(X, y, k)
        call["rawFitted"] = floats(raw.fitted)
        call["rawCost"] = float(raw.loss)
    if isinstance(confint, ll.WaldConfint):
        call["confintParams"] = {
            key: getattr(confint, key)
            for key in ["level", "scale", "pseudofrac", "quasi", "method"]
        }
        call["fisherCondition"] = fisher_condition(
            confint, X * k[:, None], result.fitted
        )
    return call


# --- writing -----------------------------------------------------------------------------------


def floats(values):
    return [
        None if not np.isfinite(v) else float(v) for v in np.ravel(values).astype(float)
    ]


def table_json(df):
    """A data frame as {columns, rows}: dates as ISO days, NaN and ±inf as None."""
    df = df.replace([np.inf, -np.inf], np.nan).astype(object)
    df = df.where(df.notna(), None)
    rows = [
        [
            (
                v.strftime("%Y-%m-%d")
                if isinstance(v, pd.Timestamp)
                else v.item() if isinstance(v, np.generic) else v
            )
            for v in values
        ]
        for values in df.itertuples(index=False, name=None)
    ]
    return {"columns": list(df.columns), "rows": rows}


def work(args):
    case_seed, preprint_path, out = args
    try:
        case = run_case(case_seed, preprint_path)
    except Exception:  # noqa: BLE001 - a broken scenario shouldn't stop the others
        return f"case-{case_seed}: generator failed\n{traceback.format_exc()}"
    with gzip.open(Path(out) / f"case-{case_seed}.json.gz", "wt") as file:
        json.dump(case, file, allow_nan=False, separators=(",", ":"))
    expected = case["expected"]
    summary = (
        "error" if "error" in expected else f"{len(expected['output']['rows'])} rows"
    )
    return f"case-{case_seed}: {case['source']:<11} {case['configKind']:<36} {summary:<10} {case['seconds']}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="the first case's seed")
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

    preprint = args.preprint
    if not (preprint.exists() and preprint.stat().st_size > 1000):
        print(
            f"note: no preprint data at {preprint} (Git LFS), skipping it",
            file=sys.stderr,
        )
        preprint = None
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(args.seed + i, preprint, str(args.out)) for i in range(args.count)]
    failed = 0
    with multiprocessing.Pool(args.jobs) as pool:
        for line in pool.imap_unordered(work, tasks):
            print(line, flush=True)
            failed += "generator failed" in line
    if failed:
        print(f"{failed} cases failed to generate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
