# Fuzz test

A differential test of the TypeScript port against LolliPop itself: random data and random settings
are run through LolliPop's real `deconvolute` command, and the port has to reproduce what it did.

```sh
npm test                    # the checked-in corpus (fuzz/corpus), with the unit tests
npm run fuzz                # 100 new cases from a random seed
npm run fuzz -- --count 500 --seed 7 --preprint path/to/tallymut_line_full.tsv.zst --jobs 8
```

`npm run fuzz` needs [uv](https://docs.astral.sh/uv/). On its first run, it creates a Python
environment in `fuzz/.venv` with LolliPop from this repository (and `pandas < 3`, see below). It then
generates the cases into `fuzz/out/<seed>`, evaluates the port on them, prints the cases that
failed, and writes a report to `fuzz/out/<seed>/report.json`. The same seed gives the same cases.

## Files

The pipeline, in the order it runs:

| File                          | What it does                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| `scenarios.py`                | random scenarios: a tally table and the settings for `lollipop deconvolute`            |
| `run_lollipop.py`             | runs LolliPop on each scenario (in-process, instrumented), and writes it out as a case |
| `runPort.ts`                  | the format of a case, and running the port on it                                       |
| `evaluate.ts`                 | the checks: whether the port reproduces what LolliPop did                              |
| `cli.ts`                      | `npm run fuzz`: sets up Python, generates cases, evaluates them, writes the report     |
| `fuzz.spec.ts`                | `npm test`: evaluates the corpus                                                       |
| `corpus/`                     | a fixed set of cases                                                                   |
| `data/wasap_zurich_2026.json` | a snapshot of W-ASAP data, one of the sources                                          |

## A case

Each case is drawn from its own seed. It has some data and some settings.

**The data** comes from one of these sources:

- **synthetic**: a random panel of lineages along a random tree, with random prevalence curves at
  a few locations. The read counts are binomial, with dropouts, outliers, deletions and
  `mut`/`shared`/`subset`/`revert` annotations. Sometimes signatures are indistinguishable or
  empty.
- **preprint**: a random slice (location, time window, positions, variants) of the tally table of
  the LolliPop preprint. That table is in Git LFS (`preprint/data/tallymut_line_full.tsv.zst`), so
  pass `--preprint` with the real file. Without it, this source is skipped.
- **wasap**: a random slice of the W-ASAP snapshot in `data/`.
- **auto_no_loc**: a random slice of LolliPop's own test tally, which has no location column.

**The settings** are random too:

- variant renaming, variants lists and `to_drop`;
- start and end dates, and removing deletions;
- missing locations or dates (`no_loc`, `no_date`, `location_code`);
- periods with their own variants (`var_dates`);
- mutation filters, including the preprint's;
- the deconvolution configuration. This is either one of LolliPop's presets, or a random kernel,
  bandwidth, `min_tol`, `renormalize`, regressor, loss and `f_scale`, with random Wald intervals
  (scale, level, `pseudofrac`, quasi-likelihood, method) or a bootstrap.

`run_lollipop.py` writes the tally and the configuration files, and runs `lollipop deconvolute` on
them in-process. It seeds NumPy's global generator first, so the bootstrap can be replayed. It
patches LolliPop to record what happens inside the run. Each case is saved as
`case-<seed>.json.gz` and holds:

- `expected`: the output, or the error LolliPop raised.
- `draws`: the mutations each bootstrap round drew.
- `calls`: a sample of single regressions, with their inputs, LolliPop's fit, and the confidence
  band.
- `losses`: the loss of every regression, in order. With the robust regressor, `tightLosses` too:
  each regression solved to the minimum (SciPy with tight tolerances and the exact Jacobian).
- `ties`: where a value isn't determined (see below).
- `variantOrder` and `locationOrder`: the order LolliPop used. Without a variants list or `--loc`,
  that order comes from a Python set and changes from run to run, so the test reuses LolliPop's.

## The checks

For each case, `evaluate.ts` checks:

1. **The bootstrap draws.** Every round draws the same mutations as LolliPop.
2. **Single regressions**, replayed with LolliPop's inputs:
    - NNLS gives the same residual.
    - The robust regression reaches a loss at least as low as SciPy's minimum.
    - LolliPop's coefficients give the same confidence band.
3. **Every regression's loss**, in order:
    - NNLS: the same as LolliPop's.
    - Robust: as low as SciPy's minimum.
4. **The output**: the same rows and the same values, to 1e-8. Where LolliPop raised an error, the
   port has to throw too.

Not every value is determined, and the test only compares the ones that are. `run_lollipop.py`
records where they aren't:

- **Ties.** Where variants have the same signature over the rows a fit sees, only their sum is
  determined.
- **Ill-conditioning.** Where the design (with tied variants merged) or the Fisher information is
  nearly singular, the split between variants, or the Wald interval, is rounding noise in NumPy too.
  With a condition κ, least squares coefficients are only determined up to about κ²ε, and the
  tolerance grows accordingly; the losses still have to agree.
- **SciPy's NNLS bug.** In about 1 case in 400, SciPy 1.18's `nnls` returns a non-optimal
  solution. There, the port finds a lower loss than LolliPop.

**The robust regressor** gets a weaker check. SciPy's solver stops early, and where the minimum is
flat, the coefficients aren't pinned down by the losses. So with the robust regressor:

- the losses are checked as above;
- the output's rows are checked;
- the output's values aren't. Their differences from LolliPop's go into the report
  (`maxDifference`).

With the non-convex losses (`cauchy`, `arctan`), the two solvers can stop at different local minima,
so only the output's rows are compared.

## When a case fails

Evaluate it again, alone (without generating anything):

```sh
npm run fuzz -- --dir fuzz/out/<seed> --case case-<n>
```

Once it's fixed, copy the case into `corpus/`, so `npm test` keeps checking it.

`report.json` lists, per case, the source, the configuration, the outcome, the run time and the
largest differences. It's the place to look for drift that is still within the tolerances.

## Python versions

The cases come from the LolliPop in this repository. With pandas 3, its logit-scale intervals fail:
`DataFrame.applymap` was removed. That's why the environment pins `pandas < 3`.

SciPy is whatever uv installs (1.18 as of writing). A newer SciPy may fix the NNLS bug, or change
where its robust solver stops. The checks above don't depend on either.
