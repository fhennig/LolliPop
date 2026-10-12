# LolliPop in TypeScript

A TypeScript port of [LolliPop](../README.md)'s kernel deconvolution — from mutation frequencies in
wastewater samples to the prevalences of a panel of variants over time — as a library, without the
command line. It runs in the browser or in Node, and has no dependencies.

Licensed under the GPL-3.0-or-later, like LolliPop.

## Layout

The port follows the Python code file by file, so that each piece can be read side by side with its
original:

| TypeScript                          | Python                                                                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `src/kernels.ts`                    | `lollipop/kernels.py`                                                                                            |
| `src/regressors.ts`                 | `lollipop/regressors.py`                                                                                         |
| `src/confints.ts`                   | `lollipop/confints.py`                                                                                           |
| `src/kernelDeconv.ts`               | `lollipop/kerneldeconv.py`                                                                                       |
| `src/preprocessors.ts`              | `lollipop/preprocessors.py`                                                                                      |
| `src/deconvolute.ts`                | `lollipop/cli/deconvolute.py`, without the files, the progress bars and the parallel processing                  |
| `src/presets.ts`                    | `presets/*.yaml`                                                                                                 |
| `src/numeric/nnls.ts`               | `scipy.optimize.nnls` (a port of SciPy's C code)                                                                 |
| `src/numeric/leastSquares.ts`       | `scipy.optimize.least_squares`, for the bounded robust regression of `RobustReg` (a different solver, see below) |
| `src/numeric/randomState.ts`        | `numpy.random.seed` / `randint` (MT19937, draw for draw)                                                         |
| `src/numeric/stats.ts`, `linalg.ts` | `scipy.stats.norm.ppf`, `np.quantile`, `np.linalg.inv`                                                           |
| `src/deconvolve.ts`                 | not in LolliPop: the entry point for dashboards (signatures and frequencies in, prevalences out)                 |

Names stay close to the Python ones (`KernelDeconv`, `deconvAll`, `generalPreprocess`, …), and the
configuration objects use the keys of LolliPop's YAML files (`variants_pangolin`, `kernel_params`,
`min_tol`, …), so a YAML file parsed into an object can be passed as it is. A table (pandas' data
frame) is an array of rows, `Record<string, string | number | null>`, with `null` for NaN.

## Usage

The way a dashboard uses it (this is what the W-ASAP dashboard does):

```ts
import { deconvolve, PRESETS } from 'lollipop';

const result = deconvolve({
    signatures: { XFG: ['C241T', …], 'NB.1.8.1': […] },
    frequencies: [{ date: '2026-05-04', mutation: 'C241T', frac: 0.42 }, …],
    config: PRESETS.deconv_bootstrap_cowwid, // the default
    random: 0, // the seed of the bootstrap
});
// result.dates[i].estimates[j]: { variant, proportion, lower, upper }
```

The whole pipeline of `lollipop deconvolute`, on a tally table:

```ts
import { deconvolute, PRESETS, RandomState } from 'lollipop';

const rows = deconvolute({
    tally, // the rows of the tally TSV
    columns, // its header, in order (the filter syntax depends on the column order, as in LolliPop)
    variantsConfig: { variants_pangolin: { al: 'B.1.1.7', … }, variants_list: [...], to_drop: ['subset'] },
    deconvConfig: PRESETS.deconv_linear_logit,
    variantsDates: { var_dates: { '2021-01-01': ['B.1.1.7', …] } },
    filters: { amplicon75: ['proto v3', 'pos >= 22428', 'pos <= 22785'] },
    random: new RandomState(42),
});
// rows: { location, variant, date, proportion, proportionLower?, proportionUpper? }, sorted like LolliPop's output
```

The building blocks (`KernelDeconv`, the kernels, regressors, confidence intervals and the
preprocessor) are exported too.

## Differences from LolliPop

- **The robust regression is solved to the minimum.** `RobustReg` uses SciPy's trust-region solver,
  which stops where its tolerances say the loss is flat enough. The port uses an active-set Newton
  method that goes on to the minimum: in the fuzz test, every fit reaches a loss at least as low as
  SciPy solving to the minimum (tight tolerances, exact Jacobian). Where the loss is flat (few samples
  in the kernel, a small `f_scale`, nearly indistinguishable variants), the coefficients at the minimum
  are barely determined, and LolliPop's own results can be several percentage points off it.
- **The non-convex losses** (`cauchy`, `arctan`, which no preset uses) have local minima, and which one
  a solver finds depends on its path; the port's and SciPy's can differ.
- **NNLS: a SciPy bug is fixed.** SciPy 1.18's `nnls` (its C rewrite of Lawson and Hanson) sometimes
  returns a non-optimal solution, with a residual norm that isn't the solution's: when it rejects a
  candidate column, it restores only the pivot, not the rest of the column that LAPACK's `dlarfgp`
  has scaled in place. The port restores the column. (In the fuzz test, about 1 case in 400 hits this.)
- **The bootstrap draws the same rounds for the same seed** — NumPy's legacy generator is ported — but
  LolliPop's command line doesn't actually seed it (`--seed` seeds a `default_rng` that is never used),
  so its own bootstrap isn't reproducible.
- **Without a `variants_list`** (or `--loc`), LolliPop takes the variants (or locations) in the order
  of a Python set, which changes from run to run; the port takes them in the order of the configuration
  and the data. The order only matters where variants can't be told apart (below), and for which
  location's bootstrap draws come first.
- **Indistinguishable variants.** Where variants have the same signature over the mutations a fit
  sees, the split between them isn't determined, and SciPy's NNLS breaks the tie by rounding noise;
  the port's NNLS can break it differently, the sum is the same. Likewise, Wald intervals from a
  singular Fisher information are noise in NumPy (and NaN here).
- **Wald intervals without a `scale`** are on the linear scale (`WaldConfint`'s default). LolliPop
  computes them and then fails reading `confint_params["scale"]` (`KeyError`), so its presets
  `deconv_linear_wald`, `deconv_linear_wald_quasibinom` and `deconv_linear_quasi_strat` don't run.
- **An empty fit** (no rows above `min_tol` on a date) gives NaN, so 0 after LolliPop's `fillna`;
  SciPy's NNLS returns uninitialized memory there.
- Not ported: reading and writing files, the JSON and column-per-variant outputs, `n_cores`, and the
  other commands (`generate-mutlist`, `getmutations-from-basecount`).

LolliPop itself needs pandas < 3 for the logit-scale intervals (`DataFrame.applymap` is gone in
pandas 3).

## Tests

```sh
npm test           # unit tests, and the fuzz test on the corpus in fuzz/corpus
npm run fuzz       # the fuzz test on new cases: needs Python (uv), see below
```

### The fuzz test

The fuzz test runs random data and random settings through LolliPop's real `deconvolute` command, and
checks that the port reproduces it: the same bootstrap draws, the same losses for every regression,
and the same output, where the output is determined. About 8,000 random cases pass as of this
writing. How it works, what it checks, and what to do when a case fails is in
[`fuzz/README.md`](fuzz/README.md).
