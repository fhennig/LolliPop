// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The checks of the fuzz test: whether the port reproduces what LolliPop did with a case. Each
 * check returns its problems (none: passed), see README.md.
 */

import { NnlsReg, NullConfint, RobustReg, WaldConfint, type DeconvolutedRow, type Loss } from '../src/index.js';
import { runPort, type FuzzCase, type PortRun, type RecordedCall, type Table } from './runPort.js';

export const tolerances = {
    /**
     * The output, where the port should agree up to rounding (NNLS, Wald intervals). (Bootstrap
     * means can be off a little more: where a column barely enters NNLS's active set in a round.)
     */
    output: 1e-8,
    /** A confidence band from the same coefficients. */
    confint: 1e-9,
};

/** The losses with local minima, where solvers can end up in different ones. */
const NON_CONVEX = ['cauchy', 'arctan'];

export type Evaluation = {
    /** Per check, its problems. */
    problems: Record<string, string[]>;
    report: CaseReport;
};

export type CaseReport = {
    id: string;
    source: string;
    configKind: string;
    regressor: string;
    outcome: string;
    milliseconds?: number;
    /** The largest difference from LolliPop's output (with the robust regressor: not checked). */
    maxDifference?: number;
    /** The largest ratio of a loss to LolliPop's (robust: to SciPy's minimum). */
    maxLossRatio?: number;
    tiedValues?: number;
    skippedValues?: number;
    calls: { regressor: string; difference: number; costRatio?: number; confintDifference?: number }[];
};

export function evaluateCase(fuzzCase: FuzzCase): Evaluation {
    const robust = fuzzCase.deconvConfig.regressor === 'robust';
    const report: CaseReport = {
        id: fuzzCase.id,
        source: fuzzCase.source,
        configKind: fuzzCase.configKind,
        regressor: fuzzCase.deconvConfig.regressor ?? 'nnls',
        outcome: 'ok',
        calls: [],
    };
    const run = runPort(fuzzCase);
    report.milliseconds = run.milliseconds;

    if ('error' in fuzzCase.expected) {
        report.outcome = 'python error';
        return {
            problems: { 'fails like LolliPop': 'output' in run.result ? [`no error, unlike LolliPop (${fuzzCase.expected.error})`] : [] }, // prettier-ignore
            report,
        };
    }
    if ('error' in run.result) {
        report.outcome = 'port error';
        return { problems: { runs: [`fails: ${run.result.error}`] }, report };
    }
    const problems: Record<string, string[]> = {
        'draws the same bootstrap rounds': checkDraws(fuzzCase, run),
    };
    fuzzCase.calls.forEach((call, i) => {
        const { problems: callProblems, report: callReport } = checkCall(call);
        problems[`regression ${i}`] = callProblems;
        report.calls.push(callReport);
    });
    const lossType = fuzzCase.deconvConfig.regressor_params?.loss_type ?? 'soft_l1';
    const convex = !robust || !NON_CONVEX.includes(lossType);
    if (convex) {
        problems['reaches the same losses'] = checkLosses(fuzzCase, run, report);
    }

    // The output. With the robust regressor, SciPy stops short of the minimum, and where the
    // minimum is flat the coefficients aren't determined beyond the losses (checked above): only
    // the rows are compared, the differences are in the report. Likewise with non-convex losses,
    // where the solvers can stop at different local minima.
    const comparison = compareOutputs(run.result.output, fuzzCase.expected.output, fuzzCase, robust ? Infinity : tolerances.output); // prettier-ignore
    Object.assign(report, {
        maxDifference: comparison.maxDifference,
        tiedValues: comparison.tiedValues,
        skippedValues: comparison.skippedValues,
    });
    if (robust) {
        report.outcome = convex ? 'values not compared (robust)' : 'values not compared (non-convex loss)';
    }
    problems['gives the same output'] = comparison.mismatches;
    return { problems, report };
}

/** Every bootstrap round draws the same mutations as LolliPop. */
function checkDraws(fuzzCase: FuzzCase, run: PortRun): string[] {
    const expected = fuzzCase.draws.map(({ idcs }) => idcs);
    if (run.draws.length !== expected.length) {
        return [`${run.draws.length} bootstrap rounds instead of ${expected.length}`];
    }
    const round = run.draws.findIndex((draws, i) => draws.join() !== expected[i].join());
    return round === -1 ? [] : [`round ${round} draws ${run.draws[round].join()} instead of ${expected[round].join()}`];
}

/**
 * Every regression reaches the same loss as LolliPop's (NNLS), or one as low as SciPy's minimum
 * (robust regression), and not much lower: that would be another objective.
 */
function checkLosses(fuzzCase: FuzzCase, run: PortRun, report: CaseReport): string[] {
    const robust = fuzzCase.deconvConfig.regressor === 'robust';
    const reference = (robust ? fuzzCase.tightLosses : null) ?? fuzzCase.losses;
    if (run.losses.length !== reference.length) {
        return [`${run.losses.length} regressions instead of ${reference.length}`];
    }
    // (SciPy's NNLS sometimes misses the minimum, see `check_nnls`: there, the port's is lower.)
    const scipyWrong = !robust && Object.values(fuzzCase.ties).some((ties) => ties.scipyNnlsWrong);
    let maxLossRatio = 0;
    const problems: string[] = [];
    run.losses.forEach((loss, i) => {
        const target = reference[i];
        if (target === null) {
            return;
        }
        maxLossRatio = Math.max(maxLossRatio, loss / Math.max(target, 1e-300));
        const above = loss > target * (1 + 1e-9) + 1e-12;
        // (SciPy solving to the minimum still stops up to about 1e-4 short of it, relatively.)
        const below = !scipyWrong && loss < target * (1 - (robust ? 1e-2 : 1e-9)) - (robust ? 1e-9 : 1e-12);
        if (above || below) {
            problems.push(`regression ${i}: loss ${loss} instead of ${target}`);
        }
    });
    report.maxLossRatio = maxLossRatio;
    return problems.slice(0, 5);
}

/**
 * A regression of LolliPop's, replayed on its own: the same NNLS residual, or a robust cost as
 * low as SciPy's minimum; and the same confidence band from LolliPop's coefficients.
 */
function checkCall(call: RecordedCall): { problems: string[]; report: CaseReport['calls'][number] } {
    const problems: string[] = [];
    let report: CaseReport['calls'][number];
    const columns = call.variantNames.length;
    if (call.regressor === 'NnlsReg') {
        const fit = new NnlsReg().fit(call.X, call.y, call.k, columns);
        // The minimum of NNLS is unique only where the columns are independent; the residual is.
        if (fit.loss > call.rawCost! * (1 + 1e-9) + 1e-12) {
            problems.push(`NNLS residual ${fit.loss} instead of ${call.rawCost}`);
        }
        report = {
            regressor: 'NnlsReg',
            difference: Math.max(...fit.fitted.map((value, j) => difference(value, call.rawFitted![j]))),
        };
    } else {
        const { loss_type, f_scale } = call.regressorParams!;
        const fit = new RobustReg({ loss_type: loss_type as Loss, f_scale }).fit(call.X, call.y, call.k, columns);
        const cost = robustCost(call, fit.fitted);
        // (for the non-convex losses, a local minimum: no telling)
        if (!NON_CONVEX.includes(loss_type) && cost > call.tightCost! * (1 + 1e-9) + 1e-12) {
            problems.push(`robust cost ${cost} above SciPy's minimum ${call.tightCost}`);
        }
        report = {
            regressor: `RobustReg ${loss_type}`,
            difference: Math.max(...fit.fitted.map((value, j) => difference(value, call.tightFitted![j]))),
            costRatio: cost / call.tightCost!,
        };
    }

    const confint = call.confint === 'WaldConfint' ? new WaldConfint(call.confintParams) : new NullConfint();
    const band = confint.confint({
        X: call.X.map((row, i) => row.map((value) => value * call.k[i])),
        coefs: call.fitted.map((value) => value ?? NaN),
        y: call.y,
        kvals: call.k,
    });
    report.confintDifference = Math.max(
        0,
        ...(['lower', 'upper'] as const).flatMap((bound) =>
            band[bound].map((value, j) => difference(value, call.confBand[bound][j])),
        ),
    );
    // An ill-conditioned Fisher information loses digits in the inversion, like in NumPy.
    const tolerance = Math.max(tolerances.confint, 1e-15 * (call.fisherCondition ?? Infinity));
    if ((call.fisherCondition ?? Infinity) < 1e12 && report.confintDifference > tolerance) {
        problems.push(`confidence band off by ${report.confintDifference}`);
    }
    return { problems, report };
}

type Comparison = {
    mismatches: string[];
    maxDifference: number;
    /** Values compared only as sums over variants that some fit couldn't tell apart. */
    tiedValues: number;
    /** Values not compared, where some fit was degenerate or empty. */
    skippedValues: number;
};

const VALUES = ['proportion', 'proportionLower', 'proportionUpper'] as const;

/**
 * Compares the port's output with LolliPop's table, row by row (both are sorted the same).
 *
 * Where a fit couldn't tell some variants apart (`ties`), the split between them is down to
 * rounding, in SciPy as in the port: there, only their summed proportion is compared, not their
 * intervals (and no Wald interval on the date); where a fit was degenerate otherwise, or empty, or
 * SciPy's NNLS got it wrong, nothing is.
 */
export function compareOutputs(
    actual: DeconvolutedRow[],
    expected: Table,
    fuzzCase: FuzzCase,
    baseTolerance: number,
): Comparison {
    const mismatches: string[] = [];
    let maxDifference = 0;
    let tiedValues = 0;
    let skippedValues = 0;
    if (actual.length !== expected.rows.length) {
        mismatches.push(`${actual.length} rows instead of ${expected.rows.length}`);
    }
    const column = (name: string) => expected.columns.indexOf(name);
    const value = (values: Table['rows'][number], name: string) => values[column(name)] as number | null;

    // the sums over tied variants, per location and date: [port, LolliPop, tolerance]
    const tiedSums = new Map<string, [number, number, number]>();
    expected.rows.slice(0, actual.length).forEach((values, i) => {
        const row = actual[i];
        for (const key of ['location', 'variant', 'date'] as const) {
            if (column(key) !== -1 && String(values[column(key)]) !== row[key]) {
                mismatches.push(`row ${i}: ${key} ${row[key]} instead of ${values[column(key)]}`);
            }
        }
        const ties = fuzzCase.ties[row.date];
        if (ties?.all || ties?.empty || ties?.scipyNnlsWrong) {
            skippedValues++;
            return;
        }
        // An ill-conditioned design loses digits in any solver: with a residual, least squares
        // coefficients are only determined up to about κ² ε (Wedin), where κ is the design's
        // condition. (Past that, the losses still have to agree.)
        const condition = ties?.designCondition ?? 0;
        const tolerance = Math.max(baseTolerance, 1e-14 * condition, Number.EPSILON * condition * condition);
        // An ill-conditioned Fisher information amplifies rounding, in NumPy too, by its condition.
        const intervalTolerance =
            ties?.fisherCondition === null ? Infinity : Math.max(tolerance, (ties?.fisherCondition ?? 0) * 1e-14);
        // With ties, the Fisher information that Wald intervals invert is singular: every
        // interval on the date is rounding noise.
        const intervalsComparable = !(ties?.groups.length && fuzzCase.deconvConfig.confint === 'wald');
        const group = ties?.groups.find((members) => members.includes(row.variant));
        if (group !== undefined) {
            tiedValues++;
            const key = JSON.stringify([row.location, row.date, group]);
            const sums = tiedSums.get(key) ?? [0, 0, tolerance];
            sums[0] += row.proportion;
            sums[1] += value(values, 'proportion') ?? NaN;
            tiedSums.set(key, sums);
            return;
        }
        for (const key of VALUES) {
            if (column(key) === -1) {
                continue;
            }
            // (Past a relative error of 1, a numerically singular interval: noise, also NaN or not.)
            if (key !== 'proportion' && (!intervalsComparable || intervalTolerance >= 1)) {
                skippedValues++;
                continue;
            }
            const d = difference(row[key], value(values, key));
            maxDifference = Math.max(
                maxDifference,
                key === 'proportion' ? d : Math.min(d, (d * tolerance) / intervalTolerance),
            );
            if (d > (key === 'proportion' ? tolerance : intervalTolerance)) {
                mismatches.push(
                    `row ${i} (${row.location} ${row.variant} ${row.date}): ${key} ${row[key]} instead of ${value(values, key)}`,
                );
            }
        }
    });
    for (const [key, [port, lollipop, tolerance]] of tiedSums) {
        const d = difference(port, lollipop);
        maxDifference = Math.max(maxDifference, d);
        if (d > tolerance) {
            mismatches.push(`tied ${key}: ${port} instead of ${lollipop}`);
        }
    }
    if (mismatches.length > 5) {
        mismatches.splice(5, mismatches.length - 5, `… (${mismatches.length} mismatches)`);
    }
    return { mismatches, maxDifference, tiedValues, skippedValues };
}

/** SciPy's cost of the robust regression at `beta`. */
function robustCost(call: RecordedCall, beta: readonly number[]): number {
    const { loss_type: loss, f_scale: f } = call.regressorParams!;
    let total = 0;
    call.X.forEach((row, i) => {
        const residual = call.k[i] * row.reduce((sum, value, j) => sum + value * beta[j], 0) - call.k[i] * call.y[i];
        const z = (residual / f) ** 2;
        total +=
            loss === 'linear'
                ? z
                : loss === 'soft_l1'
                  ? 2 * (Math.sqrt(1 + z) - 1)
                  : loss === 'huber'
                    ? z <= 1
                        ? z
                        : 2 * Math.sqrt(z) - 1
                    : loss === 'cauchy'
                      ? Math.log1p(z)
                      : Math.atan(z);
    });
    return 0.5 * f * f * total;
}

/** A missing value in the output: NaN, or ±inf (which the CSV round trip turns into null). */
function isNullish(value: number | null | undefined): boolean {
    return value === null || value === undefined || !Number.isFinite(value);
}

/** `|a − b|`, scaled down for large values; 0 if both are missing, ∞ if one is. */
function difference(a: number | null | undefined, b: number | null | undefined): number {
    if (isNullish(a) || isNullish(b)) {
        return isNullish(a) && isNullish(b) ? 0 : Infinity;
    }
    return Math.abs(a! - b!) / Math.max(1, Math.abs(a!), Math.abs(b!));
}
