// SPDX-License-Identifier: GPL-3.0-or-later

import { type DeconvolutedRow } from '../src/index.js';
import { difference, type FuzzCase, type RecordedCall, type Table } from './fuzzCase.js';

export const tolerances = {
    /**
     * The output, where the port should agree up to rounding (NNLS, Wald intervals). (Bootstrap
     * means can be off a little more: where a column barely enters NNLS's active set in a round.)
     */
    output: 1e-8,
    /** The output with the robust regressor, against LolliPop solving it to the minimum. */
    robustOutput: 1e-6,
    /** The output with the robust regressor, against LolliPop as it is (SciPy's tolerances). */
    robustOutputStock: 0.05,
    /** A confidence band from the same coefficients. */
    confint: 1e-9,
};

export type Comparison = {
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
    { usePrecision = false } = {},
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

    // the sums over tied variants, per location and date: [port, LolliPop]
    const tiedSums = new Map<string, [number, number, number]>();
    expected.rows.slice(0, actual.length).forEach((values, i) => {
        const row = actual[i];
        for (const key of ['location', 'variant', 'date'] as const) {
            if (column(key) !== -1 && String(values[column(key)]) !== row[key]) {
                mismatches.push(`row ${i}: ${key} ${row[key]} instead of ${values[column(key)]}`);
            }
        }
        const ties = fuzzCase.ties[row.date];
        // How precisely the robust minimum is determined on the date: SciPy's own spread.
        const precision = usePrecision ? fuzzCase.precision[row.date] : 0;
        if (ties?.all || ties?.empty || ties?.scipyNnlsWrong || precision === null) {
            skippedValues++;
            return;
        }
        // An ill-conditioned design loses digits in any solver.
        const tolerance = Math.max(baseTolerance, 10 * (precision ?? 0), 1e-14 * (ties?.designCondition ?? 0));
        // An ill-conditioned Fisher information amplifies rounding (in NumPy too), and any
        // imprecision of the coefficients it is computed from, by its condition.
        const intervalTolerance =
            ties?.fisherCondition === null
                ? Infinity
                : Math.max(tolerance, (ties?.fisherCondition ?? 0) * Math.max(1e-14, 10 * (precision ?? 0)));
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
            if (key !== 'proportion' && !intervalsComparable) {
                skippedValues++;
                continue;
            }
            const d = difference(row[key], value(values, key));
            // (Past a relative error of 1, a numerically singular one: noise, also NaN or not.)
            if (key !== 'proportion' && intervalTolerance >= 1) {
                skippedValues++;
                continue;
            }
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
export function robustCost(call: RecordedCall, beta: readonly number[]): number {
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
