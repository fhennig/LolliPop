// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Differential fuzz test: replays the cases of `generate.py` — LolliPop's own results on random
 * data and settings — on the port. The cases are in `fuzz/corpus` (a fixed set, checked in), or
 * in the directory `FUZZ_DIR` (a fresh run, see `run.mjs`).
 *
 * Per case:
 * - draws: the bootstrap draws the same mutations as LolliPop, round by round;
 * - regressions: a sample of LolliPop's fits, replayed one by one with LolliPop's inputs — the
 *   same NNLS solution, a robust cost as low as SciPy's minimum, the same confidence band from
 *   the same coefficients;
 * - losses: every fit of the pipeline reaches the same loss (NNLS), or one as low as SciPy's
 *   minimum (robust regression);
 * - output: the whole pipeline gives the same table (or fails where LolliPop fails). With the
 *   robust regressor, the values are compared with SciPy solving it to the minimum, but only
 *   reported (`flatDifferences`): where the minimum is flat, the losses are the same and the
 *   coefficients aren't determined.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

import { afterAll, describe, expect, test } from 'vitest';

import {
    deconvolute,
    NnlsReg,
    NullConfint,
    RobustReg,
    WaldConfint,
    type DeconvolutedRow,
    type Loss,
} from '../src/index.js';
import { compareOutputs, robustCost, tolerances } from './compare.js';
import { difference, loadCases, tallyRows, type FuzzCase, type RecordedCall } from './fuzzCase.js';

const NON_CONVEX = ['cauchy', 'arctan'];

const directory = process.env.FUZZ_DIR ?? new URL('corpus', import.meta.url).pathname;
const cases = loadCases(directory);

type CallReport = { regressor: string; difference: number; costRatio?: number; confintDifference?: number };
type CaseReport = {
    id: string;
    source: string;
    configKind: string;
    regressor: string;
    outcome: string;
    milliseconds?: number;
    maxDifference?: number;
    maxDifferenceTight?: number;
    flatDifferences?: string[];
    maxLossRatio?: number;
    tiedValues?: number;
    skippedValues?: number;
    calls: CallReport[];
};
const reports: CaseReport[] = [];

afterAll(() => {
    const path = process.env.FUZZ_REPORT;
    if (path) {
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, JSON.stringify(reports, null, 1));
    }
});

type PortRun = { output: DeconvolutedRow[]; draws: number[][]; losses: number[]; milliseconds: number };

function runPort(fuzzCase: FuzzCase): PortRun {
    const variantsConfig = structuredClone(fuzzCase.variantsConfig);
    if (variantsConfig.variants_list == null && fuzzCase.variantsDates === null && fuzzCase.variantOrder !== null) {
        // Without a list, LolliPop takes the variants in the order of a Python set, which varies
        // from run to run (and the order breaks ties): take LolliPop's.
        variantsConfig.variants_list = fuzzCase.variantOrder;
    }
    const draws: number[][] = [];
    const losses: number[] = [];
    const started = performance.now();
    const output = deconvolute({
        tally: tallyRows(fuzzCase.tally),
        columns: fuzzCase.tally.columns,
        variantsConfig,
        deconvConfig: structuredClone(fuzzCase.deconvConfig),
        variantsDates: fuzzCase.variantsDates,
        filters: fuzzCase.filters,
        // Likewise the locations, without `loc` (which then only checks that they're there).
        loc: fuzzCase.loc ?? (fuzzCase.locationOrder.length > 1 ? fuzzCase.locationOrder : null),
        random: fuzzCase.pythonSeed,
        onResample: (randIdcs) => draws.push(randIdcs),
        onLoss: (loss) => losses.push(loss),
    });
    return { output, draws, losses, milliseconds: performance.now() - started };
}

describe.each(cases.map((fuzzCase) => [fuzzCase.id, fuzzCase] as const))('%s', (_, fuzzCase) => {
    const report: CaseReport = {
        id: fuzzCase.id,
        source: fuzzCase.source,
        configKind: fuzzCase.configKind,
        regressor: fuzzCase.deconvConfig.regressor ?? 'nnls',
        outcome: 'error' in fuzzCase.expected ? 'python error' : 'ok',
        calls: [],
    };
    reports.push(report);

    if ('error' in fuzzCase.expected) {
        const pythonError = fuzzCase.expected.error;
        test(`fails like LolliPop (${pythonError.slice(0, 60)})`, () => {
            expect(() => runPort(fuzzCase)).toThrow();
        });
        return;
    }
    const expected = fuzzCase.expected.output;
    const robust = fuzzCase.deconvConfig.regressor === 'robust';
    const lossType = fuzzCase.deconvConfig.regressor_params?.loss_type ?? 'soft_l1';
    const convex = !robust || !NON_CONVEX.includes(lossType);

    let run: PortRun | undefined;
    const port = () => {
        run ??= runPort(fuzzCase);
        report.milliseconds = run.milliseconds;
        return run;
    };

    test('draws the same bootstrap rounds', () => {
        expect(port().draws).toEqual(fuzzCase.draws.map(({ idcs }) => idcs));
    });

    test.each(fuzzCase.calls.map((call, i) => [i, call] as const))('regression %i', (_, call) => {
        report.calls.push(checkCall(call));
    });

    test.runIf(convex)('reaches the same losses', () => {
        const { losses } = port();
        const reference = (robust ? fuzzCase.tightLosses : null) ?? fuzzCase.losses;
        expect(losses.length).toBe(reference.length);
        let maxLossRatio = 0;
        const problems: string[] = [];
        losses.forEach((loss, i) => {
            const target = reference[i];
            if (target === null) {
                return;
            }
            maxLossRatio = Math.max(maxLossRatio, loss / Math.max(target, 1e-300));
            // As low as LolliPop's (for the robust regression, SciPy's minimum), and not much
            // lower: that would be another objective.
            // (SciPy's NNLS sometimes misses the minimum, see `check_nnls` in generate.py: there,
            // the port's loss is lower.)
            const scipyWrong = !robust && Object.values(fuzzCase.ties).some((ties) => ties.scipyNnlsWrong);
            const above = loss > target * (1 + 1e-9) + 1e-12;
            // (SciPy solving to the minimum still stops up to about 1e-4 short of it, relatively.)
            const below = !scipyWrong && loss < target * (1 - (robust ? 1e-2 : 1e-9)) - (robust ? 1e-9 : 1e-12);
            if (above || below) {
                problems.push(`regression ${i}: loss ${loss} instead of ${target}`);
            }
        });
        report.maxLossRatio = maxLossRatio;
        expect(problems.slice(0, 5), `${fuzzCase.id}:\n${problems.slice(0, 5).join('\n')}`).toEqual([]);
    });

    test('gives the same output', () => {
        const actual = port().output;
        if (!convex) {
            // Non-convex losses have local minima, and which one a solver finds depends on its
            // path: SciPy's and the port's differ. Only the rows are compared.
            report.outcome = 'values not compared (non-convex loss)';
            const rows = compareOutputs(actual, expected, fuzzCase, Infinity);
            expect(rows.mismatches, `${fuzzCase.id}:\n${rows.mismatches.join('\n')}`).toEqual([]);
            return;
        }
        if (robust && fuzzCase.expectedTight !== null && 'output' in fuzzCase.expectedTight) {
            const tight = compareOutputs(actual, fuzzCase.expectedTight.output, fuzzCase, tolerances.robustOutput, {
                usePrecision: true,
            });
            const stock = compareOutputs(actual, expected, fuzzCase, Infinity);
            Object.assign(report, {
                maxDifference: stock.maxDifference,
                maxDifferenceTight: tight.maxDifference,
                flatDifferences: tight.mismatches,
                tiedValues: tight.tiedValues,
                skippedValues: tight.skippedValues,
            });
            // the rows themselves have to be the same
            expect(stock.mismatches, `${fuzzCase.id}:\n${stock.mismatches.join('\n')}`).toEqual([]);
        } else {
            const stock = compareOutputs(actual, expected, fuzzCase, tolerances.output);
            Object.assign(report, {
                maxDifference: stock.maxDifference,
                tiedValues: stock.tiedValues,
                skippedValues: stock.skippedValues,
            });
            expect(stock.mismatches, `${fuzzCase.id}:\n${stock.mismatches.join('\n')}`).toEqual([]);
        }
    });
});

function checkCall(call: RecordedCall): CallReport {
    const columns = call.variantNames.length;
    if (call.regressor === 'NnlsReg') {
        const fit = new NnlsReg().fit(call.X, call.y, call.k, columns);
        // The minimum of NNLS is unique only where the columns are independent; the residual is
        // always the same.
        expect(fit.loss).toBeLessThanOrEqual(call.rawCost * (1 + 1e-9) + 1e-12);
        const result = {
            regressor: 'NnlsReg',
            difference: Math.max(...fit.fitted.map((value, j) => difference(value, call.rawFitted[j]))),
        };
        checkConfint(call, result);
        return result;
    }
    const { loss_type, f_scale } = call.regressorParams!;
    const fit = new RobustReg({ loss_type: loss_type as Loss, f_scale }).fit(call.X, call.y, call.k, columns);
    const cost = robustCost(call, fit.fitted);
    // As low as SciPy's minimum (for the non-convex losses, a local minimum: no telling)
    if (!NON_CONVEX.includes(loss_type)) {
        expect(cost).toBeLessThanOrEqual(call.tightCost! * (1 + 1e-9) + 1e-12);
    }
    const result = {
        regressor: `RobustReg ${loss_type}`,
        difference: Math.max(...fit.fitted.map((value, j) => difference(value, call.tightFitted![j]))),
        costRatio: cost / call.tightCost!,
    };
    checkConfint(call, result);
    return result;
}

/** The confidence band from LolliPop's coefficients. */
function checkConfint(call: RecordedCall, result: CallReport): void {
    const confint = call.confint === 'WaldConfint' ? new WaldConfint(call.confintParams) : new NullConfint();
    const band = confint.confint({
        X: call.X.map((row, i) => row.map((value) => value * call.k[i])),
        coefs: call.fitted.map((value) => value ?? NaN),
        y: call.y,
        kvals: call.k,
    });
    // An ill-conditioned Fisher information loses digits in the inversion, like in NumPy.
    const tolerance = Math.max(tolerances.confint, 1e-15 * (call.fisherCondition ?? Infinity));
    const differences = (['lower', 'upper'] as const).flatMap((bound) =>
        band[bound].map((value, j) => difference(value, call.confBand[bound][j])),
    );
    result.confintDifference = Math.max(0, ...differences);
    if (call.fisherCondition !== null && call.fisherCondition !== undefined && call.fisherCondition < 1e12) {
        expect(result.confintDifference).toBeLessThanOrEqual(tolerance);
    }
}
