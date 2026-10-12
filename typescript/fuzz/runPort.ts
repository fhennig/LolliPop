// SPDX-License-Identifier: GPL-3.0-or-later

/** The fuzz cases that `run_lollipop.py` writes, and running the port on them. */

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';

import {
    deconvolute,
    type DeconvConfig,
    type DeconvolutedRow,
    type Filters,
    type Row,
    type VarDates,
    type VariantsConfig,
} from '../src/index.js';

export type Table = { columns: string[]; rows: (string | number | boolean | null)[][] };

export type FuzzCase = {
    id: string;
    seed: number;
    source: string;
    notes: string | null;
    configKind: string;
    /** The scenario: what `lollipop deconvolute` got. */
    tally: Table;
    variantsConfig: VariantsConfig;
    deconvConfig: DeconvConfig;
    variantsDates: VarDates | null;
    filters: Filters | null;
    loc: string[] | null;
    /** The seed of NumPy's global generator, which the bootstrap draws from. */
    pythonSeed: number;
    /** What LolliPop did with it. */
    expected: { output: Table } | { error: string; traceback: string };
    draws: { half: number; idcs: number[] }[];
    calls: RecordedCall[];
    /** The dates where a value isn't determined (see `notable` in run_lollipop.py). */
    ties: Record<string, Ties>;
    /** The loss of every regression, in order: LolliPop's, and the minimum's (robust only). */
    losses: (number | null)[];
    tightLosses: (number | null)[] | null;
    /** The order of the variants and locations, where LolliPop takes them from a Python set. */
    variantOrder: string[] | null;
    locationOrder: string[];
};

/** A regression in full: its inputs, LolliPop's fit, and its confidence band. */
export type RecordedCall = {
    date: string;
    minTol: number;
    renormalize: boolean;
    variantNames: string[];
    X: number[][];
    y: number[];
    k: number[];
    regressor: 'NnlsReg' | 'RobustReg';
    regressorParams?: { loss_type: string; f_scale: number };
    confint: 'NullConfint' | 'WaldConfint';
    confintParams?: {
        level: number;
        scale: 'linear' | 'logit';
        pseudofrac: number;
        quasi: boolean;
        method: 'all' | 'strat';
    };
    fisherCondition?: number | null;
    /** After renormalizing. */
    fitted: (number | null)[];
    loss: number;
    confBand: { lower: (number | null)[]; upper: (number | null)[] };
    /** NnlsReg: SciPy's fit before renormalizing, and the norm of its residual. */
    rawFitted?: (number | null)[];
    rawCost?: number;
    /** RobustReg: SciPy solving it to the minimum, and its cost. */
    tightFitted?: number[];
    tightCost?: number;
};

export type Ties = {
    /** Variants with identical columns in some fit on the date. */
    groups: string[][];
    /** Some fit's design was degenerate in another way. */
    all: boolean;
    /** Some fit had no rows. */
    empty: boolean;
    /** SciPy's NNLS returned a non-optimal solution in some fit (see `check_nnls`). */
    scipyNnlsWrong?: boolean;
    /**
     * The largest condition of the (weighted) design of a fit, with each group of tied variants
     * merged into one column, where above 1e3 (null: singular).
     */
    designCondition?: number | null;
    /** The largest condition of the Fisher information of a Wald interval (null: singular). */
    fisherCondition?: number | null;
};

export type PortRun = {
    /** The output, or the error the port threw. */
    result: { output: DeconvolutedRow[] } | { error: string };
    draws: number[][];
    losses: number[];
    milliseconds: number;
};

export function loadCases(directory: string, only?: string): FuzzCase[] {
    return readdirSync(directory)
        .filter((file) => file.endsWith('.json.gz') && (only === undefined || file === `${only}.json.gz`))
        .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }))
        .map((file) => JSON.parse(gunzipSync(readFileSync(join(directory, file))).toString()) as FuzzCase);
}

/** Runs the port's `deconvolute` on the case's scenario, like `lollipop deconvolute` was. */
export function runPort(fuzzCase: FuzzCase): PortRun {
    const variantsConfig = structuredClone(fuzzCase.variantsConfig);
    if (variantsConfig.variants_list == null && fuzzCase.variantsDates === null && fuzzCase.variantOrder !== null) {
        // Without a list, LolliPop takes the variants in the order of a Python set, which varies
        // from run to run (and the order breaks ties): take LolliPop's.
        variantsConfig.variants_list = fuzzCase.variantOrder;
    }
    const draws: number[][] = [];
    const losses: number[] = [];
    const started = performance.now();
    let result: PortRun['result'];
    try {
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
        result = { output };
    } catch (error) {
        result = { error: String(error) };
    }
    return { result, draws, losses, milliseconds: performance.now() - started };
}

export function tallyRows({ columns, rows }: Table): Row[] {
    return rows.map((values) =>
        Object.fromEntries(
            columns.map((column, i) => [column, typeof values[i] === 'boolean' ? String(values[i]) : values[i]]),
        ),
    ) as Row[];
}
