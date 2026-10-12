// SPDX-License-Identifier: GPL-3.0-or-later

/** The fuzz cases that `generate.py` writes, and how to read them. */

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { type DeconvConfig, type Filters, type Row, type VarDates, type VariantsConfig } from '../src/index.js';

export type Table = { columns: string[]; rows: (string | number | boolean | null)[][] };

export type LollipopRun = { output: Table } | { error: string; traceback: string };

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
    /** Before renormalizing. */
    rawFitted: (number | null)[];
    /** NnlsReg: the norm of the residual; RobustReg: the cost. */
    rawCost: number;
    tightFitted?: number[];
    tightCost?: number;
};

export type FuzzCase = {
    id: string;
    seed: number;
    source: string;
    notes: string | null;
    configKind: string;
    tally: Table;
    variantsConfig: VariantsConfig;
    deconvConfig: DeconvConfig;
    variantsDates: VarDates | null;
    filters: Filters | null;
    loc: string[] | null;
    pythonSeed: number;
    expected: LollipopRun;
    expectedTight: LollipopRun | null;
    draws: { half: number; idcs: number[] }[];
    calls: RecordedCall[];
    callCount: number;
    /** The dates where a fit couldn't tell variants apart, see `generate.py`. */
    ties: Record<string, Ties>;
    /** Per date, how precisely the robust minimum is determined, in the proportions (null: not). */
    precision: Record<string, number | null>;
    variantOrder: string[] | null;
    locationOrder: string[];
    /** The loss of every regression, in order: LolliPop's, and the minimum's (robust only). */
    losses: number[];
    tightLosses: number[] | null;
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

export function loadCases(directory: string): FuzzCase[] {
    return readdirSync(directory)
        .filter((file) => file.endsWith('.json.gz'))
        .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }))
        .map((file) => JSON.parse(gunzipSync(readFileSync(join(directory, file))).toString()) as FuzzCase);
}

export function tallyRows({ columns, rows }: Table): Row[] {
    return rows.map((values) =>
        Object.fromEntries(
            columns.map((column, i) => [column, typeof values[i] === 'boolean' ? String(values[i]) : values[i]]),
        ),
    ) as Row[];
}

/** A missing value in the output: NaN, or ±inf (which the CSV round trip turns into null). */
export function isNullish(value: number | null | undefined): boolean {
    return value === null || value === undefined || !Number.isFinite(value);
}

/** `|a − b|`, scaled down for large values; 0 if both are missing, ∞ if one is. */
export function difference(a: number | null | undefined, b: number | null | undefined): number {
    if (isNullish(a) || isNullish(b)) {
        return isNullish(a) && isNullish(b) ? 0 : Infinity;
    }
    return Math.abs(a! - b!) / Math.max(1, Math.abs(a!), Math.abs(b!));
}
