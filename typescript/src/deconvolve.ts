// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The deconvolution as a dashboard calls it: from the signatures of a panel of lineages and the
 * frequencies of their mutations to the prevalences over time. Not part of LolliPop: it builds
 * the tally table that LolliPop reads and runs `deconvolute` on it.
 */

import { deconvolute, type DeconvConfig } from './deconvolute.js';
import { type Log } from './preprocessors.js';
import { PRESETS } from './presets.js';
import { type RandomState } from './numeric/randomState.js';
import { type Cell, type Row } from './table.js';

/** The extra variant that takes up what the panel doesn't explain. */
export const UNDETERMINED = 'undetermined';

/** The frequency of one mutation in one sample. */
export type MutationFrequency = {
    /** ISO `yyyy-mm-dd`, the sampling date. */
    date: string;
    /** Like `C241T`: any name, as long as the signatures use the same. */
    mutation: string;
    /** Between 0 and 1. */
    frac: number;
};

export type DeconvolveInput = {
    /** The mutations of each lineage of the panel, in the order of the panel. */
    signatures: Record<string, readonly string[]>;
    frequencies: readonly MutationFrequency[];
    /** By default, LolliPop's `deconv_bootstrap_cowwid` preset, the one of the Swiss surveillance. */
    config?: DeconvConfig;
    /** The seed or generator of the bootstrap; by default, seed 0. */
    random?: RandomState | number;
    log?: Log;
    /** Called with the draws of each bootstrap round, for tests (see `deconvolute`). */
    onResample?: (randIdcs: number[], mutations: readonly Cell[]) => void;
};

export type DeconvolutionEstimate = {
    /** The lineage, or `UNDETERMINED`. */
    variant: string;
    proportion: number;
    /** With a bootstrap or Wald confidence intervals, else NaN. */
    lower: number;
    upper: number;
};

export type DeconvolutionResult = {
    /** The lineages of the panel, then `UNDETERMINED`. */
    variants: string[];
    /** The mutations that tell the lineages of the panel apart. */
    informativeMutations: string[];
    /** Every sampling date, ascending, with one estimate per variant (in the order of `variants`). */
    dates: { date: string; estimates: DeconvolutionEstimate[] }[];
};

export function deconvolve({
    signatures,
    frequencies,
    config = PRESETS.deconv_bootstrap_cowwid,
    random,
    log,
    onResample,
}: DeconvolveInput): DeconvolutionResult {
    const lineages = Object.keys(signatures);
    const variants = [...lineages, UNDETERMINED];
    const sets = lineages.map((lineage) => new Set(signatures[lineage]));

    const columns = ['date', 'frac', 'mutations', ...lineages];
    const tally: Row[] = frequencies.map(({ date, mutation, frac }) => ({
        date,
        frac,
        mutations: mutation,
        ...Object.fromEntries(lineages.map((lineage, i) => [lineage, sets[i].has(mutation) ? 'mut' : null])),
    }));

    const rows = deconvolute({
        tally,
        columns,
        variantsConfig: {
            variants_pangolin: Object.fromEntries(lineages.map((lineage) => [lineage, lineage])),
            variants_list: lineages,
            no_loc: true,
            remove_deletions: false,
        },
        deconvConfig: config,
        random,
        log,
        onResample,
    });

    const byDate = new Map<string, Map<string, DeconvolutionEstimate>>();
    for (const row of rows) {
        let estimates = byDate.get(row.date);
        if (estimates === undefined) {
            estimates = new Map();
            byDate.set(row.date, estimates);
        }
        estimates.set(row.variant, {
            variant: row.variant,
            proportion: row.proportion,
            lower: row.proportionLower ?? NaN,
            upper: row.proportionUpper ?? NaN,
        });
    }
    return {
        variants,
        informativeMutations: informativeMutations(signatures),
        dates: [...byDate.keys()].sort().map((date) => ({
            date,
            estimates: variants.map(
                (variant) => byDate.get(date)!.get(variant) ?? { variant, proportion: NaN, lower: NaN, upper: NaN },
            ),
        })),
    };
}

/** The mutations that tell the lineages of the panel apart: some lineages carry them, but not all. */
export function informativeMutations(signatures: Record<string, readonly string[]>): string[] {
    const sets = Object.values(signatures).map((mutations) => new Set(mutations));
    return [...new Set(sets.flatMap((set) => [...set]))].filter((mutation) => {
        const carriers = sets.filter((set) => set.has(mutation)).length;
        return carriers > 0 && carriers < sets.length;
    });
}
