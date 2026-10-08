// SPDX-License-Identifier: GPL-3.0-or-later
// Port of LolliPop's lollipop/confints.py.

import { inv, LinAlgError, sum, type Matrix } from './numeric/linalg.js';
import { type RandomState } from './numeric/randomState.js';
import { normPpf } from './numeric/stats.js';
import { type Cell, type Row } from './table.js';

export type ConfBand = { lower: number[]; upper: number[] };

export type ConfintArgs = {
    /** The design matrix, its rows scaled by the kernel values. */
    X: Matrix;
    coefs: readonly number[];
    y?: readonly number[];
    kvals?: readonly number[];
};

export interface Confint {
    confint(args: ConfintArgs): ConfBand;
}

/** Class to return Nones as confint */
export class NullConfint implements Confint {
    confint({ coefs }: ConfintArgs): ConfBand {
        return {
            lower: coefs.map(() => NaN),
            upper: coefs.map(() => NaN),
        };
    }
}

export type WaldConfintParams = {
    level?: number;
    scale?: 'linear' | 'logit';
    pseudofrac?: number;
    quasi?: boolean;
    method?: 'all' | 'strat';
};

/** class to compute wald se of the proportions deconvolved through a linear prob. model */
export class WaldConfint implements Confint {
    readonly level: number;
    readonly scale: 'linear' | 'logit';
    readonly pseudofrac: number;
    readonly quasi: boolean;
    readonly method: 'all' | 'strat';

    constructor({
        level = 0.95,
        scale = 'linear',
        pseudofrac = 0.001,
        quasi = false,
        method = 'all',
    }: WaldConfintParams = {}) {
        this.level = level;
        this.scale = scale;
        this.pseudofrac = pseudofrac;
        this.quasi = quasi;
        this.method = method;
    }

    /**
     * compute overdispersion according to a quasibinomial model
     *
     * Returns a single factor (`method: 'all'`), or one per variant (`method: 'strat'`).
     */
    quasibinomOverdisp(
        X: Matrix,
        y: readonly number[],
        coefs: readonly number[],
        kvals: readonly number[],
    ): number | number[] {
        let yHat = matVec(X, coefs);
        yHat = yHat.map((value) => value + this.pseudofrac);
        yHat = yHat.map((value) => value / (value + (1 - value + 2 * this.pseudofrac)));

        const expectedVar = yHat.map((value) => value * (1 - value));

        if (this.method === 'all') {
            const kvalsSum = sum(kvals);
            return Math.sqrt(
                sum(y.map((value, i) => (((value - yHat[i]) ** 2 / expectedVar[i]) * kvals[i]) / kvalsSum)),
            );
        } else {
            const n = coefs.length;
            const mutInd = X.map((row) => row[n - 1] === 0);
            const overdispVals = y.map((value, i) => ((value - yHat[i]) ** 2 / expectedVar[i]) * kvals[i]);
            const norm = new Array<number>(n).fill(0);
            const overdispAgg = new Array<number>(n).fill(0);
            X.forEach((row, i) => {
                if (mutInd[i]) {
                    for (let j = 0; j < n; j++) {
                        norm[j] += kvals[i] * row[j];
                        overdispAgg[j] += overdispVals[i] * row[j];
                    }
                }
            });
            for (let j = 0; j < n; j++) {
                overdispAgg[j] /= norm[j];
            }
            overdispAgg[n - 1] = sum(overdispVals.filter((_, i) => mutInd[i])) / sum(kvals.filter((_, i) => mutInd[i]));

            return overdispAgg.map(Math.sqrt);
        }
    }

    /** The inverse Fisher information of the linear probability model, or `undefined` if singular. */
    private inverseInfo(X: Matrix, coefs: readonly number[]): Matrix | undefined {
        const total = sum(coefs);
        const pseudocoefs = coefs.map((value) => value / total);
        // compute fitted values of mutation props.
        let fittedMut = matVec(X, pseudocoefs);
        // add pseudofrac to avoid zero fitted mutations
        fittedMut = fittedMut.map((value) => value + this.pseudofrac);
        fittedMut = fittedMut.map((value) => value / (value + (1 - value + 2 * this.pseudofrac)));
        // compute Fisher information of in terms of binomial model
        const binomInfo = fittedMut.map((value) => 1 / (value * (1 - value)));
        // compute Fisher information of in terms of linear prob. model: Xᵀ diag(binomInfo) X
        const n = coefs.length;
        const linprobInfo = Array.from({ length: n }, () => new Array<number>(n).fill(0));
        X.forEach((row, i) => {
            for (let j = 0; j < n; j++) {
                const scaled = row[j] * binomInfo[i];
                for (let l = 0; l < n; l++) {
                    linprobInfo[j][l] += scaled * row[l];
                }
            }
        });
        // try to invert Fisher information matrix
        try {
            return inv(linprobInfo);
        } catch (error) {
            if (error instanceof LinAlgError) {
                return undefined;
            }
            throw error;
        }
    }

    /** compute standard errors on the linear scale */
    standardError(X: Matrix, coefs: readonly number[]): number[] {
        const inverse = this.inverseInfo(X, coefs);
        // leave the nans in place, compute root on non-negative
        return coefs.map((_, j) => (inverse !== undefined && inverse[j][j] >= 0.0 ? Math.sqrt(inverse[j][j]) : NaN));
    }

    /** compute the standard error on the logit scale using the Delta method */
    logitStandardError(X: Matrix, coefs: readonly number[]): number[] {
        const n = coefs.length;
        const invInfo = this.inverseInfo(X, coefs) ?? Array.from({ length: n }, () => new Array<number>(n).fill(NaN));

        // construct the jacobian
        const total = sum(coefs);
        let pseudocoefs = coefs.map((value) => value / total);
        const pseudoTotal = sum(pseudocoefs);
        pseudocoefs = pseudocoefs.map((value) => (value + this.pseudofrac) / (pseudoTotal + 2 * this.pseudofrac));
        // -1/(p(1-p)) in every column of row i, and its opposite on the diagonal
        const jacobian = pseudocoefs.map((value, i) => {
            const entry = -1 * (1 / (value * (1 - value)));
            return pseudocoefs.map((_, j) => (i === j ? entry + 2 * (-1 * entry) : entry));
        });

        // project inverse fisher inf. on logit scale: diag(J inv Jᵀ)
        return jacobian.map((row) => {
            let variance = 0;
            for (let j = 0; j < n; j++) {
                let projected = 0;
                for (let l = 0; l < n; l++) {
                    projected += invInfo[j][l] * row[l];
                }
                variance += row[j] * projected;
            }
            return Math.sqrt(variance);
        });
    }

    /** compute confidence intervals on the linear scale */
    confint({ X, coefs, y, kvals }: ConfintArgs): ConfBand {
        const z = normPpf(1 - (1 - this.level) / 2);
        const withOverdisp = (se: number[]): number[] => {
            if (!this.quasi) {
                return se;
            }
            const overdisp = this.quasibinomOverdisp(X, y!, coefs, kvals!);
            return se.map((value, j) => value * (typeof overdisp === 'number' ? overdisp : overdisp[j]));
        };

        if (this.scale === 'linear') {
            const se = withOverdisp(this.standardError(X, coefs));
            return {
                lower: coefs.map((value, j) => value - z * se[j]),
                upper: coefs.map((value, j) => value + z * se[j]),
            };
        } else {
            const se = withOverdisp(this.logitStandardError(X, coefs));

            let fittedPseudo = coefs.map((value) => value + this.pseudofrac);
            const pseudoTotal = sum(fittedPseudo);
            fittedPseudo = fittedPseudo.map((value) => value / pseudoTotal);
            const logitFittedPseudo = fittedPseudo.map((value) => Math.log(value) - Math.log(1 - value));
            const logitFitted = coefs.map((value) => Math.log(value) - Math.log(1 - value));

            // like np.min / np.max over the two: NaN if either is
            return {
                lower: logitFitted.map((value, j) => Math.min(value - z * se[j], logitFittedPseudo[j] - z * se[j])),
                upper: logitFitted.map((value, j) => Math.max(value + z * se[j], logitFittedPseudo[j] + z * se[j])),
            };
        }
    }
}

function matVec(X: Matrix, v: readonly number[]): number[] {
    return X.map((row) => {
        let result = 0;
        for (let j = 0; j < v.length; j++) {
            result += row[j] * v[j];
        }
        return result;
    });
}

/**
 * Function to resample mutations by replacement (preserving mutation-complement pairs).
 * Returns a copy of the rows with a `resample_value` column indicating how many times the
 * mutation was in the resample, and the drawn indices.
 *
 * `mutations` are the unique names, the mutations first, then their complements in the same order.
 */
export function resampleMutations(
    dfCity1: readonly Row[],
    mutations: readonly Cell[],
    random: RandomState,
    namefield = 'mutations',
): { dfSampled: Row[]; randIdcs: number[] } {
    const half = Math.trunc(mutations.length / 2);
    // resample indices of mutations with replacement (warning: high is one above actual high!
    const randIdcs = random.randint(0, half, half);
    // for all mutations, count how many times they appear in the resample (0, 1, 2 ...)
    const resamplesCounts = new Array<number>(half).fill(0);
    for (const index of randIdcs) {
        resamplesCounts[index]++;
    }
    // make a dict of {mutation : occurences in the resample} pairs
    const counts = [...resamplesCounts, ...resamplesCounts];
    const resampleCoeffDict = new Map<Cell, number>();
    mutations.forEach((mutation, i) => {
        if (i < counts.length) {
            resampleCoeffDict.set(mutation, counts[i]);
        }
    });
    // make a column with coefficients for how many times a row should be accounted for according to the resample
    const dfSampled = dfCity1.map((row) => ({ ...row, resample_value: resampleCoeffDict.get(row[namefield]) ?? NaN }));

    return { dfSampled, randIdcs };
}
