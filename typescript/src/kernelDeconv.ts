// SPDX-License-Identifier: GPL-3.0-or-later
// Port of LolliPop's lollipop/kerneldeconv.py.

import { WaldConfint, type ConfBand, type Confint } from './confints.js';
import { GaussianKernel, type Kernel } from './kernels.js';
import { sum } from './numeric/linalg.js';
import { NnlsReg, type Regressor } from './regressors.js';
import { toDay, unique, type Frame } from './table.js';

export type KernelDeconvArgs = {
    /** design matrix: one row per observation, one column per variant */
    X: number[][];
    /** the names of the columns of X */
    variantNames: string[];
    /** observed mutation frequencies */
    y: number[];
    /** observation dates, ISO */
    dates: string[];
    /** weights for the observations */
    weights?: number[];
    /** object with methods to compute kernel weighting */
    kernel?: Kernel;
    /** object with methods to compute the regression */
    reg?: Regressor;
    /** object with method to compute confidence bands */
    confint?: Confint;
};

export type DeconvFit = {
    fitted: number[];
    loss: number;
    confBand: ConfBand;
};

export type DeconvParams = {
    min_tol?: number;
    renormalize?: boolean;
};

/** Compute kernel deconvolution, using specified kernel function and regressor. */
export class KernelDeconv {
    readonly X: number[][];
    readonly y: number[];
    readonly dates: string[];
    readonly weights: number[];
    readonly kernel: Kernel;
    readonly reg: Regressor;
    readonly confint: Confint;
    readonly variantNames: string[];
    private readonly days: number[];

    fitted?: Frame;
    loss?: number[];
    confBands?: { lower: Frame; upper: Frame };

    constructor({
        X,
        variantNames,
        y,
        dates,
        weights,
        kernel = new GaussianKernel(),
        reg = new NnlsReg(),
        confint = new WaldConfint(),
    }: KernelDeconvArgs) {
        this.X = X;
        this.y = y;
        this.dates = dates;
        this.weights = weights ?? y.map(() => 1);
        this.kernel = kernel;
        this.reg = reg;
        this.confint = confint;
        this.variantNames = variantNames;
        this.days = dates.map(toDay);
    }

    /** compute kernel deconvolution centered on specific date, returns fitted regression */
    deconv(date: string, { min_tol = 1e-10, renormalize = true }: DeconvParams = {}): DeconvFit {
        // compute kernel values
        const day = toDay(date);
        const kernelValues = this.kernel.values(
            0,
            this.days.map((value) => day - value),
        );
        const kvals = kernelValues.map((value, i) => value * this.weights[i]);
        const selected = kvals.flatMap((value, i) => (value >= min_tol ? [i] : []));
        const X = selected.map((i) => this.X[i]);
        const y = selected.map((i) => this.y[i]);
        const k = selected.map((i) => kvals[i]);

        // compute fitted coefs
        const regfit = this.reg.fit(X, y, k, this.variantNames.length);

        // renormalize
        if (renormalize) {
            const total = sum(regfit.fitted);
            regfit.fitted = regfit.fitted.map((value) => value / total);
        }

        // compute confint
        const confBand = this.confint.confint({
            X: X.map((row, i) => row.map((value) => value * k[i])),
            coefs: regfit.fitted,
            y,
            kvals: k,
        });

        return { ...regfit, confBand };
    }

    /** compute kernel deconvolution for all dates */
    deconvAll(params: DeconvParams = {}): this {
        const fitted: number[][] = [];
        const loss: number[] = [];
        const lower: number[][] = [];
        const upper: number[][] = [];
        const dates = unique(this.dates);
        for (const date of dates) {
            const deconv = this.deconv(date, params);
            fitted.push(deconv.fitted);
            loss.push(deconv.loss);
            lower.push(deconv.confBand.lower);
            upper.push(deconv.confBand.upper);
        }

        const frame = (values: number[][]): Frame => ({ index: dates, columns: this.variantNames, values });
        this.fitted = frame(fitted);
        this.loss = loss;
        this.confBands = { lower: frame(lower), upper: frame(upper) };

        return this;
    }

    /** renormalize variants proportion so that they sum to 1 */
    renormalize(): this {
        if (this.fitted !== undefined) {
            this.fitted = {
                ...this.fitted,
                values: this.fitted.values.map((row) => {
                    const total = sum(row);
                    return row.map((value) => value / total);
                }),
            };
        }
        return this;
    }
}
