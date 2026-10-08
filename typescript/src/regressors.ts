// SPDX-License-Identifier: GPL-3.0-or-later
// Port of LolliPop's lollipop/regressors.py.

import { leastSquares, type Loss } from './numeric/leastSquares.js';
import { nnls } from './numeric/nnls.js';

export type RegressionFit = {
    /** The fitted coefficients, one per variant. */
    fitted: number[];
    /** NnlsReg: the norm of the residual; RobustReg: SciPy's `cost`. */
    loss: number;
};

export interface Regressor {
    /**
     * fit f(X) ~= y, weighted by values of k
     *
     * X: array of variant definition (design matrix)
     * y: array of observed mutation frequencies
     * k: kernel weighting values
     * columns: the number of columns of X (which a NumPy array knows even without rows)
     */
    fit(X: readonly (readonly number[])[], y: readonly number[], k: readonly number[], columns?: number): RegressionFit;
}

/** wrapper around nnls to feed to kernel_deconv */
export class NnlsReg implements Regressor {
    fit(
        X: readonly (readonly number[])[],
        y: readonly number[],
        k: readonly number[],
        columns = X[0]?.length ?? 0,
    ): RegressionFit {
        const { x, rnorm } = nnls(
            X.map((row, i) => row.map((value) => k[i] * value)),
            y.map((value, i) => k[i] * value),
            undefined,
            columns,
        );
        return { fitted: x, loss: rnorm };
    }
}

export type RobustRegParams = {
    loss_type?: Loss;
    f_scale?: number;
};

/** wrapper around sp.optimize.least_squares to feed to kernel_deconv */
export class RobustReg implements Regressor {
    readonly lossType: Loss;
    readonly fScale: number;

    constructor({ loss_type = 'soft_l1', f_scale = 0.1 }: RobustRegParams = {}) {
        this.lossType = loss_type;
        this.fScale = f_scale;
    }

    /** b0: starting values for optimization */
    fit(
        X: readonly (readonly number[])[],
        y: readonly number[],
        k: readonly number[],
        columns = X[0]?.length ?? 0,
        b0?: number[],
    ): RegressionFit {
        // make starting values
        b0 ??= new Array<number>(columns).fill(1 / columns);
        // regress
        const ls = leastSquares({
            x: X,
            y,
            k,
            x0: b0,
            lower: 0,
            upper: 1,
            loss: this.lossType,
            fScale: this.fScale,
        });
        return { fitted: ls.x, loss: ls.cost };
    }
}
