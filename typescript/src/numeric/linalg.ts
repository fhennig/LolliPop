// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The bit of `numpy.linalg` that LolliPop uses, for its small dense problems: a handful of columns
 * (one per variant) and up to a few thousand rows. Matrices are arrays of rows.
 */

export type Matrix = number[][];

/** Thrown where `numpy.linalg` raises `LinAlgError`. */
export class LinAlgError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'LinAlgError';
    }
}

/**
 * Like `np.linalg.inv`: LU with partial pivoting (LAPACK's `gesv`), which fails only on an exactly
 * zero pivot — a nearly singular matrix gets a huge inverse, as with NumPy.
 */
export function inv(matrix: Matrix): Matrix {
    const n = matrix.length;
    const lu = matrix.map((row) => [...row]);
    const permutation = Array.from({ length: n }, (_, i) => i);
    for (let column = 0; column < n; column++) {
        let pivot = column;
        for (let row = column + 1; row < n; row++) {
            if (Math.abs(lu[row][column]) > Math.abs(lu[pivot][column])) {
                pivot = row;
            }
        }
        if (lu[pivot][column] === 0) {
            throw new LinAlgError('Singular matrix');
        }
        [lu[column], lu[pivot]] = [lu[pivot], lu[column]];
        [permutation[column], permutation[pivot]] = [permutation[pivot], permutation[column]];
        for (let row = column + 1; row < n; row++) {
            const factor = (lu[row][column] /= lu[column][column]);
            if (factor !== 0) {
                for (let k = column + 1; k < n; k++) {
                    lu[row][k] -= factor * lu[column][k];
                }
            }
        }
    }
    // Solve LU x = P e_j for every unit vector e_j.
    const result: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let j = 0; j < n; j++) {
        const x = permutation.map((i) => (i === j ? 1 : 0));
        for (let row = 0; row < n; row++) {
            for (let k = 0; k < row; k++) {
                x[row] -= lu[row][k] * x[k];
            }
        }
        for (let row = n - 1; row >= 0; row--) {
            for (let k = row + 1; k < n; k++) {
                x[row] -= lu[row][k] * x[k];
            }
            x[row] /= lu[row][row];
        }
        for (let i = 0; i < n; i++) {
            result[i][j] = x[i];
        }
    }
    return result;
}

export function dot(a: readonly number[], b: readonly number[]): number {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

export function sum(values: readonly number[]): number {
    let result = 0;
    for (const value of values) {
        result += value;
    }
    return result;
}
