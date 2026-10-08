// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Stands in for `scipy.optimize.least_squares`, which `regressors.RobustReg` uses, for the problems
 * it gets there: residuals linear in the coefficients, bounds on the coefficients, and one of
 * SciPy's robust losses.
 *
 * SciPy's trust-region solver (`trf`) has no counterpart in JavaScript, and porting it would buy
 * little: it stops where its tolerances say the loss is flat enough, not at the minimum. This
 * solves the same problem to the minimum instead, with an active-set Newton method. For the
 * convex losses (`linear`, `soft_l1`, `huber`) the minimum is unique wherever the variants can be
 * told apart, so the two agree up to SciPy's stopping point; for the non-convex ones (`cauchy`,
 * `arctan`), both find a local minimum from the same starting point.
 */

import { dot } from './linalg.js';

export type Loss = 'linear' | 'soft_l1' | 'huber' | 'cauchy' | 'arctan';

export const LOSSES: readonly Loss[] = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'];

export type LeastSquaresProblem = {
    /** The design rows `X[i]`. */
    x: readonly (readonly number[])[];
    /** The observations. */
    y: readonly number[];
    /** The row weights: the residuals are `k[i] · (X[i]·β − y[i])`. */
    k: readonly number[];
    /** The starting point. */
    x0: readonly number[];
    lower: number;
    upper: number;
    loss: Loss;
    fScale: number;
};

export type LeastSquaresResult = {
    x: number[];
    /** Like SciPy's `cost`: `½ Σ f² ρ((rᵢ/f)²)`. */
    cost: number;
    iterations: number;
};

/** ρ(z), ρ′(z) and ρ″(z) of SciPy's losses, for z = (r/f)². */
function rho(loss: Loss, z: number): [number, number, number] {
    switch (loss) {
        case 'linear':
            return [z, 1, 0];
        case 'soft_l1': {
            const t = 1 + z;
            return [2 * (Math.sqrt(t) - 1), t ** -0.5, -0.5 * t ** -1.5];
        }
        case 'huber':
            return z <= 1 ? [z, 1, 0] : [2 * Math.sqrt(z) - 1, z ** -0.5, -0.5 * z ** -1.5];
        case 'cauchy':
            return [Math.log1p(z), 1 / (1 + z), -1 / (1 + z) ** 2];
        case 'arctan': {
            const t = 1 + z * z;
            return [Math.atan(z), 1 / t, (-2 * z) / (t * t)];
        }
    }
}

const MAX_ITERATIONS = 500;
/** Stop when no coefficient moves by more than this. */
const X_TOLERANCE = 1e-14;
/** Stop when the cost decreases by less than this, relatively, three times in a row. */
const STALL_TOLERANCE = 1e-15;
/** Coefficients this close to a bound are on it. */
const BOUND_TOLERANCE = 1e-14;
/** Armijo's sufficient decrease. */
const SIGMA = 1e-4;

/**
 * `min ½ Σ f² ρ((rᵢ/f)²)` with `rᵢ = kᵢ (Xᵢ·β − yᵢ)`, subject to `lower ≤ β ≤ upper`.
 *
 * Identical design rows are summed up per row pattern, so that a step costs a few operations per
 * row: the design rows of a deconvolution are 0/1 signature rows, with few distinct ones.
 */
export function leastSquares(problem: LeastSquaresProblem): LeastSquaresResult {
    const { x: design, y, k, lower, upper, loss, fScale } = problem;
    const n = problem.x0.length;
    const f2 = fScale * fScale;

    const patternIndex = new Map<string, number>();
    const patterns: (readonly number[])[] = [];
    const rowPattern = design.map((row) => {
        const key = row.join();
        let index = patternIndex.get(key);
        if (index === undefined) {
            index = patterns.length;
            patternIndex.set(key, index);
            patterns.push(row);
        }
        return index;
    });

    const fitted = new Float64Array(patterns.length);
    const gradientSums = new Float64Array(patterns.length);
    const curvatureSums = new Float64Array(patterns.length);
    const gaussNewtonSums = new Float64Array(patterns.length);

    const evaluate = (beta: readonly number[], derivatives: boolean): number => {
        for (let p = 0; p < patterns.length; p++) {
            fitted[p] = dot(patterns[p], beta);
        }
        if (derivatives) {
            gradientSums.fill(0);
            curvatureSums.fill(0);
            gaussNewtonSums.fill(0);
        }
        let cost = 0;
        for (let i = 0; i < y.length; i++) {
            const p = rowPattern[i];
            const residual = k[i] * (fitted[p] - y[i]);
            const z = (residual * residual) / f2;
            const [value, first, second] = rho(loss, z);
            cost += value;
            if (derivatives) {
                gradientSums[p] += first * residual * k[i];
                curvatureSums[p] += (first + 2 * z * second) * k[i] * k[i];
                gaussNewtonSums[p] += first * k[i] * k[i];
            }
        }
        return 0.5 * f2 * cost;
    };

    const accumulate = (sums: Float64Array): number[][] => {
        const result = Array.from({ length: n }, () => new Array<number>(n).fill(0));
        patterns.forEach((pattern, p) => {
            for (let j = 0; j < n; j++) {
                if (pattern[j] !== 0) {
                    const scaled = sums[p] * pattern[j];
                    for (let l = 0; l < n; l++) {
                        result[j][l] += scaled * pattern[l];
                    }
                }
            }
        });
        return result;
    };

    const clip = (value: number) => Math.min(upper, Math.max(lower, value));
    let beta = problem.x0.map(clip);
    let cost = evaluate(beta, true);
    let iterations = 0;
    let stalled = 0;
    for (; iterations < MAX_ITERATIONS; iterations++) {
        const gradient = new Array<number>(n).fill(0);
        patterns.forEach((pattern, p) => {
            for (let j = 0; j < n; j++) {
                gradient[j] += gradientSums[p] * pattern[j];
            }
        });

        // (Rounding leaves coefficients a hair off a bound they reached: put them on it.)
        const snapped = beta.map((value) =>
            value - lower <= BOUND_TOLERANCE ? lower : upper - value <= BOUND_TOLERANCE ? upper : value,
        );
        if (snapped.some((value, j) => value !== beta[j])) {
            beta = snapped;
            cost = evaluate(beta, true);
            iterations--;
            continue;
        }

        // The active set: coefficients at a bound that the gradient pushes against it. The others
        // take a Newton step, as long as it doesn't leave the box from a bound (else, that one is
        // held at its bound too).
        const fixed = beta.map(
            (value, j) => (value <= lower && gradient[j] > 0) || (value >= upper && gradient[j] < 0),
        );
        let direction: number[] | undefined;
        let newton = false;
        for (;;) {
            const free = beta.flatMap((_, j) => (fixed[j] ? [] : [j]));
            if (free.length === 0 || free.every((j) => gradient[j] === 0)) {
                break;
            }
            const freeGradient = free.map((j) => gradient[j]);
            const newtonStep =
                newtonByQr(patterns, curvatureSums, gradientSums, free) ??
                solveSpd(restrict(accumulate(curvatureSums), free), freeGradient) ??
                pseudoSolve(restrict(accumulate(curvatureSums), free), freeGradient);
            newton = newtonStep !== undefined;
            const step = newtonStep ?? solveSpd(restrict(accumulate(gaussNewtonSums), free), freeGradient, true);
            const candidate = new Array<number>(n).fill(0);
            free.forEach((j, index) => (candidate[j] = step === undefined ? -gradient[j] : -step[index]));
            // not a descent direction (where the Hessian isn't positive definite): steepest descent
            if (!(free.reduce((sum, j) => sum + gradient[j] * candidate[j], 0) < 0)) {
                free.forEach((j) => (candidate[j] = -gradient[j]));
                newton = false;
            }
            const leaving = free.filter(
                (j) => (beta[j] <= lower && candidate[j] < 0) || (beta[j] >= upper && candidate[j] > 0),
            );
            if (leaving.length === 0) {
                direction = candidate;
                break;
            }
            leaving.forEach((j) => (fixed[j] = true));
        }
        if (direction === undefined) {
            break;
        }

        // The longest step that stays in the box, and the coefficient that reaches its bound.
        let longest = Infinity;
        let blocking = -1;
        direction.forEach((d, j) => {
            const room = d < 0 ? (beta[j] - lower) / -d : d > 0 ? (upper - beta[j]) / d : Infinity;
            if (room < longest) {
                longest = room;
                blocking = j;
            }
        });
        const slope = direction.reduce((sum, d, j) => sum + gradient[j] * d, 0);
        let accepted: { beta: number[]; cost: number; blocked: boolean } | undefined;
        for (let alpha = Math.min(1, longest), first = true; alpha > 1e-20; alpha /= 2, first = false) {
            const candidate = beta.map((value, j) =>
                alpha === longest && j === blocking
                    ? direction![j] < 0
                        ? lower
                        : upper
                    : clip(value + alpha * direction![j]),
            );
            const candidateCost = evaluate(candidate, false);
            // (Near the minimum, or on a short step to a bound, the decrease can be below rounding:
            // there, no increase will do.)
            const roundingFloor =
                first &&
                candidateCost <= cost &&
                (-slope * alpha <= 1e-12 * Math.max(cost, Number.MIN_VALUE) || alpha === longest);
            if (candidateCost <= cost + SIGMA * alpha * slope || roundingFloor) {
                accepted = { beta: candidate, cost: candidateCost, blocked: alpha === longest };
                break;
            }
        }
        if (accepted === undefined) {
            break;
        }
        // Without a Newton step (where the loss is linear, as in the outer part of the Huber
        // loss), a full step can be far too short: go on doubling it while the cost keeps falling.
        if (!newton && !accepted.blocked) {
            for (let alpha = 2; ; alpha *= 2) {
                const reach = Math.min(alpha, longest);
                const candidate = beta.map((value, j) =>
                    reach === longest && j === blocking
                        ? direction![j] < 0
                            ? lower
                            : upper
                        : clip(value + reach * direction![j]),
                );
                const candidateCost = evaluate(candidate, false);
                if (!(candidateCost <= cost + SIGMA * reach * slope && candidateCost < accepted.cost)) {
                    break;
                }
                accepted = { beta: candidate, cost: candidateCost, blocked: reach === longest };
                if (reach === longest) {
                    break;
                }
            }
        }
        const change = Math.max(...accepted.beta.map((value, j) => Math.abs(value - beta[j])));
        const decrease = cost - accepted.cost;
        beta = accepted.beta;
        cost = evaluate(beta, true);
        // (A step cut short by a bound only adds that bound to the active set.)
        if (change <= X_TOLERANCE && !accepted.blocked) {
            break;
        }
        // Along a direction the cost barely depends on, steps may stay large while the cost no
        // longer changes within rounding: stop when it stalls.
        stalled = decrease <= STALL_TOLERANCE * cost && !accepted.blocked ? stalled + 1 : 0;
        if (stalled >= 3) {
            break;
        }
    }
    return { x: beta, cost, iterations };
}

/**
 * `H⁻¹g` over the free coefficients (the Newton step is its opposite), as the least squares
 * solution of `min ‖B s − u‖` with `B = diag(√c) D` and `u = gradientSums / √c` (so `BᵀB = H`
 * and `Bᵀu = g`):
 * by QR, which keeps the condition of B instead of squaring it like forming H does. Undefined
 * where a pattern has no curvature but a gradient, or B is rank-deficient.
 */
function newtonByQr(
    patterns: readonly (readonly number[])[],
    curvatureSums: Float64Array,
    gradientSums: Float64Array,
    free: readonly number[],
): number[] | undefined {
    const rows: number[][] = [];
    const rhs: number[] = [];
    for (let p = 0; p < patterns.length; p++) {
        const curvature = curvatureSums[p];
        if (!(curvature > 0)) {
            if (gradientSums[p] !== 0) {
                return undefined;
            }
            continue;
        }
        const root = Math.sqrt(curvature);
        rows.push(free.map((j) => root * patterns[p][j]));
        rhs.push(gradientSums[p] / root);
    }
    return householderLstsq(rows, rhs);
}

/** `min ‖Ax − b‖` by Householder QR, or undefined if A is (numerically) rank-deficient. */
function householderLstsq(a: number[][], b: number[]): number[] | undefined {
    const m = a.length;
    const n = a[0]?.length ?? 0;
    if (m < n || n === 0) {
        return undefined;
    }
    const columns = Array.from({ length: n }, (_, j) => a.map((row) => row[j]));
    const rhs = [...b];
    const scale = Math.max(...columns.map((column) => Math.sqrt(column.reduce((sum, v) => sum + v * v, 0))));
    const diagonal = new Array<number>(n);
    for (let j = 0; j < n; j++) {
        const column = columns[j];
        let sumOfSquares = 0;
        for (let i = j; i < m; i++) {
            sumOfSquares += column[i] * column[i];
        }
        const columnNorm = Math.sqrt(sumOfSquares);
        if (!(columnNorm > 1e-15 * scale)) {
            return undefined;
        }
        const alpha = column[j] > 0 ? -columnNorm : columnNorm;
        // The reflection's vector v = column[j:] − alpha e₁ replaces the column below the diagonal.
        const head = column[j];
        column[j] -= alpha;
        const vNormSquared = sumOfSquares - head * head + column[j] * column[j];
        const reflect = (target: number[]) => {
            let projection = 0;
            for (let i = j; i < m; i++) {
                projection += column[i] * target[i];
            }
            const factor = (2 * projection) / vNormSquared;
            for (let i = j; i < m; i++) {
                target[i] -= factor * column[i];
            }
        };
        for (let k = j + 1; k < n; k++) {
            reflect(columns[k]);
        }
        reflect(rhs);
        diagonal[j] = alpha;
    }
    const x = new Array<number>(n).fill(0);
    for (let j = n - 1; j >= 0; j--) {
        let value = rhs[j];
        for (let k = j + 1; k < n; k++) {
            value -= columns[k][j] * x[k];
        }
        x[j] = value / diagonal[j];
    }
    return x.every(Number.isFinite) ? x : undefined;
}

/**
 * `H⁺g` for a symmetric positive semidefinite `H`: the minimum-norm Newton step where H is
 * singular, as where variants can't be told apart (the cost is flat along their difference).
 * Undefined if H has a negative eigenvalue, or the gradient has a part H doesn't see.
 */
function pseudoSolve(matrix: number[][], b: readonly number[]): number[] | undefined {
    const { values, vectors } = symmetricEigen(matrix);
    const largest = Math.max(...values.map(Math.abs));
    if (!(largest > 0) || values.some((value) => value < -1e-12 * largest)) {
        return undefined;
    }
    const result = new Array<number>(b.length).fill(0);
    values.forEach((value, i) => {
        if (value > 1e-12 * largest) {
            const projection = vectors.reduce((sum, row, j) => sum + row[i] * b[j], 0) / value;
            vectors.forEach((row, j) => (result[j] += projection * row[i]));
        }
    });
    // Only where the gradient lies in the range of H (as where the flat directions are those of
    // identical columns); else, as in the linear part of the Huber loss, it isn't a Newton step.
    const residual = matrix.map((row, j) => row.reduce((sum, value, l) => sum + value * result[l], 0) - b[j]);
    const scale = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
    if (!(Math.sqrt(residual.reduce((sum, value) => sum + value * value, 0)) <= 1e-8 * scale)) {
        return undefined;
    }
    return result.every(Number.isFinite) ? result : undefined;
}

/** Eigenvalues and eigenvectors (the columns of `vectors`) of a small symmetric matrix, by Jacobi. */
function symmetricEigen(matrix: number[][]): { values: number[]; vectors: number[][] } {
    const n = matrix.length;
    const a = matrix.map((row) => [...row]);
    const vectors = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j): number => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
        let offDiagonal = 0;
        for (let p = 0; p < n; p++) {
            for (let q = p + 1; q < n; q++) {
                offDiagonal += a[p][q] ** 2;
            }
        }
        if (offDiagonal === 0) {
            break;
        }
        for (let p = 0; p < n; p++) {
            for (let q = p + 1; q < n; q++) {
                if (a[p][q] === 0) {
                    continue;
                }
                const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                const c = 1 / Math.sqrt(t * t + 1);
                const s = t * c;
                for (let k = 0; k < n; k++) {
                    const akp = a[k][p];
                    const akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (let k = 0; k < n; k++) {
                    const apk = a[p][k];
                    const aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (let k = 0; k < n; k++) {
                    const vkp = vectors[k][p];
                    const vkq = vectors[k][q];
                    vectors[k][p] = c * vkp - s * vkq;
                    vectors[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    return { values: a.map((row, i) => row[i]), vectors };
}

function restrict(matrix: number[][], indices: readonly number[]): number[][] {
    return indices.map((j) => indices.map((l) => matrix[j][l]));
}

/**
 * Solves `A s = b` for a symmetric `A` by Cholesky, or `undefined` if `A` isn't positive definite.
 * With `damped`, a Levenberg damping is added until it is.
 */
function solveSpd(matrix: number[][], b: readonly number[], damped = false): number[] | undefined {
    const n = b.length;
    const scale = Math.max(...matrix.map((row, j) => Math.abs(row[j])));
    if (!(scale > 0)) {
        return undefined;
    }
    // No damping, then 1e-12 of the diagonal, growing a hundredfold up to the diagonal itself.
    for (let attempt = 0; attempt <= 7; attempt++) {
        const damping = attempt === 0 ? 0 : 1e-12 * scale * 100 ** (attempt - 1);
        const factor = cholesky(matrix.map((row, j) => row.map((value, l) => (j === l ? value + damping : value))));
        if (factor !== undefined) {
            const z = new Array<number>(n);
            for (let j = 0; j < n; j++) {
                let value = b[j];
                for (let l = 0; l < j; l++) {
                    value -= factor[j][l] * z[l];
                }
                z[j] = value / factor[j][j];
            }
            const s = new Array<number>(n);
            for (let j = n - 1; j >= 0; j--) {
                let value = z[j];
                for (let l = j + 1; l < n; l++) {
                    value -= factor[l][j] * s[l];
                }
                s[j] = value / factor[j][j];
            }
            return s.every(Number.isFinite) ? s : undefined;
        }
        if (!damped) {
            return undefined;
        }
    }
    return undefined;
}

function cholesky(matrix: number[][]): number[][] | undefined {
    const n = matrix.length;
    const factor = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let j = 0; j < n; j++) {
        let diagonal = matrix[j][j];
        for (let l = 0; l < j; l++) {
            diagonal -= factor[j][l] ** 2;
        }
        if (!(diagonal > 1e-14 * Math.abs(matrix[j][j]))) {
            return undefined;
        }
        factor[j][j] = Math.sqrt(diagonal);
        for (let i = j + 1; i < n; i++) {
            let value = matrix[i][j];
            for (let l = 0; l < j; l++) {
                value -= factor[i][l] * factor[j][l];
            }
            factor[i][j] = value / factor[j][j];
        }
    }
    return factor;
}
