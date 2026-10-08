// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Stands in for `scipy.optimize.nnls`, which `regressors.NnlsReg` uses: non-negative least squares
 * `min ‖Ax − b‖` subject to `x ≥ 0`.
 *
 * A port of SciPy's `__nnls` (scipy/optimize/src/nnls.c, SciPy 1.18), itself a translation of
 * Lawson and Hanson's Fortran: the same active-set steps with the same Householder and Givens
 * transformations, statement by statement, with the BLAS and LAPACK routines it calls (`ddot`,
 * `dnrm2`, `dlarfgp`, `dlarf`, `dlartgp`) written out below. The BLAS that SciPy is built with
 * may round differently, which only matters where the problem has no unique solution: SciPy then
 * breaks the ties by rounding noise.
 *
 * One deliberate difference: SciPy's code doesn't restore a candidate column that it rejects
 * (see below), and then returns a non-optimal solution with a wrong residual norm; this does.
 */

import { type Matrix } from './linalg.js';

export type NnlsResult = {
    x: number[];
    /** `‖Ax − b‖`, the norm of the residual (not its square). */
    rnorm: number;
};

/**
 * Like `scipy.optimize.nnls(A, b, maxiter)`, with the same default of `3 n` iterations, and the
 * same error when they run out. `columns` is the number of columns of `A`, for when it has no
 * rows. (SciPy then returns uninitialized memory as the solution; this returns zeros.)
 */
export function nnls(
    aRows: Matrix,
    bIn: readonly number[],
    maxiter?: number,
    columns = aRows[0]?.length ?? 0,
): NnlsResult {
    const m = aRows.length;
    const n = columns;
    if (n === 0) {
        return { x: [], rnorm: dnrm2(bIn, 0, bIn.length) };
    }
    const iterationLimit = maxiter || 3 * n;
    const x = new Array<number>(n).fill(0);
    if (m <= 0) {
        return { x, rnorm: 0 };
    }

    // Column-major, like the Fortran: a[i + j*m]
    const a = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            a[i + j * m] = aRows[i][j];
        }
    }
    const b = Float64Array.from(bIn);
    const w = new Float64Array(n);
    const zz = new Float64Array(m);
    const indices = Array.from({ length: n }, (_, i) => i);

    let indz = 0;
    let iteration = 0;
    let iz: number;
    let izmax = 0;
    let j = 0;
    let jj = 0;
    let tau = 0;
    let pivot = 1;

    // Outer loop
    outer: while (indz < Math.min(m, n)) {
        // Compute the dual vector components in set Z.
        for (let i = indz; i < n; i++) {
            j = indices[i];
            w[j] = ddot(a, indz + j * m, b, indz, m - indz);
        }

        // Find the next linearly independent column that corresponds to the largest entry in
        // the dual vector w.
        for (;;) {
            let wmax = 0.0;
            for (let k = indz; k < n; k++) {
                j = indices[k];
                if (w[j] > wmax) {
                    wmax = w[j];
                    izmax = k;
                }
            }
            // If wmax <= 0.0, terminate since this is a KKT certificate.
            if (wmax <= 0.0) {
                break outer;
            }
            iz = izmax;
            j = indices[iz];

            // The sign of wmax is OK for j to be moved to set p. Begin the transformation and
            // check new diagonal element to avoid near-linear dependence.
            // (dlarfgp scales the column below the pivot in place: keep it, to restore it if j is
            // rejected. SciPy restores only the pivot, as Lawson and Hanson's H12 allowed, and
            // goes on with a corrupted column — a non-optimal solution and a wrong rnorm.)
            const saved = a.slice(indz + 1 + j * m, (j + 1) * m);
            const reflector = dlarfgp(m - indz, a[indz + j * m], a, indz + 1 + j * m);
            pivot = reflector.beta;
            tau = reflector.tau;

            // Compute the norm of a[0:indz, j] to check for linear dependence.
            const unorm = indz > 0 ? dnrm2(a, j * m, indz) : 0.0;
            const spacing = unorm > 0.0 ? nextUp(unorm) - unorm : 0.0;

            // Test for independence by checking the pivot for zero.
            if (Math.abs(pivot) > 100.0 * spacing) {
                // Column j is sufficiently independent. Copy b into zz and solve for ztest which
                // is the new prospective value for x[j].
                zz.set(b);
                const pivot2 = a[indz + j * m];
                a[indz + j * m] = 1.0;
                dlarf(m - indz, a, indz + j * m, tau, zz, indz);
                // See if ztest is positive.
                const ztest = zz[indz] / pivot;
                if (ztest > 0.0) {
                    break;
                } else {
                    a[indz + j * m] = pivot2;
                }
            }
            a.set(saved, indz + 1 + j * m);
            // Reject j as a candidate to be moved from set z to set p.
            w[j] = 0.0;
        }

        // The index j=indices[iz] has been selected to be moved from set z to set p. Update b,
        // update indices, apply householder transformations to cols in new set z, zero
        // subdiagonal elements in col j, set w(j)=0.
        b.set(zz);
        indices[iz] = indices[indz];
        indices[indz] = j;
        indz++;
        // Apply the householder transformation to the remaining columns.
        if (indz < n) {
            for (let k = indz; k < n; k++) {
                jj = indices[k];
                dlarf(m - indz + 1, a, indz - 1 + j * m, tau, a, indz - 1 + jj * m);
            }
        }
        // Restore the pivot element into a, zero the subdiagonal elements in col j
        a[indz - 1 + j * m] = pivot;
        for (let i = indz; i < m; i++) {
            a[j * m + i] = 0.0;
        }
        // Zero the dual coefficient for the column.
        w[j] = 0.0;

        // Solve the permuted triangular system, store in zz.
        jj = solveTriangular(a, m, indices, indz, zz, jj);

        // ****** Inner loop ******
        for (;;) {
            iteration++;
            if (iteration >= iterationLimit) {
                throw new Error('Maximum number of iterations reached.');
            }

            // See if all new constrained coefficients are feasible, if not compute alpha
            let alpha = 2.0;
            for (let ip = 0; ip < indz; ip++) {
                const k = indices[ip];
                if (zz[ip] <= 0.0) {
                    const t = -x[k] / (zz[ip] - x[k]);
                    if (alpha > t) {
                        alpha = t;
                        jj = ip;
                    }
                }
            }
            // If all new constrained coefficients are feasible, alpha is still 2.0. If so exit
            // from secondary loop to main loop.
            if (alpha === 2.0) {
                break;
            }

            // Otherwise interpolate between old x and zz.
            for (let ip = 0; ip < indz; ip++) {
                const k = indices[ip];
                x[k] = x[k] + alpha * (zz[ip] - x[k]);
            }

            // Modify a, b, and the indices to move coefficient i from set p to set z.
            let i = indices[jj];
            for (;;) {
                x[i] = 0.0;

                if (jj !== indz - 1) {
                    jj++;
                    for (let row = jj; row < indz; row++) {
                        const ii = indices[row];
                        indices[row - 1] = ii;
                        const { cs, sn, r } = dlartgp(a[row - 1 + ii * m], a[row + ii * m]);
                        a[row - 1 + ii * m] = r;
                        a[row + ii * m] = 0.0;
                        // Apply the Givens rotation to all columns except ii.
                        for (let k = 0; k < n; k++) {
                            if (k !== ii) {
                                const tmp = a[row - 1 + k * m];
                                a[row - 1 + k * m] = cs * tmp + sn * a[row + k * m];
                                a[row + k * m] = -sn * tmp + cs * a[row + k * m];
                            }
                        }
                        const tmp = b[row - 1];
                        b[row - 1] = cs * tmp + sn * b[row];
                        b[row] = -sn * tmp + cs * b[row];
                    }
                }
                indz--;
                indices[indz] = i;

                // See if remaining coefficients in set P are feasible since determination of
                // alpha guarantees it. If still there are infeasible ones, they are due to
                // numerical noise. Any that are nonpositive will be set to zero and moved from
                // set p to set z.
                let nobreak = false;
                for (jj = 0; jj < indz; jj++) {
                    i = indices[jj];
                    if (x[i] <= 0.0) {
                        break;
                    }
                    if (jj === indz - 1) {
                        nobreak = true;
                    }
                }
                // If for loop completes without break, then leave the while loop
                if (nobreak) {
                    break;
                }
            }

            zz.set(b);
            jj = solveTriangular(a, m, indices, indz, zz, jj);
            // ****** end of inner loop ******
        }

        // Back in the outer loop
        for (let k = 0; k < indz; k++) {
            x[indices[k]] = zz[k];
        }
        // ****** end of outer loop ******
    }

    // Compute the residual vector and its norm.
    const rnorm = indz < m ? dnrm2(b, indz, m - indz) : 0.0;
    return { x, rnorm };
}

/** Solves the permuted triangular system in place in zz; returns the `jj` the C code leaves. */
function solveTriangular(
    a: Float64Array,
    m: number,
    indices: readonly number[],
    indz: number,
    zz: Float64Array,
    jj: number,
): number {
    for (let k = 0; k < indz; k++) {
        // ip traverses the indices of P set in reverse
        const ip = indz - 1 - k;
        if (k !== 0) {
            for (let i = 0; i <= ip; i++) {
                zz[i] = zz[i] - a[i + jj * m] * zz[ip + 1];
            }
        }
        jj = indices[ip];
        zz[ip] = zz[ip] / a[ip + jj * m];
    }
    return jj;
}

/** BLAS `ddot`: `Σ x[xOffset + i] · y[yOffset + i]` for `i < n`. */
function ddot(x: Float64Array, xOffset: number, y: Float64Array, yOffset: number, n: number): number {
    let result = 0;
    for (let i = 0; i < n; i++) {
        result += x[xOffset + i] * y[yOffset + i];
    }
    return result;
}

/** BLAS `dnrm2`: the Euclidean norm of `x[offset .. offset + n)`, without overflow. */
function dnrm2(x: ArrayLike<number>, offset: number, n: number): number {
    let scale = 0;
    for (let i = 0; i < n; i++) {
        scale = Math.max(scale, Math.abs(x[offset + i]));
    }
    if (scale === 0 || !Number.isFinite(scale)) {
        return scale;
    }
    let sumOfSquares = 0;
    for (let i = 0; i < n; i++) {
        const scaled = x[offset + i] / scale;
        sumOfSquares += scaled * scaled;
    }
    return scale * Math.sqrt(sumOfSquares);
}

/** LAPACK `dlapy2`: `√(x² + y²)` without overflow. */
function dlapy2(x: number, y: number): number {
    const xabs = Math.abs(x);
    const yabs = Math.abs(y);
    const w = Math.max(xabs, yabs);
    const z = Math.min(xabs, yabs);
    return z === 0 || w > Number.MAX_VALUE ? w : w * Math.sqrt(1 + (z / w) ** 2);
}

/** Fortran's `SIGN(a, b)`: `|a|` with the sign of `b`. */
function sign(a: number, b: number): number {
    return b < 0 || Object.is(b, -0) ? -Math.abs(a) : Math.abs(a);
}

/** `dlamch('S') / dlamch('E')`: the safe minimum over the machine epsilon. */
const SMLNUM = 2.2250738585072014e-308 / 1.1102230246251565e-16;

/**
 * LAPACK `dlarfgp`: the elementary reflector `H = I − τ v vᵀ` with `H (α; x) = (β; 0)` and
 * `β ≥ 0`, for the `n − 1` elements of `x` at `a[xOffset ..]`. Overwrites `x` with `v[1 ..]`
 * (`v[0]` is 1); returns β and τ.
 */
function dlarfgp(n: number, alphaIn: number, a: Float64Array, xOffset: number): { beta: number; tau: number } {
    let alpha = alphaIn;
    if (n <= 0) {
        return { beta: alpha, tau: 0 };
    }
    let xnorm = dnrm2(a, xOffset, n - 1);
    if (xnorm === 0) {
        // H = [+/-1, 0; I], sign chosen so alpha >= 0
        if (alpha >= 0) {
            return { beta: alpha, tau: 0 };
        }
        for (let i = 0; i < n - 1; i++) {
            a[xOffset + i] = 0;
        }
        return { beta: -alpha, tau: 2 };
    }
    // general case
    let beta = sign(dlapy2(alpha, xnorm), alpha);
    let knt = 0;
    if (Math.abs(beta) < SMLNUM) {
        // xnorm, beta may be inaccurate; scale x and recompute them
        const bignum = 1 / SMLNUM;
        do {
            knt++;
            for (let i = 0; i < n - 1; i++) {
                a[xOffset + i] *= bignum;
            }
            beta *= bignum;
            alpha *= bignum;
        } while (Math.abs(beta) < SMLNUM && knt < 20);
        xnorm = dnrm2(a, xOffset, n - 1);
        beta = sign(dlapy2(alpha, xnorm), alpha);
    }
    const savealpha = alpha;
    alpha = alpha + beta;
    let tau: number;
    if (beta < 0) {
        beta = -beta;
        tau = -alpha / beta;
    } else {
        alpha = xnorm * (xnorm / alpha);
        tau = alpha / beta;
        alpha = -alpha;
    }
    if (Math.abs(tau) <= SMLNUM) {
        // In the case where the computed TAU ends up being a denormalized number, it loses
        // relative accuracy.
        if (savealpha >= 0) {
            tau = 0;
        } else {
            tau = 2;
            for (let i = 0; i < n - 1; i++) {
                a[xOffset + i] = 0;
            }
            beta = -savealpha;
        }
    } else {
        // This is the general case.
        const scale = 1 / alpha;
        for (let i = 0; i < n - 1; i++) {
            a[xOffset + i] *= scale;
        }
    }
    // If BETA is subnormal, it may lose relative accuracy
    for (let k = 0; k < knt; k++) {
        beta *= SMLNUM;
    }
    return { beta, tau };
}

/**
 * LAPACK `dlarf('L', m, 1, v, 1, tau, c)`: applies `H = I − τ v vᵀ` to the column
 * `c[cOffset .. cOffset + m)`, with `v` at `v[vOffset ..]`.
 */
function dlarf(m: number, v: Float64Array, vOffset: number, tau: number, c: Float64Array, cOffset: number): void {
    if (tau === 0) {
        return;
    }
    // Look for the last non-zero row in v.
    let lastv = m;
    while (lastv > 0 && v[vOffset + lastv - 1] === 0) {
        lastv--;
    }
    // w = C' v, C = C - tau v w'
    const w = ddot(c, cOffset, v, vOffset, lastv);
    for (let i = 0; i < lastv; i++) {
        c[cOffset + i] -= tau * w * v[vOffset + i];
    }
}

/** LAPACK `dlartgp`: the plane rotation with `[cs sn; −sn cs] (f; g) = (r; 0)` and `r ≥ 0`. */
function dlartgp(f: number, g: number): { cs: number; sn: number; r: number } {
    if (g === 0) {
        return { cs: sign(1, f), sn: 0, r: Math.abs(f) };
    }
    if (f === 0) {
        return { cs: 0, sn: sign(1, g), r: Math.abs(g) };
    }
    const r = Math.sqrt(f * f + g * g);
    return { cs: f / r, sn: g / r, r };
}

/** The next double above a positive `value`, like C's `nextafter(value, 2 * value)`. */
function nextUp(value: number): number {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, value);
    view.setBigUint64(0, view.getBigUint64(0) + 1n);
    return view.getFloat64(0);
}
