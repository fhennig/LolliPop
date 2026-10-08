// SPDX-License-Identifier: GPL-3.0-or-later

// The stand-ins for NumPy and SciPy, against values from NumPy 2.5 and SciPy 1.18.

import { describe, expect, test } from 'vitest';

import { LOSSES } from '../src/index.js';
import { leastSquares } from '../src/numeric/leastSquares.js';
import { inv, LinAlgError } from '../src/numeric/linalg.js';
import { nnls } from '../src/numeric/nnls.js';
import { RandomState } from '../src/numeric/randomState.js';
import { normPpf, quantile } from '../src/numeric/stats.js';

describe('RandomState', () => {
    test('draws like np.random.seed(42); np.random.randint', () => {
        const random = new RandomState(42);
        expect(random.randint(0, 10, 8)).toEqual([6, 3, 7, 4, 6, 9, 2, 6]);
        expect(random.randint(0, 3, 4)).toEqual([2, 2, 0, 2]);
    });

    test('rejects seeds NumPy rejects', () => {
        expect(() => new RandomState(-1)).toThrow();
        expect(() => new RandomState(2 ** 32)).toThrow();
    });
});

describe('nnls', () => {
    test('solves the examples of the SciPy documentation', () => {
        const a = [
            [1, 0],
            [1, 0],
            [0, 1],
        ];
        expect(nnls(a, [2, 1, 1])).toEqual({
            x: [expect.closeTo(1.5, 15), expect.closeTo(1, 15)],
            rnorm: expect.closeTo(0.7071067811865477, 15),
        });
        expect(nnls(a, [-1, -1, -1])).toEqual({ x: [0, 0], rnorm: expect.closeTo(1.7320508075688772, 15) });
    });

    test('keeps a coefficient at its bound', () => {
        const { x, rnorm } = nnls(
            [
                [0.9, 0.1, 0.3],
                [0.2, 0.8, 0.5],
                [0.4, 0.4, 0.9],
                [0.1, 0.3, 0.2],
            ],
            [0.5, 0.6, 0.1, 0.9],
        );
        expect(x[0]).toBeCloseTo(0.35946991, 8);
        expect(x[1]).toBeCloseTo(0.75759249, 8);
        expect(x[2]).toBe(0);
        expect(rnorm).toBeCloseTo(0.7362019925445837, 14);
    });

    test('gives zeros without rows (where SciPy returns uninitialized memory)', () => {
        expect(nnls([], [], undefined, 3)).toEqual({ x: [0, 0, 0], rnorm: 0 });
    });
});

describe('leastSquares', () => {
    test.each(LOSSES)('finds the exact fit with the %s loss', (loss) => {
        const x = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ];
        const beta = [0.2, 0.5, 0.3];
        const y = x.map((row) => row.reduce((sum, value, j) => sum + value * beta[j], 0));
        const result = leastSquares({
            x,
            y,
            k: [1, 2, 0.5, 1],
            x0: [1 / 3, 1 / 3, 1 / 3],
            lower: 0,
            upper: 1,
            loss,
            fScale: 0.1,
        });
        result.x.forEach((value, j) => expect(value).toBeCloseTo(beta[j], 10));
        expect(result.cost).toBeCloseTo(0, 20);
    });

    test('respects the bounds', () => {
        const result = leastSquares({
            x: [[1], [1]],
            y: [2, 3],
            k: [1, 1],
            x0: [0.5],
            lower: 0,
            upper: 1,
            loss: 'soft_l1',
            fScale: 0.1,
        });
        expect(result.x).toEqual([1]);
    });
});

test('inv is like np.linalg.inv', () => {
    const result = inv([
        [4, 7],
        [2, 6],
    ]);
    expect(result[0][0]).toBeCloseTo(0.6, 15);
    expect(result[0][1]).toBeCloseTo(-0.7, 15);
    expect(result[1][0]).toBeCloseTo(-0.2, 15);
    expect(result[1][1]).toBeCloseTo(0.4, 15);
    expect(() =>
        inv([
            [1, 2],
            [2, 4],
        ]),
    ).toThrow(LinAlgError);
});

test('normPpf is like scipy.stats.norm.ppf', () => {
    const expected: [number, number][] = [
        [0.975, 1.959963984540054],
        [0.95, 1.6448536269514722],
        [0.995, 2.5758293035489004],
        [0.6, 0.2533471031357997],
    ];
    for (const [q, value] of expected) {
        expect(normPpf(q)).toBeCloseTo(value, 14);
    }
});

test('quantile is like np.quantile', () => {
    expect(quantile([0.1, 0.5, 0.2, 0.9, 0.33], 0.025)).toBeCloseTo(0.11, 15);
    expect(quantile([0.1, 0.5, 0.2, 0.9, 0.33], 0.975)).toBeCloseTo(0.86, 15);
    expect(quantile([1, NaN], 0.5)).toBeNaN();
});
