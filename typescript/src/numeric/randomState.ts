// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Stands in for `numpy.random.RandomState` (the legacy global generator behind `np.random.seed`
 * and `np.random.randint`), which `confints.resample_mutations` draws from: the Mersenne Twister
 * MT19937 with NumPy's seeding and bounded-integer algorithm, so that the same seed gives the same
 * draws as LolliPop.
 */

const N = 624;
const M = 397;

export class RandomState {
    private readonly mt = new Uint32Array(N);
    private index = N;

    constructor(seed = 0) {
        this.seed(seed);
    }

    /** Like `np.random.seed(seed)` with an integer seed: MT19937's `init_genrand`. */
    seed(seed: number): void {
        if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
            throw new RangeError(`Seed must be between 0 and 2**32 - 1, got ${seed}`);
        }
        this.mt[0] = seed >>> 0;
        for (let i = 1; i < N; i++) {
            const previous = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
            // 1812433253 * previous + i, modulo 2**32 (split, to stay within exact doubles)
            this.mt[i] = (Math.imul(1812433253, previous) + i) >>> 0;
        }
        this.index = N;
    }

    /** The next 32 random bits, as an unsigned integer: MT19937's `genrand_int32`. */
    nextUint32(): number {
        if (this.index >= N) {
            this.twist();
        }
        let y = this.mt[this.index++];
        y ^= y >>> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >>> 18;
        return y >>> 0;
    }

    /**
     * Like `np.random.randint(low, high, size)`: `size` integers in `[low, high)`. NumPy's legacy
     * generator draws them by masking 32-bit draws to the next power of two and rejecting those
     * out of range.
     */
    randint(low: number, high: number, size: number): number[] {
        if (size === 0) {
            // NumPy checks the range only when it draws
            return [];
        }
        if (high <= low) {
            throw new RangeError(`low >= high (${low} >= ${high})`);
        }
        const range = high - 1 - low;
        if (range > 0xffffffff) {
            throw new RangeError('Ranges above 2**32 are not supported');
        }
        const result = new Array<number>(size);
        if (range === 0) {
            return result.fill(low);
        }
        if (range === 0xffffffff) {
            for (let i = 0; i < size; i++) {
                result[i] = low + this.nextUint32();
            }
            return result;
        }
        let mask = range;
        mask |= mask >>> 1;
        mask |= mask >>> 2;
        mask |= mask >>> 4;
        mask |= mask >>> 8;
        mask |= mask >>> 16;
        mask >>>= 0;
        for (let i = 0; i < size; i++) {
            let value: number;
            do {
                value = (this.nextUint32() & mask) >>> 0;
            } while (value > range);
            result[i] = low + value;
        }
        return result;
    }

    private twist(): void {
        const mt = this.mt;
        for (let i = 0; i < N; i++) {
            const y = (mt[i] & 0x80000000) | (mt[(i + 1) % N] & 0x7fffffff);
            mt[i] = mt[(i + M) % N] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
        }
        this.index = 0;
    }
}
