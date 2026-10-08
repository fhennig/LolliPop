// SPDX-License-Identifier: GPL-3.0-or-later
// Port of LolliPop's lollipop/kernels.py.

export interface Kernel {
    /** The kernel weights between `y1` and each of `y2`. */
    values(y1: number, y2: readonly number[]): number[];
}

/** compute gaussian kernel weighting between y1 and y2 */
export class GaussianKernel implements Kernel {
    /**
     * bandwidth (float): bandwidth argument to kernel, default = 1.0 — the kernel's variance, not
     * its standard deviation
     */
    constructor(readonly bandwidth = 1.0) {}

    /** compute gaussian kernel between y1 and y2 */
    values(y1: number, y2: readonly number[]): number[] {
        return y2.map((value) => Math.exp(-((y1 - value) ** 2) / 2 / this.bandwidth));
    }
}

/** compute box kernel between y1 and y2 */
export class BoxKernel implements Kernel {
    /** bandwidth (float): bandwidth argument to kernel, default = 1.0 — the box's full width */
    constructor(readonly bandwidth = 1.0) {}

    /** compute box kernel between y1 and y2 */
    values(y1: number, y2: readonly number[]): number[] {
        return y2.map((value) => 1.0 * Number(Math.abs(y1 - value) <= this.bandwidth / 2));
    }
}
