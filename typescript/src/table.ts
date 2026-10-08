// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The little of pandas that the port needs. LolliPop works on data frames; here, a table is an
 * array of rows, a row a record of cells, and a missing value (pandas' NaN) is `null` or `NaN`.
 * Dates are ISO strings, `yyyy-mm-dd`.
 */

export type Cell = string | number | null;

export type Row = Record<string, Cell>;

/** A numeric data frame with labelled rows (an index) and columns. */
export type Frame = {
    index: string[];
    columns: string[];
    /** `values[i][j]`: row `index[i]`, column `columns[j]`. */
    values: number[][];
};

export function isMissing(value: Cell | undefined): value is null | undefined {
    return value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value));
}

/** Like `Series.unique()`: in the order of first appearance. */
export function unique<T>(values: readonly T[]): T[] {
    return [...new Set(values)];
}

/** The days since the epoch of an ISO date (or date-time, taken as UTC). */
export function toDay(date: string): number {
    const time = Date.parse(date.length === 10 ? `${date}T00:00:00Z` : date);
    if (Number.isNaN(time)) {
        throw new RangeError(`Not a date: ${date}`);
    }
    return time / 86_400_000;
}

/** The ISO date (`yyyy-mm-dd`) a number of days after the epoch. */
export function fromDay(day: number): string {
    return new Date(day * 86_400_000).toISOString().slice(0, 10);
}
