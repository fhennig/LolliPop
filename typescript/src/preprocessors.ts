// SPDX-License-Identifier: GPL-3.0-or-later
// Port of LolliPop's lollipop/preprocessors.py.

import { isMissing, toDay, type Cell, type Row } from './table.js';

export type GeneralPreprocessArgs = {
    variants_list: string[];
    variants_pangolin: Record<string, string>;
    variants_not_reported: string[];
    to_drop: Cell[];
    start_date?: string | null;
    end_date?: string | null;
    no_date?: boolean;
    remove_deletions?: boolean;
    make_complement?: boolean;
    namefield?: string;
};

/** A filter: statements that all have to hold for a row to be removed. */
export type Filters = Record<string, string[]>;

export type Log = (message: string) => void;

/** General class to preprocess tallymut data before deconvolution. */
export class DataPreprocesser {
    dfTally: Row[];
    /** The columns of the table, in pandas' order (which the filter syntax depends on). */
    columns: string[];

    /**
     * columns: the columns of the table in order, like the header of the TSV; by default, the
     * keys of the rows in the order they first appear.
     */
    constructor(
        dfTally: readonly Row[],
        columns?: readonly string[],
        private readonly log: Log = () => {},
    ) {
        this.columns = columns !== undefined ? [...columns] : [...new Set(dfTally.flatMap(Object.keys))];
        this.dfTally = dfTally.map((row) =>
            Object.fromEntries(this.columns.map((column) => [column, row[column] ?? null])),
        );
    }

    /** return the complement of mutations signatures and mutations fracs */
    makeComplement(dfTally: readonly Row[], variantsList: readonly string[], namefield: string): Row[] {
        return dfTally.map((row) => {
            const tData: Row = { ...row };
            if (this.columns.includes(namefield)) {
                tData[namefield] = concat('-', row[namefield]);
            }
            tData.frac = 1 - (row.frac as number);
            for (const variant of variantsList) {
                tData[variant] = 1 - (row[variant] as number);
            }
            tData.undetermined = 1;
            return tData;
        });
    }

    /** General preprocessing steps */
    generalPreprocess({
        variants_list,
        variants_pangolin,
        variants_not_reported,
        to_drop,
        start_date = null,
        end_date = null,
        no_date = false,
        remove_deletions = true,
        make_complement = true,
        namefield = 'mutations',
    }: GeneralPreprocessArgs): this {
        // rename columns
        const pangolinValues = Object.values(variants_pangolin);
        if (pangolinValues.length !== new Set(pangolinValues).size) {
            throw new Error(`duplicate values in:\n${JSON.stringify(variants_pangolin)}`);
        }
        this.columns = this.columns.map((column) => variants_pangolin[column] ?? column);
        this.dfTally = this.dfTally.map((row) =>
            Object.fromEntries(
                Object.entries(row).map(([column, value]) => [variants_pangolin[column] ?? column, value]),
            ),
        );
        // drop non reported variants
        this.dropColumns(variants_not_reported);
        // drop rows without estimated frac or date
        const required = no_date ? ['frac'] : ['frac', 'date'];
        this.dfTally = this.dfTally.filter((row) => required.every((column) => !isMissing(row[column])));
        // create column with mutation signature
        if (this.columns.includes('base') && this.columns.includes('pos')) {
            // NOTE if cojac-based instead of SNV-bsed deconvolution: there is no single mutation cojac tabmut has an option to generate mutations strings
            this.setColumn('mutations', (row) => concat(pythonStr(row.pos), row.base));
        }
        // filter by minimum and maximum dates
        if (!no_date) {
            if (start_date !== null) {
                const start = toDay(start_date);
                this.dfTally = this.dfTally.filter((row) => toDay(row.date as string) >= start);
            }
            if (end_date !== null) {
                const end = toDay(end_date);
                this.dfTally = this.dfTally.filter((row) => toDay(row.date as string) < end);
            }
        }
        // remove deletions
        if (remove_deletions) {
            if (this.columns.includes('base')) {
                this.dfTally = this.dfTally.filter((row) => row.base !== '-');
            } else {
                // NOTE if cojac-based instead of SNV-bsed deconvolution: there is no single mutation
                this.log(
                    `Warning, 'remove_deletions' is set in configuration, but no 'base' column is present in columns ${this.columns}`,
                );
            }
        }

        // delete lines with mutation of the type that we want to delete (to_drop)
        // e.g.: remove all 'subset' mutations
        const absentcol = variants_list.filter((variant) => !this.columns.includes(variant));
        if (absentcol.length > 0) {
            // check for missing
            this.log(`Warning, variants_list's ${absentcol} is not present in columns ${this.columns}`);
        }
        for (const v of variants_list) {
            if (this.columns.includes(v)) {
                this.dfTally = this.dfTally.filter((row) => !isIn(row[v], to_drop));
            }
        }

        // this should be done very differently: create 0-1 matrix of definitions
        const ones = ['extra', 'mut', 'shared', 'revert', 'subset'];
        this.dfTally = this.dfTally.map((row) =>
            Object.fromEntries(
                Object.entries(row).map(([column, value]) => [
                    column,
                    isMissing(value) ? 0 : typeof value === 'string' && ones.includes(value) ? 1 : value,
                ]),
            ),
        );

        // remove uninformative mutations
        const variantsColumns = [...new Set(variants_list)].filter((variant) => this.columns.includes(variant));
        this.dfTally = removeUninformative(this.dfTally, variantsColumns);

        // make complement of mutation signatures for undetermined cases
        this.columns.splice(Math.max(this.columns.length - 1, 0), 0, 'undetermined');
        this.dfTally = this.dfTally.map((row) => ({ ...row, undetermined: 0 }));
        if (make_complement) {
            this.dfTally = [...this.dfTally, ...this.makeComplement(this.dfTally, variantsColumns, namefield)];
        }

        return this;
    }

    /** filter out hardcoded problematic mutations */
    filterMutations(filters?: Filters | null): this {
        if (!filters || Object.keys(filters).length === 0) {
            return this;
        }

        const numeric = new Set(
            this.columns.filter((column) => this.dfTally.every((row) => typeof row[column] === 'number')),
        );

        const rxprser = new RegExp(
            String.raw`^ *(?<qc>['"]?)(?:(?<col>` +
                this.columns.join('|') +
                String.raw`)|(?<bad>\w+))\k<qc> *(?<op>in|[<>=~!]*) *(?<qv>['"]?)(?<val>.+)\k<qv> *$`,
        );

        /** parse a single statement from a filter and apply it, returning a boolean per row */
        const applyFilterStatement = (name: string, fs: string): boolean[] => {
            const match = rxprser.exec(fs);
            if (match === null) {
                throw new Error(`Cannot parse statement <${fs}> in filter ${name}`);
            }
            const m = match.groups as Record<'qc' | 'col' | 'bad' | 'op' | 'qv' | 'val', string | undefined>;
            if (!m.col) {
                throw new Error(
                    `bad column name ${m.bad}, not in list: ${this.columns}, while parsing statement <${fs}> in filter ${name}`,
                );
            }
            const column = m.col;
            const rawVal = m.val!;

            // HACK handle 'date' column differently, to force datatypes
            const isDate = column === 'date';
            const col: Cell[] = this.dfTally.map((row) => (isDate ? toDay(row.date as string) : row[column]));
            const val: Cell = isDate ? toDay(rawVal) : numeric.has(column) ? toNumeric(rawVal) : rawVal;

            // apply operator
            const op = m.op ?? '';
            switch (op) {
                case '=':
                case '==':
                case '':
                    if (op === '' && typeof val === 'string' && val.includes(' ')) {
                        throw new Error(
                            `Do not use values with space <${val}> when using no operator (implicit 'equals'). (while parsing statement <${fs}> in filter ${name})`,
                        );
                    }
                    return col.map((value) => value === val);
                case '!=':
                case '!':
                    return col.map((value) => value !== val);
                case '<':
                    return col.map((value) => compare(value, val, (a, b) => a < b));
                case '<=':
                case '=<':
                    return col.map((value) => compare(value, val, (a, b) => a <= b));
                case '>=':
                    return col.map((value) => compare(value, val, (a, b) => a >= b));
                case '>':
                    return col.map((value) => compare(value, val, (a, b) => a > b));
                case 'in': {
                    // unpack list
                    const list = String(val)
                        .replace(/^[[ ]+/, '')
                        .replace(/[ \]]+$/, '')
                        .split(',')
                        .map((v) => v.replace(/^["' ]+|["' ]+$/g, ''));
                    return col.map((value) => typeof value === 'string' && list.includes(value));
                }
                case '~':
                case '=~':
                case '~=':
                case '!~':
                case '~!': {
                    if (typeof val !== 'string' || col.some((value) => typeof value !== 'string')) {
                        throw new Error(`Can only use ~ with string values, in statement <${fs}> in filter ${name}`);
                    }
                    const pattern =
                        !m.qv && val[0] === val[val.length - 1] && '/@'.includes(val[val.length - 1])
                            ? val.slice(1, -1)
                            : val;
                    const regex = new RegExp(pattern);
                    const contains = col.map((value) => regex.test(value as string));
                    return op === '!~' || op === '~!' ? contains.map((value) => !value) : contains;
                }
                default:
                    throw new Error(`unknown operator ${op}, while parsing statement <${fs}> in filter ${name}`);
            }
        };

        for (const [name, fl] of Object.entries(filters)) {
            this.log(`filter ${name}`);
            const statements = fl.map((fstatmt) => applyFilterStatement(name, fstatmt));
            this.dfTally = this.dfTally.filter((_, i) => !statements.every((result) => result[i]));
        }

        return this;
    }

    private dropColumns(columns: readonly string[]): void {
        const dropped = new Set(columns);
        this.columns = this.columns.filter((column) => !dropped.has(column));
        this.dfTally = this.dfTally.map((row) =>
            Object.fromEntries(Object.entries(row).filter(([column]) => !dropped.has(column))),
        );
    }

    private setColumn(column: string, value: (row: Row) => Cell): void {
        if (!this.columns.includes(column)) {
            this.columns.push(column);
        }
        this.dfTally = this.dfTally.map((row) => ({ ...row, [column]: value(row) }));
    }
}

/** Removes the rows that all or none of the variants carry. */
export function removeUninformative(rows: readonly Row[], variantsColumns: readonly string[]): Row[] {
    return rows.filter((row) => {
        let total = 0;
        for (const variant of variantsColumns) {
            total += row[variant] as number;
        }
        return total !== 0 && total !== variantsColumns.length;
    });
}

/** Like pandas' `isin`: NaN matches NaN. */
function isIn(value: Cell, values: readonly Cell[]): boolean {
    return values.some((candidate) => candidate === value || (isMissing(candidate) && isMissing(value)));
}

/** String concatenation of pandas columns: NaN if either is. */
function concat(first: Cell, second: Cell): Cell {
    return isMissing(first) || isMissing(second) ? null : `${first}${second}`;
}

/** Like Python's `str()` of a number in a column of integers or floats. */
function pythonStr(value: Cell): Cell {
    return typeof value === 'number' && Number.isInteger(value) ? String(value) : value;
}

/** Like `pd.to_numeric` on a single string. */
function toNumeric(value: string): number {
    const number = Number(value);
    if (value.trim() === '' || Number.isNaN(number)) {
        throw new Error(`Unable to parse string "${value}"`);
    }
    return number;
}

/** A comparison of pandas: false with NaN. */
function compare(a: Cell, b: Cell, comparison: (a: number | string, b: number | string) => boolean): boolean {
    if (isMissing(a) || isMissing(b)) {
        return false;
    }
    if (typeof a !== typeof b) {
        throw new TypeError(`'<' not supported between ${typeof a} and ${typeof b}`);
    }
    return comparison(a, b);
}
