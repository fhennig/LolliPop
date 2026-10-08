// SPDX-License-Identifier: GPL-3.0-or-later
// Port of LolliPop's lollipop/cli/deconvolute.py: the whole pipeline, without the files, the
// command line, the progress bars and the parallel processing.

import { NullConfint, resampleMutations, WaldConfint, type Confint, type WaldConfintParams } from './confints.js';
import { KernelDeconv, type DeconvParams } from './kernelDeconv.js';
import { BoxKernel, GaussianKernel, type Kernel } from './kernels.js';
import { RandomState } from './numeric/randomState.js';
import { nanMean, quantile } from './numeric/stats.js';
import { DataPreprocesser, removeUninformative, type Filters, type Log } from './preprocessors.js';
import { NnlsReg, RobustReg, type Regressor, type RobustRegParams } from './regressors.js';
import { fromDay, isMissing, toDay, unique, type Cell, type Frame, type Row } from './table.js';

/** The variants configuration (`--variants-config`), as in LolliPop's YAML. */
export type VariantsConfig = {
    variants_pangolin: Record<string, string>;
    variants_list?: string[] | null;
    variants_not_reported?: string[];
    to_drop?: Cell[];
    no_date?: boolean;
    no_loc?: boolean;
    start_date?: string | null;
    end_date?: string | null;
    remove_deletions?: boolean;
    locations_list?: string[] | null;
};

/** The deconvolution configuration (`--deconv-config`), as in LolliPop's YAML presets. */
export type DeconvConfig = {
    bootstrap?: number;
    kernel?: string;
    kernel_params?: { bandwidth?: number };
    confint?: string;
    confint_params?: WaldConfintParams;
    regressor?: string;
    regressor_params?: RobustRegParams;
    deconv_params?: DeconvParams;
};

/** The variants to scan per period (`--variants-dates`): from each date on, until the next. */
export type VarDates = { var_dates: Record<string, string[]> };

export type DeconvoluteArgs = {
    /** The tally table, one row per mutation and sample (`TALLY_TSV`). */
    tally: readonly Row[];
    /** The columns of the tally table in order, like its header; by default, the keys of the rows. */
    columns?: readonly string[];
    variantsConfig: VariantsConfig;
    deconvConfig: DeconvConfig;
    variantsDates?: VarDates | null;
    /** `--loc`: the locations to process. */
    loc?: string[] | null;
    filters?: Filters | null;
    /**
     * The generator of the bootstrap: a seed, or a generator, which then carries over from one
     * call to the next, like NumPy's global one in LolliPop.
     */
    random?: RandomState | number;
    namefield?: string;
    /** What LolliPop prints. */
    log?: Log;
    /** Called with the indices each bootstrap round draws, for tests. */
    onResample?: (randIdcs: number[]) => void;
    /** Called with the loss of every regression, in order, for tests. */
    onLoss?: (loss: number) => void;
};

/** A row of LolliPop's output: the estimate for a variant at a location on a date. */
export type DeconvolutedRow = {
    location: string;
    variant: string;
    /** ISO date; with `no_date`, the dummy date that stands for the sample. */
    date: string;
    proportion: number;
    /** With a bootstrap or a confidence interval. */
    proportionLower?: number;
    proportionUpper?: number;
};

const kernels: Record<string, new (bandwidth?: number) => Kernel> = {
    gaussian: GaussianKernel,
    box: BoxKernel,
};
const confints: Record<string, new (params?: WaldConfintParams) => Confint> = {
    null: NullConfint,
    wald: WaldConfint,
};
const regressors: Record<string, new (params?: RobustRegParams) => Regressor> = {
    nnls: NnlsReg,
    robust: RobustReg,
};

/** A frame of results of one location and period, with the columns `location` and `estimate`. */
type ResultFrame = { frame: Frame; location: string; estimate?: string };

type DeconvoluteBootstrapArgs = {
    location: string;
    locDf: Row[];
    columns: string[];
    bootstrap: number;
    noDate: boolean;
    dateIntervals: [string, string | null][];
    varDates: VarDates;
    kernel: new (bandwidth?: number) => Kernel;
    kernelParams: { bandwidth?: number };
    regressor: new (params?: RobustRegParams) => Regressor;
    regressorParams: RobustRegParams;
    confint: new (params?: WaldConfintParams) => Confint;
    confintParams: WaldConfintParams;
    deconvParams: DeconvParams;
    haveConfint: boolean;
    confintName: string | null;
    namefield: string;
    random: RandomState;
    onResample?: (randIdcs: number[]) => void;
    onLoss?: (loss: number) => void;
};

/** Deconvolute the data for a given location and bootstrap iteration. */
function deconvoluteBootstrap({
    location,
    locDf,
    columns,
    bootstrap,
    noDate,
    dateIntervals,
    varDates,
    kernel,
    kernelParams,
    regressor,
    regressorParams,
    confint,
    confintParams,
    deconvParams,
    haveConfint,
    confintName,
    namefield,
    random,
    onResample,
    onLoss,
}: DeconvoluteBootstrapArgs): ResultFrame[] {
    // deconvolution results
    const deconv: ResultFrame[] = [];

    for (let b = 0; b < (bootstrap > 1 ? bootstrap : 1); b++) {
        let tempDfb: Row[];
        if (bootstrap > 1) {
            // resample if we're doing bootstrapping
            if (!columns.includes(namefield)) {
                throw new Error(
                    `bootstrapping needs a column with names for the entries of the tally table, but no column '${namefield}' found. Use option '--namefield' to specify`,
                );
            }
            const mutations = unique(locDf.map((row) => row[namefield]));
            const { dfSampled, randIdcs } = resampleMutations(locDf, mutations, random, namefield);
            onResample?.(randIdcs);
            tempDfb = dfSampled;
        } else {
            // just run one on everything
            tempDfb = locDf;
        }

        for (const [mindate, maxdate] of dateIntervals) {
            let tempDf2: Row[];
            if (!noDate) {
                // filter by time period for period-specific variants list
                const min = toDay(mindate);
                if (maxdate !== null) {
                    const max = toDay(maxdate);
                    tempDf2 = tempDfb.filter((row) => {
                        const day = toDay(row.date as string);
                        return day >= min && day < max;
                    });
                } else {
                    tempDf2 = tempDfb.filter((row) => toDay(row.date as string) >= min);
                }
            } else {
                // no date => no filtering
                tempDf2 = tempDfb;
            }
            if (tempDf2.length === 0) {
                continue;
            }

            // remove uninformative mutations (present either always or never)
            const intervalVariants = varDates.var_dates[mindate];
            const variantsColumns = unique(intervalVariants).filter((variant) => variant in tempDf2[0]);
            tempDf2 = removeUninformative(tempDf2, variantsColumns);
            if (tempDf2.length === 0) {
                continue;
            }

            // define deconvolution kernel
            const variantNames = [...intervalVariants, 'undetermined'];
            for (const variant of variantNames) {
                if (!(variant in tempDf2[0])) {
                    throw new Error(`KeyError: "['${variant}'] not in index"`);
                }
            }
            const tKdec = new KernelDeconv({
                X: tempDf2.map((row) => variantNames.map((variant) => row[variant] as number)),
                variantNames,
                y: tempDf2.map((row) => row.frac as number),
                dates: tempDf2.map((row) => row.date as string),
                // resampling weights
                weights: bootstrap > 1 ? tempDf2.map((row) => row.resample_value as number) : undefined,
                kernel: new kernel(kernelParams.bandwidth),
                reg: new regressor(regressorParams),
                confint: new confint(confintParams),
            });
            // do the deconvolution
            tKdec.deconvAll(deconvParams);
            tKdec.loss!.forEach((loss) => onLoss?.(loss));

            if (haveConfint) {
                // with conf int
                deconv.push({ frame: tKdec.fitted!, location, estimate: 'MSE' });
                deconv.push({ frame: tKdec.confBands!.lower, location, estimate: `${confintName}_lower` });
                deconv.push({ frame: tKdec.confBands!.upper, location, estimate: `${confintName}_upper` });
            } else {
                // without conf int
                deconv.push({ frame: tKdec.fitted!, location });
            }
        }
    }

    return deconv;
}

/** Deconvolution for Wastewater Genomics: LolliPop's `deconvolute` command, as a function. */
export function deconvolute({
    tally,
    columns: tallyColumns,
    variantsConfig: confYaml,
    deconvConfig: deconv,
    variantsDates = null,
    loc = null,
    filters = null,
    random: randomOrSeed,
    namefield = 'mutations',
    log = () => {},
    onResample,
    onLoss,
}: DeconvoluteArgs): DeconvolutedRow[] {
    const variantsPangolin = confYaml.variants_pangolin;
    let variantsList = confYaml.variants_list != null ? [...confYaml.variants_list] : null;
    const variantsNotReported = confYaml.variants_not_reported ?? [];
    const toDrop = confYaml.to_drop ?? [];
    let noDate = confYaml.no_date ?? false;
    let noLoc = confYaml.no_loc ?? false;
    const startDate = confYaml.start_date ?? null;
    const endDate = confYaml.end_date ?? null;
    const removeDeletions = confYaml.remove_deletions ?? true;
    let locationsList = loc && loc.length ? [...loc] : (confYaml.locations_list ?? null);

    const columns = tallyColumns !== undefined ? [...tallyColumns] : [...new Set(tally.flatMap(Object.keys))];
    let dfTally: Row[] = tally.map((row) => Object.fromEntries(columns.map((column) => [column, row[column] ?? null])));
    const setColumn = (column: string, value: (row: Row, i: number) => Cell) => {
        if (!columns.includes(column)) {
            columns.push(column);
        }
        dfTally = dfTally.map((row, i) => ({ ...row, [column]: value(row, i) }));
    };

    // handle location
    if (!noLoc && !columns.includes('location')) {
        if (columns.includes('location_code')) {
            log('NOTE: No location fullnames, using codes instead');
            setColumn('location', (row) => row.location_code);
        } else if (locationsList === null) {
            log(
                'WARNING: No location in input data. Either pass one with `--loc`/`locations_list` parameter  or set true the `no_loc` parameter',
            );
            noLoc = true;
        } else if (locationsList.length === 1) {
            log(`WARNING: No location in input data, assuming everything is ${locationsList[0]}`);
            const location = locationsList[0];
            setColumn('location', () => location);
        } else {
            throw new Error(
                'ERROR: No location in input data. Either pass exactly one with `--loc`/`locations_list` parameter or set true the `no_loc` parameter',
            );
        }
    }

    if (noLoc) {
        if (columns.includes('location')) {
            const present = locationNames(dfTally);
            if (present.length) {
                log(`WARNING: no_loc is set, but there are still locations in input: ${present}`);
            }
        } else {
            log('no_loc: ignoring location information and treating all input as a single location');
        }
        setColumn('location', () => 'location');
        locationsList = ['location'];
    }

    if (locationsList === null) {
        // remember to remove empty cells: nan or empty cells
        // (in the order of appearance; LolliPop takes them in the order of a Python set)
        locationsList = locationNames(dfTally);
    } else {
        const present = new Set(dfTally.map((row) => row.location));
        const badLocations = locationsList.filter((location) => !present.has(location));
        if (badLocations.length !== 0) {
            throw new Error(`Bad locations in list: ${badLocations}, please fix the variants config.`);
        }
    }

    // check if dates are present
    if (!columns.includes('date') || dfTally.every((row) => isMissing(row.date))) {
        if (!noDate) {
            noDate = true;
            log('WARNING: No dates found in input data, automatically switching `no_date` !!!');
        }
    }
    // no date!!!
    if (noDate) {
        log('no_date: special mode for deconvoluting without time component');
        // HACK dummy date to keep the deconvolution kernel happy
        const samples = unique(dfTally.map((row) => row.sample));
        const dateDict = new Map(samples.map((sample, i) => [sample, fromDay(toDay('1999-12-01') + i)]));
        setColumn('date', (row) => dateDict.get(row.sample)!);
    }

    // dates intervals for which to apply different variants as discovered using cojac
    let varDates: VarDates;
    if (variantsDates) {
        if (noDate) {
            log('WARNING: running in `no_date` mode, still var_dates specified');
        }
        varDates = variantsDates;

        const allVarDates = unique(Object.values(varDates.var_dates).flat());

        if (variantsList === null) {
            // build list of all variants from var_dates (if we did lack one)
            variantsList = allVarDates;
        } else {
            // have list => double - check it against var_dates
            const list = variantsList;
            const notOnDate = list.filter((variant) => !allVarDates.includes(variant));
            if (notOnDate.length) {
                log(`NOTE: ${notOnDate} never used in var_dates, despite being in variants_list`);
            }
            const notOnList = allVarDates.filter((variant) => !list.includes(variant));
            if (notOnList.length) {
                log(`WARNING: var_dates lists variants: ${notOnList}, but they are not in variants_list`);
                variantsList = [...list, ...notOnList];
            }
        }
    } else {
        if (variantsList === null) {
            // build list of all variants from lineage map (if we did lack one)
            variantsList = unique(Object.values(variantsPangolin));
        }

        if (noDate) {
            // dummy date
            varDates = { var_dates: { '1999-12-01': variantsList } };
        } else {
            // search for all, always
            varDates = { var_dates: { [confYaml.start_date ?? '2020-01-01']: variantsList } };
            log(
                'NOTE: deconvoluting for all variants on all dates. Consider writing a var_dates YAML based on cojac detections',
            );
        }
    }

    // build the intervals pairs
    const d = Object.keys(varDates.var_dates);
    const dateIntervals = d.map((mindate, i): [string, string | null] => [mindate, d[i + 1] ?? null]);
    if (!noDate) {
        for (const [mindate, maxdate] of dateIntervals) {
            if (maxdate !== null && !(mindate < maxdate)) {
                throw new Error(`out of order dates: ${mindate} >= ${maxdate}. Please fix the content of var_dates`);
            }
        }
    }

    const preproc = new DataPreprocesser(dfTally, columns, log)
        .generalPreprocess({
            variants_list: variantsList,
            variants_pangolin: variantsPangolin,
            variants_not_reported: variantsNotReported,
            to_drop: toDrop,
            start_date: startDate,
            end_date: endDate,
            no_date: noDate,
            remove_deletions: removeDeletions,
            namefield,
        })
        .filterMutations(filters);

    const random = randomOrSeed instanceof RandomState ? randomOrSeed : new RandomState(randomOrSeed ?? 0);

    // bootstrap
    const bootstrap = deconv.bootstrap ?? 0;
    // kernel
    const kernel = kernels[deconv.kernel ?? ''] ?? GaussianKernel;
    const kernelParams = { ...deconv.kernel_params };
    if (noDate) {
        log('no_date: overriding kernel bandwidth');
        kernelParams.bandwidth = 1e-17;
    }
    // confint
    const confint = confints[deconv.confint ?? ''] ?? NullConfint;
    const haveConfint = confint !== NullConfint;
    if (haveConfint && bootstrap > 1) {
        throw new Error(
            `either use bootstrapping or a confint class, not both at the same time.\nbootstrap: ${bootstrap}, confint: ${deconv.confint}`,
        );
    }
    const confintName = haveConfint ? capitalize(deconv.confint!) : null;
    const confintParams = deconv.confint_params ?? {};
    // regressor
    const regressor = regressors[deconv.regressor ?? ''] ?? NnlsReg;
    const regressorParams = deconv.regressor_params ?? {};
    // deconv
    const deconvParams = deconv.deconv_params ?? {};

    // CORE DECONVOLUTION
    // iterate over locations
    const allDeconv = locationsList.flatMap((location) =>
        deconvoluteBootstrap({
            location,
            locDf: noLoc ? preproc.dfTally : preproc.dfTally.filter((row) => row.location === location),
            columns: preproc.columns,
            bootstrap,
            noDate,
            dateIntervals,
            varDates,
            kernel,
            kernelParams,
            regressor,
            regressorParams,
            confint,
            confintParams,
            deconvParams,
            haveConfint,
            confintName,
            namefield,
            random,
            onResample,
            onLoss,
        }),
    );

    // concatenate all the results; a variant missing from a frame is NaN, like with pd.concat
    if (allDeconv.length === 0) {
        throw new Error('No objects to concatenate');
    }
    const foundColumns = new Set(allDeconv.flatMap(({ frame }) => frame.columns));
    // variants actually in dataframe
    const foundVar = variantsList.filter((variant) => foundColumns.has(variant));
    if (foundVar.length < variantsList.length) {
        log(`some variants never found in dataset ${variantsList.filter((variant) => !foundColumns.has(variant))}`);
    }
    const valueVars = [...foundVar, 'undetermined'];

    // deconv output, melted: one value per location, estimate, variant and date
    type Melted = { location: string; estimate?: string; variant: string; date: string; frac: number };
    const flat: Melted[] = [];
    for (const { frame, location, estimate } of allDeconv) {
        for (const variant of valueVars) {
            const j = frame.columns.indexOf(variant);
            frame.index.forEach((date, i) => {
                let frac = j === -1 ? NaN : frame.values[i][j];
                if (!haveConfint && Number.isNaN(frac)) {
                    frac = 0;
                }
                flat.push({ location, estimate, variant, date, frac });
            });
        }
    }

    // aggregation
    const groups = new Map<string, Melted[]>();
    for (const row of flat) {
        const key = JSON.stringify([row.location, row.variant, row.date]);
        let group = groups.get(key);
        if (group === undefined) {
            group = [];
            groups.set(key, group);
        }
        group.push(row);
    }
    let deconvDfAgg: DeconvolutedRow[];
    if (bootstrap > 1) {
        // bootstrap => mean + quantiles
        deconvDfAgg = [...groups.values()].map((group) => {
            const values = group.map((row) => row.frac);
            return {
                location: group[0].location,
                variant: group[0].variant,
                date: group[0].date,
                proportion: nanMean(values),
                proportionLower: quantile(values, 0.025),
                proportionUpper: quantile(values, 0.975),
            };
        });
    } else if (haveConfint) {
        // wald => pivot
        deconvDfAgg = [...groups.values()].map((group) => {
            const byEstimate = (estimate: string) => {
                const matches = group.filter((row) => row.estimate === estimate);
                if (matches.length > 1) {
                    throw new Error('Index contains duplicate entries, cannot reshape');
                }
                return matches[0]?.frac ?? NaN;
            };
            return {
                location: group[0].location,
                variant: group[0].variant,
                date: group[0].date,
                proportion: byEstimate('MSE'),
                proportionLower: byEstimate(`${confintName}_lower`),
                proportionUpper: byEstimate(`${confintName}_upper`),
            };
        });
    } else {
        // no conf => as-is
        deconvDfAgg = flat.map(({ location, variant, date, frac }) => ({ location, variant, date, proportion: frac }));
    }
    deconvDfAgg.sort(
        (a, b) =>
            compareStrings(a.location, b.location) ||
            compareStrings(a.variant, b.variant) ||
            compareStrings(a.date, b.date),
    );
    // reverse logit scale
    // (LolliPop reads `confint_params["scale"]` here, and fails without it: here, it is linear,
    // WaldConfint's default.)
    if (haveConfint && confintParams.scale === 'logit') {
        const expit = (x: number) => {
            const clipped = Math.min(100, Math.max(-100, x));
            return Math.exp(clipped) / (1 + Math.exp(clipped));
        };
        for (const row of deconvDfAgg) {
            row.proportionLower = expit(row.proportionLower!);
            row.proportionUpper = expit(row.proportionUpper!);
        }
    }

    return deconvDfAgg;
}

/** The locations in the data, without empty ones, in the order of appearance. */
function locationNames(rows: readonly Row[]): string[] {
    return unique(rows.map((row) => row.location)).filter(
        (location): location is string => !isMissing(location) && location !== '',
    ) as string[];
}

function capitalize(value: string): string {
    return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
}

function compareStrings(a: string, b: string): number {
    return a < b ? -1 : a > b ? 1 : 0;
}
