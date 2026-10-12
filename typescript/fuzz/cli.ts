// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * The fuzz test, end to end: generates cases with LolliPop (run_lollipop.py), evaluates the port
 * on them (evaluate.ts), prints what failed and writes a report. See README.md.
 *
 *     npm run fuzz -- [--count N] [--seed S] [--preprint PATH] [--jobs J]
 *     npm run fuzz -- --dir fuzz/out/<seed> [--case case-<n>]     (evaluate existing cases again)
 */

import { spawnSync } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { evaluateCase, type CaseReport } from './evaluate.js';
import { loadCases } from './runPort.js';

const here = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
    options: {
        count: { type: 'string', default: '100' },
        seed: { type: 'string', default: String(Math.floor(Math.random() * 1e9)) },
        preprint: { type: 'string' },
        jobs: { type: 'string' },
        dir: { type: 'string' },
        case: { type: 'string' },
    },
});

/** Runs a command; exits if it can't, or (unless `mayFail`) if it fails. */
function run(command: string, commandArgs: string[], mayFail = false): boolean {
    const result = spawnSync(command, commandArgs, { stdio: 'inherit' });
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0 && !mayFail) {
        process.exit(result.status ?? 1);
    }
    return result.status === 0;
}

let generatorFailed = false;

/** Generates the cases into `fuzz/out/<seed>`, with a Python environment for LolliPop. */
function generate(): string {
    const venv = join(here, '.venv');
    const python = join(venv, 'bin', 'python');
    if (!existsSync(python)) {
        run('uv', ['venv', '--python', '3.12', venv]);
        // (LolliPop's logit intervals use DataFrame.applymap, which pandas 3 removed)
        run('uv', ['pip', 'install', '--python', python, '-e', `${join(here, '..', '..')}[cli]`, 'pandas<3']);
    }
    const out = join(here, 'out', args.seed);
    const generatorArgs = [join(here, 'run_lollipop.py'), '--seed', args.seed, '--count', args.count, '--out', out];
    if (args.preprint) generatorArgs.push('--preprint', args.preprint);
    if (args.jobs) generatorArgs.push('--jobs', args.jobs);
    console.log(`generating ${args.count} cases from seed ${args.seed} into ${out}`);
    // (where some cases fail to generate, evaluate the others, and fail at the end)
    generatorFailed = !run(python, generatorArgs, true);
    return out;
}

const directory = args.dir ?? generate();
const reports: CaseReport[] = [];
let failed = 0;
for (const fuzzCase of loadCases(directory, args.case)) {
    const { problems, report } = evaluateCase(fuzzCase);
    reports.push(report);
    const failures = Object.entries(problems).filter(([, list]) => list.length > 0);
    if (failures.length > 0) {
        failed++;
        console.log(`FAIL ${fuzzCase.id} (${fuzzCase.source}, ${fuzzCase.configKind})`);
        for (const [check, list] of failures) {
            console.log(`  ${check}:\n${list.map((problem) => `    ${problem}`).join('\n')}`);
        }
    }
}
writeFileSync(join(directory, 'report.json'), JSON.stringify(reports, null, 1));
console.log(
    `${reports.length - failed} of ${reports.length} cases passed; report in ${join(directory, 'report.json')}`,
);
if (generatorFailed) {
    console.log('some cases failed to generate (see above)');
}
process.exitCode = failed > 0 || generatorFailed ? 1 : 0;
