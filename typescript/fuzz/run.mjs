// SPDX-License-Identifier: GPL-3.0-or-later

// npm run fuzz -- [--count N] [--seed S] [--preprint PATH] [--jobs J]
//
// Sets up the Python environment (uv: LolliPop from this repository), generates fuzz cases with
// generate.py into fuzz/out/<seed>, and runs the fuzz test on them.

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({
    options: {
        count: { type: 'string', default: '100' },
        seed: { type: 'string', default: String(Math.floor(Math.random() * 1e9)) },
        preprint: { type: 'string' },
        jobs: { type: 'string' },
    },
});

function run(command, args, options = {}) {
    const result = spawnSync(command, args, { stdio: 'inherit', ...options });
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        process.exit(result.status ?? 1);
    }
}

const venv = join(here, '.venv');
const python = join(venv, 'bin', 'python');
if (!existsSync(python)) {
    run('uv', ['venv', '--python', '3.12', venv]);
    run('uv', ['pip', 'install', '--python', python, '-e', `${join(here, '..', '..')}[cli]`, 'pandas<3']);
}

const out = join(here, 'out', values.seed);
const args = [join(here, 'generate.py'), '--seed', values.seed, '--count', values.count, '--out', out];
if (values.preprint) args.push('--preprint', values.preprint);
if (values.jobs) args.push('--jobs', values.jobs);
console.log(`generating ${values.count} cases from seed ${values.seed} into ${out}`);
run(python, args);

run('npx', ['vitest', 'run', 'fuzz'], {
    cwd: join(here, '..'),
    env: { ...process.env, FUZZ_DIR: out, FUZZ_REPORT: join(out, 'report.json') },
});
