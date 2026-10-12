// SPDX-License-Identifier: GPL-3.0-or-later

/** The fuzz test on the corpus (`fuzz/corpus`), for `npm test`. New cases: `npm run fuzz`. */

import { describe, expect, test } from 'vitest';

import { evaluateCase } from './evaluate.js';
import { loadCases } from './runPort.js';

const cases = loadCases(new URL('corpus', import.meta.url).pathname);

describe('fuzz corpus', () => {
    test.each(cases.map((fuzzCase) => [fuzzCase.id, fuzzCase] as const))('%s', (_, fuzzCase) => {
        const { problems } = evaluateCase(fuzzCase);
        const failures = Object.fromEntries(Object.entries(problems).filter(([, list]) => list.length > 0));
        expect(failures).toEqual({});
    });
});
