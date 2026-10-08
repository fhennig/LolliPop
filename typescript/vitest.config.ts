import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        include: ['src/**/*.spec.ts', 'test/**/*.spec.ts', 'fuzz/**/*.spec.ts'],
        exclude: ['**/node_modules/**', 'fuzz/.venv/**'],
        testTimeout: 120_000,
    },
});
