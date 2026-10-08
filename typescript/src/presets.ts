// SPDX-License-Identifier: GPL-3.0-or-later
// LolliPop's presets/*.yaml.

import { type DeconvConfig } from './deconvolute.js';

export const PRESETS = {
    /** Bootstrap Confints */
    deconv_bootstrap: {
        kernel_params: { bandwidth: 30 },
        bootstrap: 1000,
        deconv_params: { min_tol: 1e-3 },
    },
    /** The one of the Swiss surveillance (COWWID). */
    deconv_bootstrap_cowwid: {
        bootstrap: 100,
        kernel_params: { bandwidth: 10 },
        regressor: 'robust',
        regressor_params: { f_scale: 0.01 },
        deconv_params: { min_tol: 1e-3 },
    },
    /** Deconvolution with linear reg / LS loss */
    deconv_linear: {
        kernel_params: { bandwidth: 10 },
    },
    /** Logit parametrized Wald confints */
    deconv_linear_logit: {
        kernel_params: { bandwidth: 30 },
        confint: 'wald',
        confint_params: { scale: 'logit', pseudofrac: 0.01 },
        deconv_params: { min_tol: 1e-3 },
    },
    /** Logit parametrised confidence intervals with quasilikelihood */
    deconv_linear_logit_quasi: {
        kernel_params: { bandwidth: 30 },
        confint: 'wald',
        confint_params: { quasi: true, scale: 'logit', pseudofrac: 0.01 },
        deconv_params: { min_tol: 1e-3 },
    },
    /** Logit parametrised confidence intervals with quasilikelihood, stratified */
    deconv_linear_logit_quasi_strat: {
        kernel_params: { bandwidth: 30 },
        confint: 'wald',
        confint_params: { quasi: true, scale: 'logit', pseudofrac: 0.01, method: 'strat' },
        deconv_params: { min_tol: 1e-3 },
    },
    /** Wald confint, quasibinomial stratified */
    deconv_linear_quasi_strat: {
        kernel_params: { bandwidth: 30 },
        confint: 'wald',
        confint_params: { quasi: true, method: 'strat' },
        deconv_params: { min_tol: 1e-3 },
    },
    /** Deconvolution with Wald Confints */
    deconv_linear_wald: {
        kernel_params: { bandwidth: 30 },
        confint: 'wald',
        deconv_params: { min_tol: 1e-3 },
    },
    /** Wald confints, quasibinomial */
    deconv_linear_wald_quasibinom: {
        kernel_params: { bandwidth: 30 },
        confint: 'wald',
        confint_params: { quasi: true },
        deconv_params: { min_tol: 1e-3 },
    },
    deconv_mcgill: {
        kernel: 'gaussian',
        kernel_params: { bandwidth: 1e-17 },
        confint: 'null',
        regressor: 'robust',
    },
} satisfies Record<string, DeconvConfig>;

export type PresetName = keyof typeof PRESETS;
