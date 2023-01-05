"""
Generate risk scenarios for Value-at-Risk (VaR) calculation; compute statistics
for VaR backtesting.
"""

# Copyright 2022 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import itertools
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm

from scipy import interpolate
from tqdm import tqdm
from arbitragerepair import repair, constraints
from marketmodel.utils import Finance


def calibrate_factors(cs_t, G_full, W_full, b_full, n_factor_pri, n_factor_sec):
    """
    Calibrate primary and secondary factors from normalised call option prices.
    Two-step calibration: first primary factors and then secondary ones.

    """

    # step 1: calibrate primary factors
    y1 = (cs_t - G_full[0]).reshape((-1, 1))
    x1 = G_full[1:1+n_factor_pri].T
    mdl1 = sm.OLS(y1, x1).fit()
    xi_pri = mdl1.params

    # step 2: calibrate seconary factors
    y2 = mdl1.resid.reshape((-1, 1))
    x2 = G_full[1+n_factor_pri:].T
    mdl2 = sm.OLS(y2, x2).fit()
    xi_sec = mdl2.params

    # repair arbitrage in the calibrated factors
    xi = np.hstack((xi_pri, xi_sec))

    if np.sum(xi.dot(W_full.T) - b_full < 0) > 0:
        # add artificial spread to discourage perturbing primary factors
        spread = np.hstack((np.ones((2, n_factor_pri)) * 1e-4,
                            np.ones((2, n_factor_sec))))
        eps = repair.l1ba(W_full, b_full, xi, spread)
        xi = xi + eps

    return xi


def calculate_base_normc(S_t, cs_t, S_next, cs_next, ks, Ts, mpor, dt):
    """
    Calculate/approximate realised PnL of normalised calls.

    """

    Cs = cs_t * S_t
    Ks = ks * S_t
    Ts_risk = Ts - mpor * dt

    ks_realised = Ks / S_next
    Ts_realised = Ts_risk

    # interpolate normalised call prices
    alpha_k = 1. / (np.max(ks) - np.min(ks))
    alpha_T = 1. / (np.max(Ts) - np.min(Ts))

    interp_c = interpolate.Rbf(ks * alpha_k, Ts * alpha_T, cs_next,
                               function='multiquadric', epsilon=.01)

    cs_realised_ = interp_c(ks_realised * alpha_k, Ts_realised * alpha_T)

    Cs_realised = cs_realised_ * S_next

    return Cs, Cs_realised


def calculate_scenario_normc(xi_scenario, S_scenario, G_full, W_full, b_full,
                             S_t, tolerance, cs_weights, ks, Ts, mpor, dt,
                             n_factor_pri, n_factor_sec, repair_arbitrage=False):
    """
    Calculate scenario PnLs of normalised calls. The scenarios are generated
    from trained neural-SDE models (of xi and S).

    """

    idxs_arb = np.where(
        np.any(xi_scenario.dot(W_full.T) - b_full[None, :] < -tolerance,
               axis=1))[0]

    spread = np.hstack((np.ones((2, n_factor_pri)) * 1e-4,
                        np.ones((2, n_factor_sec))))

    epsilons = np.zeros_like(xi_scenario)
    for i in tqdm(idxs_arb):
        eps = repair.l1ba(W_full, b_full, xi_scenario[i], spread)
        epsilons[i] = eps

    xi_scenario += epsilons

    # reconstruct option prices from all scenario factors
    cs_scenario = (xi_scenario.dot(G_full[1:]) + G_full[0][None,:]) / cs_weights

    # interpolate scenario normalised call prices
    cs_scenario_decayT = np.zeros_like(cs_scenario)
    cs_scenario_decayT_arbfree = np.zeros_like(cs_scenario)

    n_scenario = xi_scenario.shape[0]
    alpha_k = 1. / (np.max(ks) - np.min(ks))
    alpha_T = 1. / (np.max(Ts) - np.min(Ts))
    for i_scenario in tqdm(range(n_scenario)):

        cs_i_anchor = cs_scenario[i_scenario]
        interp_c = interpolate.Rbf(ks * alpha_k, Ts * alpha_T, cs_i_anchor,
                                   function='multiquadric', epsilon=.01)

        ks_scenario = ks * S_t / S_scenario[i_scenario]
        Ts_scenario = Ts - mpor * dt
        cs_i = interp_c(ks_scenario * alpha_k, Ts_scenario * alpha_T)
        cs_scenario_decayT[i_scenario] = cs_i

        # repair arbitrage
        if repair_arbitrage:
            mat_A_i, vec_b_i, _, n = constraints.detect(
                Ts_scenario, ks_scenario, cs_i, verbose=False)
            if np.sum(n) > 0:
                eps = repair.l1(mat_A_i, vec_b_i, cs_i)
                cs_scenario_decayT_arbfree[i_scenario] = cs_i + eps
            else:
                cs_scenario_decayT_arbfree[i_scenario] = cs_i
        else:
            cs_scenario_decayT_arbfree[i_scenario] = cs_i

    return xi_scenario, cs_scenario_decayT, cs_scenario_decayT_arbfree


def get_ts_pnl_var(securityid, mpor, confidence,
                   cs_ts_test, S_ts_test, ks, dir_var_scenario):
    """
    Compute time series of realised PnLs and concurrent VaRs (long and short)
    for a list of trading strategies (option portfolios).

    """

    idxs_delta = [0, 2, 4, 6, 8, 10, 12]  # deltas for defining strategies
    n_exp = 10
    n_mny = 13

    n_delta = len(idxs_delta)
    n_test = cs_ts_test.shape[0] - mpor

    # placeholders for outrights
    idxs_or = []
    for i_exp in range(n_exp):
        for i in idxs_delta:
            idxs_or.append(i + i_exp * n_mny)
    n_or = len(idxs_delta) * n_exp
    or_ts_var_long = np.zeros((n_or, n_test))
    or_ts_var_short = np.zeros((n_or, n_test))
    or_ts_pnl = np.zeros((n_or, n_test))

    # placeholders for delta spreads
    ls_ds_pair1 = list(itertools.combinations(range(n_delta), 2))
    ls_ds_pair = []
    for i_exp in range(n_exp):
        for idx1, idx2 in ls_ds_pair1:
            ls_ds_pair.append(
                (idx1 + i_exp * n_delta, idx2 + i_exp * n_delta))
    n_ds = len(ls_ds_pair)
    ds_ts_var_long = np.zeros((n_ds, n_test))
    ds_ts_var_short = np.zeros((n_ds, n_test))
    ds_ts_pnl = np.zeros((n_ds, n_test))

    # placeholders for delta butterflies
    n_bf_per_exp = n_delta // 2
    n_bf = n_bf_per_exp * n_exp
    ls_bf_triplet1 = [(i, n_bf_per_exp, n_delta - i - 1) for i in
                      range(n_bf_per_exp)]
    ls_bf_triplet = []
    for i_exp in range(n_exp):
        for idx1, idx2, idx3 in ls_bf_triplet1:
            ls_bf_triplet.append((
                                 idx1 + i_exp * n_delta, idx2 + i_exp * n_delta,
                                 idx3 + i_exp * n_delta))
    bf_ts_var_long = np.zeros((n_bf, n_test))
    bf_ts_var_short = np.zeros((n_bf, n_test))
    bf_ts_pnl = np.zeros((n_bf, n_test))

    # placeholder for delta-hedged options
    n_dh = n_exp
    ls_dh = [n_delta // 2 + i * n_delta for i in range(n_dh)]
    dh_ts_var_long = np.zeros((n_dh, n_test))
    dh_ts_var_short = np.zeros((n_dh, n_test))
    dh_ts_pnl = np.zeros((n_dh, n_test))

    # placeholders for delta-neutral strangles
    n_dns_per_exp = n_delta // 2
    n_dns = n_dns_per_exp * n_exp
    ls_dns_pair1 = [(i, n_delta - i - 1) for i in range(n_dns_per_exp)]
    ls_dns_pair = []
    for i_exp in range(n_exp):
        for idx1, idx2 in ls_dns_pair1:
            ls_dns_pair.append((idx1 + i_exp * n_delta, idx2 + i_exp * n_delta))
    dns_ts_var_long = np.zeros((n_dns, n_test))
    dns_ts_var_short = np.zeros((n_dns, n_test))
    dns_ts_pnl = np.zeros((n_dns, n_test))

    # placeholders for risk reversals
    n_rr_per_exp = n_delta // 2
    n_rr = n_rr_per_exp * n_exp
    ls_rr_pair1 = [(i, n_delta - i - 1) for i in range(n_rr_per_exp)]
    ls_rr_pair = []
    for i_exp in range(n_exp):
        for idx1, idx2 in ls_rr_pair1:
            ls_rr_pair.append((idx1 + i_exp * n_delta, idx2 + i_exp * n_delta))
    rr_ts_var_long = np.zeros((n_rr, n_test))
    rr_ts_var_short = np.zeros((n_rr, n_test))
    rr_ts_pnl = np.zeros((n_rr, n_test))

    # placeholders for calendar spreads
    ls_cs_pair1 = list(itertools.combinations(range(n_exp), 2))
    ls_cs_pair = [(n_delta // 2 + i * n_delta, n_delta // 2 + j * n_delta) for
                  i, j in ls_cs_pair1]
    n_cs = len(ls_cs_pair)
    cs_ts_var_long = np.zeros((n_cs, n_test))
    cs_ts_var_short = np.zeros((n_cs, n_test))
    cs_ts_pnl = np.zeros((n_cs, n_test))

    # placeholders for VIX
    k_arr = ks[idxs_delta]
    vix_ts_var_long = np.zeros(n_test)
    vix_ts_var_short = np.zeros(n_test)
    vix_ts_pnl = np.zeros(n_test)

    # placeholders for S
    S_ts_var_long = np.zeros(n_test)
    S_ts_var_short = np.zeros(n_test)
    S_ts_pnl = np.zeros(n_test)

    for t in tqdm(range(n_test)):

        # load scenario prices
        fname_scenarios = f'{dir_var_scenario}normc_scenarios_t{t}_{securityid}.csv'
        fname_S_scenarios = f'{dir_var_scenario}S_scenarios_t{t}_{securityid}.csv'
        try:
            S_scenario = pd.read_csv(fname_S_scenarios).values
        except:
            print(f'Missing file {fname_S_scenarios}.')
            continue
        try:
            cs_scenario = pd.read_csv(fname_scenarios).values
        except:
            print(f'Missing file {fname_scenarios}.')
            continue

        Cs_scenario = cs_scenario * S_scenario
        or_Cs_scenarios = Cs_scenario[:, idxs_or]

        # load base and realised prices
        fname_base = f'{dir_var_scenario}outrights_base_t{t}_{securityid}.csv'
        df1 = pd.read_csv(fname_base)
        or_Cs_base = df1['base'].values[idxs_or]
        or_Cs_realised = df1['realised'].values[idxs_or]

        S_base = S_ts_test[t]
        S_realised = S_ts_test[t + mpor]

        # caluclating VaR for outright strategies
        or_pnl_scenario = or_Cs_scenarios - or_Cs_base[None, :]
        or_pnl_realised = or_Cs_realised - or_Cs_base

        var_long, var_short = Finance.calc_var(or_pnl_scenario, confidence)

        or_ts_var_long[:, t] = var_long
        or_ts_var_short[:, t] = var_short
        or_ts_pnl[:, t] = or_pnl_realised

        # caluclating VaR for delta spread strategies
        n_scenario = or_Cs_scenarios.shape[0]
        ds_Cs_scenario = np.zeros((n_scenario, n_ds))
        ds_Cs_base = np.zeros(n_ds)
        ds_Cs_realised = np.zeros(n_ds)

        for i_dspread in range(n_ds):
            idx1, idx2 = ls_ds_pair[i_dspread]
            ds_Cs_scenario[:, i_dspread] = \
                or_Cs_scenarios[:,idx1] - or_Cs_scenarios[:, idx2]
            ds_Cs_base[i_dspread] = or_Cs_base[idx1] - or_Cs_base[idx2]
            ds_Cs_realised[i_dspread] = \
                or_Cs_realised[idx1] - or_Cs_realised[idx2]

        ds_pnl_scenario = ds_Cs_scenario - ds_Cs_base[None, :]
        ds_pnl_realised = ds_Cs_realised - ds_Cs_base

        var_long, var_short = Finance.calc_var(ds_pnl_scenario, confidence)

        ds_ts_var_long[:, t] = var_long
        ds_ts_var_short[:, t] = var_short
        ds_ts_pnl[:, t] = ds_pnl_realised

        # caluclate VaR for delta butterfly strategies
        bf_Cs_scenario = np.zeros((n_scenario, n_bf))
        bf_Cs_base = np.zeros(n_bf)
        bf_Cs_realised = np.zeros(n_bf)

        for i_bf in range(n_bf):
            idx1, idx2, idx3 = ls_bf_triplet[i_bf]
            bf_Cs_scenario[:, i_bf] = \
                or_Cs_scenarios[:, idx1] - 2 * or_Cs_scenarios[:, idx2] + \
                or_Cs_scenarios[:, idx3]
            bf_Cs_base[i_bf] = or_Cs_base[idx1] - 2 * or_Cs_base[idx2] + \
                               or_Cs_base[idx3]
            bf_Cs_realised[i_bf] = or_Cs_realised[idx1] - 2 * or_Cs_realised[
                idx2] + or_Cs_realised[idx3]

        bf_pnl_scenario = bf_Cs_scenario - bf_Cs_base[None, :]
        bf_pnl_realised = bf_Cs_realised - bf_Cs_base

        var_long, var_short = Finance.calc_var(bf_pnl_scenario, confidence)

        bf_ts_var_long[:, t] = var_long
        bf_ts_var_short[:, t] = var_short
        bf_ts_pnl[:, t] = bf_pnl_realised

        # caluclate VaR for delta-hedged strategies
        dh_Cs_scenario = np.zeros((n_scenario, n_dh))
        dh_Cs_base = np.zeros(n_dh)
        dh_Cs_realised = np.zeros(n_dh)

        for i_dh in range(n_dh):
            idx = ls_dh[i_dh]
            dh_Cs_scenario[:, i_dh] = \
                or_Cs_scenarios[:, idx] - 0.5 * S_scenario[:, 0]
            dh_Cs_base[i_dh] = or_Cs_base[idx] - 0.5 * S_base
            dh_Cs_realised[i_dh] = or_Cs_realised[idx] - 0.5 * S_realised

        dh_pnl_scenario = dh_Cs_scenario - dh_Cs_base[None, :]
        dh_pnl_realised = dh_Cs_realised - dh_Cs_base

        var_long, var_short = Finance.calc_var(dh_pnl_scenario, confidence)

        dh_ts_var_long[:, t] = var_long
        dh_ts_var_short[:, t] = var_short
        dh_ts_pnl[:, t] = dh_pnl_realised

        # calculate VaR for delta-neutral strangles
        dns_Cs_scenario = np.zeros((n_scenario, n_dns))
        dns_Cs_base = np.zeros(n_dns)
        dns_Cs_realised = np.zeros(n_dns)

        for i_dns in range(n_dns):
            idx1, idx2 = ls_dns_pair[i_dns]
            dns_Cs_scenario[:, i_dns] = \
                or_Cs_scenarios[:, idx1] + or_Cs_scenarios[:, idx2] - \
                S_scenario[:, 0]
            dns_Cs_base[i_dns] = or_Cs_base[idx1] + or_Cs_base[idx2] - S_base
            dns_Cs_realised[i_dns] = or_Cs_realised[idx1] + or_Cs_realised[
                idx2] - S_realised

        dns_pnl_scenario = dns_Cs_scenario - dns_Cs_base[None, :]
        dns_pnl_realised = dns_Cs_realised - dns_Cs_base

        var_long, var_short = Finance.calc_var(dns_pnl_scenario, confidence)

        dns_ts_var_long[:, t] = var_long
        dns_ts_var_short[:, t] = var_short
        dns_ts_pnl[:, t] = dns_pnl_realised

        # calculate VaR for risk reversals
        rr_Cs_scenario = np.zeros((n_scenario, n_rr))
        rr_Cs_base = np.zeros(n_rr)
        rr_Cs_realised = np.zeros(n_rr)

        for i_rr in range(n_rr):
            idx1, idx2 = ls_rr_pair[i_rr]
            rr_Cs_scenario[:, i_rr] = \
                or_Cs_scenarios[:, idx1] - or_Cs_scenarios[:, idx2] + \
                S_scenario[:, 0]
            rr_Cs_base[i_rr] = or_Cs_base[idx1] - or_Cs_base[idx2] + S_base
            rr_Cs_realised[i_rr] = or_Cs_realised[idx1] - or_Cs_realised[
                idx2] + S_realised

        rr_pnl_scenario = rr_Cs_scenario - rr_Cs_base[None, :]
        rr_pnl_realised = rr_Cs_realised - rr_Cs_base

        var_long, var_short = Finance.calc_var(rr_pnl_scenario, confidence)

        rr_ts_var_long[:, t] = var_long
        rr_ts_var_short[:, t] = var_short
        rr_ts_pnl[:, t] = rr_pnl_realised

        # calculate VaR for calendar spreads
        cs_Cs_scenario = np.zeros((n_scenario, n_cs))
        cs_Cs_base = np.zeros(n_cs)
        cs_Cs_realised = np.zeros(n_cs)

        for i_cs in range(n_cs):
            idx1, idx2 = ls_cs_pair[i_cs]
            cs_Cs_scenario[:, i_cs] = or_Cs_scenarios[:,
                                      idx1] - or_Cs_scenarios[:, idx2]
            cs_Cs_base[i_cs] = or_Cs_base[idx1] - or_Cs_base[idx2]
            cs_Cs_realised[i_cs] = or_Cs_realised[idx1] - or_Cs_realised[idx2]

        cs_pnl_scenario = cs_Cs_scenario - cs_Cs_base[None, :]
        cs_pnl_realised = cs_Cs_realised - cs_Cs_base

        var_long, var_short = Finance.calc_var(cs_pnl_scenario, confidence)

        cs_ts_var_long[:, t] = var_long
        cs_ts_var_short[:, t] = var_short
        cs_ts_pnl[:, t] = cs_pnl_realised

        # calculate VaR for VIX
        vix_scenario = Finance.calc_vix(
            k_arr, S_scenario, or_Cs_scenarios[:, :n_delta], normc=False)
        vix_base = Finance.calc_vix(
            k_arr, S_base, or_Cs_base[:n_delta], normc=False)
        vix_realised = Finance.calc_vix(
            k_arr, S_realised, or_Cs_realised[:n_delta], normc=False)

        vix_pnl_scenario = vix_scenario - vix_base
        vix_pnl_realised = vix_realised - vix_base

        var_long, var_short = Finance.calc_var(vix_pnl_scenario, confidence)

        vix_ts_var_long[t] = var_long
        vix_ts_var_short[t] = var_short
        vix_ts_pnl[t] = vix_pnl_realised

        # calculate VaR for S
        S_pnl_scenario = S_scenario - S_base
        S_pnl_realised = S_realised - S_base

        var_long, var_short = Finance.calc_var(S_pnl_scenario, confidence)

        S_ts_var_long[t] = var_long
        S_ts_var_short[t] = var_short
        S_ts_pnl[t] = S_pnl_realised

    ls_portfolioname = ['Ourtight', 'Delta spread', 'Butterfly spread',
                        'Delta-hedged option', 'Delta-neutral straggle',
                        'Risk reversal', 'Calendar spread', 'VIX', 'S']
    ls_portcode = ['or', 'ds', 'bf', 'dh', 'dns', 'rr', 'cs', 'vix', 'S']
    ls_ts_pnl, ls_ts_var_long, ls_ts_var_short = [[], [], []]
    for pc in ls_portcode:
        ls_ts_pnl.append(eval(f'{pc}_ts_pnl'))
        ls_ts_var_long.append(eval(f'{pc}_ts_var_long'))
        ls_ts_var_short.append(eval(f'{pc}_ts_var_short'))

    return ls_portfolioname, ls_ts_pnl, ls_ts_var_long, ls_ts_var_short


def gen_backtest_stats(confidence, ls_ts_pnl, ls_ts_var_long, ls_ts_var_short):
    """
    Compute VaR backtesting statistics:
    - coverage
    - independence
    - procyclicality (trough-to-peak ratio)

    """

    n_test = ls_ts_pnl[0].shape[1]

    # coverage statistics
    ls_n_breach_short = []
    ls_n_breach_long = []

    for i in range(len(ls_ts_pnl)):
        ts_pnl = ls_ts_pnl[i]
        ts_var_long = ls_ts_var_long[i]
        ts_var_short = ls_ts_var_short[i]
        if ts_pnl.ndim == 1:
            ls_n_breach_short.append(np.sum(ts_pnl > ts_var_short))
            ls_n_breach_long.append(np.sum(ts_pnl < ts_var_long))
        else:
            ls_n_breach_short.append(np.sum(ts_pnl > ts_var_short, axis=1))
            ls_n_breach_long.append(np.sum(ts_pnl < ts_var_long, axis=1))

    n_breach_long = np.hstack(ls_n_breach_long)
    n_breach_short = np.hstack(ls_n_breach_short)

    coverage_long = 1 - n_breach_long / n_test
    coverage_short = 1 - n_breach_short / n_test

    coverage = np.hstack((coverage_short, coverage_long))

    # independence statistics
    def lr_bt(hits, alpha):
        """Likelihood ratio framework of Christoffersen (1998)"""
        tr = hits[1:] - hits[:-1]  # Sequence to find transitions

        # Transitions: nij denotes state i is followed by state j nij times
        n01, n10 = (tr == 1).sum(), (tr == -1).sum()
        n11, n00 = (hits[1:][tr == 0] == 1).sum(), (
                    hits[1:][tr == 0] == 0).sum()

        # Times in the states
        n0, n1 = n01 + n00, n10 + n11
        n = n0 + n1

        # Probabilities of the transitions from one state to another
        p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
        p = n1 / n

        if n1 > 0:
            # Unconditional Coverage
            uc_h0 = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
            uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
            uc = -2 * (uc_h0 - uc_h1)

            # Independence
            ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
            ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(
                1 - p11)
            if p11 > 0:
                ind_h1 += n11 * np.log(p11)
            ind = -2 * (ind_h0 - ind_h1)

            # Conditional coverage
            cc = uc + ind

            pval_cci = 1 - sp.stats.chi2.cdf(ind, 1)
            pval_cc = 1 - sp.stats.chi2.cdf(cc, 2)

        else:

            pval_cci = 1
            pval_cc = 1

        return pval_cci, pval_cc

    ls_pval_cci_long = []  # Christofersen's test
    ls_pval_cci_short = []

    ls_pval_cc_long = []  # Conditional coverage mixed test
    ls_pval_cc_short = []

    for i in range(len(ls_ts_pnl)):
        ts_pnl = ls_ts_pnl[i]
        ts_var_long = ls_ts_var_long[i]
        ts_var_short = ls_ts_var_short[i]

        mat_breach_short = (ts_pnl > ts_var_short) * 1
        mat_breach_long = (ts_pnl < ts_var_long) * 1

        if  ts_pnl.ndim == 1:
            pval_cci_short, pval_cc_short = lr_bt(mat_breach_short, 1 - confidence)
            pval_cci_long, pval_cc_long = lr_bt(mat_breach_long, 1 - confidence)

            ls_pval_cci_short.append(pval_cci_short)
            ls_pval_cci_long.append(pval_cci_long)
            ls_pval_cc_short.append(pval_cc_short)
            ls_pval_cc_long.append(pval_cc_long)
        else:
            for j in range(ls_ts_pnl[i].shape[0]):
                pval_cci_short, pval_cc_short = lr_bt(mat_breach_short[j], 1 - confidence)
                pval_cci_long, pval_cc_long = lr_bt(mat_breach_long[j], 1 - confidence)

                ls_pval_cci_short.append(pval_cci_short)
                ls_pval_cci_long.append(pval_cci_long)
                ls_pval_cc_short.append(pval_cc_short)
                ls_pval_cc_long.append(pval_cc_long)

    pval_cci = np.hstack((ls_pval_cci_short, ls_pval_cci_long))
    pval_cc = np.hstack((ls_pval_cc_short, ls_pval_cc_long))

    # procyclicality stats
    ls_ttp_long = []
    ls_ttp_short = []

    for i in range(len(ls_ts_pnl)):
        ts_pnl = ls_ts_pnl[i]
        ts_var_long = ls_ts_var_long[i]
        ts_var_short = ls_ts_var_short[i]

        if ts_pnl.ndim == 1:
            ls_ttp_short.append(np.min(ts_var_short) / np.max(ts_var_short))
            ls_ttp_long.append(np.max(ts_var_long) / np.min(ts_var_long))
        else:
            ls_ttp_short.append(
                np.min(ts_var_short, axis=1) / np.max(ts_var_short, axis=1))
            ls_ttp_long.append(
                np.max(ts_var_long, axis=1) / np.min(ts_var_long, axis=1))

    ttp_short = np.hstack(ls_ttp_short)
    ttp_long = np.hstack(ls_ttp_long)

    ttp = np.hstack((ttp_short, ttp_long))

    return coverage, pval_cci, pval_cc, ttp