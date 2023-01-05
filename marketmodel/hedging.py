import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm

from arbitragerepair import repair
from tqdm import tqdm
from scipy import interpolate
from marketmodel import utils
from marketmodel.neuralsde import Simulate, Train


def prep_hedging_data(mpor: int,
                      df_opt, Ts_liq, ms_liq,
                      G_full, cs_weights, W_full, b_full,
                      model_xi, model_S, W, b,
                      dist_multiplier, X_interior, G_raw, pca_xi, scales_X,
                      **kwargs):
    """
    Prepare various data for hedging analysis.

    """

    # construct price basis functions as interpolators of price basis matrices
    interp_Gs, interp_Gs_w, interp_csweight, alpha_T, alpha_m = \
        construct_price_basis(G_full, cs_weights, Ts_liq, ms_liq)
    interp_Gs_raw, _, _, _, _ = \
        construct_price_basis(G_raw, cs_weights, Ts_liq, ms_liq)

    # check whether to use the restrictive structure of NN covariance matrix
    restrictive = kwargs.get('restrictive', False)

    # check whether to compute heston greeks
    heston = kwargs.get('heston', False)
    df_heston = kwargs.get('df_heston', None)
    if heston and (df_heston is not None):
        print('Enable heston greeks.')
        import matlab.engine
        eng = matlab.engine.start_matlab()

    ls_date = df_opt['date'].unique()
    ls_date.sort()
    n_test = len(ls_date) - mpor

    ls_df_opt_date = []
    ls_df_opt_next = []
    ts_mat_vol = np.zeros((n_test, 3, 3))
    ts_mu = np.zeros((n_test, 3))
    ts_vol_S = np.zeros(n_test)
    ts_xi = np.zeros((n_test, 2))
    for t in tqdm(range(n_test)):

        date = ls_date[t]

        mask = df_opt['date'] == date
        df_opt_date = df_opt.loc[mask, :]

        # get spot price
        S = df_opt_date['underlyinglast'].values[0]

        # filter by the pre-defined liquid range
        df_opt_date = filter_liq_range(df_opt_date, Ts_liq, ms_liq, S)
        df_opt_date.drop_duplicates(subset=['strike', 'T_days'], inplace=True)
        df_opt_date.sort_values(by=['T_days', 'strike'], inplace=True)

        # compute option prices from IVs
        cs, Cs = calc_pv(df_opt_date, S)
        df_opt_date['pv'] = Cs

        # compute option deltas
        ks = df_opt_date['strike'].values / S
        Ts = df_opt_date['T_days'].values / 365.
        ms = np.log(ks)
        ivs = df_opt_date['impliedvolatility'].values
        ls_dc_dm_bs, _, _ = utils.Finance.\
            calc_normc_derivatives_bs(ivs.reshape((1, -1)), Ts, ms)
        deltas = cs - ls_dc_dm_bs[0]
        df_opt_date['delta'] = deltas

        # compute option vegas
        vegas = np.zeros_like(cs)
        for i in range(vegas.shape[0]):
            tau = Ts[i]
            iv = ivs[i]
            k = ks[i]
            a = iv * np.sqrt(tau)
            d1 = -np.log(k) / a + 0.5 * a
            vegas[i] = S * sp.stats.norm.pdf(d1) * np.sqrt(tau)
        df_opt_date['vega'] = vegas

        # compute option heston deltas and vegas
        if heston and (df_heston is not None):
            v, theta, kappa, sigma, rho, _, _ = df_heston.iloc[t]
            Ts_days = df_opt_date['T_days'].values
            Ks = df_opt_date['strike'].values
            ds, vs, vlts = eng.optSensByHestonNI(
                0., matlab.double([S]), 0., matlab.double(Ts_days.tolist()),
                'call', matlab.double(Ks.tolist()), matlab.double([v]),
                matlab.double([theta]), matlab.double([kappa]),
                matlab.double([sigma]), matlab.double([rho]),
                'OutSpec', ["delta", "vega", "vegalt"], nargout=3)
            ds = np.array(ds).ravel()
            vs = np.array(vs).ravel()
            vlts = np.array(vlts).ravel()
            df_opt_date['delta_heston'] = ds
            df_opt_date['vega_heston'] = vs
            df_opt_date['vegalt_heston'] = vlts
            df_opt_date['delta_heston_mv'] = ds + vs * rho * sigma / S

        # compute option xi-exposures
        df_opt_date['xi1_sens'] = S * interp_Gs_w[1](ms * alpha_m, Ts * alpha_T)
        df_opt_date['xi2_sens'] = S * interp_Gs_w[2](ms * alpha_m, Ts * alpha_T)
        df_opt_date['xi_sens'] = np.sqrt(df_opt_date['xi1_sens'] ** 2 +
                                         df_opt_date['xi2_sens'] ** 2)

        # compute option nSDE-factor-MV hedge units
        ## get weighted option prices
        cs_weights_traded = interp_csweight(ms * alpha_m, Ts * alpha_T)
        cs_w = cs * cs_weights_traded

        ## get price basis matrices
        n_opt = len(ms)
        n_G = G_full.shape[0]
        G_full_traded = np.zeros((n_G, n_opt))
        for i in range(n_G):
            G_full_traded[i] = interp_Gs[i](ms * alpha_m, Ts * alpha_T)

        G_pri = np.zeros((3, n_opt))
        for i in range(3):
            G_pri[i] = interp_Gs_raw[i](ms * alpha_m, Ts * alpha_T)

        ts_mat_vol[t], ts_vol_S[t], ts_xi[t], ts_mu[t] = calc_nnvol_xiS(
            S, cs_w, G_full_traded, W_full, b_full, model_xi, W, b,
            dist_multiplier, X_interior, G_pri, pca_xi, scales_X,
            restrictive=restrictive)

        # get next period values
        ls_optid_date = df_opt_date['optionid'].unique()

        # get options of the next date
        date_next = ls_date[t + mpor]
        mask = df_opt['date'] == date_next
        df_opt_next = df_opt.loc[mask, :]

        mask_opt = df_opt_next['optionid'].isin(ls_optid_date)
        df_opt_next = df_opt_next[mask_opt]

        # sort
        df_opt_next.sort_values(by=['T_days', 'strike'], inplace=True)

        # compute option prices from IVs
        S_next = df_opt_next['underlyinglast'].values[0]
        _, Cs_next = calc_pv(df_opt_next, S_next)
        df_opt_next['pv'] = Cs_next

        # assign next day value
        df_opt_date.set_index(['optionid'], inplace=True)
        df_opt_next.set_index(['optionid'], inplace=True)
        df_opt_date['pv_next'] = df_opt_next['pv']

        ls_df_opt_date.append(df_opt_date)
        ls_df_opt_next.append(df_opt_next)

    return ls_df_opt_date, ls_df_opt_next, ts_mat_vol, ts_vol_S, ts_xi, ts_mu


def calc_portfolio_risks(ls_df_opt_date, ls_df_opt_next, portfolio):
    """
    Compute risk exposures for various option portfolios.

    """

    n_test = len(ls_df_opt_date)

    # get underlying prices
    ts_S, ts_S_next = np.zeros((2, n_test))
    for t in range(n_test):
        df_opt_date = ls_df_opt_date[t]
        df_opt_next = ls_df_opt_next[t]
        ts_S[t] = df_opt_date['underlyinglast'].values[0]
        ts_S_next[t] = df_opt_next['underlyinglast'].values[0]

    ts_pv, ts_pv_next, ts_delta, ts_delta_mv, ts_vega, ts_xi1_sens, ts_xi1_mv, \
    ts_xi2_sens = np.zeros((8, n_test))

    for t in range(n_test):

        df_opt_date = ls_df_opt_date[t]

        if portfolio == 'naive':

            ts_pv[t], ts_delta[t], ts_delta_mv[t], ts_vega[t], \
            ts_xi1_sens[t], ts_xi1_mv[t], ts_xi2_sens[t], ts_pv_next[t] = \
                (df_opt_date.loc[:, ['pv', 'delta', 'delta_mv', 'vega',
                                     'xi1_sens', 'xi1_mv', 'xi2_sens',
                                     'pv_next']]).sum(axis=0)

        elif portfolio == 'naive_vega':

            vegas = df_opt_date['vega'].values
            ts_pv[t], ts_delta[t], ts_delta_mv[t], ts_vega[t], \
            ts_xi1_sens[t], ts_xi1_mv[t], ts_xi2_sens[t], ts_pv_next[t] = \
                (df_opt_date.loc[:, ['pv', 'delta', 'delta_mv', 'vega',
                                     'xi1_sens', 'xi1_mv', 'xi2_sens',
                                     'pv_next']] / vegas[:, None]).sum(axis=0)

        elif portfolio == 'vix':

            S = ts_S[t]
            Ks = df_opt_date['strike'].values
            Ts = df_opt_date['T_days'].values

            # identify the traded contracts
            tenor = 30
            target_T = Ts[np.argmin(abs(Ts - tenor))]
            mask_T = df_opt_date['T_days'] == target_T
            df = df_opt_date[mask_T]

            # vix value
            arr_weights_c, arr_weights_s, const = calc_vix_coefficients(
                S, tenor / 365., Ks[mask_T])
            vix = np.sqrt(np.sum(
                arr_weights_c * df['pv'].values + arr_weights_s * S) + const)

            S_next = ts_S_next[t]
            vix_next = np.sqrt(np.sum(arr_weights_c * df[
                'pv_next'].values + arr_weights_s * S_next) + const)

            ts_pv[t] = vix
            ts_pv_next[t] = vix_next

            # delta
            dV_dS = np.sum(arr_weights_c * df['delta'].values + arr_weights_s)
            ts_delta[t] = dV_dS / 2 / vix
            ts_delta_mv[t] = np.sum(
                arr_weights_c * df['delta_mv'].values + arr_weights_s) / 2 / vix

            # vega
            ts_vega[t] = np.sum(arr_weights_c * df['vega'].values) / 2 / vix

            # xi_sens
            ts_xi1_sens[t] = np.sum(
                arr_weights_c * df['xi1_sens'].values) / 2 / vix
            ts_xi1_mv[t] = np.sum(
                arr_weights_c * df['xi1_mv'].values) / 2 / vix
            ts_xi2_sens[t] = np.sum(
                arr_weights_c * df['xi2_sens'].values) / 2 / vix

        else:
            raise NotImplementedError()

    return ts_S, ts_S_next, ts_pv, ts_pv_next, ts_delta, ts_delta_mv, ts_vega, \
           ts_xi1_sens, ts_xi2_sens, ts_xi1_mv


def calc_nnvol_xiS(S, cs_w, G_full_traded, W_full, b_full, model_xiS, W, b,
                   dist_multiplier, X_interior, G_pri, pca_xi, scales_X,
                   restrictive=False):
    """ Compute neural-network predicted covariance matrices. """

    # calibrate factors
    n_factor_pri = W.shape[1]
    n_factor_sec = W_full.shape[1] - n_factor_pri
    xi = calibrate_factors(
        cs_w, G_full_traded, W_full, b_full, n_factor_pri, n_factor_sec,
        G_pri, pca_xi, scales_X)
    xi_pri = xi[:n_factor_pri]

    # predict diffusion terms
    X0 = xi_pri
    n_dim = X0.shape[0] + 1
    n_varcov, mask_diagonal = Train._identify_diagonal_entries(n_dim)
    mask_diagonal = mask_diagonal[:-1]  # remove the placeholder for mu_S

    # get drift and diffusion of xi
    x_xi = X0
    gamma_nn = model_xiS.predict(x_xi.reshape(1, -1), verbose=0)[0]
    if restrictive:
        gamma_nn_l = gamma_nn[:2]
        gamma_nn_r = gamma_nn[2:]
        gamma_nn_l = np.hstack((gamma_nn_l, 0))
        gamma_nn = np.hstack((gamma_nn_l, gamma_nn_r))

    gamma_nn[np.array(mask_diagonal).ravel()] = np.exp(
        gamma_nn[np.array(mask_diagonal).ravel()])

    sigma_term = gamma_nn[:n_varcov]
    xc = np.concatenate([sigma_term, sigma_term[n_dim:][::-1]])
    g = np.reshape(xc, [n_dim, n_dim])
    sigma_tilde = np.triu(g, k=0).T

    mu_S = 0.0  # the value does not matter
    mu_tilde = np.hstack((mu_S, gamma_nn[n_varcov:]))

    # scale diffusion and correct drift
    proj_scale = 1
    rho_star = 1e-4
    epsmu_star = 10
    xiS = np.hstack((S, X0))
    W_bar = np.hstack((np.zeros((W.shape[0], 1)), W))
    mu, mat_vol = Simulate.scale_drift_diffusion_xiS(
                    xiS, mu_tilde, sigma_tilde, W, W_bar, b,
                    dist_multiplier, proj_scale,
                    rho_star, epsmu_star, X_interior)

    vol_S = mat_vol[0, 0]

    return mat_vol, vol_S, xi_pri, mu


def calc_pv(df, S):
    """ Calculate option present values from implied volatilities. """

    ks = df['strike'].values / S
    Ts = df['T_days'].values / 365.

    n_opt = len(ks)
    ms = np.log(ks)
    ivs = df['impliedvolatility'].values
    cs = np.zeros(n_opt)

    for i_opt in range(n_opt):
        c = utils.Finance.calc_normcallprice(ivs[i_opt], Ts[i_opt], ms[i_opt])
        cs[i_opt] = c

    Cs = cs * S
    return cs, Cs


def filter_liq_range(df_opt_date, Ts_liq, ms_liq, S):
    """
    Filter the option dataframe by removing options outside the liquid range
    specified by the given list of time-to-maturities and moneynesses.

    """

    from scipy.spatial import Delaunay
    ks_liq = np.exp(ms_liq)
    points = np.vstack((ks_liq, Ts_liq)).T
    hull = Delaunay(points)

    ks = df_opt_date['strike'].values / S
    Ts = df_opt_date['T_days'].values / 365.
    points_kT = np.vstack((ks, Ts)).T
    mask_liquid = hull.find_simplex(points_kT) >= 0

    df_opt_date = df_opt_date[mask_liquid]

    return df_opt_date


def construct_price_basis(G_full, cs_weights, Ts_liq, ms_liq):
    """
    Construct smooth price basis functions from price basis matrices.

    """

    # normalise x and y values
    alpha_m = 1. / (np.max(ms_liq) - np.min(ms_liq))
    alpha_T = 1. / (np.max(Ts_liq) - np.min(Ts_liq))

    n_G = G_full.shape[0]

    # price basis functions (one for each factor)
    interp_Gs = []
    for i in range(n_G):
        interp_Gi = interpolate.Rbf(
            ms_liq * alpha_m, Ts_liq * alpha_T, G_full[i],
            function='multiquadric', epsilon=.001)
        interp_Gs.append(interp_Gi)

    # reversely-scaled price basis functions (one for each factor)
    interp_Gs_w = []
    G_full_w = G_full / cs_weights[None, :]
    for i in range(n_G):
        interp_Gi = interpolate.Rbf(
            ms_liq * alpha_m, Ts_liq * alpha_T, G_full_w[i],
            function='multiquadric', epsilon=.001)
        interp_Gs_w.append(interp_Gi)

    # normalised call price weight interpolators
    interp_csweight = interpolate.Rbf(
        ms_liq * alpha_m, Ts_liq * alpha_T, cs_weights,
        function='multiquadric', epsilon=.001)

    return interp_Gs, interp_Gs_w, interp_csweight, alpha_T, alpha_m


def calibrate_factors(cs_t, G_full, W_full, b_full,
                      n_factor_pri, n_factor_sec, G_pri, pca_xi, scales_X):
    """
    Calibrate primary and secondary factors from normalised call option prices.
    Two-step calibration: first primary factors and then secondary ones.

    """

    # step 1: calibrate primary factors
    y0 = (cs_t - G_pri[0]).reshape((-1, 1))
    x0 = G_pri[1:2].T
    mdl0 = sm.OLS(y0, x0).fit()
    xi_0 = mdl0.params

    y1 = mdl0.resid.reshape((-1, 1))
    x1 = G_pri[2:3].T
    mdl1 = sm.OLS(y1, x1).fit()
    xi_1 = mdl1.params

    xi_pri0 = np.hstack((xi_0, xi_1))
    xi_pri1 = pca_xi.transform(xi_pri0.reshape((1, -1)))
    xi_pri = xi_pri1.ravel() * scales_X

    # step2: calibrate seconary factors
    y2 = mdl1.resid.reshape((-1, 1))
    x2 = G_full[1 + n_factor_pri:].T
    mdl2 = sm.OLS(y2, x2).fit()
    xi_sec = mdl2.params

    # repair arbitrage in the calibrated factors
    xi = np.hstack((xi_pri, xi_sec))
    if np.sum(xi.dot(W_full.T) - b_full < 0) > 0:
        # artificial spread to discourage perturbing primary factors
        spread = np.hstack((np.ones((2, n_factor_pri)) * 1e-4,
                            np.ones((2, n_factor_sec))))

        eps = repair.l1ba(W_full, b_full, xi, spread)
        xi = xi + eps

    return xi


def get_df_hedge(ls_df_opt_date, risk_tuple, T):
    """ Construct the dataframe for the hedge. """

    risk, _, smile = risk_tuple
    n_test = len(ls_df_opt_date)

    df_hedge = pd.DataFrame()
    for t in tqdm(range(n_test)):

        # select the hedging option with the closest maturity
        df_opt_date = ls_df_opt_date[t]
        Ts_day = df_opt_date['T_days'].values
        target_T = Ts_day[np.argmin(np.abs(Ts_day - T))]
        df1 = df_opt_date[df_opt_date['T_days'] == target_T]

        # select the hedging option with the desirable strike
        if smile == 'max':  # maximal risk exposures
            hedge = df1.iloc[np.argmax(abs(df1[risk]))]
        elif smile == 'atm':  # ATM (typically most liquid)
            hedge = df1.iloc[
                np.argmin(np.abs(df1['strike'] / df1['underlyinglast'] - 1))]
        else:
            raise NotImplementedError()

        df_hedge = df_hedge.append(hedge)

    return df_hedge


def calc_pnl_hedge1(df_hedge, risk, ts_risk, ts_delta, pnl_naked, pnl_S):
    """ Compute hedged PnL time series (hedge one risk factor). """

    # get the unit of the specified hedge by eliminating the specified risk
    pnl_hedge = (df_hedge['pv_next'] - df_hedge['pv']).values
    risk_hedge = df_hedge[risk].values
    unit_hedge = ts_risk / risk_hedge

    # get the unit of the underlying by eliminating delta
    delta_hedge = df_hedge['delta'].values
    unit_S = ts_delta - delta_hedge * unit_hedge

    # compute the hedged PnLs
    pnl_h = pnl_naked - pnl_hedge * unit_hedge - pnl_S * unit_S

    return pnl_h


def calc_pnl_hedge2(df_hedge1, df_hedge2, risk1, risk2, ts_risk1, ts_risk2,
                    ts_delta, pnl_naked, pnl_S):
    """ Compute hedged PnL time series (hedge two risk factors). """

    # compute PnLs of the two hedging instruments
    pnl_hedge1 = (df_hedge1['pv_next'] - df_hedge1['pv']).values
    pnl_hedge2 = (df_hedge2['pv_next'] - df_hedge2['pv']).values

    # compute risk exposures of the two hedging instruments
    delta_hedge1 = df_hedge1['delta'].values
    delta_hedge2 = df_hedge2['delta'].values
    risk1_hedge1 = df_hedge1[risk1].values
    risk1_hedge2 = df_hedge2[risk1].values
    risk2_hedge1 = df_hedge1[risk2].values
    risk2_hedge2 = df_hedge2[risk2].values
    ts_risk = np.vstack((ts_delta, ts_risk1, ts_risk2)).T

    # solve for hedge ratios
    n_test = len(ts_delta)
    units_Sh1h2 = np.zeros((n_test, 3))
    for t in range(n_test):
        y = ts_risk[t]
        A = np.array([[1, delta_hedge1[t], delta_hedge2[t]],
                      [0, risk1_hedge1[t], risk1_hedge2[t]],
                      [0, risk2_hedge1[t], risk2_hedge2[t]]])
        x = np.linalg.inv(A).dot(y)
        units_Sh1h2[t] = x

    pnls = np.vstack((pnl_S, pnl_hedge1, pnl_hedge2)).T
    pnl_dxi1xi2h = pnl_naked - np.sum(pnls * units_Sh1h2, axis=1)

    return pnl_dxi1xi2h


def calc_pnl_std_insturmentT(ls_df_opt_date, risk_tuple, ts_delta,
                             pnl_naked, pnl_S, ls_Ts,
                             xi1_mv=False, **kwargs):

    risk, ts_risk, smile = risk_tuple
    ls_std_pnl_h = []

    for T in ls_Ts:
        df_hedge = get_df_hedge(ls_df_opt_date, risk_tuple, T)

        if xi1_mv:
            ts_mvxi1_S = kwargs.get('ts_mvxi1_S')

            # xi1-mv-hedge
            x0 = np.vstack((ts_delta, ts_mvxi1_S, ts_risk)).T
            x0 = np.pad(x0, [[0, 0], [1, 0]])
            x0[:, 0] = 1
            x0 = x0.reshape((-1, 2, 2))

            y0 = np.vstack((ts_delta, ts_risk)).T
            y0 = y0.reshape((-1, 2, 1))

            unit_S, unit_hedge = np.squeeze(np.linalg.solve(x0, y0)).T

            pnl_hedge = (df_hedge['pv_next'] - df_hedge['pv']).values
            pnl_h = pnl_naked - pnl_hedge * unit_hedge - pnl_S * unit_S

        else:
            pnl_h = calc_pnl_hedge1(
                df_hedge, risk, ts_risk, ts_delta, pnl_naked, pnl_S)

        ls_std_pnl_h.append(np.std(pnl_h))

    return np.array(ls_std_pnl_h)


def calc_vix_coefficients(S, T1, k_arr):
    """ Calculate coefficients for option components of the VIX portfolio. """

    # calculate Delta-k arrays
    dk_arr = np.zeros_like(k_arr)
    for i in range(len(dk_arr)):
        if i == 0:
            dk_arr[i] = k_arr[i + 1] - k_arr[i]
        elif i == len(dk_arr) - 1:
            dk_arr[i] = k_arr[i] - k_arr[i - 1]
        else:
            dk_arr[i] = 0.5 * (k_arr[i + 1] - k_arr[i - 1])

    arr_weights_c = dk_arr / k_arr ** 2 * 2 / T1
    mask_itm = k_arr < S
    arr_weights_s = np.zeros_like(arr_weights_c)
    arr_weights_s[mask_itm] = - arr_weights_c[mask_itm]

    const = np.sum(-arr_weights_s * k_arr) - 0.01 ** 2 / T1

    return arr_weights_c, arr_weights_s, const

