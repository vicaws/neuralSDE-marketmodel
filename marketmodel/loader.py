"""
Load simulated data from the Heston-SLV model described in the paper. 
"""

# Copyright 2021 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import numpy as np
import pickle

from arbitragerepair import constraints


class DataHestonSlv(object):
    """
    Data object for data generated from Heston-SLV models.
    """
    def __init__(self, list_exp, list_mny, St, vt, r,
                 v0, theta, kappa, sigma, rho,
                 leverage_range_exp, leverage_range_stk, leverage_value,
                 Cs_hestonSLV, ivs_hestonSLV, Cs_heston_SLV_ts):

        # simulated trajectory
        self.list_exp = list_exp
        self.list_mny = list_mny

        self.Cs_heston_SLV_ts = Cs_heston_SLV_ts
        self.St = St
        self.vt = vt
        self.r = r

        # calibrated Heston model and data
        self.heston_v0 = v0
        self.heston_theta = theta
        self.heston_kappa = kappa
        self.heston_sigma = sigma
        self.heston_rho = rho

        # calibrated leverage function
        self.leverage_range_exp = leverage_range_exp
        self.leverage_range_stk = leverage_range_stk
        self.leverage_value = leverage_value

        # Heston SLV initial data
        self.hestonSLV_Cs = Cs_hestonSLV    # call price surface
        self.hestonSLV_ivs = ivs_hestonSLV  # implied vol surface

        # Heston SLV simulated data
        self.Cs_heston_SLV_ts = Cs_heston_SLV_ts


def load_hestonslv_data(fname):
    """
    Load data objects for data generated from Heston-SLV models.

    Parameters
    ----------
    fname: string
        Path of the pickled data object.

    Returns
    -------
    St: numpy.array, 1D, shape = (L+1, )
        Time series of underlying stock price. L is a positive integer.

    vt: numpy.array, 1D, shape = (L+1, )
        Time series of Heston-SLV instantaneous variance.

    list_exp: numpy.array, 1D, shape = (n_expiry, )
        List of time-to-expiries (number of days).

    list_mny: numpy.array, 1D, shape = (n_mny, )
        List of moneynesses (relative moneyness).

    cs_ts_raw: numpy.array, 2D, shape = (L+1, N)
        Time series of normalised call price surfaces.
        N is the number of options and N = n_mny x n_exp

    cs_ts: numpy.array, 2D, shape = (L+1, n_opt)
        Time series of normalised call price surfaces, where small values
        (<= 1e-5) are truncated. N is the number of options.

    mask_quality_value: numpy.array, 1D, shape = (N, )
        Boolean logical mask of qualified values.

    Ts: numpy.array, 1D, shape = (n_opt, )
        List of time-to-expiries corresponding to the n_opt options.

    ks: numpy.array, 1D, shape = (n_opt, )
        List of moneynesses (relative moneyness) corresponding to the n_opt
        options.

    mat_A: numpy.array, 2D, shape = (R, n_opt)
        Coefficient matrix. R is the number of constraints.

    vec_b: numpy.array, 1D, shape = (R, )
        Vector of constant terms. R is the number of constraints.

    """

    # load
    infile = open(fname, 'rb')
    data_cache = pickle.load(infile)
    infile.close()

    # retrieve useful info
    Cs_heston_SLV_ts = data_cache.Cs_heston_SLV_ts

    list_exp = np.array(data_cache.list_exp)
    list_mny = np.array(data_cache.list_mny)
    ks, Ts = np.meshgrid(list_mny, list_exp)
    ks = ks.flatten()
    Ts = Ts.flatten() / 365.

    St = np.array(data_cache.St)  # underlying stock price
    vt = np.array(data_cache.vt)  # instantaneous variance

    # normalise call option prices
    cs_ts_raw = np.array(Cs_heston_SLV_ts) / np.array(St)[:, None]

    # remove options whose prices are too close to zero
    mask_quality_value = np.min(cs_ts_raw, axis=0) > 1e-5
    cs_ts = cs_ts_raw[:, mask_quality_value]
    Ts = Ts[mask_quality_value]
    ks = ks[mask_quality_value]

    # get static arbitrage constraints
    mat_A, vec_b, _, _ = constraints.detect(Ts, ks, cs_ts[0], verbose=False)

    return St, vt, list_exp, list_mny, cs_ts_raw, cs_ts, mask_quality_value, \
           Ts, ks, mat_A, vec_b
