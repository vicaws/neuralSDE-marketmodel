"""
Utility methods, including:
  - Configurations
  - Library of plot methods
"""

# Copyright 2022 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec


class Config(object):
    """
    Library of hyperparameters.
    """

    hp_sde_transform = {
        'norm_factor': .1,
        'frac_critical_threshold': 0.9,
        'critical_value': 0.95,
        'proj_scale': .1,
        'rho_star': 1e-4,
        'epsmu_star': 10
    }

    hp_model_S = {
        'pruning_sparsity': .5,
        'validation_split': .1,
        'batch_size': 512,
        'epochs': 500
    }

    hp_model_mu = {
        'validation_split': .2,
        'batch_size': 512,
        'epochs': 200
    }

    hp_model_xi = {
        'pruning_sparsity': .5,
        'validation_split': .1,
        'batch_size': 512,
        'epochs': 20000,
        'factor_multiplier': 250,
        'lbd_penalty_eq': 0,
        'lbd_penalty_sz': 0,
    }


class ConfigOm(object):
    """
    Library of hyperparameters (for OptionMetrics data).
    """

    hp_sde_transform = {
        'norm_factor_pri': .1,
        'norm_factor_sec': .05,
        'frac_critical_threshold': 0.9,
        'critical_value': 0.95,
        'proj_scale': 1,
        'rho_star': 1e-4,
        'epsmu_star': 10
    }

    hp_model_lnS = {
        'pruning_sparsity': .5,
        'validation_split': .1,
        'batch_size': 512,
        'epochs': 1000
    }

    hp_model_xi = {
        'pruning_sparsity': .5,
        'validation_split': .1,
        'batch_size': 512,
        'epochs': 20000,
        'factor_multiplier': 250
    }


class PlotLib(object):
    """
    Library of plot methods.
    """

    @staticmethod
    def plot_loss_over_epochs(history, save=False, out_fname=None):

        hist = pd.DataFrame(history.history)

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(hist['loss'], label='Training loss')
        ax.plot(hist['val_loss'], label='Validation loss')
        ax.set_xlabel('Epochs')
        ax.legend()
        plt.grid()
        plt.tight_layout()

        if save:
            plt.savefig(out_fname, dpi=500)
            plt.close()

    @staticmethod
    def plot_in_sample_model_xi(vt, idxs_remove, X, data_cache, mu, sigma_L):

        # exploit the linear relationship between the Heston-SLV instantaneous
        # variance and the first factor
        variances = np.delete(vt[:-1], idxs_remove)
        reg = LinearRegression().fit(X[:, 0].reshape((-1, 1)),
                                     variances.reshape((-1, 1)))

        variances_approx = reg.coef_ * X[:, 0] + reg.intercept_
        variances_approx[variances_approx < 0] = 0
        vols_approx = np.sqrt(variances_approx)

        # heston drift and diffusion
        heston_drift = data_cache.heston_kappa * (
                data_cache.heston_theta - variances_approx) / reg.coef_
        heston_diffusion = data_cache.heston_sigma * vols_approx / reg.coef_

        # plot
        fig = plt.figure(figsize=(8, 2.8))

        ax = fig.add_subplot(121)
        ax.set_title(r'Drift of $\xi_1$')
        ax.scatter(X[:, 0], mu[:, 0], color='k', s=1, alpha=1,
                   label=r'$\phi_\mu^{\xi_1, \theta}(\tilde{\xi})$')
        ax.scatter(X[:, 0], heston_drift,
                   color='r', s=1, alpha=1, label=r'$\mu^{\xi_1}(\xi_1)$')
        ax.set_xlabel(r'$\xi_{1t}$')
        ax.set_ylabel(r'$\mu_t$')
        ax.legend()

        ax = fig.add_subplot(122)
        ax.set_title(r'Diffusion of $\xi_1$')
        ax.scatter(X[:, 0], sigma_L[:, 0, 0], color='k', s=1, alpha=1,
                   label=r'$\phi_\sigma^{\xi_1, \theta}(\tilde{\xi})$')
        ax.scatter(X[:, 0], heston_diffusion,
                   color='r', s=1, alpha=1, label=r'$\sigma^{\xi_1}(\xi_1)$')
        ax.set_xlabel(r'$\xi_{1t}$')
        ax.set_ylabel(r'$\sigma_t$')
        ax.legend()

        plt.tight_layout()

    @staticmethod
    def plot_xi_drift_diffusion(X, mu, sigma_L, n_plot_points, W, b,
                                mu_scale=1.2*1e-4, sigma_scale=2*1e-2,
                                sigma_hw_relative_scale=1./3,
                                idxs_bdy_plot=None):

        if idxs_bdy_plot is None:
            idxs_bdy_plot = [0, 1, 3, 5]

        n_obs = X.shape[0]

        # randomly sample a few data points
        idxs_random = np.random.choice(range(n_obs), n_plot_points)

        xi_rnd = X[idxs_random, :]
        mu_rnd = mu[idxs_random, :]
        sigma_rnd = sigma_L[idxs_random, :, :]

        # create diffusion matrix ellipses
        ells = []
        for i in range(xi_rnd.shape[0]):
            v = sigma_rnd[i]
            evv, ev = np.linalg.eig(v.dot(v.T))

            U1 = np.array((ev[0, 0]*sigma_scale*evv[0]**sigma_hw_relative_scale))
            V1 = np.array((ev[1, 0]*sigma_scale*evv[0]**sigma_hw_relative_scale))
            U2 = np.array((ev[0, 1]*sigma_scale*evv[1]**sigma_hw_relative_scale))
            V2 = np.array((ev[1, 1]*sigma_scale*evv[1]**sigma_hw_relative_scale))

            angle = V1 / U1 * 180 / np.pi
            width = np.sqrt(U1**2 + V1**2)
            height = np.sqrt(U2**2 + V2**2)
            e = Ellipse(xy=(xi_rnd[i, 0], xi_rnd[i, 1]),
                        width=width, height=height, angle=angle)
            ells.append(e)

        # plot
        fig = plt.figure(figsize=(8, 3))

        ax = fig.add_subplot(121)
        ax.set_title('Drift')
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.scatter(xi_rnd[:, 0], xi_rnd[:, 1],
                   facecolors='green', edgecolors='green', s=20, alpha=.5)

        PlotLib.__plot_bdy(W, b, idxs_bdy_plot, ax)

        for i in range(xi_rnd.shape[0]):
            Xx = np.array((xi_rnd[i, 0]))
            Y = np.array((xi_rnd[i, 1]))
            U = np.array((mu_rnd[i, 0]*mu_scale))
            V = np.array((mu_rnd[i, 1]*mu_scale))
            ax.quiver(Xx, Y, U, V, units='xy', width=.0002, scale=.01, alpha=1)

        ax = fig.add_subplot(122)
        ax.set_title('Diffusion')
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.scatter(xi_rnd[:,0], xi_rnd[:,1],
                   facecolors='green', edgecolors='black', s=1, alpha=1)
        PlotLib.__plot_bdy(W, b, idxs_bdy_plot, ax)

        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(.5)

        plt.tight_layout()

    @staticmethod
    def plot_xi_drift_diffusion_om(X, mu, sigma_L, n_plot_points, W, b,
                                   mu_scale=1 * 1e-4, sigma_scale=7 * 1e-2,
                                   sigma_hw_relative_scale=1. / 2,
                                   plot_fname=None):

        n_obs = X.shape[0]

        # randomly sample a few data points
        np.random.seed(9)
        idxs_random = np.random.choice(range(n_obs), n_plot_points)

        xi_rnd = X[idxs_random, :]
        mu_rnd = mu[idxs_random, :]
        sigma_rnd = sigma_L[idxs_random, :, :]

        # create diffusion matrix ellipses
        ells = []
        for i in range(xi_rnd.shape[0]):
            v = sigma_rnd[i]
            evv, ev = np.linalg.eig(v.dot(v.T))

            U1 = np.array(
                (ev[0, 0] * sigma_scale * evv[0] ** sigma_hw_relative_scale))
            V1 = np.array(
                (ev[1, 0] * sigma_scale * evv[0] ** sigma_hw_relative_scale))
            U2 = np.array(
                (ev[0, 1] * sigma_scale * evv[1] ** sigma_hw_relative_scale))
            V2 = np.array(
                (ev[1, 1] * sigma_scale * evv[1] ** sigma_hw_relative_scale))

            angle = V1 / U1 * 180 / np.pi
            width = np.sqrt(U1 ** 2 + V1 ** 2)
            height = np.sqrt(U2 ** 2 + V2 ** 2)
            e = Ellipse(xy=(xi_rnd[i, 0], xi_rnd[i, 1]),
                        width=width, height=height, angle=angle)
            ells.append(e)

        # plot
        fig = plt.figure(figsize=(9, 3))

        ax = fig.add_subplot(121)
        ax.set_title('Drift')
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.scatter(xi_rnd[:, 0], xi_rnd[:, 1],
                   facecolors='green', edgecolors='green', s=20, alpha=.5)

        PlotLib.__plot_bdy_om(W, b, ax)

        for i in range(xi_rnd.shape[0]):
            Xx = np.array((xi_rnd[i, 0]))
            Y = np.array((xi_rnd[i, 1]))
            U = np.array((mu_rnd[i, 0] * mu_scale))
            V = np.array((mu_rnd[i, 1] * mu_scale))
            ax.quiver(Xx, Y, U, V, units='xy', width=.0002, scale=.01, alpha=1)

        ax = fig.add_subplot(122)
        ax.set_title('Diffusion')
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.scatter(xi_rnd[:, 0], xi_rnd[:, 1],
                   facecolors='green', edgecolors='black', s=1, alpha=1)
        PlotLib.__plot_bdy_om(W, b, ax)

        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(.5)

        plt.tight_layout()
        if plot_fname is not None:
            plt.savefig(plot_fname, dpi=500)
            plt.close()

    @staticmethod
    def plot_simulated_xi(st, xit, X, plot_fname):

        fig = plt.figure(figsize=(12, 4))
        gs = GridSpec(3, 2)
        ax = fig.add_subplot(gs[:, 0])
        ax.scatter(xit[0], xit[1], s=.1)
        ax.scatter(X[:, 0], X[:, 1], s=.1, alpha=.2)
        ax = fig.add_subplot(gs[0, 1:])
        ax.plot(xit[0])
        ax = fig.add_subplot(gs[1, 1:])
        ax.plot(xit[1])
        ax = fig.add_subplot(gs[2, 1:])
        ax.plot(st)

        plt.tight_layout()
        plt.savefig(plot_fname, dpi=500)
        plt.close()

    @staticmethod
    def plot_xi12(X, xit):

        mask1 = (X[:, 0] < .05) & (X[:,1] < 0.06)
        mask2 = (xit[0] < .05) & (xit[1] < 0.06)

        df_temp1 = pd.DataFrame(np.vstack((X[mask1,0], X[mask1,1])).T,
                                columns=['xi1', 'xi2'])
        df_temp1['Data'] = 'Heston-SLV'
        df_temp2 = pd.DataFrame(np.vstack((xit[0, mask2], xit[1, mask2])).T,
                                columns=['xi1', 'xi2'])  # change
        df_temp2['Data'] = 'Simulation'
        df_temp = pd.concat((df_temp1, df_temp2), axis=0)
        df_temp.reset_index(inplace=True)

        h = sns.jointplot(data=df_temp, x='xi1', y='xi2', hue='Data',
                          kind='kde', marginal_kws=dict(bw_method=0.4), height=4,
                          levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        h.set_axis_labels(r'$\xi_1$', r'$\xi_2$', fontsize=12)

        plt.tight_layout()

    @staticmethod
    def plot_simulated_paths(St, X, st, xit, W, b):
        np.random.seed(100)
        mask_sample = np.random.choice(range(9951), 1000, replace=False)

        idxs_bdy_plot = [0, 1, 3, 5]

        fig = plt.figure(figsize=(8,3))
        gs = GridSpec(3, 7)
        ax = fig.add_subplot(gs[:, :3])
        ax.set_title(r'Trajectory of $\xi$')
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.scatter(X[mask_sample, 0], X[mask_sample, 1], s=2, alpha=.5, label='Input data')
        ax.scatter(xit[0, mask_sample], xit[1, mask_sample], marker='x',
                   s=2, linewidth=.5, alpha=.8, label='Simulation')
        PlotLib.__plot_bdy(W, b, idxs_bdy_plot, ax=ax)
        ax.legend()

        ax = fig.add_subplot(gs[0, 3:5])
        ax.set_title(r'Simulated $S_t$')
        ax.plot(st, 'k', linewidth=.5)
        ax.set_xticks([])
        ax = fig.add_subplot(gs[1, 3:5])
        ax.set_title(r'Simulated $\xi_{1t}$')
        ax.plot(xit[0], 'k', linewidth=.5)
        ax.set_xticks([])
        ax.set_ylim([-0.04, 0.09])
        ax = fig.add_subplot(gs[2, 3:5])
        ax.set_title(r'Simulated $\xi_{2t}$')
        ax.plot(xit[1], 'k', linewidth=.5)
        ax.set_xlabel(r'$t$')
        ax.set_ylim([-0.04, 0.09])

        ax = fig.add_subplot(gs[0, 5:])
        ax.set_title(r'Heston-SLV $S_t$')
        ax.plot(St, 'k', linewidth=.5)
        ax.set_xticks([])
        ax = fig.add_subplot(gs[1, 5:])
        ax.set_title(r'Heston-SLV $\xi_{1t}$')
        ax.plot(X[:, 0], 'k', linewidth=.5)
        ax.set_xticks([])
        ax.set_ylim([-0.04, 0.09])
        ax = fig.add_subplot(gs[2, 5:])
        ax.set_title(r'Heston-SLV $\xi_{2t}$')
        ax.plot(X[:, 1], 'k', linewidth=.5)
        ax.set_xlabel(r'$t$')
        ax.set_ylim([-0.04, 0.09])

        plt.tight_layout()

    @staticmethod
    def plot_om2_timeseries(
            headers, ls_securityid, ls_dates, ls_S, ls_C, Ts, med_ks):

        fig = plt.figure(figsize=(10, 3.5))
        g = GridSpec(2, 3, fig)
        ax = fig.add_subplot(g[0, :2])
        ax.set_title('Index price')
        for i in range(len(ls_securityid)):
            df = pd.DataFrame(ls_S[i], index=ls_dates[i],
                              columns=[headers[i]])
            df.index = pd.to_datetime(df.index)
            df.plot(ax=ax, linewidth=1)
        ax = fig.add_subplot(g[1, :2])
        ax.set_title('1M ATM call option price (normalised)')
        for i in range(len(ls_securityid)):
            df = pd.DataFrame(ls_C[i][:, 8].T, index=ls_dates[i],
                              columns=[headers[i]])
            df.index = pd.to_datetime(df.index)
            df.plot(ax=ax, linewidth=1)
        ax = fig.add_subplot(g[:, 2])
        ax.scatter(med_ks, Ts, s=3, color='k')
        ax.set_xlabel(r'Relative moneyness $K/F(T)$')
        ax.set_ylabel(r'Time-to-expiries')

        plt.tight_layout()

    @staticmethod
    def plot_om2_factors(X, W, b):
        import pypoman
        from scipy.spatial import ConvexHull

        fig = plt.figure(figsize=(9, 3))
        ax = fig.add_subplot(111)

        # plot scattergram of factors
        ax.scatter(X[:, 0], X[:, 1], color='k', s=.05, alpha=.5)

        # plot arbitrage boundaries
        xs = np.linspace(-0.022, .095, 50)
        for i in range(W.shape[0]):
            wi = W[i]
            bi = b[i]
            ys = bi / wi[1] - wi[0] / wi[1] * xs
            mask = (ys < 0.12) & (ys > -0.085)
            ax.plot(xs[mask], ys[mask], '--r', linewidth=2)
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')

        # plot zoom-in
        axins = ax.inset_axes([0.78, 0.03, 0.21, 0.47])
        data_mask = X[:, 0] < -0.01
        data_mask &= X[:, 1] > -0.01
        axins.scatter(X[data_mask, 0], X[data_mask, 1], color='k', s=.05,
                      alpha=.5)
        xs = np.linspace(-0.021, -0.015, 100)
        for i in range(W.shape[0]):
            wi = W[i]
            bi = b[i]
            ys = bi / wi[1] - wi[0] / wi[1] * xs
            mask = (ys < 0.035) & (ys > -0.005)
            axins.plot(xs[mask], ys[mask], '--r', linewidth=2)
        axins.set_xticks([])
        axins.set_yticks([])
        ax.indicate_inset_zoom(axins)

        # fill color for the no-arbitrage region
        vertices = np.array(pypoman.compute_polytope_vertices(-W, -b))
        hull = ConvexHull(vertices)
        ax.fill(vertices[hull.vertices, 0], vertices[hull.vertices, 1], 'green',
                alpha=0.1)

        plt.tight_layout()

    @staticmethod
    def __plot_bdy(W, b, idxs_bdy_plot, ax):
        xs = np.linspace(-0.032, 0.01, 100)
        for i in idxs_bdy_plot:
            wi = W[i]
            bi = b[i]
            ys = bi/wi[1] - wi[0]/wi[1] * xs
            mask = (ys<0.06) & (ys>-0.03)
            ax.plot(xs[mask], ys[mask], '--r',linewidth=2)

    @staticmethod
    def __plot_bdy_om(W, b, ax):
        xs = np.linspace(-0.022, 0.078, 100)
        for i in range(W.shape[0]):
            wi = W[i]
            bi = b[i]
            ys = bi / wi[1] - wi[0] / wi[1] * xs
            mask = (ys < 0.045) & (ys > -0.024)
            ax.plot(xs[mask], ys[mask], '--r', linewidth=2)


class Finance(object):
    """
    Library of methods for computing various financial terms.
    """

    @staticmethod
    def calc_normcallprice(sigma, T, m):
        a = - m / sigma / np.sqrt(T)
        b = 0.5 * sigma * np.sqrt(T)
        v1 = sp.stats.norm.cdf(a + b)
        v2 = sp.stats.norm.cdf(a - b) * np.exp(m)
        return v1 - v2

    @staticmethod
    def calc_iv(c_tilde, T, m):
        """
        Calculate implied volatility from the normalised call price
        """

        def solve_iv(c_tilde, T, m):
            def obj_func(sigma):
                return Finance.calc_normcallprice(sigma, T, m) - c_tilde

            return sp.optimize.brentq(obj_func, 1e-6, 1.)

        iv = solve_iv(c_tilde, T, m)

        return iv

    @staticmethod
    def calc_vix(k_arr, St, cs_ts, normc=True):
        """
        Calculate the CBOE VIX index.
        """

        # calculate Delta-k arrays
        dk_arr = np.zeros_like(k_arr)
        for i in range(len(dk_arr)):
            if i == 0:
                dk_arr[i] = k_arr[i + 1] - k_arr[i]
            elif i == len(dk_arr) - 1:
                dk_arr[i] = k_arr[i] - k_arr[i - 1]
            else:
                dk_arr[i] = 0.5 * (k_arr[i + 1] - k_arr[i - 1])

        # only use the 30-day option prices
        T1 = 30 / 365.25

        if np.isscalar(St):
            F = St
            K_arr = k_arr * F
            vs = cs_ts.copy()
            if normc:
                vs *= F
            dK_arr = dk_arr * F

            # caluclate OTM option prices using put-call parity
            for i in range(vs.shape[0]):
                if K_arr[i] < F:
                    vs[i] = vs[i] - F + K_arr[i]

            a1 = np.sum(dK_arr / K_arr ** 2 * vs) * 2 / T1
            b1 = 0.01 ** 2 / T1
            sigma1 = np.sqrt(a1 - b1)

            # compute vix
            vix = 100 * sigma1

            return vix

        ls_vix = []
        for t in range(St.shape[0]):
            F = St[t]
            K_arr = k_arr * F
            vs = cs_ts[t, :].copy()
            if normc:
                 vs *= F
            dK_arr = dk_arr * F

            # caluclate OTM option prices using put-call parity
            for i in range(vs.shape[0]):
                if K_arr[i] < F:
                    vs[i] = vs[i] - F + K_arr[i]

            a1 = np.sum(dK_arr / K_arr ** 2 * vs) * 2 / T1
            b1 = 0.01 ** 2 / T1
            sigma1 = np.sqrt(a1 - b1)

            # compute vix
            vix = 100 * sigma1

            ls_vix.append(vix)

        ls_vix = np.array(ls_vix)

        return ls_vix

    @staticmethod
    def calc_var(pnl_scenario, confidence):
        var_long, var_short = np.quantile(
            pnl_scenario, [1 - confidence, confidence], axis=0)
        var_long = np.minimum(var_long, 0)
        var_short = np.maximum(var_short, 0)
        return var_long, var_short

    @staticmethod
    def calc_normc_derivatives_bs(ivs_ts, Ts, ms):
        """
        Calculate partial derivatives of the normalised call prices
        """

        ls_dc_dm_bs = np.zeros_like(ivs_ts)
        ls_d2c_dm2_bs = np.zeros_like(ivs_ts)
        ls_dc_dT_bs = np.zeros_like(ivs_ts)

        for t in range(ivs_ts.shape[0]):
            ivs = ivs_ts[t]

            a = ivs * np.sqrt(Ts)
            b = np.exp(ms)
            d1 = -ms / a + 0.5 * a
            d2 = -ms / a - 0.5 * a

            phi_d1 = sp.stats.norm.pdf(d1)
            phi_d2 = sp.stats.norm.pdf(d2)
            Phi_d2 = sp.stats.norm.cdf(d2)

            dc_dm = -b * Phi_d2
            d2c_dm2 = b / a * phi_d2 + dc_dm

            dd1_dT = (0.5 * ms / a + 0.25 * a) / Ts
            dd2_dT = (0.5 * ms / a - 0.25 * a) / Ts

            dc_dT = phi_d1 * dd1_dT - b * phi_d2 * dd2_dT

            ls_dc_dm_bs[t, :] = dc_dm
            ls_d2c_dm2_bs[t, :] = d2c_dm2
            ls_dc_dT_bs[t, :] = dc_dT

        return ls_dc_dm_bs, ls_d2c_dm2_bs, ls_dc_dT_bs