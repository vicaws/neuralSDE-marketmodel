"""
Utility methods, including:
  - Configurations
  - Library of plot methods
"""

# Copyright 2021 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import numpy as np
import pandas as pd
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
    def __plot_bdy(W, b, idxs_bdy_plot, ax):
        xs = np.linspace(-0.032, 0.01, 100)
        for i in idxs_bdy_plot:
            wi = W[i]
            bi = b[i]
            ys = bi/wi[1] - wi[0]/wi[1] * xs
            mask = (ys<0.06) & (ys>-0.03)
            ax.plot(xs[mask], ys[mask], '--r',linewidth=2)
