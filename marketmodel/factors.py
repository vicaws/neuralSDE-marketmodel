"""
Decode factors and pre-calculate related variables for preparing training data
for estimations of nerual-SDE models.
"""

# Copyright 2021 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import numpy as np
import pybobyqa
import pypoman

from scipy import interpolate
from sklearn.decomposition import PCA
from tqdm import tqdm
from cvxopt import matrix, solvers


class PrepTrainData(object):
    """
    Library of methods for preparing training data for neural network models.
    """

    @staticmethod
    def prep_data_model_S_initial(St, cs_ts, max_PC: int, factor_multiplier=1):
        """
        Prepare training data for the initial neural-SDE model of S, the
        underlying stock price.

        Parameters
        ----------
        St: numpy.array, 1D, shape = (L+1, )
            Time series of underlying stock price. L is a positive integer.

        cs_ts: numpy.array, 2D, shape = (L+1, N)
            Time series of normalised call price surfaces. N is the number of
            options.

        max_PC: int
            The maximal amount of principal components used for representing
            the option prices.

        factor_multiplier: optional, float
            A positive constant scaler applied to the factor data to improve
            numerical stability of the NN training process.

        Returns
        -------
        X_S: numpy.array, 2D, shape = (L, max_PC+1)
            Feature data for the neural-SDE model for S, which could be viewed
            as a supervised learning model.

        Y_S: numpy.array, 2D, shape = (L, 2)
            Label data for the neural-SDE model for S.

        """

        # extract factors using PCA
        pca = PCA(n_components=max_PC)
        pca.fit(cs_ts)
        PCs_full = pca.transform(cs_ts)
        xi_S = PCs_full[:-1, :]

        # process S
        St = np.array(St)
        S = St[:-1]
        dS = St[1:] - St[:-1]

        # combine
        X_S = np.hstack((S[:, None] / factor_multiplier, xi_S))
        Y_S = np.vstack((dS, S)).T / factor_multiplier

        return X_S, Y_S

    @staticmethod
    def prep_data_model_S(S, dS, X, factor_multiplier=1):
        """
        Prepare training data for the neural-SDE model of S, the underlying
        stock price.

        Parameters
        ----------
        S, dS: numpy.array, 1D, shape = (n_time, )
            Time series of the underlying stock price S and its first order
            difference.

        X: numpy.array, 2D, shape = (n_time, n_factor)
            Decoded factor data.

        factor_multiplier: optional, float
            A positive constant scaler applied to the factor data to improve
            numerical stability of the NN training process.

        Returns
        -------
        X_S: numpy.array, 2D, shape = (n_time, n_factor+1)
            Feature data for the neural-SDE model for S, which could be viewed
            as a supervised learning model.

        Y_S: numpy.array, 2D, shape = (n_time, 2)
            Label data for the neural-SDE model for S.

        """

        X_S = np.hstack((S[:, None] / factor_multiplier, X))
        Y_S = np.vstack((dS, S)).T / factor_multiplier
        return X_S, Y_S

    @staticmethod
    def prepare_data_model_xi(S, X,
                              proj_dX, Omegas, det_Omega,
                              corr_dirs, epsmu, mu_base, z_ts,
                              factor_multiplier=1):
        """
        Prepare training data for the neural-SDE model of S, the underlying
        stock price.

        Parameters
        ----------
        S: numpy.array, 1D, shape = (n_time, )
            Time series of the underlying stock price S.

        X: numpy.array, 2D, shape = (n_time, n_factor)
            Decoded factor data.

        proj_dX, Omegas, det_Omega, corr_dirs, epsmu, mu_base, z_ts:
            Various pre-calculated training data for model of xi.

        factor_multiplier: optional, float
            A positive constant scaler applied to the factor data to improve
            numerical stability of the NN training process.

        Returns
        -------
        X_xi: numpy.array, 2D, shape = (n_time, n_factor+1)
            Feature data for the neural-SDE model for xi, which could be viewed
            as a supervised learning model.

        Y_xi: numpy.array, 2D, shape = (n_time, 2)
            Label data for the neural-SDE model for xi.

        """

        X_xi = np.hstack((S[:, None] / factor_multiplier, X))
        Y_xi = np.hstack((proj_dX, Omegas, det_Omega, corr_dirs, epsmu,
                          mu_base, z_ts))

        return X_xi, Y_xi

    @staticmethod
    def calc_call_derivatives(list_mny, list_exp, cs_ts_raw,
                              mask_qualify_value):
        """
        Calculate partial derivatives of call price surfaces.
        *Note: currently this method only supports option prices over a grid of
        parameters.

        Parameters
        __________
        list_mny: numpy.array, 1D, shape = (n_mny, )
            List of moneynesses (relative moneyness).

        list_exp: numpy.array, 1D, shape = (n_expiry, )
            List of time-to-expiries (number of days).

        cs_ts_raw: numpy.array, 2D, shape = (L+1, N)
            Time series of normalised call price surfaces.
            Here N = n_mny x n_exp.

        mask_quality_value: numpy.array, 1D, shape = (N, )
            Boolean logical mask of qualified values.

        Returns
        _______
        cT_ts: numpy.array, 2D, shape = (L, n_opt)
            The partial derivative dc/dT for call options on the grid over time.

        cm_ts: numpy.array, 2D, shape = (L, n_opt)
            The partial derivative dc/dm for call options on the grid over time.

        cmm_ts: numpy.array, 2D, shape = (L, n_opt)
            The second order partial derivative d2c/dm2 for call options on the
            grid over time.

        """
        ms_arr = np.log(np.array(list_mny))
        Ts_arr = np.array(list_exp) / 365.

        L = cs_ts_raw.shape[0]-1
        N = np.sum(mask_qualify_value)

        cm_ts = np.zeros((L, N))
        cT_ts = np.zeros((L, N))
        cmm_ts = np.zeros((L, N))
        for idx_date in tqdm(range(L)):

            # construct the interpolator and supply with input data
            ct_grid = cs_ts_raw[idx_date].reshape((len(list_exp), len(list_mny)))
            call_interp = NormedCallInterp(ms_arr, Ts_arr, ct_grid)
            Ts_ext_arr = call_interp.Ts_ext_arr

            # compute first derivatives
            cm_grid, cT_grid = call_interp.dz(ms_arr, Ts_ext_arr)
            # compute second derivatives
            cmm_grid = call_interp.d2z_dm2(ms_arr, Ts_ext_arr)

            # remove unqualified values
            cm_t = (cm_grid[1:]).ravel()[mask_qualify_value]
            cT_t = (cT_grid[1:]).ravel()[mask_qualify_value]
            cmm_t = (cmm_grid[1:]).ravel()[mask_qualify_value]

            cm_ts[idx_date, :] = cm_t
            cT_ts[idx_date, :] = cT_t
            cmm_ts[idx_date, :] = cmm_t

        return cT_ts, cm_ts, cmm_ts

    @staticmethod
    def normalise_dist_drift(rho, rho_star, epsmu_star):
        c = epsmu_star / (np.exp(rho_star) - 1.)
        return c * (np.exp(rho) - 1.)

    @staticmethod
    def normalise_dist_diffusion(rho, dist_multiplier, proj_scale):
        return proj_scale * (1 - 1. / (rho * dist_multiplier + 1))

    @staticmethod
    def calc_diffusion_scaling(W, b, X, dX, dist_multiplier, proj_scale):
        """
        Pre-calculate diffusion shrinking transformation matrix and other
        related data.

        Parameters
        __________
        W: numpy.array, 2D, shape = (n_constraint, n_factor)
            The coefficient matrix of the static arbitrage constraints in terms
            of the factors.

        b: numpy.array, 1D, shape = (n_constraint, )
            The constant vector term of the static arbitrage constraints in
            terms of the factors.

        X: numpy.array, 2D, shape = (n_time, n_factor)
            Decoded factor data.

        dX: numpy.array, 2D, shape = (n_time, n_factor)
            First-order difference of the decoded factor data.

        dist_multiplier, proj_scale: float
            Hyper-parameters that are used to normalise distance between the
            process to the static arbitrage boundaries. The maximal normalised
            distance is proj_scale, and dist_multiplier adjusts the convergence
            rate to zero when distance is dropping to zero.

        Returns
        _______
        Omegas: numpy.array, 2D, shape = (n_time, n_factor x n_factor)
            Diffusion shrinking matrices (flattened) over time.

        det_Omega: numpy.array, 2D, shape = (n_time, 1)
            The determinants of diffusion shrinking matrices over time.

        proj_dX: numpy.array, 2D, shape = (n_time, n_factor)
            The term in the likelihood function of xi that relates the diffusion
            shrinking matrix and dX.

        """

        # normalise boundary coefficients
        norm_W = np.linalg.norm(W, axis=1)
        assert np.max(np.abs(norm_W - 1.)) < 1e-12

        # compute distances
        dist_X = np.abs(W.dot(X.T) - b[:, None]) / \
                 np.linalg.norm(W, axis=1, keepdims=True)

        # compute normalised distances
        epsilons = PrepTrainData.normalise_dist_diffusion(
            dist_X, dist_multiplier, proj_scale)

        n_obs, n_dim = X.shape
        proj_dX = np.zeros((n_obs, n_dim))
        Omegas = np.zeros((n_obs, n_dim * n_dim))
        det_Omega = np.zeros((n_obs, 1))

        for idx_obs in tqdm(range(n_obs)):
            epsilon = epsilons[:, idx_obs]
            idxs_sorted_eps = np.argsort(epsilon)
            idxs_use = idxs_sorted_eps[:n_dim]

            if np.max(epsilon[idxs_use]) < 1e-8:
                raise ValueError('Some data in the sample path is on corners!')
            else:  # if the anchor point is not on the corner
                # compute new bases
                V = np.linalg.qr(W[idxs_use].T)[0].T
                Omega = np.diag(np.sqrt(epsilon[idxs_use])).dot(V)
                Omegas[idx_obs, :] = Omega.flatten()
                det_Omega[idx_obs, 0] = abs(np.linalg.det(Omega))
                try:
                    proj_dX[idx_obs, :] = np.linalg.solve(Omega.T, dX[idx_obs, :])
                except:
                    raise ValueError('Some data is on boundaries!')

        return Omegas, det_Omega, proj_dX

    @staticmethod
    def calc_drift_correction(W, b, X, rho_star, epsmu_star):
        """
        Pre-calculate polytope interior points, drift correction directions and
        correction weights.

        Parameters
        __________
        W: numpy.array, 2D, shape = (n_constraint, n_factor)
            The coefficient matrix of the static arbitrage constraints in terms
            of the factors.

        b: numpy.array, 1D, shape = (n_constraint, )
            The constant vector term of the static arbitrage constraints in
            terms of the factors.

        X: numpy.array, 2D, shape = (n_time, n_factor)
            Decoded factor data.

        rho_star, epsmu_star: float
            Hyper-parameters that are used to normalise distance between the
            process to the static arbitrage boundaries. The normalised distance
            becomes greater than epsmu_star, monotonically from zero, when the
            distance is beyond rho_star.

        Returns
        _______
        X_interior: numpy.array, 2D, shape = (n_constraint, n_factor)
            List of interior points in the polytope defined by Wx >= b.

        corr_dirs: numpy.array, 3D, shape = (n_time, n_constraint, n_factor)
            List of drift correction directions for all samples.

        epsmu: numpy.array, 2D, shape = (n_time, n_constraint)
            List of normalised distances to each of constraint (i.e. boundary of
            the polytope defined by Wx >= b) for all samples.

        """

        # identify all vertices of the polytope defined by Wx >= b using the
        # double description method.
        vertices = pypoman.compute_polytope_vertices(-W, -b)

        # for each boundary, identify an interior point
        n_bdy, n_dim = W.shape
        X_interior = np.zeros((n_bdy, n_dim))
        for k in range(n_bdy):
            wk = W[k, :]
            bk = b[k]

            # identify vertices on the k-th boundary
            vk = []
            for v in vertices:
                if np.abs(v.dot(wk) - bk) < 1e-12:
                    vk.append(v)
            vk = np.array(vk)

            # calculate the mid-point
            mk = np.mean(vk, axis=0)

            # calculate the furthest interior point normal to the boundary
            # intersecting at the mid-point
            W_rmv_k = np.delete(W, k, axis=0)
            b_rmv_k = np.delete(b, k) + rho_star

            coeffs = W_rmv_k.dot(wk.reshape((-1, 1))).T
            values = b_rmv_k - W_rmv_k.dot(mk.reshape((-1, 1))).T
            mask_neg_coeffs = coeffs < 0
            if np.sum(~mask_neg_coeffs) == 0:
                max_lb = 0.
            else:
                max_lb = np.max(
                    values[~mask_neg_coeffs] / coeffs[~mask_neg_coeffs])
            min_ub = np.min(values[mask_neg_coeffs] / coeffs[mask_neg_coeffs])

            if max_lb > min_ub:
                raise ValueError(f'The greatest lower bound {max_lb} is greater'
                                 f' than the smallest upper bound {min_ub}! Try'
                                 f' to use a smaller rho_star parameter.')

            X_interior[k] = mk + min_ub * wk

        # compute distance
        dist_X = np.abs(W.dot(X.T) - b[:, None])/\
                 np.linalg.norm(W, axis=1, keepdims=True)

        # pre-compute correction directions
        n_obs = X.shape[0]
        corr_dirs = (X_interior[None, :, :] - X[:, None, :]).reshape((n_obs,-1))
        epsmu = PrepTrainData.normalise_dist_drift(
            dist_X, rho_star, epsmu_star).T

        return X_interior, corr_dirs, epsmu

    @staticmethod
    def calc_baseline_drift(cT_ts, cm_ts, cmm_ts, model_S, X_S, G, scales_X):

        # calculate zt
        z_ts = PrepTrainData.calc_zt(cT_ts, cm_ts, cmm_ts, model_S, X_S)

        # calculate reversely scaled price basis vector
        g = G[1:, :] * scales_X[:, None] ** 2

        return - z_ts.dot(g.T)

    @staticmethod
    def calc_zt(cT_ts, cm_ts, cmm_ts, model_S, X_S):

        # calculate gamma_t, volatility of S
        Y_pred = model_S.predict(X_S)
        gamma_ts = np.sqrt(np.exp(-Y_pred[:, 0]))

        # calculate z_t
        z_ts = -cT_ts + 0.5*(cmm_ts - cm_ts)*gamma_ts[:, None]**2

        return z_ts


class DecodeFactor(object):
    """
    Library of methods for decoding factors from prices.
    """

    @staticmethod
    def decode_factor_dasa(cs_ts, St,
                           model_S_initial, X_S, cT_ts, cm_ts, cmm_ts,
                           mat_A, vec_b, norm_factor):
        """
        Decode one dynamic arbitrage factor and one static arbitrage factor in
        order. Our paper https://arxiv.org/abs/2105.11053 uses this decoding.

        Parameters
        __________
        cs_ts: numpy.array, 2D, shape = (n_time, n_opt)
            Time series of normalised call price surfaces.

        St: numpy.array, 1D, shape = (n_time, )
            Time series of underlying stock price.

        model_S_initial: tensorflow NN model
            The initial neural SDE model of S.

        X_S: numpy.array, 2D, shape = (n_time, n_feature)
            The feature data used to train the initial neural SDE model of S.

        cT_ts, cm_ts, cmm_ts: numpy.array, 2D, shape = (n_time, n_opt)
            Partial derivatives of the call option prices.

        mat_A: numpy.array, 2D, shape = (n_constraint, n_opt)
            The coefficient matrix of static arbitrage constraints linear
            inequalities.

        vec_b: numpy.array, 1D, shape = (n_constraint, )
            The right-hand-side constant vector term of static arbitrage
            constraints linear inequalities.

        norm_factor: float
            The max-min norm for factors. Factors are scaled to the same
            max-min norm to improve model estimation.

        Returns
        _______
        G_scaled: numpy.array, 2D, shape = (n_factor, n_opt)
            Price basis vector.

        X_scaled: numpy.array, 2D, shape = (n_time, n_factor)
            Decoded factor data, scaled to have the same norm.

        dX_scaled: numpy.array, 2D, shape = (n_time, n_factor)
            First-order difference of the decoded factor data.

        S, dS: numpy.array, 1D, shape = (n_time, )
            Time series of the underlying stock price S and its first order
            difference.

        W: numpy.array, 2D, shape = (n_constraint, n_factor)
            The coefficient matrix of the static arbitrage constraints in terms
            of the factors.

        b: numpy.array, 1D, shape = (n_constraint, )
            The constant vector term of the static arbitrage constraints in
            terms of the factors.

        idxs_remove: numpy.array, 1D
            The list of time indexes of factors to remove due to the presence
            of static arbitrage.

        scales_X: numpy.array, 1D, shape = (n_factor, )
            The list of scaler numbers that are used to scale each factor to
            the same max-min norm.

        """

        # parameters
        n_da_factor = 1  # number of dynamic arbitrage factor
        n_sa_factor = 1  # number of static arbitrage factor
        n_PC = 6         # number of PC constituents for the sa factor
        weights_ = np.array([[0.96461803, -0.25353509, -0.53838957,
                              0.05788529, 0.04651391, -0.01545341]])

        # calculate the constant term G0
        G0 = cs_ts.mean(axis=0)
        res0 = cs_ts - G0[None, :]

        # decode dynamic arbitrage factors
        G_da, xi_da = DecodeFactor.decode_dynarb_factor(
            res0, n_da_factor, model_S_initial, X_S, cT_ts, cm_ts, cmm_ts)
        res1 = res0 - xi_da.dot(G_da)

        # remove redundant static arbitrage constraints
        mask_redundant = DecodeFactor.find_redundant_constraints(-mat_A, -vec_b)
        mat_A_nec = mat_A[~mask_redundant, :]
        vec_b_nec = vec_b[~mask_redundant]

        # decode static arbitrage factors
        rhs0 = vec_b_nec - G0.dot(mat_A_nec.T)
        rhs1 = rhs0[:, None] - mat_A_nec.dot(G_da.reshape((-1, 1))). \
            dot(xi_da.reshape((1, -1)))

        G_sa, xi_sa, weights = DecodeFactor.decode_stcarb_factor(
            res1, n_sa_factor, n_PC, mat_A_nec, rhs1, weights_)

        # combine factors
        G_comb = np.vstack((G0, G_da, G_sa))
        xi_comb = np.hstack((xi_da, xi_sa))

        # orthogonalise factors
        pca_xi = PCA(n_components=2)
        pca_xi.fit(xi_comb)
        H = pca_xi.components_

        xi = pca_xi.transform(xi_comb)
        G = np.vstack((G0, H.dot(G_comb[1:,:])))

        # remove arbitrageable factors
        cs_ts_recnst = G[0] + xi.dot(G[1:, :])
        idxs_ab = np.where(~np.all(
            mat_A_nec.dot(cs_ts_recnst.T) - vec_b_nec[:, None] >= 0, axis=0))[0]
        idxs_ab_prev = idxs_ab - 1
        idxs_remove = np.union1d(idxs_ab, idxs_ab_prev)
        idxs_remove = idxs_remove[idxs_remove != xi.shape[0]-1]

        X = np.delete(xi[:-1, :], idxs_remove, axis=0)
        dX = np.delete(np.diff(xi, axis=0), idxs_remove, axis=0)

        S = np.delete(St[:-1], idxs_remove)
        dS = np.delete(np.diff(np.log(St)), idxs_remove)

        # scale factors to the same max-min norm
        scales_X = norm_factor / (np.max(X, axis=0) - np.min(X, axis=0))
        X_scaled = X * scales_X
        dX_scaled = dX * scales_X
        G_scaled = np.vstack((G0, G[1:, :]/scales_X[:, None]))

        # define static arbitrage constraints in terms of factors
        A_tilde = mat_A_nec.dot(G_scaled.T)
        b_tilde = vec_b_nec - A_tilde[:, 0]
        mask_redundant = DecodeFactor.find_redundant_constraints(
            -A_tilde[:, 1:], -b_tilde)
        W = A_tilde[~mask_redundant, 1:]
        b = b_tilde[~mask_redundant]

        norm_W = np.linalg.norm(W, axis=1)
        W = W / norm_W[:, None]
        b = b / norm_W

        return G_scaled, X_scaled, dX_scaled, S, dS, W, b, idxs_remove, scales_X

    @staticmethod
    def decode_factor_pcasa_om(cs_ts, mat_A, vec_b):
        """
        Decode one PCA factor and one static arbitrage factor in order for the
        EURO STOXX 50 and DAX index option prices collected from OptionMetrix.
        Our paper https://arxiv.org/abs/2205.15991 uses this decoding.

        Parameters
        __________
        cs_ts: numpy.array, 2D, shape = (n_time, n_opt)
            Time series of normalised call price surfaces.

        mat_A: numpy.array, 2D, shape = (n_constraint, n_opt)
            The coefficient matrix of static arbitrage constraints linear
            inequalities.

        vec_b: numpy.array, 1D, shape = (n_constraint, )
            The right-hand-side constant vector term of static arbitrage
            constraints linear inequalities.

        Returns
        _______
        G: numpy.array, 2D, shape = (n_factor, n_opt)
            Price basis vector.

        pca_xi: numpy.array, 2D, shape = (n_time, n_factor)
            Decoded factor data (unscaled).
        """

        # parameters
        n_pca_factor = 1  # number of statistical accuracy factor
        n_sa_factor = 1   # number of static arbitrage factor
        n_PC = 4          # number of PC constituents for the sa factor
        weights_ = np.array([[0.12001689, -0.57931713, 0.43636627, 0.18497003]])

        # calculate the constant term G0
        G0 = cs_ts.mean(axis=0)
        res0 = cs_ts - G0[None, :]

        # decode statistical accuracy factors
        G_pca, xi_pca = DecodeFactor.decode_pca_factor(res0, n_pca_factor)
        res1 = res0 - xi_pca.dot(G_pca)

        # decode static arbitrage factors
        rhs0 = vec_b - G0.dot(mat_A.T)
        rhs1 = rhs0[:, None] - mat_A.dot(G_pca.reshape((-1, 1))). \
            dot(xi_pca.reshape((1, -1)))

        G_sa, xi_sa, weights = DecodeFactor.decode_stcarb_factor(
            res1, n_sa_factor, n_PC, mat_A, rhs1, weights_)

        # combine factors
        Gx = np.vstack((G_pca, G_sa))
        xi = np.hstack((xi_pca, xi_sa))
        G = np.vstack((G0, Gx))

        # orthogonalise factors
        pca_xi = PCA(n_components=2)
        pca_xi.fit(xi)

        return G, pca_xi

    @staticmethod
    def decode_dynarb_factor(residuals, n_da_factor: int,
                             model_S_initial, X_S, cT_ts, cm_ts, cmm_ts):
        """
        Decode dynamic-arbitrage factors from price residual data.

        Parameters
        __________
        residuals: numpy.array, 2D, shape = (n_time, n_opt)
            Call option price residual data.

        n_da_factor: int
            The number of dynamic-arbitrage factors to be decoded.

        model_S_initial: tensorflow NN model
            The initial neural SDE model of S.

        X_S: numpy.array, 2D, shape = (n_time, n_feature)
            The feature data used to train the initial neural SDE model of S.

        cT_ts, cm_ts, cmm_ts: numpy.array, 2D, shape = (n_time, n_opt)
            Partial derivatives of the call option prices.

        Returns
        _______
        G_da: numpy.array, 2D, shape = (n_da_factor, n_opt)
            Price basis vector.

        xi_da: numpy.array, 2D, shape = (n_time, n_da_factor)
            Factor time series data.

        """

        # calculate z_t
        z_ts = PrepTrainData.calc_zt(cT_ts, cm_ts, cmm_ts, model_S_initial, X_S)

        # calculate dynamic-arbitrage factor
        pca_z = PCA(n_components=3)
        pca_z.fit(z_ts)
        v_z = pca_z.components_

        G_da = v_z[:n_da_factor]
        xi_da = residuals.dot(G_da.T)

        return G_da, xi_da

    @staticmethod
    def decode_stcarb_factor(residuals, n_sa_factor: int, n_PC: int,
                             mat_A, rhs, weights_=None):
        """
        Decode static-arbitrage factors from price residual data.

        Parameters
        __________
        residuals: numpy.array, 2D, shape = (n_time, n_opt)
            Call option price residual data.

        n_sa_factor: int
            The number of static-arbitrage factors to be decoded.

        n_PC: int
            The number of principal components of the residual data that are
            used as constituents of the static-arbitrage factor.

        mat_A: numpy.array, 2D, shape = (n_constraint, n_opt)
            The coefficient matrix of static arbitrage constraints linear
            inequalities.

        rhs: numpy.array, 1D, shape = (n_constraint, )
            The right-hand-side constant vector term of static arbitrage
            constraints linear inequalities.

        weights_: optional, numpy.array, 2D, shape = (n_pre_sa_factor, n_PC)
            Pre-calculated weights to speed up factor decoding.

        Returns
        _______
        G_sa: numpy.array, 2D, shape = (n_sa_factor, n_opt)
            Price basis vector.

        xi_sa: numpy.array, 2D, shape = (n_time, n_sa_factor)
            Factor time series data.

        """
        res_i = residuals
        rhs_i = rhs

        G_sa = np.zeros((n_sa_factor, res_i.shape[1]))
        xi_sa = np.zeros((res_i.shape[0], n_sa_factor))

        # check if pre-calculated weights are supplied and correctly formatted
        if weights_ is None:
            n_given_weights = 0
        else:
            if not weights_.shape[1] == n_PC:
                raise ValueError(f'The supplied weights have a wrong dimension.'
                                 f' It should have {n_PC} components.')
            n_given_weights = weights_.shape[0]

        weights = np.zeros((n_sa_factor, n_PC))

        # find static arbitrage iteratively
        for i in range(n_sa_factor):
            # find principal components which are constituents of the static
            # arbitrage factor
            pca_c = PCA(n_components=n_PC)
            pca_c.fit(res_i)
            G_sub = pca_c.components_
            xi_PC = pca_c.transform(res_i)

            # solve the optimisation problem only when no pre-calculated weight
            # is supplied
            if (n_given_weights < n_sa_factor) and i >= n_given_weights:

                def obj(weight):
                    # normalise weight
                    weight /= np.linalg.norm(weight)
                    # construct the weighted direction and score
                    G_i = np.sum(G_sub * weight[:, None], axis=0)
                    xi_i = np.sum(xi_PC * weight[None, :], axis=1)
                    lhs = mat_A.dot(G_i.reshape((-1, 1))).dot(
                        xi_i.reshape((1, -1)))
                    recnst_err = np.linalg.norm(
                        res_i - xi_i[:, None]*G_i[None, :])
                    return -np.sum(np.all(lhs - rhs_i > 0, axis=0)) + recnst_err

                weight0 = np.ones(n_PC) * .1  # initialisation
                lbs = -np.ones(n_PC)          # lower bounds
                ubs = np.ones(n_PC)           # upper bounds
                print('Searching for the static-arbitrage minimisation '
                      'direction for projection.')
                soln = pybobyqa.solve(obj, weight0, bounds=(lbs, ubs),
                                      maxfun=10000, seek_global_minimum=True)
                print(soln)

                weight_opt = soln.x / np.linalg.norm(soln.x)
            else:
                weight = weights_[i, :]
                weight_opt = weight / np.linalg.norm(weight)

            weights[i, :] = weight_opt

            G_sa_i = np.sum(G_sub * weight_opt[:, None], axis=0)
            xi_sa_i = np.sum(xi_PC * weight_opt[None, :], axis=1)

            # update residual and right-hand-side of the arbitrage inequalities
            res_i -= xi_sa_i[:, None] * G_sa_i[None, :]
            rhs_i -= mat_A.dot(G_sa_i.reshape((-1, 1))).dot(
                xi_sa_i.reshape((1, -1)))

            G_sa[i, :] = G_sa_i
            xi_sa[:, i] = xi_sa_i

        return G_sa, xi_sa, weights

    @staticmethod
    def decode_pca_factor(residuals, n_pca_factor: int):
        """
        Decode statistical accuracy factors from price residual data.

        Parameters
        __________
        residuals: numpy.array, 2D, shape = (n_time, n_opt)
            Call option price residual data.

        n_pca_factor: int
            The number of statistical accuracy factors to be decoded.


        Returns
        _______
        G_pca: numpy.array, 2D, shape = (n_pca_factor, n_opt)
            Price basis vector.

        xi_pca: numpy.array, 2D, shape = (n_time, n_pca_factor)
            Factor time series data.

        """
        pca = PCA(n_components=n_pca_factor)
        pca.fit(residuals)

        G_pca = pca.components_
        xi_pca = pca.transform(residuals)

        return G_pca, xi_pca

    @staticmethod
    def append_pca_secondary_factors(X, G, cs_ts_X, n_factor_sec,
                                     norm_factor_sec, mat_A, vec_b):
        from sklearn.decomposition import PCA

        # compute residuals
        G0 = G[0]
        Gx = G[1:]
        cs_ts_rcnst = X.dot(Gx) + G0[None, :]
        res = cs_ts_X - cs_ts_rcnst

        # de-mean residuals
        res_cnst = np.mean(res, axis=0)
        G0_adj = G[0] + res_cnst
        G_adj = np.vstack((G0_adj, Gx))
        res_adj = res - res_cnst[None, :]

        # decompose the residuals using PCA
        pca_res = PCA(n_components=n_factor_sec)
        pca_res.fit(res_adj)
        x = pca_res.transform(res_adj)
        v = pca_res.components_

        scale_x = norm_factor_sec / (np.max(x, axis=0) - np.min(x, axis=0))
        X_post = np.hstack((X, x * scale_x[None, :]))
        G_post = np.vstack((G_adj, v / scale_x[:, None]))

        A_tilde = mat_A.dot(G_post.T)
        b_tilde = vec_b - A_tilde[:, 0]

        W_full = A_tilde[:, 1:]
        b_full = b_tilde

        # normalise these constraints
        norm_W = np.linalg.norm(W_full, axis=1)
        W_full /= norm_W[:, None]
        b_full /= norm_W

        return X_post, G_post, W_full, b_full

    @staticmethod
    def find_redundant_constraints(A, b):
        """
        Find redundant constraints of the linear inequality system Ax <= b.
        The method is called the Linear Programming method.
        See https://www.hindawi.com/journals/mpe/2010/723402/ for more details.

        Returns
        _______
        mask_redundant: numpy.array, 1D, shape = (n_constraint, )
            Boolean logical mask indicating whether a constraint is redundant.
        """
        n_constraint = A.shape[0]
        mask_redundant = np.ones(n_constraint, dtype=bool)

        for i_constraint in tqdm(range(n_constraint)):
            Ai = np.delete(A, i_constraint, axis=0)
            bi = np.delete(b, i_constraint)
            ai= A[i_constraint, :]

            G = matrix(Ai)
            h = matrix(bi)
            c = matrix(-ai)

            try:
                sol = solvers.lp(c, G, h, solver='glpk')
                if np.array(sol['x']).flatten().dot(-c) > b[i_constraint]:
                    mask_redundant[i_constraint] = False
            except:
                mask_redundant[i_constraint] = False
        return mask_redundant

    @staticmethod
    def report_factor_metrics(G0, Gx, X, cs_ts, mat_A, vec_b, z_ts,
                              verbose=False):

        # compute reconstructed prices
        cs_ts_rcst = X.dot(Gx) + G0[None, :]

        # compute residual norm
        res = np.linalg.norm(cs_ts_rcst - cs_ts)

        # compute MAPE
        mape = np.mean(np.abs(cs_ts_rcst - cs_ts)/cs_ts)

        # compute PSAS
        psas = np.sum(np.any(mat_A.dot(cs_ts_rcst.T) -
                             vec_b[:, None]<0, axis=0)) / 10000

        # compute PDA

        g1 = Gx / np.linalg.norm(Gx, axis=1)[:, None]
        z_ts_demean = z_ts - np.mean(z_ts, axis=0)[None,:]
        z_var = z_ts_demean.T.dot(z_ts_demean)
        xx = z_ts_demean.dot(g1.T)
        pda = 1 - np.cumsum(np.diagonal(xx.T.dot(xx)) / np.trace(z_var))

        if verbose:
            print('Residual norm: {:.4f}'.format(res))
            print('MAPE:          {:.2f}%'.format(mape*100))
            print('PSAS:          {:.2f}%'.format(psas*100))
            print('PDA:           {:.2f}%'.format(pda[-1]*100))

        return res, mape, psas, pda


class NormedCallInterp:
    """
    Interpolator object for normalised call option price surfaces.
    """

    def __init__(self, ms_arr, Ts_arr, cs_grid):
        '''
        ms_arr: array of unique log-moneynesses ln(K/F)
        Ts_arr: array of unique time-to-expiry in year fractions
        cs_grid: rectangular grid of normalised call prices
        '''
        ms_arr.sort()  # sorting is necessary for the interpolation to work
        Ts_arr.sort()

        self.ms_arr = ms_arr
        self.Ts_arr = Ts_arr
        self.cs_grid = cs_grid

        # array of moneyness K/F
        self.ks_arr = np.exp(ms_arr)

        # include zero-strike option
        self.Ts_ext_arr = np.hstack((0., Ts_arr))
        self.cs_ext_grid = np.vstack((np.maximum(1-self.ks_arr, 0.), cs_grid))

        # construct an interpolator over (k, T)
        self.interp = interpolate.interp2d(self.ks_arr, self.Ts_ext_arr,
                                           self.cs_ext_grid, kind='cubic')

        # grid
        self.ks_ext_grid, self.Ts_ext_grid = np.meshgrid(self.ks_arr, self.Ts_ext_arr)

        # prepare for RBF interpolation
        ks_ext_flattened = self.ks_ext_grid.ravel()
        Ts_ext_flattened = self.Ts_ext_grid.ravel()
        self.alpha_k = 1./(np.max(ks_ext_flattened) - np.min(ks_ext_flattened))
        self.alpha_T = 1./(np.max(Ts_ext_flattened) - np.min(Ts_ext_flattened))

        # construct an interpolater for dz/dT
        ## compute dz/dT on the anchor points
        dz_dT_anchor = self.interp.__call__(self.ks_arr, self.Ts_ext_arr, dy=1)
        ## correct dz/dT based on static arbitrage constraints
        dz_dT_anchor[dz_dT_anchor < 0.] = 0.
        self.interp_dz_dT = interpolate.Rbf(ks_ext_flattened*self.alpha_k,
                                            Ts_ext_flattened*self.alpha_T,
                                            dz_dT_anchor.ravel(),
                                            function='multiquadric', epsilon=.01)

        # construct an interpolater for dz/dk
        ## compute dz/dk on the anchor points
        dz_dk_anchor = self.interp.__call__(self.ks_arr, self.Ts_ext_arr, dx=1)
        ## correct dz/dk based on static arbitrage constraints
        dz_dk_anchor[dz_dk_anchor > 0.] = 0.
        dz_dk_anchor[0, self.ks_arr<=1] = -1.  # Expirying in-the-money call payoff
        dz_dk_anchor[0, self.ks_arr>1] = 0.    # Expirying out-of-the-money call payoff
        self.interp_dz_dk = interpolate.Rbf(ks_ext_flattened*self.alpha_k,
                                            Ts_ext_flattened*self.alpha_T,
                                            dz_dk_anchor.ravel(),
                                            function='multiquadric', epsilon=.01)

        # construct an interpolater for ddz/dkk
        ## compute ddz/dkk on the anchor points
        ddz_dkk_anchor = self.interp.__call__(self.ks_arr, self.Ts_ext_arr, dx=2)
        ## correct ddz/dkk based on static arbitrage constraints
        ddz_dkk_anchor[ddz_dkk_anchor < 0.] = 0.
        ddz_dkk_anchor[0, 0] = 0.
        ddz_dkk_anchor[0, -1] = 0.
        self.interp_ddz_dkk = interpolate.Rbf(ks_ext_flattened*self.alpha_k,
                                              Ts_ext_flattened*self.alpha_T,
                                              ddz_dkk_anchor.ravel(),
                                              function='multiquadric', epsilon=.01)


    def z(self, in_ms_arr, in_Ts_arr):
        ''' Return interpolated values
        '''
        in_ks_arr = np.exp(in_ms_arr)
        return self.interp(in_ks_arr, in_Ts_arr)

    def dz(self, in_ms_arr, in_Ts_arr):
        ''' Return interpolated first order partial derivatives
        '''
        in_ks_arr = np.exp(in_ms_arr)
        in_ks_grid, in_Ts_grid = np.meshgrid(in_ks_arr, in_Ts_arr)
        in_ks_flattened = in_ks_grid*self.alpha_k
        in_Ts_flattened = in_Ts_grid*self.alpha_T

        # dz/dT
        dz_dT = self.interp_dz_dT(in_ks_flattened, in_Ts_flattened)
        dz_dT = dz_dT.reshape((len(in_Ts_arr), len(in_ks_arr)))
        dz_dT[dz_dT<0.] = 0.

        # dz/dm
        ## first, compute dz/dk
        dz_dk = self.interp_dz_dk(in_ks_flattened, in_Ts_flattened)
        dz_dk = dz_dk.reshape((len(in_Ts_arr), len(in_ks_arr)))
        dz_dk[dz_dk>0.] = 0.
        ## second, compute dz/dm
        dz_dm = dz_dk * in_ks_arr[None, :]

        return dz_dm, dz_dT

    def d2z_dm2(self, in_ms_arr, in_Ts_arr):
        ''' Return interpolated second order partial derivatives
        '''
        in_ks_arr = np.exp(in_ms_arr)
        in_ks_grid, in_Ts_grid = np.meshgrid(in_ks_arr, in_Ts_arr)
        in_ks_flattened = in_ks_grid*self.alpha_k
        in_Ts_flattened = in_Ts_grid*self.alpha_T

        # dz/dk
        dz_dk = self.interp_dz_dk(in_ks_flattened, in_Ts_flattened)
        dz_dk = dz_dk.reshape((len(in_Ts_arr), len(in_ks_arr)))
        dz_dk[dz_dk>0.] = 0.

        # ddz/dkk
        ddz_dkk = self.interp_ddz_dkk(in_ks_flattened, in_Ts_flattened)
        ddz_dkk = ddz_dkk.reshape((len(in_Ts_arr), len(in_ks_arr)))
        ddz_dkk[ddz_dkk<0.] = 0.

        # ddz/dmm
        ddz_dmm = dz_dk * in_ks_arr[None, :] + ddz_dkk * (in_ks_arr**2)[None, :]
        ## correct values at T=0
        if 0. in in_Ts_arr:
            ddz_dmm[in_Ts_arr==0.] = -np.exp(in_ms_arr)
            ddz_dmm[in_Ts_arr==0., in_ms_arr>0] = 0.
        return ddz_dmm
