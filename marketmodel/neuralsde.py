"""
Construct, train neural-SDE models and simulate trajectories from the learnt
models.
"""

# Copyright 2021 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_model_optimization as tfmot

import marketmodel.utils as utils

from glob import glob
from tqdm import tqdm
from marketmodel.factors import PrepTrainData


class Loss(object):
    """
    Library of loss functions for neural SDE models.
    """

    @staticmethod
    def loss_S(dt):
        """
        Loss function for the neural SDE model of S.

        Parameters
        __________
        dt: float
            Time increment.

        Returns
        _______
        loss: method
            Loss function.

        """
        def loss(y_true, y_pred):
            # extract data
            alpha = y_pred[:, 0]
            beta = y_pred[:, 1]
            dS = y_true[:, 0]
            S = y_true[:, 1]

            # compute drift
            mu = beta * S  # drift term

            # compute log-likelihood
            l = tf.reduce_sum(2*tf.math.log(S)-alpha + tf.square(dS - mu*dt) *
                              tf.exp(alpha) / dt / S**2)
            return l
        return loss

    @staticmethod
    def loss_xi(dt, n_dim, n_varcov, mask_diagonal, W, G,
                lbd_penalty_eq, lbd_penalty_sz):
        """
        Loss function for the neural SDE model of xi.

        """

        def loss(y_true, y_pred):
            # get diffusion terms in the predicted values; in particular,
            # diagonal terms of the diffusion matrix are taken exponentials
            sigma_term = tf.transpose(
                tf.where(tf.constant(mask_diagonal),
                         tf.transpose(tf.exp(y_pred)), tf.transpose(y_pred)))[:,
                         :n_varcov]

            # construct the transposed diffusion matrix
            sigma_tilde_T = tfp.math.fill_triangular(sigma_term, upper=True)

            # get diagonal terms of the diffusion matrix
            sigma_term_diagonal = tf.where(tf.constant(mask_diagonal),
                                           tf.transpose(y_pred), 0.)

            # get drift terms in the predicted values
            mu_residuals = y_pred[:, n_varcov:]

            # get pre-calculated terms from the inputs

            ## regarding diffusion scaling
            proj_dX = y_true[:, :n_dim]
            Omega = tf.reshape(y_true[:, n_dim:n_dim+n_dim**2],
                               shape=[-1, n_dim, n_dim])
            det_Omega = y_true[:, n_dim+n_dim**2:n_dim+n_dim**2+1]
            n1 = n_dim+n_dim**2+1

            ## regarding drift correction
            n_bdy = W.shape[0]
            corr_dirs = tf.reshape(y_true[:, n1:n1+n_dim*n_bdy],
                                   shape=[-1, n_bdy, n_dim])
            epsmu = y_true[:, n1+n_dim*n_bdy:n1+n_dim*n_bdy+n_bdy]
            n2 = n1+n_dim*n_bdy+n_bdy

            ## regarding baseline drift
            mu_base = y_true[:, n2:n2+n_dim]
            n3 = n2+n_dim

            ## regarding MPR penalty
            zed = tf.expand_dims(y_true[:, n3:], axis=-1)

            # compute corrected drifts

            ## compute drift
            mu_term = mu_base * mu_residuals

            ## compute weights assigned to each correction direction
            mu_tilde_inner_W = tf.matmul(
                mu_term, tf.constant(W.T, dtype=tf.float32))
            corr_dir_inner_W = tf.reduce_sum(
                corr_dirs * tf.constant(W, dtype=tf.float32), axis=-1)
            gamma = tf.maximum(-mu_tilde_inner_W - epsmu, 0.) / corr_dir_inner_W

            ## compute corrected drift
            mu_tf = mu_term + tf.reduce_sum(
                tf.expand_dims(gamma, axis=-1) * corr_dirs, axis=1)
            mu_tf = tf.expand_dims(mu_tf, axis=-1)

            # compute log likelihood
            Omega_T = tf.transpose(Omega, perm=[0, 2, 1])
            sigma_tilde = tf.transpose(sigma_tilde_T, perm=[0, 2, 1])

            proj_mu = tf.linalg.solve(Omega_T, mu_tf)
            sol_mu = tf.linalg.triangular_solve(
                sigma_tilde, proj_mu, lower=True)
            sol_mu = tf.squeeze(sol_mu)

            proj_dX_tf = tf.expand_dims(proj_dX, axis=-1)
            sol_dX = tf.linalg.triangular_solve(
                sigma_tilde, proj_dX_tf, lower=True)
            sol_dX = tf.squeeze(sol_dX)

            l1 = 2 * tf.reduce_sum(tf.math.log(det_Omega)) + \
                 2 * tf.reduce_sum(sigma_term_diagonal)

            l2 = 1./dt * tf.reduce_sum(tf.square(sol_dX))

            l3 = dt * tf.reduce_sum(tf.square(sol_mu))

            l4 = -2 * tf.reduce_sum(sol_mu * sol_dX)

            # compute the penalty term

            ## evaluate the X variable in the regression problem
            sigma = tf.matmul(Omega_T, sigma_tilde)
            G_tf = tf.expand_dims(tf.constant(G[1:], dtype=tf.float32), axis=0)
            reg_Xt = tf.matmul(sigma, G_tf, transpose_a=True)

            ## evaluate the Y variable in the regression problem
            reg_Y = tf.matmul(G_tf, mu_tf, transpose_a=True) - zed

            ## evaluate the OLS estimates of the regression problem
            reg_XtY = tf.matmul(reg_Xt, reg_Y)
            reg_XtX = tf.matmul(reg_Xt, reg_Xt, transpose_b=True)
            reg_psi = tf.linalg.solve(reg_XtX, reg_XtY)  #
            reg_err = reg_Y - tf.matmul(reg_Xt, reg_psi, transpose_a=True)

            pnty = lbd_penalty_eq * tf.reduce_sum(tf.square(reg_err)) + \
                   lbd_penalty_sz * tf.reduce_sum(tf.square(reg_psi))

            return l1 + l2 + l3 + l4 + pnty

        return loss


class Model(object):
    """
    Library of constructing neural network models.
    """

    @staticmethod
    def construct_S(dim_input, n_obs,
                    pruning_sparsity, validation_split, batch_size, epochs):

        # construct the fully connected model
        dim_output = 2
        model_S = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(dim_input,),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(dim_output)])

        # prune the model
        n_obs_train = n_obs * (1 - validation_split)
        end_step = np.ceil(n_obs_train / batch_size).astype(np.int32) * epochs

        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0, final_sparsity=pruning_sparsity,
            begin_step=0, end_step=end_step
        )

        model_S_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            model_S, pruning_schedule
        )

        return model_S_pruning

    @staticmethod
    def construct_mu(dim_input):

        # construct the fully connected model
        dim_output = 2
        model_mu = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(dim_input,),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(dim_output)])

        return model_mu

    @staticmethod
    def construct_xi(dim_input, dim_output, n_obs,
                     pruning_sparsity, validation_split, batch_size, epochs):

        # construct the fully connected model
        model_xi = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(dim_input,),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(dim_output)])

        # prune the model
        n_obs_train = n_obs * (1 - validation_split)
        end_step = np.ceil(n_obs_train / batch_size).astype(np.int32) * epochs

        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=pruning_sparsity,
            begin_step=0, end_step=end_step
        )

        model_xi_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            model_xi, pruning_schedule
        )

        return model_xi_pruning


class Train(object):
    """
    Library of training methods for neural SDE models.
    """

    @staticmethod
    def train_S(X_S, Y_S,
                pruning_sparsity=0.5, validation_split=0.1,
                batch_size=512, epochs=500, rand_seed=0,
                force_fit=False, model_name='model_S',
                out_dir='output/checkpoint/'):

        n_obs, dim_input = X_S.shape

        # construct the neural network model
        model_S = Model.construct_S(
            dim_input, n_obs,
            pruning_sparsity, validation_split, batch_size, epochs)

        # compile the neural network model
        model_S.compile(
            loss=Loss.loss_S(1e-3),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
        )

        # set up I/O
        tag = out_dir + model_name + '_' + str(rand_seed)
        checkpoint_filepath_model_S = tag
        checkpoint_filepath_model_S_all = tag + '*'
        csv_fname = tag + '_history.csv'

        pruning_dir = out_dir + 'pruning_summary/'
        if not os.path.exists(pruning_dir):
            os.mkdir(pruning_dir)

        # train the pruned model
        tf.random.set_seed(rand_seed)
        if glob(checkpoint_filepath_model_S_all) and not force_fit:
            model_S.load_weights(checkpoint_filepath_model_S)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath_model_S,
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            csv_logger = tf.keras.callbacks.CSVLogger(
                filename=csv_fname,
                separator=',',
                append=False
            )

            history = model_S.fit(
                X_S, Y_S,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                shuffle=True,
                verbose=True,
                callbacks=[
                    model_checkpoint_callback,
                    csv_logger,
                    tfmot.sparsity.keras.UpdatePruningStep(),
                    tfmot.sparsity.keras.PruningSummaries(log_dir=pruning_dir)]
            )

            # plot training loss history
            plot_fname = tag + '_history.png'
            utils.PlotLib.plot_loss_over_epochs(history, True, plot_fname)

        return model_S

    @staticmethod
    def train_mu(X_S, mu_base,
                 validation_split=0.1, batch_size=512,
                 epochs=200, rand_seed=0, force_fit=False,
                 model_name='model_mu', out_dir='output/checkpoint/'):

        dim_input = X_S.shape[1]

        # construct the neural network model
        model_mu = Model.construct_mu(dim_input)

        model_mu.compile(loss='mean_absolute_error', optimizer='adam')

        # set up I/O
        tag = out_dir + model_name + '_' + str(rand_seed)
        checkpoint_filepath_model_mu = tag
        checkpoint_filepath_model_mu_all = tag + '*'
        csv_fname = tag + '_history.csv'

        # train the model
        if glob(checkpoint_filepath_model_mu_all) and not force_fit:
            model_mu.load_weights(checkpoint_filepath_model_mu)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath_model_mu,
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            csv_logger = tf.keras.callbacks.CSVLogger(
                filename=csv_fname,
                separator=',',
                append=False
            )
            history = model_mu.fit(
                X_S, mu_base,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                shuffle=True, verbose=True,
                callbacks=[model_checkpoint_callback,
                           csv_logger]
            )

            # plot training loss history
            plot_fname = tag + '_history.png'
            utils.PlotLib.plot_loss_over_epochs(history, True, plot_fname)

        return model_mu

    @staticmethod
    def train_xi(X_xi, Y_xi, W, G,
                 lbd_penalty_eq, lbd_penalty_sz,
                 pruning_sparsity=0.5, validation_split=0.1,
                 batch_size=512, epochs=20000, rand_seed=0,
                 force_fit=False, model_name='model_xi',
                 out_dir='output/checkpoint/'):

        n_bdy, n_dim = W.shape
        n_varcov, mask_diagonal = Train._identify_diagonal_entries(n_dim)

        # construct the neural network model
        model_xi_pruning = Model.construct_xi(
            n_dim + 1, n_dim + n_varcov, X_xi.shape[0],
            pruning_sparsity, validation_split, batch_size, epochs)

        model_xi_pruning.compile(
            loss=Loss.loss_xi(1e-3, n_dim, n_varcov, mask_diagonal, W, G,
                              lbd_penalty_eq, lbd_penalty_sz),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        )

        # set up I/O
        tag = out_dir + model_name + '_' + str(rand_seed)
        checkpoint_filepath = tag
        checkpoint_filepath_all = tag + '*'
        csv_fname = tag + '_history.csv'

        pruning_dir = out_dir + 'pruning_summary/'
        if not os.path.exists(pruning_dir):
            os.mkdir(pruning_dir)

        # train the pruned model
        tf.random.set_seed(rand_seed)
        if glob(checkpoint_filepath_all) and not force_fit:
            model_xi_pruning.load_weights(checkpoint_filepath)
        else:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            csv_logger = tf.keras.callbacks.CSVLogger(
                filename=csv_fname,
                separator=',',
                append=False
            )

            history = model_xi_pruning.fit(
                X_xi, Y_xi,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                shuffle=True,
                verbose=True,
                callbacks=[
                    model_checkpoint_callback,
                    csv_logger,
                    tfmot.sparsity.keras.UpdatePruningStep(),
                    tfmot.sparsity.keras.PruningSummaries(log_dir=pruning_dir)]
            )

            # plot training loss history
            plot_fname = tag + '_history.png'
            utils.PlotLib.plot_loss_over_epochs(history, True, plot_fname)

        return model_xi_pruning

    @staticmethod
    def predict_in_sample_model_xi(model_xi, X_xi, Y_xi, W, G):
        n_dim = X_xi.shape[1] - 1
        n_bdy = W.shape[0]

        n_varcov, mask_diagonal = Train._identify_diagonal_entries(n_dim)

        # predict underlying functions using the learnt NN
        y_pred_nn = model_xi.predict(X_xi)

        # get diffusion terms
        mask_diagonal_np = [m[0] for m in mask_diagonal]
        sigma_term = y_pred_nn.copy()
        sigma_term[:, mask_diagonal_np] = np.exp(sigma_term[:, mask_diagonal_np])
        sigma_term = sigma_term[:, :n_varcov]
        sigma_tilde_T = Train._fill_triu(sigma_term, n_dim)

        # get drift terms
        mu_residuals = y_pred_nn[:, n_varcov:]

        # get inputs for scaling diffusions and correcting drifts
        ## regarding diffusions
        Omega = np.reshape(Y_xi[:, n_dim:n_dim+n_dim**2],
                           newshape=[-1, n_dim, n_dim])
        ## regarding drifts
        n1 = n_dim+n_dim**2+1
        corr_dirs = np.reshape(Y_xi[:, n1:n1+n_dim*n_bdy],
                               newshape=[-1, n_bdy, n_dim])
        epsmu = Y_xi[:, n1+n_dim*n_bdy:n1+n_dim*n_bdy+n_bdy]

        n2 = n1+n_dim*n_bdy+n_bdy
        mu_base = Y_xi[:, n2:n2+n_dim]

        # compute drift term
        mu_tilde = mu_base * mu_residuals

        # scale diffusion
        sigma_T = np.matmul(sigma_tilde_T, Omega)

        # correct drift
        mu_tilde_inner_W = mu_tilde.dot(W.T)
        corr_dir_inner_W = np.sum(corr_dirs * W[None, :, :], axis=-1)
        gamma = np.maximum(- mu_tilde_inner_W - epsmu, 0.) / corr_dir_inner_W
        mu = mu_tilde + np.sum(gamma[:, :, None] * corr_dirs, axis=1)

        # LU deconposition of diffusion matrices
        mat_cov = np.matmul(np.transpose(sigma_T, axes=[0, 2, 1]), sigma_T)
        sigma_L = np.linalg.cholesky(mat_cov)

        return mu_tilde, sigma_tilde_T, mu, sigma_T, sigma_L

    @staticmethod
    def _identify_diagonal_entries(n_dim):
        """
        Return the Boolean logical mask array that indicates diagonal terms in
        a diffusion matrix.

        """

        # get the number of unknowns in the diffusion matrix
        n_varcov = int(n_dim*(n_dim+1)/2)

        # construct the diagonal entry mask
        x = np.arange(n_varcov)
        xc = np.concatenate([x, x[n_dim:][::-1]])
        idxs_diagonal = [xc[i * (n_dim + 1)] for i in range(n_dim)]
        mask_diagonal = np.zeros(n_varcov + n_dim, dtype=bool)
        mask_diagonal[idxs_diagonal] = True
        mask_diagonal = [[m] for m in mask_diagonal]

        return n_varcov, mask_diagonal

    @staticmethod
    def _fill_triu(arrs_sigma, n_dim):
        """
        Return a list of upper triangular diffusion matrices, given a list of
        flat arrays that contain non-zero elements of the diffusion matrices.
        """

        n_obs = arrs_sigma.shape[0]
        mats_sigma = np.zeros((n_obs, n_dim, n_dim))

        for i in range(n_obs):
            arr_sigma = arrs_sigma[i]
            xc = np.concatenate([arr_sigma, arr_sigma[n_dim:][::-1]])
            g = np.reshape(xc, [n_dim, n_dim])
            mats_sigma[i] = np.triu(g, k=0)

        return mats_sigma


class Simulate(object):
    """
    Library of forward-simulation methods.
    """

    @staticmethod
    def simulate_S_xi_lite(dt, N, model_S, model_xi, model_mu,
                           S0, X0, W, b, factor_multiplier,
                           dist_multiplier, proj_scale,
                           rho_star, epsmu_star, X_interior, reflect=False):

        # simulate innovations
        n_dim = X0.shape[0]
        dW = np.random.normal(0, np.sqrt(dt), (n_dim + 1, N + 1))

        # initialise
        st = np.ones(N+1) * np.nan
        xit = np.ones((n_dim, N+1)) * np.nan
        st[0] = S0
        xit[:, 0] = X0

        mus_sim = []
        vols_sim = []

        n_varcov, mask_diagonal = Train._identify_diagonal_entries(n_dim)
        n_reflect = 0

        for i in tqdm(range(1, N+1)):

            try:

                # get drift and diffusion of S
                xi = xit[:, i-1]
                x_S = np.hstack((st[i-1]/factor_multiplier, xi))
                pred_S = model_S.predict(x_S.reshape(1, -1))[0]
                vol_S = np.sqrt(np.exp(-pred_S[0])) * st[i-1]
                mu_S = pred_S[1] * st[i-1]

                # simulate S
                S_ = st[i-1] + mu_S * dt + vol_S * dW[0, i]

                # get baseline drift
                x_mu = np.hstack((st[i-1]/factor_multiplier, xi))
                pred_mu_base = model_mu.predict(x_mu.reshape(1, -1))[0]

                # get drift and diffusion of xi
                x_xi = np.hstack((st[i-1]/factor_multiplier, xi))
                gamma_nn = model_xi.predict(x_xi.reshape(1,-1))[0]
                gamma_nn[np.array(mask_diagonal).ravel()] = np.exp(
                    gamma_nn[np.array(mask_diagonal).ravel()])

                sigma_term = gamma_nn[:n_varcov]
                xc = np.concatenate([sigma_term, sigma_term[n_dim:][::-1]])
                g = np.reshape(xc, [n_dim, n_dim])
                sigma_tilde = np.triu(g, k=0).T

                mu_residual = gamma_nn[n_varcov:]

                # scale diffusion and correct drift
                mu, mat_vol = Simulate.scale_drift_diffusion(
                    xi, mu_residual, sigma_tilde, W, b,
                    dist_multiplier, proj_scale,
                    rho_star, epsmu_star, X_interior, pred_mu_base)

                # tame coefficients
                mu_norm = 1. + np.linalg.norm(mu) * np.sqrt(dt)
                vol_norm = 1. + np.linalg.norm(mat_vol) * np.sqrt(dt)

                # simulate xi using Euler-scheme
                xi_ = xi + mu / mu_norm * dt + \
                      mat_vol.dot(dW[1:, i].reshape((-1, 1))).flatten()/vol_norm

                if reflect:
                    if np.any(W.dot(xi_) - b < 0):
                        n_reflect += 1
                        print(f'Reflect simulated data point at index {i}.')
                        xi_ = Simulate.reflect_data(xi, xi_, W, b)

                st[i] = S_
                xit[:, i] = xi_

                mus_sim.append(mu)
                vols_sim.append(mat_vol)

            except:

                break

        return st, xit, mus_sim, vols_sim, n_reflect

    @staticmethod
    def simulate_S_xi(dt, N,
                      model_S, model_xi, model_mu,
                      S, X, W, b, factor_multiplier,
                      dist_multiplier, proj_scale,
                      rho_star, epsmu_star, X_interior,
                      train_rand_seed, sim_rand_seed,
                      force_simulate=False, reflect=False,
                      out_dir='output/checkpoint/'):
        print(f'Simulation number: {str(train_rand_seed)}_{str(sim_rand_seed)}')

        # set I/O
        plot_fname = f'{out_dir}simulation_{str(train_rand_seed)}' + \
                     f'_{str(sim_rand_seed)}.png'
        data_fname = f'{out_dir}simulation_{str(train_rand_seed)}' + \
                     f'_{str(sim_rand_seed)}.csv'

        if os.path.exists(data_fname) and not force_simulate:
            return

        # simulate
        np.random.seed(sim_rand_seed)
        S0 = S[0]
        X0 = X[0, :]
        st, xit, mus_sim, vols_sim, n_reflect = Simulate.simulate_S_xi_lite(
            dt, N, model_S, model_xi, model_mu,
            S0, X0, W, b, factor_multiplier,
            dist_multiplier, proj_scale,
            rho_star, epsmu_star, X_interior, reflect)

        if reflect:
            plot_fname = f'{out_dir}simulation_{str(train_rand_seed)}' + \
                         f'_{str(sim_rand_seed)}_reflect_{str(n_reflect)}.png'
            data_fname = f'{out_dir}simulation_{str(train_rand_seed)}' + \
                         f'_{str(sim_rand_seed)}_reflect_{str(n_reflect)}.csv'

        # save simulated data
        out_data = np.vstack((st, xit))
        columns = ['S'] + ['xi' + str(i) for i in range(1, len(X0)+1)]
        out_data = pd.DataFrame(data=out_data.T, columns=columns)
        out_data.to_csv(data_fname, index=False)

        # plot
        utils.PlotLib.plot_simulated_xi(st, xit, X, plot_fname)

        return st, xit, mus_sim, vols_sim

    @staticmethod
    def scale_drift_diffusion(x, mu_residual, sigma_tilde, W, b,
                              dist_multiplier, proj_scale,
                              rho_star, epsmu_star, x_interior, mu_base):
        """
        Scale the drift and diffusion functions.

        Parameters
        __________

        Returns
        _______

        """

        n_dim = W.shape[1]

        # calculate the distance of the data point to each boundary
        dist_x = np.abs(W.dot(x) - b) / np.linalg.norm(W, axis=1)

        # calculate the normalised distance indicators
        epsilon_sigma = PrepTrainData.normalise_dist_diffusion(
            dist_x, dist_multiplier, proj_scale)

        # sort by distance and get first n_dim closest ones
        idxs_sorted_eps = np.argsort(epsilon_sigma)
        idxs_used_eps = idxs_sorted_eps[:n_dim]
        Wd = W[idxs_used_eps]
        epsilond_sigma = epsilon_sigma[idxs_used_eps]

        # scale the diffusions
        if np.max(epsilond_sigma) < 1e-8:  # if the anchor point is on a corner
            Omega = np.zeros((n_dim, n_dim))
        else:  # if the anchor point is not on the corner
            # compute new bases
            V = np.linalg.qr(Wd.T)[0].T
            Omega = np.diag(np.sqrt(epsilond_sigma)).dot(V)
        mat_a = Omega.T.dot(sigma_tilde).dot(sigma_tilde.T).dot(Omega)
        mat_vol = np.linalg.cholesky(mat_a)

        # scale the drifts
        ## compute drift
        mu_tilde = mu_base * mu_residual
        ## compute correction directions
        corr_dirs_x = x_interior - x[None, :]
        epsmu_x = PrepTrainData.normalise_dist_drift(
            dist_x, rho_star, epsmu_star)
        mu_tilde_inner_W = W.dot(mu_tilde)
        corr_dir_inner_W = np.sum(corr_dirs_x * W, axis=-1)
        weights_corr_dir = np.maximum(-mu_tilde_inner_W-epsmu_x, 0.) /\
                           corr_dir_inner_W

        ## compute the corrected drift
        mu = mu_tilde + np.sum(corr_dirs_x * weights_corr_dir[:, None], axis=0)

        return mu, mat_vol

    @staticmethod
    def reflect_data(x0, x1, W, b):

        mask_arb = W.dot(x1) - b < 0

        # reflect data if there is arbitrage
        if np.any(mask_arb):

            if np.sum(mask_arb) > 1:
                print('Break more than one boundaries, move to the closest '
                      'boundary.')

                wi = W[mask_arb]
                bi = b[mask_arb]
                candidates = ((bi + 1e-6 - wi.dot(x0))/wi.dot((x1-x0))).\
                                 reshape((-1, 1)) * (x1 - x0) + x0
                idx_first_qualified = np.where(
                    np.all(candidates.dot(W.T) - b[None,:] >= 0, axis=1))[0][0]

                x2 = candidates[idx_first_qualified]

            else:

                wi = W[mask_arb]
                bi = b[mask_arb]

                t = bi - wi.dot(x1)
                x2 = x1 + 2 * t * wi

                # if the reflected data point breaks any arbitrage bounds
                if np.any(x2.dot(W.T) - b < 0):
                    print('Reflect failed, move back to boundary.')
                    t = (bi - wi.dot(x1)) / (wi.dot(x1 - x0))
                    x2 = x0 + t * (x1-x0)

            return x2

        else:

            return x1
