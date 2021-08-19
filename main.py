# Copyright 2021 Sheng Wang.
# Affiliation: Mathematical Institute, University of Oxford
# Email: sheng.wang@maths.ox.ac.uk

import os
import random
import numpy as np

import marketmodel.loader as loader
import marketmodel.utils as utils

from marketmodel.loader import DataHestonSlv
from marketmodel.factors import PrepTrainData, DecodeFactor
from marketmodel.neuralsde import Train, Simulate


def run():

    # load Heston-SLV simulation data
    fname = 'input/sim_hestonslv.pkl'
    St, vt, list_exp, list_mny, cs_ts_raw, cs_ts, mask_quality_value, \
    Ts, ks, mat_A, vec_b = loader.load_hestonslv_data(fname)

    # load configurations
    hp_sde_transform = utils.Config.hp_sde_transform
    hp_model_S = utils.Config.hp_model_S
    hp_model_mu = utils.Config.hp_model_mu
    hp_model_xi = utils.Config.hp_model_xi

    # fit an initial model for S
    dir_initial_model_S = 'output/checkpoint/initial_model_S/'

    X_S, Y_S = PrepTrainData.prep_data_model_S_initial(
        St, cs_ts, max_PC=7, factor_multiplier=1e5)

    model_S_initial = Train.train_S(X_S, Y_S,
                                    hp_model_S['pruning_sparsity'],
                                    hp_model_S['validation_split'],
                                    hp_model_S['batch_size'],
                                    hp_model_S['epochs'],
                                    rand_seed=0, force_fit=False,
                                    model_name='model_S',
                                    out_dir=dir_initial_model_S)

    # calculate derivatives for the normalised call prices
    cT_ts, cm_ts, cmm_ts = PrepTrainData.calc_call_derivatives(
        list_mny, list_exp, cs_ts_raw, mask_quality_value)

    # decode factor
    G, X, dX, S, dS, W, b, idxs_remove, scales_X = \
        DecodeFactor.decode_factor_dasa(
            cs_ts, St, model_S_initial, X_S, cT_ts, cm_ts, cmm_ts, mat_A, vec_b,
            hp_sde_transform['norm_factor'])

    cT_ts = np.delete(cT_ts, idxs_remove, axis=0)
    cm_ts = np.delete(cm_ts, idxs_remove, axis=0)
    cmm_ts = np.delete(cmm_ts, idxs_remove, axis=0)

    # calibrate a hypterparameter for normalising distance
    dist_X = np.abs(W.dot(X.T) - b[:, None]) / \
             np.linalg.norm(W, axis=1, keepdims=True)
    critical_threshold = hp_sde_transform['frac_critical_threshold'] * np.min(
        np.max(dist_X, axis=1))
    dist_multiplier = (1. / (
            1 - hp_sde_transform['critical_value']) - 1) / critical_threshold

    # pre-calculate diffusion scaling data
    proj_scale = hp_sde_transform['proj_scale']
    Omegas, det_Omega, proj_dX = PrepTrainData.calc_diffusion_scaling(
        W, b, X, dX, dist_multiplier, proj_scale)

    # pre-calculate drift correction data
    X_interior, corr_dirs, epsmu = PrepTrainData.calc_drift_correction(
        W, b, X, hp_sde_transform['rho_star'], hp_sde_transform['epsmu_star'])

    # set paths
    run_batch_name = 'train_batch_1'
    out_dir = f'output/checkpoint/{run_batch_name}/'
    out_dir_plot = out_dir + 'plot/'
    print(f'Write in the folder {out_dir}.')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_dir_plot):
        os.mkdir(out_dir_plot)

    list_rand_seed = [5]
    list_sim_rand_seed = [5]

    for rand_seed in list_rand_seed:
        # set random seed
        os.environ['PYTHONHASHSEED'] = str(rand_seed)
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        # train the model for S
        X_S, Y_S = PrepTrainData.prep_data_model_S(
            S, dS, X, hp_model_xi['factor_multiplier'])
        model_S = Train.train_S(X_S, Y_S,
                                hp_model_S['pruning_sparsity'],
                                hp_model_S['validation_split'],
                                hp_model_S['batch_size'],
                                hp_model_S['epochs'],
                                rand_seed=rand_seed, force_fit=False,
                                model_name='model_S',
                                out_dir=out_dir)

        # fit model for the baseline drift
        mu_base = PrepTrainData.calc_baseline_drift(
            cT_ts, cm_ts, cmm_ts, model_S, X_S, G, scales_X)
        model_mu = Train.train_mu(X_S, mu_base,
                                  hp_model_mu['validation_split'],
                                  hp_model_mu['batch_size'],
                                  hp_model_mu['epochs'],
                                  rand_seed=rand_seed, force_fit=False,
                                  model_name='model_mu',
                                  out_dir=out_dir)

        # train the model for xi
        mu_base_est = model_mu.predict(X_S)
        z_ts = PrepTrainData.calc_zt(cT_ts, cm_ts, cmm_ts, model_S, X_S)

        X_xi, Y_xi = PrepTrainData.prepare_data_model_xi(
            S, X, proj_dX, Omegas, det_Omega, corr_dirs, epsmu, mu_base_est, z_ts,
            hp_model_xi['factor_multiplier'])
        model_xi = Train.train_xi(X_xi, Y_xi, W, G,
                                  hp_model_xi['lbd_penalty_eq'],
                                  hp_model_xi['lbd_penalty_sz'],
                                  hp_model_xi['pruning_sparsity'],
                                  hp_model_mu['validation_split'],
                                  hp_model_mu['batch_size'],
                                  hp_model_xi['epochs'],
                                  rand_seed=rand_seed, force_fit=False,
                                  model_name='model_xi', out_dir=out_dir)

        # forward simulation
        N = 10000
        dt = 1e-3

        for sim_rand_seed in list_sim_rand_seed:
            out_dir_sim = out_dir + 'sim/'
            if not os.path.exists(out_dir_sim):
                os.mkdir(out_dir_sim)

            Simulate.simulate_S_xi(
                dt, N, model_S, model_xi, model_mu, S, X, W, b,
                hp_model_xi['factor_multiplier'], dist_multiplier, proj_scale,
                hp_sde_transform['rho_star'], hp_sde_transform['epsmu_star'],
                X_interior,
                rand_seed, sim_rand_seed,
                force_simulate=True, reflect=False, out_dir=out_dir_sim)


if __name__ == '__main__':
    run()


