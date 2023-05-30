# All metrics needs obs and sim shape: (batch_size, pred_len, tgt_size)

import numpy as np
import torch


def calc_nse(obs: np.array, sim: np.array) -> np.array:
    denominator = np.sum((obs - np.mean(obs, axis=0)) ** 2, axis=0)
    numerator = np.sum((sim - obs) ** 2, axis=0)
    nse = 1 - numerator / denominator

    nse_mean = np.mean(nse)
    # Return mean NSE, and NSE of all locations, respectively
    return nse_mean, nse[:, 0]


def calc_kge(obs: np.array, sim: np.array):
    mean_obs = np.mean(obs, axis=0)
    mean_sim = np.mean(sim, axis=0)

    std_obs = np.std(obs, axis=0)
    std_sim = np.std(sim, axis=0)

    beta = mean_sim / mean_obs
    alpha = std_sim / std_obs
    numerator = np.mean(((obs - mean_obs) * (sim - mean_sim)), axis=0)
    denominator = std_obs * std_sim
    gamma = numerator / denominator
    kge = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (gamma - 1) ** 2)

    kge_mean = np.mean(kge)
    # Return mean KEG, and KGE of all locations, respectively
    return kge_mean, kge[:, 0]


def calc_tpe(obs: np.array, sim: np.array, alpha):
    sort_index = np.argsort(obs, axis=0)
    obs_sort = np.take_along_axis(obs, sort_index, axis=0)
    sim_sort = np.take_along_axis(sim, sort_index, axis=0)
    top = int(obs.shape[0] * alpha)
    obs_t = obs_sort[-top:, :]
    sim_t = sim_sort[-top:, :]
    numerator = np.sum(np.abs(sim_t - obs_t), axis=0)
    denominator = np.sum(obs_t, axis=0)
    tpe = numerator / denominator

    tpe_mean = np.mean(tpe)
    # Return mean TPE, and TPE of all locations, respectively
    return tpe_mean, tpe[:, 0]


def calc_bias(obs: np.array, sim: np.array):
    numerator = np.sum(sim - obs, axis=0)
    denominator = np.sum(obs, axis=0)
    bias = numerator / denominator

    bias_mean = np.mean(bias)
    # Return mean bias, and bias of all locations, respectively
    return bias_mean, bias[:, 0]


def calc_mse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)

    mse_mean = np.mean(mse)
    # Return mean MSE, and MSE of all locations, respectively
    return mse_mean, mse[:, 0]


def cacl_rmse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)
    rmse = np.sqrt(mse)

    rmse_mean = np.mean(rmse)
    # Return mean RMSE, and RMSE of all locations, respectively
    return rmse_mean, rmse[:, 0]


def cacl_nrmse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)
    rmse = np.sqrt(mse)
    obs_mean = np.mean(obs, axis=0)
    nrmse = rmse / obs_mean

    nrmse_mean = np.mean(nrmse)
    # Return mean NRMSE, and NRMSE of all locations, respectively
    return nrmse_mean, nrmse[:, 0]


def calc_nse_torch(obs, sim):
    with torch.no_grad():
        denominator = torch.sum((obs - torch.mean(obs, dim=0)) ** 2, dim=0)
        numerator = torch.sum((sim - obs) ** 2, dim=0)
        nse = torch.tensor(1).to(sim.device) - numerator / denominator

        nse_mean = torch.mean(nse)
        # Return mean NSE, and NSE of all locations, respectively
        return nse_mean, nse[:, 0]
