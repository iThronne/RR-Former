import torch
from torch import nn
from utils.metrics import calc_nse_torch


def eval_model(model, data_loader, decode_mode, device):
    """
    Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param data_loader: A PyTorch DataLoader, providing the data.
    :param decode_mode: autoregressive or non-autoregressive
    :param device: device for data and models

    :return: mse_mean, nse_mean
    """
    # set model to eval mode (important for dropout)
    model.eval()
    mse = nn.MSELoss()
    cnt = 0
    mse_mean = 0
    nse_mean = 0
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future, _ in data_loader:
            x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
            batch_size = y_seq_past.shape[0]
            tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
            tgt_size = y_seq_future.shape[2]
            pred_len = y_seq_future.shape[1]

            enc_inputs = x_seq
            dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            dec_inputs[:, :-pred_len, :] = y_seq_past
            # get model predictions
            if decode_mode == "NAR":
                y_hat = model(enc_inputs, dec_inputs)
                y_hat = y_hat[:, -pred_len:, :]
            elif decode_mode == "AR":
                for i in range(tgt_len - pred_len, tgt_len):
                    decoder_predict = model(enc_inputs, dec_inputs)
                    dec_inputs[:, i, :] = decoder_predict[:, i - 1, :]
                y_hat = dec_inputs[:, -pred_len:, :]
            else:  # Model is not Transformer
                y_hat = model(x_seq, y_seq_past)

            # calculate loss
            mse_value = mse(y_hat, y_seq_future).item()
            nse_value, _ = calc_nse_torch(y_hat, y_seq_future)
            cnt += 1
            mse_mean = mse_mean + (mse_value - mse_mean) / cnt  # Welford’s method
            nse_mean = nse_mean + (nse_value - nse_mean) / cnt  # Welford’s method

    return mse_mean, nse_mean


def eval_model_obs_preds(model, data_loader, decode_mode, device):
    """
    Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param data_loader: A PyTorch DataLoader, providing the data.
    :param decode_mode: autoregressive or non-autoregressive
    :param device: device for data and models

    :return: Two torch Tensors, containing the observations and model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future, _ in data_loader:
            x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
            batch_size = y_seq_past.shape[0]
            tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
            tgt_size = y_seq_future.shape[2]
            pred_len = y_seq_future.shape[1]

            enc_inputs = x_seq
            dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            dec_inputs[:, :-pred_len, :] = y_seq_past
            # get model predictions
            if decode_mode == "NAR":
                y_hat = model(enc_inputs, dec_inputs)
                y_hat = y_hat[:, -pred_len:, :]
            elif decode_mode == "AR":
                for i in range(tgt_len - pred_len, tgt_len):
                    decoder_predict = model(enc_inputs, dec_inputs)
                    dec_inputs[:, i, :] = decoder_predict[:, i - 1, :]
                y_hat = dec_inputs[:, -pred_len:, :]
            else:  # Model is not Transformer
                y_hat = model(x_seq, y_seq_past)

            obs.append(y_seq_future.to("cpu"))
            preds.append(y_hat.to("cpu"))

    obs_all = torch.cat(obs)
    preds_all = torch.cat(preds)

    return obs_all, preds_all
