import random
import os
import numpy as np
import pandas as pd
import torch
import warnings

from torch.utils.tensorboard import SummaryWriter


class SeedMethods:
    @staticmethod
    def seed_torch(seed):
        if seed is None:
            return
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class BoardWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def write_board(self, msg, metric_value, epoch, every=2):
        if epoch % every == 0:
            self.writer.add_scalar(msg, metric_value, global_step=epoch)
            self.writer.close()


def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def saving_obs_pred(obs, pred, start_date, end_date, past_len, pred_len, saving_root, date_index=None):
    if len(obs.shape) == 3 and obs.shape[-1] == 1:
        obs = obs.squeeze(-1)
    if len(pred.shape) == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if date_index is not None:
        pd_range = pd.date_range(start_date + pd.Timedelta(days=past_len), end_date - pd.Timedelta(days=pred_len - 1))
        date_index_seq = date_index[past_len:-pred_len + 1]
        if (len(date_index_seq) != len(pd_range)) or ((date_index_seq == pd_range).min is False):
            warnings.warn("The missing blocks are not contiguous and may cause some errors!")
        obs_pd = pd.DataFrame(obs, columns=[f"obs{i}" for i in range(obs.shape[1])], index=date_index_seq)
        pred_pd = pd.DataFrame(pred, columns=[f"pred{i}" for i in range(pred.shape[1])], index=date_index_seq)
    else:
        obs_pd = pd.DataFrame(obs, columns=[f"obs{i}" for i in range(obs.shape[1])])
        pred_pd = pd.DataFrame(pred, columns=[f"pred{i}" for i in range(pred.shape[1])])
    saving_root.mkdir(parents=True, exist_ok=True)
    obs_pd.to_csv(saving_root / f"obs.csv", index=True, index_label="start_date")
    pred_pd.to_csv(saving_root / f"pred.csv", index=True, index_label="start_date")
