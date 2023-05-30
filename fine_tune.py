import torch
import os
import numpy as np
import json
import importlib
from shutil import copytree

from torch.utils.data import DataLoader
from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.fine_tune_config import FineTuneConfig
from utils.tools import SeedMethods
from utils.lr_strategies import SchedulerFactory
from utils.train_full import train_full
from data.dataset import DatasetFactory

project_root = ProjectConfig.project_root
device = ProjectConfig.device
num_workers = ProjectConfig.num_workers

past_len = DataShapeConfig.past_len
pred_len = DataShapeConfig.pred_len
src_len = DataShapeConfig.src_len
tgt_len = DataShapeConfig.tgt_len
src_size = DataShapeConfig.src_size
tgt_size = DataShapeConfig.tgt_size
use_future_fea = DataShapeConfig.use_future_fea
use_static = DataShapeConfig.use_static

used_model = FineTuneConfig.used_model
decode_mode = FineTuneConfig.decode_mode
exps_config = FineTuneConfig.exps_config
loss_func = FineTuneConfig.loss_func
n_epochs = FineTuneConfig.n_epochs
batch_size = FineTuneConfig.batch_size
learning_rate = FineTuneConfig.learning_rate
scheduler_paras = FineTuneConfig.scheduler_paras

seed = FineTuneConfig.seed
pre_model_path = FineTuneConfig.pre_model_path
fine_tune_root = FineTuneConfig.fine_tune_root

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)
    fine_tune_root.mkdir(exist_ok=True)
    print(fine_tune_root)
    # Saving config files
    configs_path = project_root / "configs"
    configs_saving = fine_tune_root / "configs"
    if configs_saving.exists():
        raise RuntimeError("config files already exists!")
    copytree(configs_path, configs_saving)

    # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)

    # Dataset
    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)

    # Test for each basin
    exps_num = len(exps_config)
    for idx, exp_config in enumerate(exps_config):
        print(f"==========Now process: {idx} / {exps_num}===========")
        SeedMethods.seed_torch(seed=seed)
        root_now = fine_tune_root / exp_config["tag"]
        root_now.mkdir(exist_ok=True, parents=True)

        # Training data
        ds_train = DS.get_instance(past_len, pred_len, "train", specific_cfg=exp_config["ft_train_config"])
        train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        # We use the feature means/stds of the training data for normalization in val and test stage
        train_x_mean, train_y_mean = ds_train.get_means()
        train_x_std, train_y_std = ds_train.get_stds()
        y_stds_dict = ds_train.y_stds_dict
        # Saving training mean and training std
        train_means = np.concatenate((train_x_mean, train_y_mean), axis=0)
        train_stds = np.concatenate((train_x_std, train_y_std), axis=0)
        np.savetxt(root_now / "train_means.csv", train_means)
        np.savetxt(root_now / "train_stds.csv", train_stds)
        with open(root_now / "y_stds_dict.json", "wt") as f:
            json.dump(y_stds_dict, f)

        # Validation data (needs training mean and training std)
        ds_val = DS.get_instance(past_len, pred_len, "val", specific_cfg=exp_config["ft_val_config"],
                                 x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                                 y_stds_dict=y_stds_dict)
        val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        # Testing data (needs training mean and training std)
        ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=exp_config["ft_test_config"],
                                  x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                                  y_stds_dict=y_stds_dict)
        test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        # Model, optimizer, scheduler, loss
        pre_model = Model().to(device)
        pre_model.load_state_dict(torch.load(pre_model_path, map_location=device))
        optimizer = torch.optim.Adam(pre_model.parameters(), lr=learning_rate)
        scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
        loss_func = loss_func.to(device)

        # Training and validation
        train_full(pre_model, decode_mode, train_loader, val_loader, optimizer, scheduler, loss_func, n_epochs, device,
                   root_now)
