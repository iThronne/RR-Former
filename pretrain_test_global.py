import torch
import numpy as np
import json
import importlib
import os

from torch.utils.data import DataLoader

from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.pretrain_config import PretrainConfig
from utils.tools import SeedMethods
from utils.test_full import test_full
from data.dataset import DatasetFactory

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

seed = PretrainConfig.seed
saving_message = PretrainConfig.saving_message
saving_root = PretrainConfig.saving_root
used_model = PretrainConfig.used_model
decode_mode = PretrainConfig.decode_mode
pre_test_config = PretrainConfig.pre_test_config
batch_size = PretrainConfig.batch_size

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)
    print(saving_root)
    # Model
    # # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    best_path = list(saving_root.glob(f"(max_nse)*.pkl"))
    assert (len(best_path) == 1)
    best_path = best_path[0]
    best_model = Model().to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    # Dataset
    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)
    # # Needs training mean and training std
    train_means = np.loadtxt(saving_root / "train_means.csv", dtype="float32")
    train_stds = np.loadtxt(saving_root / "train_stds.csv", dtype="float32")
    train_x_mean = train_means[:-1]
    train_y_mean = train_means[-1]
    train_x_std = train_stds[:-1]
    train_y_std = train_stds[-1]
    with open(saving_root / "y_stds_dict.json", "rt") as f:
        y_stds_dict = json.load(f)

    # # Testing data
    ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=pre_test_config,
                              x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                              y_stds_dict=y_stds_dict)
    test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    test_full(best_model, decode_mode, test_loader, device, saving_root,
              only_metrics=False)  # TODO: only_metrics usually equals True here.
