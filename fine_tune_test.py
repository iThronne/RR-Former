import torch
import os
import importlib

from torch.utils.data import DataLoader
from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.fine_tune_config import FineTuneConfig
from utils.tools import SeedMethods
from utils.test_full import test_full
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
batch_size = FineTuneConfig.batch_size

seed = FineTuneConfig.seed
pre_model_path = FineTuneConfig.pre_model_path
fine_tune_root = FineTuneConfig.fine_tune_root

if __name__ == '__main__':
    print("pid:", os.getpid())
    print(fine_tune_root)
    SeedMethods.seed_torch(seed=seed)

    # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)

    # Dataset
    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)

    # Fine tune for each basin
    exps_num = len(exps_config)
    for idx, exp_config in enumerate(exps_config):
        print(f"==========Now process: {idx} / {exps_num}===========")
        SeedMethods.seed_torch(seed=seed)
        root_now = fine_tune_root / exp_config["tag"]

        # Testing data (needs training mean and training std)
        ds_train = DS.get_instance(past_len, pred_len, "train", specific_cfg=exp_config["ft_train_config"])
        # We use the feature means/stds of the training data for normalization in val and test stage
        train_x_mean, train_y_mean = ds_train.get_means()
        train_x_std, train_y_std = ds_train.get_stds()
        y_stds_dict = ds_train.y_stds_dict
        ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=exp_config["ft_test_config"],
                                  x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                                  y_stds_dict=y_stds_dict)
        test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        date_index = test_loader.dataset.date_index_dict[exp_config["tag"]]

        # Loading best model and testing
        best_model_path = list(root_now.glob(f"(max_nse)*.pkl"))[0]
        best_model = Model().to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_full(best_model, decode_mode, test_loader, device, root_now, False, date_index=date_index)
