import torch.nn as nn
import importlib

from ..project_config import ProjectConfig
from ..dataset_config import DatasetConfig
from utils import nseloss


class FineTuneLearningConfig:
    loss_type = "MSE"  # TODO: loss function type, chose in ["NSELoss" ,"MSE"]
    loss_functions = {"MSE": nn.MSELoss(), "NSELoss": nseloss.NSELoss()}
    loss_func = loss_functions[loss_type]

    scale_factor = 1
    n_epochs = 200  # TODO
    batch_size = 512 // scale_factor  # TODO
    learning_rate = 0.001 / scale_factor  # TODO: usually, the bath_size bigger is, the learning_rate larger will have to be.
    # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
    scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.99}
    # scheduler_paras = {"scheduler_type": "none"}
    # scheduler_paras = {"scheduler_type": "exp_decay", "decay_epoch": n_epochs * 0.5, "decay_rate": 0.99}
    # scheduler_paras = {"scheduler_type": "cos_anneal", "cos_anneal_t_max": 32}

    learning_config_info = f"{loss_type}_n{n_epochs}_bs{batch_size}_lr{learning_rate}_{scheduler_paras['scheduler_type']}"


class FineTuneConfig(FineTuneLearningConfig):
    seed = None  # Random seed
    pre_saving_message = ""  # TODO
    used_model = "Transformer"  # TODO

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode

    exps_config = list()
    for basin in DatasetConfig.global_basins_list:
        exp_config = dict()
        exp_config["tag"] = basin
        exp_config["ft_train_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.train_start,
            "end_date": DatasetConfig.train_end,
            "final_data_path": None
        }

        exp_config["ft_val_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.val_start,
            "end_date": DatasetConfig.val_end,
            "final_data_path": None
        }

        exp_config["ft_test_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.test_start,
            "end_date": DatasetConfig.test_end,
            "final_data_path": None
        }
        exps_config.append(exp_config)

    if pre_saving_message != "":
        pre_saving_root = ProjectConfig.run_root / pre_saving_message
        pre_model_path = list(pre_saving_root.glob(f"(max_nse)*.pkl"))
        assert (len(pre_model_path) == 1)
        pre_model_path = pre_model_path[0]
        fine_tune_root = pre_saving_root / "fine_tune"
