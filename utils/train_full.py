import torch
import time
import os
from utils.train_epoch import train_epoch
from utils.eval_model import eval_model
from utils.tools import BoardWriter, count_parameters


class BestModelLog:
    def __init__(self, init_model, saving_root, metric_name, high_better: bool):
        self.high_better = high_better
        self.saving_root = saving_root
        self.metric_name = metric_name
        worst = float("-inf") if high_better else float("inf")
        self.best_epoch = -1
        self.best_value = worst
        self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
        torch.save(init_model.state_dict(), self.best_model_path)

    def update(self, model, new_value, epoch):
        if ((self.high_better is True) and (new_value > self.best_value)) or \
                ((self.high_better is False) and (new_value < self.best_value)):
            os.remove(self.best_model_path)
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_model_path = self.saving_root / f"({self.metric_name})_{self.best_epoch}_{self.best_value}.pkl"
            torch.save(model.state_dict(), self.best_model_path)


def train_full(model, decode_mode, train_loader, val_loader, optimizer, scheduler, loss_func, n_epochs, device,
               saving_root, using_board=False):
    print(f"Parameters count:{count_parameters(model)}")
    log_file = saving_root / "log_train.csv"
    with open(log_file, "wt") as f:
        f.write(f"parameters_count:{count_parameters(model)}\n")
        f.write("epoch,train_loss_iterated,train_mse,val_mse,train_nse,val_nse\n")
    if using_board:
        tb_root = saving_root / "tb_log"
        writer = BoardWriter(tb_root)
    else:
        writer = None

    mse_log = BestModelLog(model, saving_root, "min_mse", high_better=False)
    nse_log = BestModelLog(model, saving_root, "max_nse", high_better=True)
    newest_log = BestModelLog(model, saving_root, "newest", high_better=True)

    t1 = time.time()
    for i in range(n_epochs):
        print(f"Training progress: {i} / {n_epochs}")
        train_loss_iterated = train_epoch(model, train_loader, optimizer, scheduler, loss_func, decode_mode, device)
        mse_train, nse_train = '', ''
        # mse_train, nse_train is not need to be calculated (via eval_model function),
        # and you can comment the next line to speed up
        mse_train, nse_train = eval_model(model, train_loader, decode_mode, device)
        mse_val, nse_val = eval_model(model, val_loader, decode_mode, device)
        if writer is not None:
            if (mse_train != '') and (nse_train != ''):
                writer.write_board(f"train_mse", metric_value=mse_train, epoch=i)
                writer.write_board(f"train_nse", metric_value=nse_train, epoch=i)
            writer.write_board(f"train_loss(iterated)", metric_value=train_loss_iterated, epoch=i)
            writer.write_board("val_mse", metric_value=mse_val, epoch=i)
            writer.write_board("val_nse", metric_value=nse_val, epoch=i)
        with open(log_file, "at") as f:
            f.write(f"{i},{train_loss_iterated},{mse_train},{mse_val},{nse_train},{nse_val}\n")
        mse_log.update(model, mse_val, i)
        nse_log.update(model, nse_val, i)
        newest_log.update(model, i, i)
    t2 = time.time()
    print(f"Training used time:{t2 - t1}")
