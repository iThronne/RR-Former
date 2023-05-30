import time

from utils.eval_model import eval_model_obs_preds
from utils.metrics import calc_nse, calc_mse
from utils.tools import count_parameters, saving_obs_pred


def test_full(test_model, decode_mode, test_loader, device, saving_root, only_metrics: bool, date_index=None):
    """
    date_index will be used if only_metric == False
    """
    t1 = time.time()
    print(f"Parameters count:{count_parameters(test_model)}")
    log_file = saving_root / f"log_test.csv"
    obs, pred = eval_model_obs_preds(test_model, test_loader, decode_mode, device)
    # Calculate nse after rescale (But if you take the same mean and std, it's equivalent before and after)
    obs = obs.numpy()
    pred = pred.numpy()
    _, mses_test = calc_mse(obs, pred)
    test_mse_mean = mses_test.mean()

    obs_rescaled = test_loader.dataset.local_rescale(obs, variable='output')
    pred_rescaled = test_loader.dataset.local_rescale(pred, variable='output')
    _, nses_test = calc_nse(obs_rescaled, pred_rescaled)
    test_nse_mean = nses_test.mean()
    print(f"Testing mean mse: {test_mse_mean}, mean nse:{test_nse_mean}")

    with open(log_file, "wt") as f:
        f.write(f"parameters_count:{count_parameters(test_model)}\n")
        f.write(f"test_mse_mean:{test_mse_mean}\n")
        f.write(f"{','.join(f'mse{i + 1}' for i in range(len(mses_test)))}\n")
        f.write(f"{','.join(str(i) for i in list(mses_test))}\n")
        f.write(f"test_nse_mean:{test_nse_mean}\n")
        f.write(f"{','.join(f'nse{i + 1}' for i in range(len(nses_test)))}\n")
        f.write(f"{','.join(str(i) for i in list(nses_test))}\n")

    if not only_metrics:
        obs_pred_root = saving_root / "obs_pred"
        start_date = test_loader.dataset.dates[0]
        end_date = test_loader.dataset.dates[1]
        past_len = test_loader.dataset.past_len
        pred_len = test_loader.dataset.pred_len
        saving_obs_pred(obs_rescaled, pred_rescaled, start_date, end_date, past_len, pred_len, obs_pred_root,
                        date_index=date_index)

    t2 = time.time()
    print(f"Testing used time:{t2 - t1}")
