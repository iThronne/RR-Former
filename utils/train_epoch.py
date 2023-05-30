import torch


# Train utilities
def train_epoch(model, data_loader, optimizer, scheduler, loss_func, decode_mode, device):
    """
    Train model for a single epoch.

    :param model: A torch.nn.Module implementing the Transformer model.
    :param data_loader: A PyTorch DataLoader, providing the trainings data in mini batches.
    :param optimizer: One of PyTorch optimizer classes.
    :param scheduler: scheduler of learning rate.
    :param loss_func: The loss function to minimize.
    :param decode_mode: decoding mode in Transformer.
    :param device: device for data and models
    """
    # set model to train mode (important for dropout)
    model.train()
    cnt = 0
    loss_mean = 0
    for x_seq, y_seq_past, y_seq_future, y_stds in data_loader:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
        batch_size = y_seq_past.shape[0]
        past_len = y_seq_past.shape[1]
        pred_len = y_seq_future.shape[1]
        tgt_len = past_len + pred_len
        tgt_size = y_seq_future.shape[2]
        if decode_mode == "NAR":
            enc_inputs = x_seq
            dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            dec_inputs[:, :-pred_len, :] = y_seq_past
            # get model predictions
            y_hat = model(enc_inputs, dec_inputs)
            y_hat = y_hat[:, -pred_len:, :]
        elif decode_mode == "AR":
            enc_inputs = x_seq
            dec_inputs = torch.cat((y_seq_past, y_seq_future), dim=1)
            # get model predictions
            y_hat = model(enc_inputs, dec_inputs)
            y_hat = y_hat[:, -pred_len - 1:-1, :]
        else:  # Model is not Transformer
            y_hat = model(x_seq, y_seq_past)

        # calculate loss
        if type(loss_func).__name__ == "NSELoss":
            y_stds = y_stds.to(device)
            loss = loss_func(y_hat, y_seq_future, y_stds)
        else:
            loss = loss_func(y_hat, y_seq_future)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()

        # calculate mean loss
        cnt += 1
        loss_mean = loss_mean + (loss.item() - loss_mean) / cnt  # Welford’s method
    scheduler.step()

    return loss_mean
