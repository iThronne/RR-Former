import torch


class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1):
        super().__init__()
        eps = torch.tensor(eps, dtype=torch.float32)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true) ** 2
        self.eps = self.eps.to(q_stds.device)
        weights = 1 / (q_stds + self.eps) ** 2
        weights = weights.reshape(-1, 1, 1)
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)
