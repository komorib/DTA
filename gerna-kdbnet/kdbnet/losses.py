import torch
import torch.nn.functional as F


def gaussian_nll_loss(mu, logvar, target, logvar_min=-7.0, logvar_max=7.0,
                      reduction='mean'):
    """Heteroscedastic Gaussian negative log-likelihood.

    loss = 0.5 * exp(-logvar) * (mu - target)**2 + 0.5 * logvar

    Parameters
    ----------
    mu, logvar, target : torch.Tensor
        Broadcastable tensors. ``logvar`` is clamped to
        ``[logvar_min, logvar_max]`` before use for numerical stability.
    logvar_min, logvar_max : float or None
        Clamp bounds on the predicted log-variance. Pass ``None`` for BOTH to
        DISABLE clamping entirely (no bound on sigma). A one-sided bound is
        applied if only one is None.
    reduction : str
        One of ``'mean'``, ``'sum'``, or ``'none'``.
    """
    if logvar_min is not None or logvar_max is not None:
        logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
    loss = 0.5 * torch.exp(-logvar) * (mu - target) ** 2 + 0.5 * logvar
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction: {}".format(reduction))


class HeteroscedasticCriterion(object):
    """Callable criterion for a 2-channel (mu, logvar) prediction head.

    Supports an MSE warm-up: for the first ``warmup_epochs`` epochs only the
    mu channel is trained with ``F.mse_loss`` (logvar is ignored), after which
    the full Gaussian NLL is used. ``current_epoch`` is expected to be set by
    the training loop at the start of each epoch (1-indexed).
    """

    def __init__(self, warmup_epochs=10, logvar_min=-7.0, logvar_max=7.0):
        self.current_epoch = 1
        self.warmup_epochs = warmup_epochs
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def __call__(self, pred, target):
        # pred is [B, 2]: column 0 = mu, column 1 = logvar
        mu = pred[:, 0:1]
        logvar = pred[:, 1:2]
        target = target.reshape(-1, 1)
        if self.current_epoch <= self.warmup_epochs:
            return F.mse_loss(mu, target)
        return gaussian_nll_loss(
            mu, logvar, target,
            logvar_min=self.logvar_min, logvar_max=self.logvar_max,
            reduction='mean',
        )
