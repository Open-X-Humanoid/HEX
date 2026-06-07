'''
@article{bai2026reshaping,
  title={Reshaping Action Error Distributions for Reliable Vision-Language-Action Models},
  author={Bai, Shuanghao and Wang, DaKai and Chi, Cheng and Zhou, Wanqi and Lyu, Jing and Zhao, Xiaoguang and Wang, Pengwei and Wang, Zhongyuan and Xing, Lei and Zhang, Shanghang and Chen, Badong},
  journal={arXiv preprint arXiv:2602.04228},
  year={2026}
}

Adapted from https://github.com/Cognition2Action-Lab/VLA-TMEE
'''

import torch


def simple_mee_loss_hex(all_pred, all_vel, sigma=1.0, eps=1e-8):
    """
    all_pred: list of tensors, each with shape [B_k, H, D_k]
    all_vel:  list of tensors, each with shape [B_k, H, D_k]

    Returns:
        A scalar MEE loss.
    """
    mee_total = 0.0
    sample_total = 0

    for pred, vel in zip(all_pred, all_vel):
        err = pred - vel                       # [B_k, H, D_k]
        err = err.reshape(-1, err.shape[-1])   # [N_k, D_k], where N_k = B_k * H

        if err.shape[0] <= 1:
            continue

        # Compute pairwise squared distances: [N_k, N_k]
        dist2 = torch.cdist(err, err, p=2).pow(2)

        # Compute the Gaussian kernel matrix.
        K = torch.exp(-dist2 / (2 * sigma * sigma))

        # T-MEE = -log(mean kernel value)
        mee = -torch.log(K.mean() + eps)

        # Weight each group by the number of samples.
        n = err.shape[0]
        mee_total += mee * n
        sample_total += n

    if sample_total == 0:
        return torch.tensor(0.0, device=all_pred[0].device)

    return mee_total / sample_total


def simple_mee_loss(y_pred, y_true, sigma: float = 0.5, eps: float = 1e-8):
    """
    Standard Minimum Error Entropy (MEE) loss with a Gaussian kernel.

    Args:
        y_pred, y_true: Tensors of shape (B, T, D)
        sigma: Kernel bandwidth
        eps: Numerical stability constant
    """
    e = (y_pred - y_true).view(-1, y_pred.shape[-1])  # [N, D]

    diff = e.unsqueeze(1) - e.unsqueeze(0)            # [N, N, D]
    dist_sq = (diff ** 2).sum(dim=-1)                 # [N, N]
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))   # K_ij

    loss = -torch.log(kernel.mean() + eps)
    return loss


def adaptive_mee_loss_chunk(y_pred, y_true, sigma: float = 0.5, eps: float = 1e-8):
    """
    Chunk-level adaptive MEE with sample-wise weighting.
    """
    e = (y_pred - y_true).view(-1, y_pred.shape[-1])  # [N, D]
    n = e.size(0)

    diff = e.unsqueeze(1) - e.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

    sigma_w = torch.sqrt(torch.tensor(n / 1000.0, dtype=e.dtype, device=e.device))
    distance = torch.norm(e, dim=1)
    w = torch.exp(-distance ** 2 / (2 * sigma_w ** 2))
    w = w / (w.sum(dim=0, keepdim=True) + eps)

    # Renyi quadratic entropy estimator
    loss = -torch.log((w * kernel).sum() / (n ** 2) + eps)
    return loss


def adaptive_mee_loss_element(
    y_pred,
    y_true,
    sigma: float = 0.5,
    sigma_w: float = 0.5,
    eps: float = 1e-8,
):
    """
    Element-level adaptive MEE with pairwise reweighting.
    """
    e = (y_pred - y_true).view(-1, y_pred.shape[-1])  # [N, D]
    n = e.size(0)

    diff = e.unsqueeze(1) - e.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

    distance = torch.norm(e, dim=1)
    # sigma_w = torch.sqrt(torch.tensor(n / 1000.0, dtype=e.dtype, device=e.device))    # if adaptive
    w = torch.exp(-distance ** 2 / (2 * sigma_w ** 2))
    w = w / (w.sum() + eps)

    W = w.view(-1, 1) * w.view(1, -1)
    V_w = (W * kernel).sum()

    loss = -torch.log(V_w + eps)
    return loss