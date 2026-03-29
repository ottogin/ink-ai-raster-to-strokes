"""MDN loss functions following the DSD paper.

The stroke generation loss has two components:
1. Location loss: negative log-likelihood of (dx, dy) under the Gaussian mixture
2. Pen loss: binary cross-entropy for pen up/down state
"""

import math
import torch
import torch.nn as nn


def gaussian_2d(x1, x2, mu1, mu2, s1, s2, rho):
    """Probability density of bivariate Gaussian.

    All inputs are (B, T, K) except x1, x2 which are (B, T, 1).
    Returns (B, T, K).
    """
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2
    neg_rho_sq = 1 - rho ** 2
    numerator = torch.exp(-z / (2 * neg_rho_sq))
    denominator = 2 * math.pi * s1s2 * torch.sqrt(neg_rho_sq)
    return numerator / denominator


def mdn_loss(mdn_params, target, lengths):
    """Compute MDN loss over a batch with variable-length sequences.

    Args:
        mdn_params: dict from model forward pass with keys:
            eos_logit: (B, T, 1)
            mu:        (B, T, K, 2)
            sigma:     (B, T, K, 2)
            rho:       (B, T, K)
            pi:        (B, T, K)
        target: (B, T, 3) ground truth (dx, dy, pen_state)
        lengths: (B,) actual sequence lengths

    Returns:
        dict with:
            loss: scalar total loss
            loc_loss: scalar location NLL
            pen_loss: scalar pen BCE
    """
    B, T, _ = target.shape
    device = target.device

    # Create mask for valid timesteps
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
    mask_float = mask.float()
    num_valid = mask_float.sum().clamp(min=1)

    # Extract targets
    target_x = target[:, :, 0:1]  # (B, T, 1)
    target_y = target[:, :, 1:2]  # (B, T, 1)
    target_pen = target[:, :, 2]  # (B, T)

    # Extract MDN params
    mu1 = mdn_params["mu"][:, :, :, 0]    # (B, T, K)
    mu2 = mdn_params["mu"][:, :, :, 1]    # (B, T, K)
    sig1 = mdn_params["sigma"][:, :, :, 0]
    sig2 = mdn_params["sigma"][:, :, :, 1]
    rho = mdn_params["rho"]               # (B, T, K)
    pi = mdn_params["pi"]                 # (B, T, K)
    eos_logit = mdn_params["eos_logit"].squeeze(-1)  # (B, T)

    # Location loss: -log p(dx, dy | mixture)
    gaussian = gaussian_2d(target_x, target_y, mu1, mu2, sig1, sig2, rho)
    mixture_prob = torch.sum(pi * gaussian, dim=-1)  # (B, T)
    loc_loss_per_step = -torch.log(mixture_prob + 1e-6)  # (B, T)
    loc_loss = (loc_loss_per_step * mask_float).sum() / num_valid

    # Pen loss: BCE for pen up/down
    pen_loss_per_step = nn.functional.binary_cross_entropy_with_logits(
        eos_logit, target_pen, reduction="none"
    )
    pen_loss = (pen_loss_per_step * mask_float).sum() / num_valid

    total_loss = loc_loss + pen_loss

    return {
        "loss": total_loss,
        "loc_loss": loc_loss,
        "pen_loss": pen_loss,
    }
