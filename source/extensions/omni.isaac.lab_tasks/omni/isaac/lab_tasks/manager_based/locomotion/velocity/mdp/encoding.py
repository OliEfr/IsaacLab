import torch
from torch import Tensor


def sinusodial_encoding_3d(position: torch.Tensor, period: torch.Tensor):
    """
    Computes a sinusoidal encoding for a 3D position.

    Args:
        position: Tensor of shape [3], representing the (x, y, z) position.
        period: Tensor of shape [3], representing the period along each axis.

    Returns:
        A tensor containing the sinusoidal encoding of shape [6].
    """
    period[period == 0.0] = float("inf")
    frequency = 2 * torch.pi / period  # Convert period to frequency
    angles = position * frequency  # Compute the scaled position

    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return encoding
