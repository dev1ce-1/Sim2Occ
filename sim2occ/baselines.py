"""CCSA and Deep CORAL feature losses, and the FDA image transformation."""
import numpy as np
import torch
from torch.nn import functional as F

def covariance(features):
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("CORAL covariance requires at least two feature vectors")
    centered = features - features.mean(dim=0)
    return (centered.T @ centered) / (len(features) - 1)


def coral(target, source_covariance):
    channels = target.shape[1]
    return (covariance(target) - source_covariance).square().sum() / (4 * channels**2)


def ccsa(target, positive, negative, margin=1.0):
    target, positive, negative = (F.normalize(x, dim=-1) for x in (target, positive, negative))
    near = (target - positive).square().sum(dim=-1)
    far = torch.linalg.norm(target - negative, dim=-1)
    return .5 * (near.mean() + (margin - far).clamp(min=0).square().mean())


def fda(source, donor, beta=0.05):
    """Replace low spatial frequencies in the amplitude, retaining source phase."""
    from scipy.fft import irfft2, rfft2

    height, width = source.shape[:2]
    if donor.shape != source.shape:
        import cv2
        donor = cv2.resize(donor, (width, height), interpolation=cv2.INTER_LINEAR)
    band = int(min(height, width) * beta)
    rows = np.r_[0:band + 1, height - band:height]
    spec = rfft2(source.astype(np.float32), axes=(0, 1), workers=1)
    donor_spec = rfft2(donor.astype(np.float32), axes=(0, 1), workers=1)
    block = spec[rows, :band + 1]
    spec[rows, :band + 1] = np.abs(donor_spec[rows, :band + 1]) * np.exp(1j * np.angle(block))
    restored = irfft2(spec, s=(height, width), axes=(0, 1), workers=1)
    return np.rint(restored.clip(0, 255)).astype(np.uint8)
