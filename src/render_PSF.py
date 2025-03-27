import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F


# ================================================
# PSF convolution
# ================================================
def render_psf(img, psf):
    """Render rgb image batch with rgb PSF.

    Args:
        img (torch.Tensor): [B, C, H, W]
        psf (torch.Tensor): [C, ks, ks]

    Returns:
        img_render (torch.Tensor): [B, C, H, W]
    """
    # Same type
    img = img.to(psf.dtype)  # Convert img to match the dtype of psf
    # Padding
    _, ks, ks = psf.shape
    padding = int(ks / 2)
    psf = torch.flip(psf, [1, 2])  # flip the PSF because F.conv2d use cross-correlation
    psf = psf.unsqueeze(1)  # shape [C, 1, ks, ks]
    img_pad = F.pad(img, (padding, padding, padding, padding), mode="reflect")

    # Convolution
    img_render = F.conv2d(img_pad, psf, groups=img.shape[1], padding=0, bias=None)
    return img_render
