import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift


def calc_psf(u, phase, wvl, z, sensor_res, ps):
    """
    Compute the point spread function (PSF) by propagating the wavefront.

    Args:
        u (torch.Tensor): Incident complex wavefront [H, W].
        phase (torch.Tensor): Phase modulation of the metalens [H, W].
        wavelength (float): Wavelength in micrometers.
        z (float): Propagation distance in micrometers.
        sensor_res (float): Sensor resolution scaling factor.
        pixel_size (float): Pixel size in micrometers.

    Returns:
        torch.Tensor: Computed PSF with shape [ks, ks] or [H, W].
    """

    phi_mod = torch.remainder(phase, 2 * torch.pi)
    wavefront = u * torch.exp(1j * phi_mod)

    # Wave propagation using the Angular Spectrum Method
    h, w = wavefront.shape
    wavefront = F.pad(
        wavefront.unsqueeze(0).unsqueeze(0),
        [h // 2, h // 2, w // 2, w // 2],
        mode="constant",
        value=0,
    )


    # use different propagation method
    # sensor_field = AngularSpectrumMethod(wavefront, z = z, wvln = wvl, ps = ps * 1e3, n = 1.0, padding = False)
    sensor_field = FresnelDiffraction(wavefront, z = z, wvln = wvl, ps = ps * 1e3, n = 1.0, padding = False)

    # Calculate PSF intensity
    psf_inten = sensor_field.abs() ** 2
    psf_inten = (
        F.interpolate(
            psf_inten,
            scale_factor=sensor_res / h,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    # Crop the PSF
    ks = 201        # kernel size for cropping the PSF
    psfc = torch.tensor([0, 0], dtype=torch.float32)
    if ks is not None:
        h, w = psf_inten.shape[-2:]
        psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
        psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

        # Pad to avoid invalid edge region
        psf_inten_pad = F.pad(
            psf_inten,
            [ks // 2, ks // 2, ks // 2, ks // 2],
            mode="constant",
            value=0,
        )
        psf = psf_inten_pad[
            psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
        ]
    else:
        h, w = psf_inten.shape[-2:]
        psf = psf_inten[
            int(h / 2 - h / 4) : int(h / 2 + h / 4),
            int(w / 2 - w / 4) : int(w / 2 + w / 4),
        ]
    # Normalize and convert to float precision
    psf /= psf.sum()  # shape of [ks, ks] or [h, w]
    # psf = diff_float(psf)

    return psf

def AngularSpectrumMethod(u, z, wvln=0.489, ps=0.001, n=1.0, padding=True):
    """Angular spectrum method.

    Reference:
        [1] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py#L293
        [2] https://blog.csdn.net/zhenpixiaoyang/article/details/111569495

    Args:
        u (tesor): complex field, shape [H, W] or [B, 1, H, W]
        z (float): propagation distance in [mm]
        wvln (float): wavelength in [um]
        ps (float): pixel size in [mm]
        n (float): refractive index
        padding (bool): padding or not

    Returns:
        u: complex field, shape [H, W] or [B, 1, H, W]
    """
    assert wvln > 0.1 and wvln < 10, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3  # [um] to [mm]
    k = 2 * n * np.pi / wvln_mm  # [mm]-1

    # Shape
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)

    # Padding
    if padding:
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad), mode="constant", value=0)
    else:
        Wimg, Himg = Worg, Horg

    # Propagation with angular spectrum method
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5 / ps, 0.5 / ps, Wimg, device=u.device),
        torch.linspace(0.5 / ps, -0.5 / ps, Himg, device=u.device),
        indexing="xy",
    )
    square_root = torch.sqrt(1 - wvln_mm**2 * (fx**2 + fy**2))
    H = torch.exp(1j * k * z * square_root)
    H = fftshift(H)

    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    del fx, fy
    return u


def FresnelDiffraction(u, z, wvln, ps, n=1.0, padding=True, TF=None):
    """Fresnel propagation with FFT.

    Ref: Computational fourier optics : a MATLAB tutorial
         https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m

    Args:
        u: complex field, shape [H, W] or [B, C, H, W]
        z (float): propagation distance
        wvln (float): wavelength in [um]
        ps (float): pixel size
        n (float): refractive index
        padding (bool): padding or not
        TF (bool): transfer function or impulse response
    """
    # Padding
    if padding:
        try:
            _, _, Worg, Horg = u.shape
        except:
            Horg, Worg = u.shape
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        _, _, Wimg, Himg = u.shape

    # Compute H function
    assert wvln > 0.1 and wvln < 10, "wvln should be in [um]."
    wvln_mm = wvln * 1e-3  # [um] to [mm]
    k = 2 * n * np.pi / wvln_mm
    x, y = torch.meshgrid(
        torch.linspace(-0.5 * Wimg * ps, 0.5 * Himg * ps, Wimg + 1, device=u.device)[
            :-1
        ],
        torch.linspace(0.5 * Wimg * ps, -0.5 * Himg * ps, Himg + 1, device=u.device)[
            :-1
        ],
        indexing="xy",
    )
    fx, fy = torch.meshgrid(
        torch.linspace(-0.5 / ps, 0.5 / ps, Wimg + 1, device=u.device)[:-1],
        torch.linspace(-0.5 / ps, 0.5 / ps, Himg + 1, device=u.device)[:-1],
        indexing="xy",
    )

    # TF or IR method
    # Computational fourier optics. Chapter 5, section 5.1.
    if TF is None:
        if ps > wvln_mm * np.abs(z) / (Wimg * ps):
            TF = True
        else:
            TF = False

    if TF:
        H = np.sqrt(n) * torch.exp(-1j * np.pi * wvln_mm * z * (fx**2 + fy**2) / n)
        H = fftshift(H)
    else:    
        h = n / (1j * wvln_mm * z) * torch.exp(1j * k / (2 * z) * (x**2 + y**2))
        H = fft2(fftshift(h)) * ps**2

    # Fourier transformation
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # Remove padding
    if padding:
        u = u[..., Wpad:-Wpad, Hpad:-Hpad]

    return u