"""
Structured-noise utilities for the Klein img2img distillation pipeline.
Copied from auto_remaster/sandbox/diffusers_flux2/train_dreambooth_lora_flux2_klein_img2img.py
"""

import torch


def create_frequency_soft_cutoff_mask(
    height: int,
    width: int,
    cutoff_radius: float,
    transition_width: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")

    center_u, center_v = height // 2, width // 2
    frequency_radius = torch.sqrt(
        (u - center_u).float() ** 2 + (v - center_v).float() ** 2
    )

    mask = torch.exp(
        -((frequency_radius - cutoff_radius) ** 2) / (2 * transition_width**2)
    )
    mask = torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)
    return mask


def _clip_magnitude(t, clip_percentile=0.95):
    threshold = torch.quantile(t, clip_percentile)
    return torch.clamp(t, max=threshold)


def generate_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    input_noise: torch.Tensor = None,
    sampling_method: str = "fft",
) -> torch.Tensor:
    """
    Generate structured noise for a batch of images using frequency-domain phase mixing.

    Args:
        image_batch      : (B, C, H, W) reference tensor (patchified latents or pixel space)
        cutoff_radius    : frequency cutoff in FFT-space pixels; None = full structure
        sampling_method  : "fft" | "cdf" | "two-gaussian"

    Returns:
        (B, C, H, W) structured noise, same dtype as image_batch
    """
    assert sampling_method in ["fft", "cdf", "two-gaussian"]

    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch_f = image_batch.float()

    pad_h = (int(height * (pad_factor - 1)) // 2) * 2
    pad_w = (int(width * (pad_factor - 1)) // 2) * 2

    padded = torch.nn.functional.pad(
        image_batch_f,
        (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
        mode="reflect",
    )
    pH, pW = height + pad_h, width + pad_w

    if cutoff_radius is not None:
        cr = min(pH / 2, pW / 2, cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            pH, pW, cr, transition_width, device
        )
    else:
        freq_mask = torch.ones(pH, pW, device=device)

    fft_s = torch.fft.fftshift(torch.fft.fft2(padded, dim=(-2, -1)), dim=(-2, -1))
    image_phases = _clip_magnitude(torch.angle(fft_s))
    image_magnitudes = torch.abs(fft_s)  # noqa: F841 — used implicitly via shape

    if input_noise is not None:
        noise_padded = torch.nn.functional.pad(
            input_noise.float(),
            (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
            mode="reflect",
        )
    else:
        noise_padded = torch.randn_like(padded)

    if sampling_method == "fft":
        ns = torch.fft.fftshift(
            torch.fft.fft2(noise_padded, dim=(-2, -1)), dim=(-2, -1)
        )
        noise_magnitudes = _clip_magnitude(torch.abs(ns))
        noise_phases = torch.angle(ns)
    else:
        N = pH * pW
        rayleigh_scale = (N / 2) ** 0.5
        if sampling_method == "cdf":
            u = torch.rand_like(image_phases)
            noise_magnitudes = _clip_magnitude(
                rayleigh_scale * torch.sqrt(-2.0 * torch.log(u + 1e-12))
            )
        else:  # two-gaussian
            u1, u2 = torch.randn_like(image_phases), torch.randn_like(image_phases)
            noise_magnitudes = _clip_magnitude(
                rayleigh_scale * torch.sqrt(u1**2 + u2**2)
            )
        if input_noise is not None:
            ns = torch.fft.fftshift(
                torch.fft.fft2(noise_padded, dim=(-2, -1)), dim=(-2, -1)
            )
            noise_magnitudes = _clip_magnitude(torch.abs(ns))
            noise_phases = torch.angle(ns)
        else:
            noise_phases = torch.rand_like(image_phases) * 2 * torch.pi - torch.pi

    noise_magnitudes = noise_magnitudes * noise_std

    fm = freq_mask.unsqueeze(0).unsqueeze(0)
    mixed_phases = fm * image_phases + (1.0 - fm) * noise_phases

    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)
    out_padded = torch.real(
        torch.fft.ifft2(torch.fft.ifftshift(fft_combined, dim=(-2, -1)), dim=(-2, -1))
    )

    clamp_mask = ((out_padded < -5) | (out_padded > 5)).float()
    out_padded = out_padded * (1 - clamp_mask) + noise_padded * clamp_mask

    out = out_padded[
        :, :, pad_h // 2 : pad_h // 2 + height, pad_w // 2 : pad_w // 2 + width
    ]
    return out.to(dtype)
