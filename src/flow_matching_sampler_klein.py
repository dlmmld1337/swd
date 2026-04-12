import torch
import torch.nn.functional as F

from diffusers import Flux2KleinPipeline
from src.utils.train_utils_klein import (
    pack_klein_input,
    patchify_normalize_klein,
    unpack_klein_output,
    vae_encode_spatial_klein,
)

########################################################################################################################
#                                IMG2IMG INFERENCE SAMPLER FOR KLEIN                                                  #
########################################################################################################################


@torch.no_grad()
def sampling_klein_img2img(
    model,
    cond_pixels,
    vae,
    bn_mean,
    bn_std,
    prompt_embeds,
    text_ids,
    solver,
    structural_noise_radius,
    generate_structured_noise_fn,
    guidance_scale=1.0,
    weight_dtype=torch.bfloat16,
):
    """
    4-step scale-wise img2img inference for the Klein student.

    Parameters
    ----------
    model        : student Flux2Transformer2DModel (with LoRA)
    cond_pixels  : (1, 3, H_full, W_full) condition image in [-1, 1], full resolution
    solver       : FlowMatchingSolver with scales and boundary_start_idx set
    generate_structured_noise_fn : callable(spatial_latents, cutoff_radius) -> noise

    Returns
    -------
    latent : (1, C, H, W) final patchified latent (call Flux2KleinPipeline._unpatchify_latents + vae.decode outside)
    """
    device = cond_pixels.device
    dtype = weight_dtype

    min_px = int(
        solver.min_scale * 16
    )  # solver.min_scale is in patchified-latent units; *16 → pixels

    # ── Step 1. Initial latent from structured noise at min scale ──
    cond_min = F.interpolate(cond_pixels, size=(min_px, min_px), mode="area")
    cond_latent_spatial = vae_encode_spatial_klein(cond_min.to(dtype), vae).to(
        device=device, dtype=dtype
    )
    cond_latent = patchify_normalize_klein(cond_latent_spatial, bn_mean, bn_std)

    noise_spatial = generate_structured_noise_fn(
        cond_latent_spatial, cutoff_radius=structural_noise_radius
    ).to(dtype=dtype, device=device)
    latent = patchify_normalize_klein(noise_spatial, bn_mean, bn_std)

    # ── Step 2. Sampler loop (mirrors original SWD inference) ──
    sigmas = solver.noise_scheduler.sigmas[solver.boundary_idx].to(
        device=device, dtype=dtype
    )
    timesteps = solver.noise_scheduler.timesteps[solver.boundary_start_idx].to(
        device=device
    )

    idx_start = 0
    idx_end = len(solver.boundary_idx) - 1
    k = 1

    try:
        next_scale = solver.scales[k]
    except IndexError:
        next_scale = solver.scales[0]

    while True:
        sigma = sigmas[idx_start]
        sigma_next = sigmas[idx_start + 1]
        timestep = timesteps[idx_start]
        T = torch.tensor([timestep], device=device)

        # Pack and call the student
        hidden_states, img_ids, latent_ids, _ = pack_klein_input(latent, cond_latent)
        guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(
                hidden_states=hidden_states.to(dtype),
                encoder_hidden_states=prompt_embeds,
                timestep=T / 1000,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=False,
            )
        pred_tokens = out[0]  # (1, 2*noisy_len, C) — has noisy+cond tokens
        # Trim to noisy-target tokens
        orig_len = latent.shape[2] * latent.shape[3]
        pred_tokens = pred_tokens[:, :orig_len, :]
        noise_pred = unpack_klein_output(pred_tokens, latent_ids)  # (1, C, H, W)

        # Euler step:  x_{t-1} = x_t - sigma * v
        latent = latent - noise_pred * sigma

        if idx_start + 1 == idx_end:
            break

        idx_start += 1

        if int(solver.scales[k - 1]) != int(next_scale):
            latent = F.interpolate(
                latent, size=(int(next_scale), int(next_scale)), mode="bicubic"
            )

        next_px = int(next_scale * 16)
        cond_rescaled = F.interpolate(cond_pixels, size=(next_px, next_px), mode="area")
        cond_latent_spatial = vae_encode_spatial_klein(cond_rescaled.to(dtype), vae).to(
            device=device, dtype=dtype
        )
        cond_latent = patchify_normalize_klein(cond_latent_spatial, bn_mean, bn_std)

        boundary_noise_spatial = generate_structured_noise_fn(
            cond_latent_spatial, cutoff_radius=structural_noise_radius
        ).to(dtype=dtype, device=device)
        boundary_noise = patchify_normalize_klein(
            boundary_noise_spatial, bn_mean, bn_std
        )
        latent = sigma_next * boundary_noise + (1.0 - sigma_next) * latent

        k += 1
        try:
            next_scale = solver.scales[k]
        except IndexError:
            next_scale = solver.scales[-1]

    return latent
