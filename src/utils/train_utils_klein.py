import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils.torch_utils import is_compiled_module

########################################################################################################################
#                                    TRAINING UTILITIES FOR KLEIN IMG2IMG                                             #
########################################################################################################################


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def vae_encode_spatial_klein(pixel_values, vae):
    """VAE-encode pixel values to Klein's raw spatial latent space."""
    with torch.no_grad():
        return vae.encode(pixel_values.to(vae.dtype)).latent_dist.mode()


def patchify_normalize_klein(latents, bn_mean, bn_std):
    """Patchify raw Klein latents and apply batch-norm statistics normalization."""
    latents = Flux2KleinPipeline._patchify_latents(latents)
    latents = (latents - bn_mean.to(latents.device, latents.dtype)) / bn_std.to(
        latents.device, latents.dtype
    )
    return latents


def vae_encode_normalize_klein(pixel_values, vae, bn_mean, bn_std):
    """
    VAE-encode pixel values to Klein's patchified normalised latent space.

    pixel_values : (B, 3, H, W) in [-1, 1]
    returns      : (B, C*4, H//2, W//2) patchified latents
    """
    z = vae_encode_spatial_klein(pixel_values, vae)
    z = patchify_normalize_klein(z, bn_mean, bn_std)
    return z  # dtype = vae.dtype (bf16)


def vae_denormalize_klein(latents, bn_mean, bn_std):
    """Invert Klein latent batch-norm statistics before VAE decode."""
    return latents * bn_std.to(latents.device, latents.dtype) + bn_mean.to(
        latents.device, latents.dtype
    )


def sample_batch_klein(
    args,
    accelerator,
    global_step,
    loader,
    fm_solver,
    vae,
    bn_mean,
    bn_std,
    prompt_embeds,
    text_ids,
    weight_dtype,
):
    """
    Sample one batch of paired img2img data at the appropriate scale boundary.

    Returns
    -------
    model_input          : target latents at current  scale  (B, C, H, W) patchified
    model_input_prev     : target latents at previous scale  (B, C, H, W) patchified
    cond_model_input     : cond   latents at current  scale  (B, C, H, W) patchified
    cond_model_input_spatial : cond latents before patchify/norm at current scale
    batch_prompt_embeds  : (B, T, D)
    batch_text_ids       : (B, T, 4)
    idx_start            : (B,) long tensor — boundary start index
    """
    batch = next(loader)

    # ── Pick boundary (cycles through all boundaries during training) ──
    idx_start = fm_solver.boundary_start_idx[global_step % args.num_boundaries]
    idx_start = torch.tensor([idx_start] * args.train_batch_size).long()

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]
    if scales[0].item() == fm_solver.min_scale:
        previous_scales = scales
    else:
        previous_scales = fm_solver._get_previous_scale(scales)

    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
    cond_pixel_values = batch["cond_pixel_values"].to(
        accelerator.device, dtype=weight_dtype
    )

    # ── Scale-wise downsampling: both images to current and previous scale ──
    # scales are in patchified-latent units; pixels = scales * 16 (VAE×2, patchify×2 reversed)
    pixel_curr = fm_solver.downscale_to_current(pixel_values, scales * 16)
    cond_curr = fm_solver.downscale_to_current(cond_pixel_values, scales * 16)
    pixel_prev = fm_solver.downscale_to_current(pixel_values, previous_scales * 16)
    # cond at previous scale is only needed in the inference sampler; here just return current.

    # ── VAE encode + normalise ──
    model_input = vae_encode_normalize_klein(pixel_curr, vae, bn_mean, bn_std)
    model_input_prev = vae_encode_normalize_klein(pixel_prev, vae, bn_mean, bn_std)
    cond_model_input_spatial = vae_encode_spatial_klein(cond_curr, vae)
    cond_model_input = patchify_normalize_klein(
        cond_model_input_spatial, bn_mean, bn_std
    )

    # ── Broadcast prompt embeddings to batch size ──
    B = args.train_batch_size
    batch_prompt_embeds = prompt_embeds.expand(B, -1, -1).contiguous()
    batch_text_ids = text_ids.expand(B, -1, -1).contiguous()

    return (
        model_input,
        model_input_prev,
        cond_model_input,
        cond_model_input_spatial,
        batch_prompt_embeds,
        batch_text_ids,
        idx_start,
    )


def pack_klein_input(noisy_target, cond_model_input):
    """
    Prepare packed + concatenated hidden_states and img_ids for the Klein transformer.
    Both inputs must have the same spatial dimensions (guaranteed when same scale).

    noisy_target     : (B, C, H, W) patchified latents
    cond_model_input : (B, C, H, W) patchified latents

    Returns
    -------
    hidden_states  : (B, 2*H*W, C) packed cat [noisy | cond]
    img_ids        : (B, 2*H*W, 4) positional IDs
    latent_ids     : (B, H*W,   4) IDs for the noisy-target tokens only (for unpacking)
    orig_noisy_len : int
    """
    device = noisy_target.device
    B = noisy_target.shape[0]

    latent_ids = Flux2KleinPipeline._prepare_latent_ids(noisy_target).to(device)

    cond_model_input_list = [cond_model_input[i : i + 1] for i in range(B)]
    cond_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(device)
    cond_ids = cond_ids.view(B, -1, latent_ids.shape[-1])

    packed_noisy = Flux2KleinPipeline._pack_latents(noisy_target)
    packed_cond = Flux2KleinPipeline._pack_latents(cond_model_input)
    orig_noisy_len = packed_noisy.shape[1]

    hidden_states = torch.cat([packed_noisy, packed_cond], dim=1)
    img_ids = torch.cat([latent_ids, cond_ids], dim=1)

    return hidden_states, img_ids, latent_ids, orig_noisy_len


def unpack_klein_output(output_tokens, latent_ids):
    """
    Unpack trimmed output tokens back to spatial (B, C, H, W) patchified latents.

    output_tokens : (B, orig_noisy_len, C) — already trimmed to target tokens
    latent_ids    : (B, orig_noisy_len, 4)
    """
    if output_tokens.shape[1] != latent_ids.shape[1]:
        raise ValueError(
            f"Token/id length mismatch in Klein unpack: {output_tokens.shape[1]} vs {latent_ids.shape[1]}"
        )
    return Flux2KleinPipeline._unpack_latents_with_ids(output_tokens, latent_ids)
