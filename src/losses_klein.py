import torch
import torch.nn.functional as F
from contextlib import contextmanager

from src.utils.train_utils_klein import (
    pack_klein_input,
    patchify_normalize_klein,
    unpack_klein_output,
)


@contextmanager
def _teacher_on_device(teacher, device):
    """Temporarily move the CPU-offloaded teacher to `device` for inference."""
    teacher.to(device)
    try:
        yield teacher
    finally:
        teacher.to("cpu")
        torch.cuda.empty_cache()


########################################################################################################################
#                        THE LOSSES NEEDED FOR KLEIN IMG2IMG DISTILLATION                                             #
#  Structure mirrors losses.py exactly; changes are:                                                                   #
#   • No pooled_prompt_embeds / uncond_prompt_embeds (Klein has none)                                                  #
#   • No disable_adapter() — teacher is a frozen separate object                                                       #
#   • CFG via guidance parameter (one forward instead of two)                                                          #
#   • Klein forward wrapped by call_klein_transformer() helper                                                         #
#   • Structured noise (generate_structured_noise_batch_vectorized) instead of randn_like                             #
########################################################################################################################


# ── re-export mmd_loss_ from losses.py so callers only need one import ──
from src.losses import gan_loss_fn, mmd_loss_  # noqa: F401


# ----------------------------------------------------------------------------------------------------------------------
def call_klein_transformer(
    transformer,
    noisy_target,
    cond_model_input,
    prompt_embeds,
    text_ids,
    timestep,
    guidance_scale,
    classify_index_block=None,
    return_only_features=False,
    return_features=True,
):
    """
    One forward pass through a Klein transformer (student, teacher, or fake/discriminator).

    Handles packing/concat/unpacking internally so that callers work with spatial
    patchified latents (B, C, H, W) throughout — same as the SD3 losses convention.

    Returns
    -------
    ((spatial_pred,), features)  when return_only_features=False
    features                     when return_only_features=True

    spatial_pred : (B, C, H, W) patchified latents
    features     : list[Tensor(B, orig_noisy_len, inner_dim)]
    """
    B, C, H, W = noisy_target.shape
    device = noisy_target.device
    dtype = next(transformer.parameters()).dtype

    hidden_states, img_ids, latent_ids, orig_noisy_len = pack_klein_input(
        noisy_target, cond_model_input
    )

    guidance = torch.full([B], guidance_scale, device=device, dtype=dtype)

    base_kwargs = dict(
        hidden_states=hidden_states.to(dtype),
        encoder_hidden_states=prompt_embeds.to(dtype),
        timestep=(timestep / 1000).to(dtype),
        img_ids=img_ids,
        txt_ids=text_ids,
        guidance=guidance,
        return_dict=False,
    )

    # TransformerClsKlein (fake/discriminator) has forward_with_feature_extraction_klein
    # injected, which accepts the extra kwargs.  Plain Flux2Transformer2DModel (student,
    # teacher) does not — pass only the standard kwargs for those.
    from src.transformer_with_discriminator_klein import TransformerClsKlein

    is_cls = isinstance(transformer, TransformerClsKlein)

    if is_cls:
        output = transformer(
            **base_kwargs,
            classify_index_block=classify_index_block or [],
            return_only_features=return_only_features,
            return_features=return_features,
        )
    else:
        output = transformer(**base_kwargs)

    if is_cls:
        if return_only_features:
            return output  # list of feature tensors
        if return_features:
            pred_tokens, features = output
            spatial_pred = unpack_klein_output(pred_tokens[0], latent_ids)
            return (spatial_pred,), features
        else:
            # return_features=False → TransformerClsKlein returns a plain 1-tuple
            spatial_pred = unpack_klein_output(output[0], latent_ids)
            return (spatial_pred,), []
    else:
        # Plain model: output is a 1-tuple (tensor,)
        pred_tokens = output[0][:, :orig_noisy_len, :]
        spatial_pred = unpack_klein_output(pred_tokens, latent_ids)
        return (spatial_pred,), []


# ----------------------------------------------------------------------------------------------------------------------
def fake_diffusion_loss_klein(
    transformer,
    transformer_fake,
    prompt_embeds,
    text_ids,
    model_input,
    cond_model_input,
    timesteps_start,
    idx_start,
    optimizer,
    params_to_optimize,
    weight_dtype,
    noise_scheduler,
    fm_solver,
    accelerator,
    args,
    generate_structured_noise_fn,
    model_input_down=None,
    cond_model_input_spatial=None,
    bn_mean=None,
    bn_std=None,
):
    """
    Train the fake/discriminator network.
    Mirrors fake_diffusion_loss() from losses.py.
    """
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer_fake.train()
    transformer.eval()

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(
            model_input, scales
        )
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    # Structured noise seeded from cond spatial latents
    if cond_model_input_spatial is not None:
        noise_spatial = generate_structured_noise_fn(
            cond_model_input_spatial, cutoff_radius=args.structural_noise_radius
        ).to(device=model_input.device, dtype=cond_model_input_spatial.dtype)
        noise = patchify_normalize_klein(noise_spatial, bn_mean, bn_std).to(
            dtype=model_input.dtype, device=model_input.device
        )
    else:
        noise = torch.randn_like(model_input_prev)

    noisy_model_input_curr = noise_scheduler.scale_noise(
        model_input_prev, timesteps_start, noise
    )

    with torch.no_grad(), accelerator.autocast():
        model_pred, _ = call_klein_transformer(
            transformer,
            noisy_model_input_curr,
            cond_model_input,
            prompt_embeds,
            text_ids,
            timesteps_start,
            guidance_scale=args.cfg_teacher,
            classify_index_block=[],
            return_features=False,
        )

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred[0].device)[
        :, None, None, None
    ]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred[0]

    ## Apply random diffusion noise to fake sample (for discriminator training)
    idx_noisy = torch.randint(0, len(noise_scheduler.timesteps), (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)
    sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred[0].device)[
        :, None, None, None
    ]

    noise2 = torch.randn_like(fake_sample)
    noisy_fake_sample = noise_scheduler.scale_noise(
        fake_sample, timesteps_noisy, noise2
    )
    ## ---------------------------------------------------------------------------

    ## STEP 2. Predict with fake net + diffusion loss
    ## ---------------------------------------------------------------------------
    with accelerator.autocast():
        fake_pred, inner_features_fake = call_klein_transformer(
            transformer_fake,
            noisy_fake_sample,
            cond_model_input,
            prompt_embeds,
            text_ids,
            timesteps_noisy,
            guidance_scale=args.cfg_teacher,
            classify_index_block=args.cls_blocks,
            return_only_features=False,
            return_features=True,
        )

    fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred[0]
    loss = F.mse_loss(fake_pred_x0.float(), fake_sample.float(), reduction="mean")
    ## ---------------------------------------------------------------------------

    ## STEP 3. Real features + GAN loss
    ## ---------------------------------------------------------------------------
    noise3 = torch.randn_like(model_input)
    noisy_true_sample = noise_scheduler.scale_noise(
        model_input, timesteps_noisy, noise3
    )
    with accelerator.autocast():
        inner_features_true = call_klein_transformer(
            transformer_fake,
            noisy_true_sample,
            cond_model_input,
            prompt_embeds,
            text_ids,
            timesteps_noisy,
            guidance_scale=args.cfg_teacher,
            classify_index_block=args.cls_blocks,
            return_only_features=True,
        )
    gan_loss = gan_loss_fn(
        transformer_fake.module.cls_pred_branch,
        inner_features_fake,
        inner_features_true,
    )
    loss = loss + gan_loss * args.disc_cls_loss_weight
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()

    transformer.train()
    transformer_fake.eval()
    return avg_loss


# ----------------------------------------------------------------------------------------------------------------------
def generator_loss_klein(
    transformer,
    transformer_fake,
    transformer_teacher,
    prompt_embeds,
    text_ids,
    model_input,
    cond_model_input,
    timesteps_start,
    idx_start,
    optimizer,
    params_to_optimize,
    weight_dtype,
    noise_scheduler,
    fm_solver,
    accelerator,
    args,
    generate_structured_noise_fn,
    model_input_down=None,
    cond_model_input_spatial=None,
    bn_mean=None,
    bn_std=None,
):
    """
    Train the student transformer.
    Mirrors generator_loss() from losses.py.
    """
    ## STEP 1. Student prediction → fake sample
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()
    transformer_fake.eval()

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(
            model_input, scales
        )
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    if cond_model_input_spatial is not None:
        noise_spatial = generate_structured_noise_fn(
            cond_model_input_spatial, cutoff_radius=args.structural_noise_radius
        ).to(device=model_input.device, dtype=cond_model_input_spatial.dtype)
        noise = patchify_normalize_klein(noise_spatial, bn_mean, bn_std).to(
            dtype=model_input.dtype, device=model_input.device
        )
    else:
        noise = torch.randn_like(model_input_prev)

    noisy_model_input_curr = noise_scheduler.scale_noise(
        model_input_prev, timesteps_start, noise
    )

    with accelerator.autocast():
        model_pred, _ = call_klein_transformer(
            transformer,
            noisy_model_input_curr,
            cond_model_input,
            prompt_embeds,
            text_ids,
            timesteps_start,
            guidance_scale=args.cfg_teacher,
            classify_index_block=[],
            return_features=False,
        )

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred[0].device)[
        :, None, None, None
    ]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred[0]
    true_sample = model_input

    if args.do_dmd_loss:
        ## DMD noise timestep
        idx_noisy = torch.randint(
            args.dmd_noise_start_idx, args.dmd_noise_end_idx, (len(fake_sample),)
        )
        sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred[0].device)[
            :, None, None, None
        ]
        timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(
            device=model_input.device
        )

        noise2 = torch.randn_like(fake_sample)
        noisy_fake_sample = noise_scheduler.scale_noise(
            fake_sample, timesteps_noisy, noise2
        )
        ## ---------------------------------------------------------------------------

        ## STEP 2. DMD loss
        ## ---------------------------------------------------------------------------
        _device = model_input.device
        with _teacher_on_device(
            transformer_teacher, _device
        ), torch.no_grad(), accelerator.autocast():
            # Teacher: frozen merged LoRA model, CPU-offloaded
            real_pred, _ = call_klein_transformer(
                transformer_teacher,
                noisy_fake_sample,
                cond_model_input,
                prompt_embeds,
                text_ids,
                timesteps_noisy,
                guidance_scale=args.cfg_teacher,
                classify_index_block=[],
                return_features=False,
            )
            real_pred_x0 = noisy_fake_sample - sigma_noisy * real_pred[0]

            fake_net_pred, _ = call_klein_transformer(
                transformer_fake,
                noisy_fake_sample,
                cond_model_input,
                prompt_embeds,
                text_ids,
                timesteps_noisy,
                guidance_scale=args.cfg_teacher,
                classify_index_block=[],
                return_features=False,
            )
            fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_net_pred[0]

        p_real = fake_sample - real_pred_x0
        p_fake = fake_sample - fake_pred_x0

        grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
        grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(
            fake_sample.float(),
            (fake_sample - grad).detach().float(),
            reduction="mean",
        )
        loss = args.dmd_loss_weight * loss
    else:
        loss = torch.zeros(1, device=fake_sample.device, dtype=fake_sample.dtype)
        noisy_fake_sample = None
        timesteps_noisy = None
        sigma_noisy = None
    ## ---------------------------------------------------------------------------

    ## STEP 3. GAN loss
    ## ---------------------------------------------------------------------------
    trainable_keys = [
        n for n, p in transformer_fake.named_parameters() if p.requires_grad
    ]
    transformer_fake.requires_grad_(False).eval()

    if args.do_gan_loss:
        assert (
            noisy_fake_sample is not None
        ), "GAN loss requires do_dmd_loss=True (needs noisy_fake_sample)"
        inner_features_fake = call_klein_transformer(
            transformer_fake,
            noisy_fake_sample,
            cond_model_input,
            prompt_embeds,
            text_ids,
            timesteps_noisy,
            guidance_scale=args.cfg_teacher,
            classify_index_block=args.cls_blocks,
            return_only_features=True,
        )
        gan_loss = gan_loss_fn(
            transformer_fake.module.cls_pred_branch,
            inner_features_fake,
            inner_features_true=None,
        )
        loss = loss + gan_loss * args.gen_cls_loss_weight
    ## ---------------------------------------------------------------------------

    ## STEP 4. MMD loss
    ## ---------------------------------------------------------------------------
    if args.do_mmd_loss:
        idx_noisy_mmd = torch.randint(
            args.mmd_noise_start_idx, args.mmd_noise_end_idx, (len(fake_sample),)
        )
        timesteps_noisy_mmd = noise_scheduler.timesteps[idx_noisy_mmd].to(
            device=model_input.device
        )

        noise3 = torch.randn_like(fake_sample)
        noisy_true_sample = noise_scheduler.scale_noise(
            true_sample, timesteps_noisy_mmd, noise3
        )
        noisy_fake_sample_mmd = noise_scheduler.scale_noise(
            fake_sample, timesteps_noisy_mmd, noise3
        )

        with accelerator.autocast():
            inner_features_fake_mmd = call_klein_transformer(
                transformer_fake,
                noisy_fake_sample_mmd,
                cond_model_input,
                prompt_embeds,
                text_ids,
                timesteps_noisy_mmd,
                guidance_scale=args.cfg_teacher,
                classify_index_block=args.mmd_blocks,
                return_only_features=True,
            )
            inner_features_real_mmd = call_klein_transformer(
                transformer_fake,
                noisy_true_sample,
                cond_model_input,
                prompt_embeds,
                text_ids,
                timesteps_noisy_mmd,
                guidance_scale=args.cfg_teacher,
                classify_index_block=args.mmd_blocks,
                return_only_features=True,
            )

        # call_klein_transformer returns a list of [B,N,D] tensors (one per block);
        # mmd_loss_ expects a single [B, N, D] tensor — concatenate along the token dim.
        inner_features_real_mmd = torch.cat(inner_features_real_mmd, dim=1)
        inner_features_fake_mmd = torch.cat(inner_features_fake_mmd, dim=1)

        mmd_loss = mmd_loss_(
            inner_features_real_mmd,
            inner_features_fake_mmd,
            kernel=args.mmd_kernel,
            sigma=args.mmd_rbf_sigma,
            do_batch_mmd=args.do_batch_mmd,
            c=args.huber_c,
        )
        loss = loss + args.mmd_loss_weight * mmd_loss
    else:
        mmd_loss = torch.zeros_like(loss)
    ## ---------------------------------------------------------------------------

    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_mmd_loss = accelerator.gather(mmd_loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    ## ---------------------------------------------------------------------------

    # Restore trainable parameters in fake model
    transformer_fake.module.cls_pred_branch.requires_grad_(True).train()
    for n, p in transformer_fake.named_parameters():
        if n in trainable_keys:
            p.requires_grad_(True)

    return avg_loss, avg_mmd_loss
