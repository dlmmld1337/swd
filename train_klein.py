import types
from pathlib import Path

import torch
import torchvision
from accelerate.logging import get_logger
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from tqdm.auto import tqdm

from src.dataset_klein import get_loader_klein
from src.flow_matching_sampler import FlowMatchingSolver
from src.flow_matching_sampler_klein import sampling_klein_img2img
from src.losses_klein import fake_diffusion_loss_klein, generator_loss_klein
from src.transformer_with_discriminator_klein import (
    TransformerClsKlein,
    forward_with_feature_extraction_klein,
)
from src.utils.prepare_utils import prepare_accelerator, prepare_optimizer
from src.utils.prepare_utils_klein import prepare_models_klein
from src.utils.setup_utils import (
    load_if_exist,
    prepare_3rd_party,
    saving,
    seed_everything,
    set_tf32,
)
from src.utils.structured_noise import generate_structured_noise_batch_vectorized
from src.utils.train_utils_klein import (
    sample_batch_klein,
    unwrap_model,
    vae_denormalize_klein,
)

logger = get_logger(__name__)


########################################################################################################################
#                                       VALIDATION HELPERS                                                             #
########################################################################################################################


@torch.no_grad()
def log_validation_klein(
    transformer_student,
    vae,
    bn_mean,
    bn_std,
    prompt_embeds,
    text_ids,
    fm_solver,
    train_dataset_hf,
    accelerator,
    args,
    global_step,
    weight_dtype,
):
    """Generate 4 triplet images [condition | generated | target] and log to tensorboard."""
    import random
    from torchvision import transforms

    transformer_student.eval()
    validation_dir = Path(args.output_dir) / "validation" / f"step-{global_step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)

    student_model = unwrap_model(transformer_student, accelerator)

    rng = random.Random(args.seed)
    n_samples = min(4, len(train_dataset_hf))
    indices = rng.sample(range(len(train_dataset_hf)), n_samples)

    validation_transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ]
    )
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    triplets = []
    for sample_index, idx in enumerate(indices):
        example = train_dataset_hf[idx]
        cond_pil = validation_transform(example["input_image"].convert("RGB"))
        target_pil = validation_transform(example["edited_image"].convert("RGB"))

        cond_tensor = (
            to_tensor(cond_pil).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        )

        sampled_latents = sampling_klein_img2img(
            model=student_model,
            cond_pixels=cond_tensor,
            vae=vae,
            bn_mean=bn_mean,
            bn_std=bn_std,
            prompt_embeds=prompt_embeds.to(weight_dtype),
            text_ids=text_ids,
            solver=fm_solver,
            structural_noise_radius=getattr(args, "structural_noise_radius", 100),
            generate_structured_noise_fn=generate_structured_noise_batch_vectorized,
            guidance_scale=args.cfg_teacher,
            weight_dtype=weight_dtype,
        )

        decoded_latents = vae_denormalize_klein(sampled_latents, bn_mean, bn_std)
        decoded_latents = Flux2KleinPipeline._unpatchify_latents(decoded_latents)
        generated_tensor = vae.decode(
            decoded_latents.to(device=accelerator.device, dtype=vae.dtype),
            return_dict=False,
        )[0]
        generated_tensor = ((generated_tensor.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        generated_tensor = generated_tensor[0].cpu()

        cond_vis = transforms.ToTensor()(cond_pil)
        target_vis = transforms.ToTensor()(target_pil)
        triplet = torch.cat([cond_vis, generated_tensor, target_vis], dim=2)
        triplets.append(triplet)
        torchvision.utils.save_image(
            triplet,
            validation_dir / f"sample_{sample_index:02d}.png",
        )

    if accelerator.is_main_process and len(triplets) > 0:
        grid = torchvision.utils.make_grid(triplets, nrow=1)  # stack vertically
        torchvision.utils.save_image(grid, validation_dir / "grid.png")

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_image("validation/triplets", grid, global_step)
            elif tracker.name == "wandb":
                import wandb

                tracker.log(
                    {
                        "validation_triplets": [
                            wandb.Image(
                                grid.permute(1, 2, 0).numpy(),
                                caption=f"step {global_step}",
                            )
                        ]
                    }
                )

        logger.info(f"Saved validation images to {validation_dir}")
    transformer_student.train()


########################################################################################################################
#                                           TRAINING                                                                   #
########################################################################################################################


def train(args):
    ## PREPARATION STAGE
    ## -----------------------------------------------------------------------------------------------
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelerator(args, logging_dir, False)

    seed_everything(args.seed, accelerator.process_index, accelerator.num_processes)
    set_tf32(tf32=True)

    ## Load models
    (
        transformer,
        transformer_teacher,
        transformer_fake,
        vae,
        bn_mean,
        bn_std,
        prompt_embeds,
        text_ids,
        noise_scheduler,
        weight_dtype,
    ) = prepare_models_klein(args, accelerator)

    ## Scale-wise flow matching solver.
    ## Klein patchifies latents (2×2 blocks), so spatial dims = VAE-latent scale // 2.
    ## Store patchified scales so downscale/upscale helpers work on the actual tensors.
    patchified_scales = [s // 2 for s in args.scales]

    ## Klein uses use_dynamic_shifting=True which requires `mu` in set_timesteps.
    ## Patch the scheduler so FlowMatchingSolver.__init__ can call set_timesteps normally.
    ## mu is computed from the max-scale sequence length (noisy tokens only).
    _max_seq_len = patchified_scales[-1] ** 2
    _orig_set_timesteps = noise_scheduler.set_timesteps

    def _set_timesteps_with_mu(num_inference_steps, **kwargs):
        if "mu" not in kwargs:
            kwargs["mu"] = compute_empirical_mu(
                image_seq_len=_max_seq_len, num_steps=num_inference_steps
            )
        return _orig_set_timesteps(num_inference_steps, **kwargs)

    noise_scheduler.set_timesteps = _set_timesteps_with_mu

    fm_solver = FlowMatchingSolver(
        noise_scheduler,
        args.num_timesteps,
        args.num_boundaries,
        patchified_scales,
        args.boundaries,
    )

    ## Inject feature-extraction forward into the fake/discriminator transformer.
    ## PEFT wraps __call__ and calls get_base_model().forward, so we must patch
    ## the base model's forward — not the PeftModel wrapper.
    transformer_fake.get_base_model().forward = types.MethodType(
        forward_with_feature_extraction_klein, transformer_fake.get_base_model()
    )
    transformer_fake = TransformerClsKlein(args, transformer_fake)

    ## Resume / initialise checkpoints
    initial_global_step = load_if_exist(args, accelerator, transformer, is_student=True)
    _ = load_if_exist(args, accelerator, transformer_fake, is_student=False)
    transformer, transformer_fake = accelerator.prepare(transformer, transformer_fake)

    ## Optimizers
    optimizer, params_to_optimize = prepare_optimizer(
        args, transformer, is_student=True
    )
    optimizer_fake, params_to_optimize_fake = prepare_optimizer(
        args, transformer_fake, is_student=False
    )
    optimizer, optimizer_fake = accelerator.prepare(optimizer, optimizer_fake)

    ## Dataset loader
    train_dataloader, train_dataset = get_loader_klein(
        batch_size=args.train_batch_size,
        resolution=getattr(args, "resolution", 512),
    )
    # Keep reference to underlying HF dataset for validation image sampling
    train_dataset_hf = train_dataset.dataset

    ## Tracker
    prepare_3rd_party(args, accelerator)
    ## -----------------------------------------------------------------------------------------------

    ## TRAINING STAGE
    ## -----------------------------------------------------------------------------------------------
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running Klein img2img distillation *****")
    logger.info(f"  Dataset size = {len(train_dataset)}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Scale boundaries = {args.scales}")
    logger.info(f"  Total steps = {args.max_train_steps}")

    global_step = initial_global_step
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    assert transformer.training

    while global_step < args.max_train_steps:

        (
            model_input,
            model_input_prev,
            cond_model_input,
            cond_model_input_spatial,
            batch_prompt_embeds,
            batch_text_ids,
            idx_start,
        ) = sample_batch_klein(
            args,
            accelerator,
            global_step,
            train_dataloader,
            fm_solver,
            vae,
            bn_mean,
            bn_std,
            prompt_embeds,
            text_ids,
            weight_dtype,
        )
        timesteps_start = noise_scheduler.timesteps[idx_start].to(
            device=model_input.device
        )

        ### Fake / Discriminator loss
        ### ----------------------------------------------------
        if args.do_dmd_loss:
            for _ in range(args.n_steps_fake_dmd):
                avg_dmd_fake_loss = fake_diffusion_loss_klein(
                    transformer,
                    transformer_fake,
                    batch_prompt_embeds,
                    batch_text_ids,
                    model_input,
                    cond_model_input,
                    timesteps_start,
                    idx_start,
                    optimizer_fake,
                    params_to_optimize_fake,
                    weight_dtype,
                    noise_scheduler,
                    fm_solver,
                    accelerator,
                    args,
                    generate_structured_noise_fn=generate_structured_noise_batch_vectorized,
                    model_input_down=model_input_prev,
                    cond_model_input_spatial=cond_model_input_spatial,
                    bn_mean=bn_mean,
                    bn_std=bn_std,
                )

                (
                    model_input,
                    model_input_prev,
                    cond_model_input,
                    cond_model_input_spatial,
                    batch_prompt_embeds,
                    batch_text_ids,
                    idx_start,
                ) = sample_batch_klein(
                    args,
                    accelerator,
                    global_step,
                    train_dataloader,
                    fm_solver,
                    vae,
                    bn_mean,
                    bn_std,
                    prompt_embeds,
                    text_ids,
                    weight_dtype,
                )
                timesteps_start = noise_scheduler.timesteps[idx_start].to(
                    device=model_input.device
                )
        else:
            avg_dmd_fake_loss = torch.zeros(1)
        ### ----------------------------------------------------

        ### Generator (student) loss
        ### ----------------------------------------------------
        avg_dmd_loss, avg_mmd_loss = generator_loss_klein(
            transformer,
            transformer_fake,
            transformer_teacher,
            batch_prompt_embeds,
            batch_text_ids,
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
            generate_structured_noise_fn=generate_structured_noise_batch_vectorized,
            model_input_down=model_input_prev,
            cond_model_input_spatial=cond_model_input_spatial,
            bn_mean=bn_mean,
            bn_std=bn_std,
        )
        ### ----------------------------------------------------

        progress_bar.update(1)
        global_step += 1

        ### Saving checkpoint
        ### ----------------------------------------------------
        if accelerator.is_main_process and global_step % args.evaluation_steps == 0:
            saving(transformer, args, accelerator, global_step, is_student=True)
            saving(transformer_fake, args, accelerator, global_step, is_student=False)
        accelerator.wait_for_everyone()
        ### ----------------------------------------------------

        ### Validation images
        ### ----------------------------------------------------
        if accelerator.is_main_process and (
            global_step == 1 or global_step % args.validation_steps == 0
        ):
            log_validation_klein(
                transformer_student=transformer,
                vae=vae,
                bn_mean=bn_mean,
                bn_std=bn_std,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                fm_solver=fm_solver,
                train_dataset_hf=train_dataset_hf,
                accelerator=accelerator,
                args=args,
                global_step=global_step,
                weight_dtype=weight_dtype,
            )
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        ### ----------------------------------------------------

        ### Loss logging
        ### ----------------------------------------------------
        if accelerator.is_main_process and global_step % args.log_steps == 0:
            logs = {
                "fake_loss": (
                    avg_dmd_fake_loss.detach().item() if args.do_dmd_loss else 0
                ),
                "dmd_loss": avg_dmd_loss.detach().item() if args.do_dmd_loss else 0,
                "mmd_loss": avg_mmd_loss.detach().item() if args.do_mmd_loss else 0,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        ### ----------------------------------------------------

    ## -----------------------------------------------------------------------------------------------

    accelerator.wait_for_everyone()
    accelerator.end_training()
