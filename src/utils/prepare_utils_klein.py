import gc
import logging
import os

import torch
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
)
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from src.utils.prepare_utils import prepare_accelerator  # reuse unchanged


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


logger = get_logger(__name__)
fallback_logger = logging.getLogger(__name__)


def _log_info(message, *args):
    try:
        logger.info(message, *args)
    except RuntimeError:
        fallback_logger.info(message, *args)


def _log_warning(message, *args):
    try:
        logger.warning(message, *args)
    except RuntimeError:
        fallback_logger.warning(message, *args)


KLEIN_BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
KLEIN_MERGED_TRANSFORMER_DIR = (
    "/code/models/flux2-klein-base-4b-merged-source-transformer"
)
KLEIN_CACHE_DIR = "/code/models"
KLEIN_PROMPT = "make this image photorealistic"


########################################################################################################################
#                                      MODEL PREPARATION FOR KLEIN                                                    #
########################################################################################################################


def _get_weight_dtype(accelerator):
    # Klein 4B requires bf16 to fit on a single 32GB GPU; always use bf16.
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _get_transformer_source():
    if not os.path.isdir(KLEIN_MERGED_TRANSFORMER_DIR):
        raise FileNotFoundError(
            "Merged Klein transformer is missing. Run merge_klein_source_lora_into_transformer.py first. "
            f"Expected directory: {KLEIN_MERGED_TRANSFORMER_DIR}"
        )
    return KLEIN_MERGED_TRANSFORMER_DIR


def prepare_models_klein(args, accelerator):
    weight_dtype = _get_weight_dtype(accelerator)

    # ------------------------------------------------------------------
    # 1. Scheduler (tiny, CPU only)
    # ------------------------------------------------------------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        KLEIN_BASE_MODEL_ID,
        subfolder="scheduler",
        cache_dir=KLEIN_CACHE_DIR,
    )

    # ------------------------------------------------------------------
    # 2. Text encoder: encode single fixed prompt FIRST, then free completely.
    #    Qwen3 is large (~14 GB CPU RAM); freeing it before any GPU allocation
    #    prevents OOM when loading 2×4B transformers onto GPU.
    # ------------------------------------------------------------------
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        KLEIN_BASE_MODEL_ID,
        subfolder="tokenizer",
        cache_dir=KLEIN_CACHE_DIR,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        KLEIN_BASE_MODEL_ID,
        subfolder="text_encoder",
        cache_dir=KLEIN_CACHE_DIR,
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)
    # Keep on CPU — encode_prompt uses text_encoder.device to pick device.
    text_encoder.to("cpu")

    text_encoding_pipeline = None
    from diffusers import Flux2KleinPipeline

    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        KLEIN_BASE_MODEL_ID,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        cache_dir=KLEIN_CACHE_DIR,
    )
    with torch.no_grad():
        prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            prompt=KLEIN_PROMPT,
            max_sequence_length=getattr(args, "max_sequence_length", 512),
        )
    # Detach from any graph / pipeline references before freeing
    prompt_embeds = prompt_embeds.detach().cpu()
    text_ids = text_ids.detach().cpu()

    del text_encoding_pipeline, text_encoder, tokenizer
    _free_memory()

    _log_info(
        f"Encoded prompt '{KLEIN_PROMPT}': embeds {prompt_embeds.shape}, text_ids {text_ids.shape}"
    )

    # Move embeddings to GPU now (they are tiny)
    prompt_embeds = prompt_embeds.to(accelerator.device)
    text_ids = text_ids.to(accelerator.device)

    # ------------------------------------------------------------------
    # 3. VAE + BN normalisation statistics
    # ------------------------------------------------------------------
    vae = AutoencoderKLFlux2.from_pretrained(
        KLEIN_BASE_MODEL_ID,
        subfolder="vae",
        cache_dir=KLEIN_CACHE_DIR,
    )
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)

    # ------------------------------------------------------------------
    # 4. Teacher: base Klein with merged LoRA (frozen, CPU-resident)
    # ------------------------------------------------------------------
    transformer_source = _get_transformer_source()
    _log_info(
        "Loading Klein transformer base from merged local snapshot: %s",
        transformer_source,
    )

    transformer_teacher = Flux2Transformer2DModel.from_pretrained(
        transformer_source,
        torch_dtype=weight_dtype,
    )
    transformer_teacher.requires_grad_(False)
    # Keep teacher on CPU — only moved to GPU transiently during loss computation.
    transformer_teacher.to("cpu", dtype=weight_dtype)
    _free_memory()

    # ------------------------------------------------------------------
    # 5. LoRA config (shared between student and fake/discriminator)
    # ------------------------------------------------------------------
    target_modules = []
    if args.apply_lora_to_attn_projections:
        target_modules.extend(["to_k", "to_q", "to_v", "to_out.0"])
    if args.apply_lora_to_add_projections:
        target_modules.extend(["add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"])
    if args.apply_lora_to_mlp_projections:
        target_modules.extend(["linear_in", "linear_out"])
    assert len(target_modules) > 0, "LoRA must be applied to at least one module type."

    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha else args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    # ------------------------------------------------------------------
    # 6. Student: fresh base Klein + new LoRA
    # ------------------------------------------------------------------
    transformer = Flux2Transformer2DModel.from_pretrained(
        transformer_source,
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer = get_peft_model(transformer, transformer_lora_config)

    # ------------------------------------------------------------------
    # 7. Fake / Discriminator: another fresh base Klein + new LoRA
    # ------------------------------------------------------------------
    transformer_fake_base = Flux2Transformer2DModel.from_pretrained(
        transformer_source,
        torch_dtype=weight_dtype,
    )
    transformer_fake_base.requires_grad_(False)
    transformer_fake_base.to(accelerator.device, dtype=weight_dtype)
    transformer_fake = get_peft_model(transformer_fake_base, transformer_lora_config)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        transformer_fake.enable_gradient_checkpointing()

    if weight_dtype == torch.float16:
        cast_training_params([transformer], dtype=torch.float32)
        cast_training_params([transformer_fake], dtype=torch.float32)

    return (
        transformer,
        transformer_teacher,
        transformer_fake,
        vae,
        latents_bn_mean,
        latents_bn_std,
        prompt_embeds,
        text_ids,
        noise_scheduler,
        weight_dtype,
    )


# Re-export prepare_optimizer for convenience
from src.utils.prepare_utils import prepare_optimizer  # noqa: F401, E402
