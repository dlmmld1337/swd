from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel


DEFAULT_BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_SOURCE_LORA_ID = (
    "dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered_noise_lora_1_klein_4B_512x512"
)
DEFAULT_CACHE_DIR = "/code/models"
DEFAULT_OUTPUT_DIR = "/code/models/flux2-klein-base-4b-merged-source-transformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the source Klein LoRA into the Klein transformer and save it locally."
    )
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--source-lora-id", default=DEFAULT_SOURCE_LORA_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if output_dir.exists() and not args.overwrite:
        print(f"Merged transformer already exists: {output_dir}")
        print("Use --overwrite to rebuild it.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.base_model_id,
        subfolder="transformer",
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    pipeline = Flux2KleinPipeline.from_pretrained(
        args.base_model_id,
        transformer=transformer,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading source LoRA: {args.source_lora_id}")
    pipeline.load_lora_weights(args.source_lora_id)
    print("Fusing LoRA into transformer")
    pipeline.fuse_lora(lora_scale=1.0)
    pipeline.unload_lora_weights()

    print(f"Saving merged transformer to: {output_dir}")
    pipeline.transformer.save_pretrained(output_dir)
    print("Done")


if __name__ == "__main__":
    main()
