"""
Standalone diagnostic for FLUX.2 Klein img2img inference on a single image.

Purpose:
- isolate whether the purple output is caused by low-step inference itself;
- reuse the same Klein base model and LoRA as the training/evaluation pipeline;
- compare multiple step counts on the exact same latent initialization.

Examples:

python test_klein_single_image_inference.py \
  --sample-index 0 \
  --steps 4 30

python test_klein_single_image_inference.py \
  --input-image /path/to/input.png \
  --target-image /path/to/target.png \
  --steps 4 8 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import DownloadConfig
from datasets import load_dataset
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps
from torchvision import transforms

from src.utils.structured_noise import generate_structured_noise_batch_vectorized


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_LORA_ID = (
    "dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered_noise_lora_1_klein_4B_512x512"
)
DEFAULT_DATASET_NAME = "dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora"
DEFAULT_PROMPT = "make this image photorealistic"
DEFAULT_MODEL_CACHE_DIR = "/code/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image diagnostic for FLUX.2 Klein img2img inference."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora-path", default=DEFAULT_LORA_ID)
    parser.add_argument("--weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cutoff-radius", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        default="results/klein_single_image_inference",
    )

    parser.add_argument("--input-image", default=None)
    parser.add_argument("--target-image", default=None)

    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cond-image-column", default="input_image")
    parser.add_argument("--image-column", default="edited_image")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-cache-dir", default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--allow-online", action="store_true")

    return parser.parse_args()


def get_weight_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def build_transforms(resolution: int):
    resize_crop = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(resolution),
        ]
    )
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return resize_crop, normalize


def load_pil_image(path: str) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_source_and_target(args: argparse.Namespace):
    resize_crop, normalize = build_transforms(args.resolution)

    if args.input_image:
        source = resize_crop(load_pil_image(args.input_image))
        target = (
            resize_crop(load_pil_image(args.target_image))
            if args.target_image
            else None
        )
        sample_name = Path(args.input_image).stem
    else:
        dataset_cache = args.cache_dir
        if dataset_cache is None:
            dataset_cache = f"/code/dataset/{args.dataset_name.split('/')[-1]}"

        dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            cache_dir=dataset_cache,
            download_config=DownloadConfig(local_files_only=not args.allow_online),
        )
        example = dataset[args.sample_index]
        source = resize_crop(example[args.cond_image_column].convert("RGB"))
        target = resize_crop(example[args.image_column].convert("RGB"))
        sample_name = f"dataset_{args.sample_index:06d}"

    source_tensor = normalize(source).unsqueeze(0)
    return source, target, source_tensor, sample_name


def resolve_local_model_path(model_id: str, model_cache_dir: str) -> str:
    model_cache_root = Path(model_cache_dir)
    repo_cache_dir = model_cache_root / f"models--{model_id.replace('/', '--')}"

    refs_main = repo_cache_dir / "refs" / "main"
    if refs_main.exists():
        snapshot_name = refs_main.read_text(encoding="utf-8").strip()
        snapshot_path = repo_cache_dir / "snapshots" / snapshot_name
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots_dir = repo_cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

    return model_id


def load_pipeline(args: argparse.Namespace, device: str, weight_dtype: torch.dtype):
    model_source = resolve_local_model_path(args.model_id, args.model_cache_dir)
    print(f"Loading base model from: {model_source}")

    pipe = Flux2KleinPipeline.from_pretrained(
        model_source,
        torch_dtype=weight_dtype,
    )
    pipe.load_lora_weights(
        args.lora_path,
        weight_name=args.weight_name,
        adapter_name="diagnostic",
    )
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


@torch.no_grad()
def build_initial_latents(
    pipe: Flux2KleinPipeline,
    source_tensor: torch.Tensor,
    cutoff_radius: float,
    device: str,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    source_tensor = source_tensor.to(device=device, dtype=pipe.vae.dtype)
    source_latents = pipe.vae.encode(source_tensor).latent_dist.mode()
    source_latents = source_latents.to(device=device, dtype=weight_dtype)

    noise_spatial = generate_structured_noise_batch_vectorized(
        source_latents.float(),
        cutoff_radius=cutoff_radius,
    ).to(device=device, dtype=weight_dtype)

    return pipe._patchify_latents(noise_spatial).to(weight_dtype)


@torch.no_grad()
def generate_for_steps(
    pipe: Flux2KleinPipeline,
    source_image: Image.Image,
    base_latents: torch.Tensor,
    prompt: str,
    guidance_scale: float,
    steps: list[int],
    seed: int,
    device: str,
):
    outputs = {}
    for num_steps in steps:
        result = pipe(
            prompt=prompt,
            image=source_image,
            latents=base_latents.clone(),
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        outputs[num_steps] = result
    return outputs


def save_panel(
    output_path: Path,
    source: Image.Image,
    generated: dict[int, Image.Image],
    target: Image.Image | None,
):
    tiles = [source] + [generated[k] for k in sorted(generated)]
    if target is not None:
        tiles.append(target)

    width, height = tiles[0].size
    panel = Image.new("RGB", (width * len(tiles), height))
    for index, tile in enumerate(tiles):
        panel.paste(tile, (index * width, 0))
    panel.save(output_path)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = get_weight_dtype(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_image, target_image, source_tensor, sample_name = load_source_and_target(
        args
    )
    pipe = load_pipeline(args, device, weight_dtype)
    base_latents = build_initial_latents(
        pipe,
        source_tensor,
        args.cutoff_radius,
        device,
        weight_dtype,
    )
    generated = generate_for_steps(
        pipe,
        source_image,
        base_latents,
        args.prompt,
        args.guidance_scale,
        args.steps,
        args.seed,
        device,
    )

    processed_source_path = output_dir / f"{sample_name}_input.png"
    source_image.save(processed_source_path)

    if target_image is not None:
        target_image.save(output_dir / f"{sample_name}_target.png")

    for num_steps, image in generated.items():
        image.save(output_dir / f"{sample_name}_steps_{num_steps:02d}.png")

    save_panel(
        output_dir / f"{sample_name}_panel.png",
        source_image,
        generated,
        target_image,
    )

    metadata = {
        "model_id": args.model_id,
        "lora_path": args.lora_path,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "seed": args.seed,
        "resolution": args.resolution,
        "cutoff_radius": args.cutoff_radius,
        "sample_name": sample_name,
        "input_image": args.input_image,
        "target_image": args.target_image,
        "dataset_name": args.dataset_name if not args.input_image else None,
        "sample_index": args.sample_index if not args.input_image else None,
    }
    (output_dir / f"{sample_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved diagnostic outputs to {output_dir}")
    print(f"Panel: {output_dir / f'{sample_name}_panel.png'}")


if __name__ == "__main__":
    main()
