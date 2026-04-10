"""
Baseline SD3.5 Medium inference with the same validation prompts used in SwD training eval.
Outputs saved to results/baseline_inference/
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline

VALIDATION_PROMPTS = [
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "A girl with pale blue hair and a cami tank top",
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "a deer with bird feathers, highly detailed, full body",
    "The interior of a mad scientists laboratory, Cluttered with science experiments, tools and strange machines, Eerie purple light, Close up, by Miyazaki",
    "a barred owl peeking out from dense tree branches",
    "a close-up of a blue dragonfly on a daffodil",
    "A green train is coming down the tracks",
    "A photograph of the inside of a subway train. There are frogs sitting on the seats. One of them is reading a newspaper. The window shows the river in the background.",
    "a family of four posing at the Grand Canyon",
    "A high resolution photo of a donkey in a clown costume giving a lecture at the front of a lecture hall. The blackboard has mathematical equations on it. There are many students in the lecture hall.",
    "A castle made of tortilla chips, in a river made of salsa. There are tiny burritos walking around the castle",
    "A castle made of cardboard.",
]

MODEL_ID = "adamo1139/stable-diffusion-3.5-medium-ungated"
OUTPUT_DIR = "results/baseline_inference"
NUM_STEPS = 28
GUIDANCE_SCALE = 4.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL_ID} ...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cpu").manual_seed(SEED)

for i, prompt in enumerate(VALIDATION_PROMPTS):
    safe_name = prompt[:50].replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{i:02d}_{safe_name}.png")
    if os.path.exists(out_path):
        print(f"[{i+1}/{len(VALIDATION_PROMPTS)}] Skip (exists): {out_path}")
        continue

    print(f"[{i+1}/{len(VALIDATION_PROMPTS)}] Generating: {prompt[:60]}")
    image = pipe(
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]
    image.save(out_path)
    print(f"  Saved: {out_path}")

print(f"\nDone. Images saved to {OUTPUT_DIR}/")
