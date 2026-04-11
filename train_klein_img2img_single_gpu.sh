#!/bin/bash
set -e

source /code/auto_remaster/sandbox/DiffSynth-Studio/.venv_diff/bin/activate
cd "$(dirname "$0")"
ARGS=(
    --model_name "klein"
    --train_batch_size 1
    --gradient_accumulation_steps 8
    --learning_rate 5e-6
    --learning_rate_cls 5e-6
    --num_boundaries 4
    --num_timesteps 28
    --scales 32 40 48 64
    --boundaries 0 7 14 21 28
    --do_dmd_loss
    --do_gan_loss
    --do_mmd_loss
    --cfg_teacher 3.5
    --cls_blocks 8
    --mmd_blocks 8
    --num_discriminator_layers 4
    --apply_lora_to_attn_projections
    --lora_rank 64
    --seed 42
    --gradient_checkpointing
    --resume_from_checkpoint "latest"
    --structural_noise_radius 100
    --resolution 512
    --output_dir "results/klein_img2img"
    --log_steps 25

    # real values:
    --validation_steps 250
    --evaluation_steps 3000
    --max_train_steps 3000

    # debug values: 3 validation/save events at steps 1, 2, 3
    # --validation_steps 1
    # --evaluation_steps 1
    # --max_train_steps 3
)

accelerate launch \
    --num_processes=1 \
    --mixed_precision bf16 \
    main.py \
    "${ARGS[@]}"
