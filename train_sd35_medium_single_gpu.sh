#!/bin/bash
source /code/auto_remaster/sandbox/DiffSynth-Studio/.venv_diff/bin/activate

accelerate launch --num_processes=1 --mixed_precision bf16 main.py \
    --model_name "medium" \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --learning_rate_cls 5e-6 \
    --num_boundaries 4 \
    --scales 64 80 96 128 \
    --boundaries 0 7 14 18 28 \
    --do_dmd_loss \
    --do_gan_loss \
    --do_mmd_loss \
    --cfg_teacher 7.0 \
    --cls_blocks 11 \
    --mmd_blocks 11 \
    --apply_lora_to_attn_projections \
    --apply_lora_to_mlp_projections \
    --seed 42 \
    --gradient_checkpointing \
    --resume_from_checkpoint "latest" \
    --validation_steps 250 \
    --evaluation_steps 3000 \
    --max_eval_samples 8 \
    --max_train_steps 3000
    # Debug flags (uncomment to test full pipeline in 4 steps):
    # --validation_steps 1 \
    # --evaluation_steps 1 \
    # --max_eval_samples 8 \
    # --max_train_steps 4
