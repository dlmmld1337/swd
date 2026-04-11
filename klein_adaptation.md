# План: Адаптация SWD-дистилляции под FLUX.2 Klein img2img

**Цель:** Взять SWD-фреймворк из `auto_remaster/sandbox/swd/` и натренировать 4-шаговый FLUX.2 Klein (img2img реастер), где учителем служит Klein + merged LoRA (`dim/nfs_pix2pix_...klein_4B_512x512`). Существующий SD3.5 код не трогаем.

---

## Как устроен SWD (детально)

**Scale-Wise Distillation** учит 4-шаговый студент-генератор через три потока потерь:

### Три лосса
| Лосс | Кто обучается | Зачем |
|---|---|---|
| **DMD** (Distribution Matching) | Студент | Студент генерирует `fake_sample`, учитель и фейк-сеть дают два score-направления; студент движется от fake-score к real-score |
| **GAN** | Дискриминатор (fake-сеть) + студент | Fake-сеть должна отличить `fake_sample` от `true_sample` по features |
| **MMD** (Maximum Mean Discrepancy) | Студент | Промежуточные features fake_sample ≈ features real_sample |

### Ключевой момент про teacher в SD3:
`transformer_teacher = transformer` — **это один и тот же объект**. Учитель = base model, студент = base + LoRA. `disable_adapter()` временно отключает LoRA → base weights → "учитель".

**Для Klein: мы загружаем ТРИ отдельных объекта** — teacher (merged LoRA), student (новая LoRA), fake (новая LoRA). `disable_adapter()` не нужен.

### CFG в DMD-лоссе (SD3):
```python
real_pred_uncond = teacher(noisy_fake, uncond_embeds, ...)  # второй forward
real_pred = real_pred_uncond + cfg_teacher * (real_pred - real_pred_uncond)
```
**Для Klein: CFG встроен в трансформер через `guidance` параметр.** Double forward не нужен, `uncond_prompt_embeds` не нужен вообще.

### Scale-wise механика:
1. `sample_batch` отдаёт `model_input` (текущий масштаб) и `model_input_prev` (предыдущий меньший масштаб)
2. В лоссах: `model_input_prev` поднимается до текущего масштаба → это имитирует точку на границе между scales
3. К ней добавляют шум boundary timestep → `noisy_model_input_curr`
4. Студент делает один шаг: `fake_sample = noisy_model_input_curr - sigma * pred`
5. `fake_sample` сравнивается с `model_input` через все три лосса

---

## Архитектурные отличия FLUX.2 Klein vs SD3.5

| | SD3.5 | FLUX.2 Klein |
|---|---|---|
| Forward sig | `(x, text_embeds, pooled, t)` | `(hidden_states=cat([noisy_tgt, cond]), t/1000, guidance, encoder_hidden_states, txt_ids, img_ids)` |
| Img conditioning | нет | token concat в `hidden_states` |
| Text encoder | CLIP × 2 + T5 | Qwen3 (один) |
| CFG | double forward с uncond_embeds | `guidance` параметр в одном forward |
| VAE norm | `(z - shift_factor) * scaling_factor` | patchify → `(z - bn_mean) / bn_std` |
| Teacher механизм | `disable_adapter()` на том же объекте | отдельный merged-объект, `no_grad()` |
| Inner dim | `transformer.inner_dim` | нужно проверить: `.inner_dim` или `.config.hidden_size` |

Изменять существующие файлы (`losses.py`, `train.py`, etc.) **нельзя** — создаём параллельные Klein-специфичные файлы.

---

## Фаза 1 — Новые вспомогательные модули

### Шаг 1. `src/dataset_klein.py`

Датасет из HuggingFace кэша:
- Загружает `dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered` из `/code/dataset/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered`
- Колонки: `input_image` (condition), `edited_image` (target)
- `transforms.Resize(512) + CenterCrop(512) + ToTensor + Normalize(0.5)` применяем к ОБОИМ
- Датасет отдаёт `{pixel_values: target_tensor, cond_pixel_values: cond_tensor}` (без caption)
- `InfiniteSampler` импортируем из `src/dataset.py` без изменений

### Шаг 2. `src/transformer_with_discriminator_klein.py`

**Архитектура `Flux2Transformer2DModel` (из исходника):**
- `self.inner_dim = num_attention_heads * attention_head_dim` — есть напрямую на модели
- `self.transformer_blocks` — `num_layers` блоков `Flux2TransformerBlock` (dual-stream: img и txt идут раздельно)
- `self.single_transformer_blocks` — `num_single_layers` блоков `Flux2SingleTransformerBlock`
  - перед ними: `hidden_states = cat([encoder_hidden_states, hidden_states], dim=1)` — txt+img сливаются
  - после: `hidden_states = hidden_states[:, num_txt_tokens:, ...]` — txt вырезается
- Timestep: модель делает `timestep * 1000` **внутри** → снаружи передаём `timesteps / 1000` (в `[0,1]`)

**Feature collection для дискриминатора — из `single_transformer_blocks`** (img+txt уже слиты, features богаче):

```python
def forward_with_feature_extraction_klein(
    self,                          # Flux2Transformer2DModel
    hidden_states,                 # packed noisy latents: (B, img_seq_len, C)
    encoder_hidden_states,         # prompt embeds: (B, txt_seq_len, joint_attention_dim)
    timestep,                      # уже в [0,1] (делим снаружи: timesteps / 1000)
    img_ids,                       # (B, img_seq_len, 4)
    txt_ids,                       # (B, txt_seq_len, 4)
    guidance=None,
    classify_index_block=None,     # list[int] — индексы в single_transformer_blocks
    return_only_features=False,
    return_dict=False,
):
    classify_index_block = classify_index_block or []
    hidden_states_collect = []
    num_txt_tokens = encoder_hidden_states.shape[1]

    # === Повторяем forward() Klein один-в-один, добавляем только сбор features ===

    # 1. Timestep embedding + modulation (Klein ожидает timestep в [0,1], сам умножает на 1000)
    temb = self.time_guidance_embed(timestep, guidance)
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod    = self.single_stream_modulation(temb)

    # 2. Input projections
    hidden_states         = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 3. RoPE
    _img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
    _txt_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
    image_rotary_emb = self.pos_embed(_img_ids)
    text_rotary_emb  = self.pos_embed(_txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # 4. Dual-stream blocks
    for block in self.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_img=double_stream_mod_img,
            temb_mod_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
        )

    # 5. Merge txt+img для single-stream
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 6. Single-stream blocks — здесь собираем features
    for index_block, block in enumerate(self.single_transformer_blocks):
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
        )
        if index_block in classify_index_block:
            # Вырезаем только img-токены (txt в начале)
            img_features = hidden_states[:, num_txt_tokens:, :]
            hidden_states_collect.append(img_features)
            if return_only_features and index_block == classify_index_block[-1]:
                return hidden_states_collect

    # 7. Убираем txt-токены, выход
    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if return_only_features:
        return hidden_states_collect
    if not return_dict:
        return (output,), hidden_states_collect
    return output, hidden_states_collect
```

`TransformerClsKlein(nn.Module)`:
- Аналог `TransformerCls`, но `inner_dim = teacher_transformer.inner_dim` — он **есть** на `Flux2Transformer2DModel`
- `forward()` делегирует в `forward_with_feature_extraction_klein`
- Discriminator-голова строится по `linspace(inner_dim → 1, num_discriminator_layers+1)`

**Timestep для Klein везде в лоссах:**
```python
# SD3 было:
transformer(noisy_input, prompt_embeds, pooled_prompt_embeds, timesteps_start)
# Klein должно быть:
forward_klein(transformer, noisy_target, cond, prompt_embeds, text_ids,
              timestep=timesteps_start / 1000,  ← обязательно делить
              guidance=guidance)
```

**Важно для `classify_index_block`:** индексы относятся к `single_transformer_blocks`. Для 4B Klein (дефолт: `num_single_layers=48`) — `--cls_blocks 8` = block 8 из 48 single-stream блоков.

### Шаг 3. `src/utils/prepare_utils_klein.py`

```python
def prepare_models_klein(args, accelerator):
    # 1. Загрузить шедулер
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        subfolder="scheduler",
        cache_dir="/code/models",
    )

    # 2. Загрузить base Klein → смержить LoRA → teacher (frozen, отдельный объект)
    transformer_teacher = Flux2Transformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B", subfolder="transformer",
        cache_dir="/code/models", torch_dtype=weight_dtype,
    )
    pipeline_for_merge = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B", transformer=transformer_teacher,
        cache_dir="/code/models",
    )
    pipeline_for_merge.load_lora_weights(
        "dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered_noise_lora_1_klein_4B_512x512"
    )
    pipeline_for_merge.fuse_lora()  # merge in-place
    transformer_teacher = pipeline_for_merge.transformer
    transformer_teacher.requires_grad_(False)

    # 3. Загрузить base Klein заново → student (новая PEFT LoRA)
    transformer = Flux2Transformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B", subfolder="transformer",
        cache_dir="/code/models", torch_dtype=weight_dtype,
    )
    transformer_lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0",
                        "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"],
    )
    transformer = get_peft_model(transformer, transformer_lora_config)

    # 4. transformer_fake — копия base + LoRA (discriminator)
    transformer_fake = copy.deepcopy(
        Flux2Transformer2DModel.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B",
            subfolder="transformer", cache_dir="/code/models", torch_dtype=weight_dtype)
    )
    transformer_fake = get_peft_model(transformer_fake, transformer_lora_config)

    # 5. VAE + normalization stats
    vae = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B", subfolder="vae",
        cache_dir="/code/models",
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std  = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)

    # 6. Один раз закодировать промпт → освободить text_encoder
    tokenizer = Qwen2TokenizerFast.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B",
        subfolder="tokenizer", cache_dir="/code/models")
    text_encoder = Qwen3ForCausalLM.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B",
        subfolder="text_encoder", cache_dir="/code/models", torch_dtype=weight_dtype)
    # encode "make this image photorealistic"
    prompt_embeds, text_ids = encode_single_prompt("make this image photorealistic", ...)
    del text_encoder  # освобождаем VRAM
    free_memory()

    return (transformer, transformer_teacher, transformer_fake,
            vae, latents_bn_mean, latents_bn_std,
            prompt_embeds, text_ids, noise_scheduler, weight_dtype)
```

### Шаг 4. `src/utils/train_utils_klein.py` — `sample_batch_klein(...)`

```python
def sample_batch_klein(args, accelerator, global_step, loader,
                        fm_solver, vae, bn_mean, bn_std, prompt_embeds,
                        text_ids, weight_dtype):
    batch = next(loader)

    # Выбираем boundary (scale + timestep)
    idx_start = fm_solver.boundary_start_idx[global_step % args.num_boundaries]
    idx_start = torch.tensor([idx_start] * args.train_batch_size).long()

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]
    previous_scales = (
        scales if scales[0] == fm_solver.min_scale
        else fm_solver._get_previous_scale(scales)
    )

    pixel_values      = batch["pixel_values"].to(accelerator.device)
    cond_pixel_values = batch["cond_pixel_values"].to(accelerator.device)

    # Scale-wise downsampling — ОБА изображения (ключевая идея SWD)
    pixel_values_curr      = fm_solver.downscale_to_current(pixel_values,      scales * 8)
    cond_pixel_values_curr = fm_solver.downscale_to_current(cond_pixel_values, scales * 8)
    pixel_values_prev      = fm_solver.downscale_to_current(pixel_values,      previous_scales * 8)
    cond_pixel_values_prev = fm_solver.downscale_to_current(cond_pixel_values, previous_scales * 8)

    # VAE encode + BN normalize
    def vae_encode_normalize(px):
        z = vae.encode(px.to(weight_dtype)).latent_dist.mode()
        z = Flux2KleinPipeline._patchify_latents(z)
        return (z - bn_mean.to(z.device)) / bn_std.to(z.device)

    model_input          = vae_encode_normalize(pixel_values_curr)
    cond_model_input     = vae_encode_normalize(cond_pixel_values_curr)
    model_input_prev     = vae_encode_normalize(pixel_values_prev)
    cond_model_input_prev = vae_encode_normalize(cond_pixel_values_prev)

    # Размножить prompt_embeds/text_ids на batch_size
    batch_prompt_embeds = prompt_embeds.repeat(args.train_batch_size, 1, 1)
    batch_text_ids      = text_ids.repeat(args.train_batch_size, 1, 1)

    return (model_input, model_input_prev,
            cond_model_input, cond_model_input_prev,
            batch_prompt_embeds, batch_text_ids, idx_start)
```

**Структурный шум:** Вместо `torch.randn_like` в лоссах используем `generate_structured_noise_batch_vectorized(cond_latent_spatial, cutoff_radius=args.structural_noise_radius)`. Это сохраняет пространственную структуру condition-изображения в начальном шуме — ключевое для img2img.

### Шаг 5. `src/losses_klein.py`

Адаптируем все три лосса. Изменения минимальны — только сигнатура вызова трансформера:

**Вместо SD3:**
```python
model_pred = transformer(noisy_input, prompt_embeds, pooled_prompt_embeds, timestep)[0]
```

**Klein:**
```python
# hidden_states = cat([noisy_target, cond]) — img2img conditioning через concat
# timestep ДЕЛИТСЯ на 1000 — модель сама умножает обратно
model_pred, _ = forward_with_feature_extraction_klein(
    transformer,
    hidden_states=cat([packed_noisy, packed_cond], dim=1),
    encoder_hidden_states=prompt_embeds,
    timestep=timestep / 1000,
    img_ids=cat([latent_ids, cond_ids], dim=1),
    txt_ids=text_ids,
    guidance=torch.full([B], cfg, device=...),
    classify_index_block=[],
    return_only_features=False,
)
# обрезаем output: model_pred = model_pred[:, :orig_noisy_len, :]
# затем unpack через _unpack_latents_with_ids
```

**Убираем:**
- `pooled_prompt_embeds` везде
- `uncond_prompt_embeds` / `uncond_pooled_prompt_embeds` из сигнатур и вызовов
- `transformer_teacher.disable_adapter()` — teacher уже merged, отдельный объект
- Двойной forward для CFG: заменяем на `guidance = torch.full([1], cfg_teacher)`

**Добавляем:**
- `cond_model_input` и `cond_model_input_prev` параметры
- Structured noise вместо `torch.randn_like` при генерации начального шума
- В `generator_loss_klein`: `cond_model_input_curr = fm_solver.upscale_to_next(cond_model_input_down, scales)` рядом с `model_input_prev`

Структура потоков `fake_diffusion_loss` / `generator_loss` (шаги 1-4) **остаётся идентичной**.

### Шаг 6. `src/flow_matching_sampler_klein.py` — `sampling_klein_img2img()`

Новый метод для инференса студента (нужен для валидации):

```python
@torch.no_grad()
def sampling_klein_img2img(
    model,                # Klein student transformer
    cond_pixels,          # condition image пикселей при полном разрешении
    vae, bn_mean, bn_std,
    prompt_embeds, text_ids,
    solver,               # FlowMatchingSolver (для сигм/timesteps/boundary)
    structural_noise_radius,
    guidance_scale=1.0,
):
    # 1. Encode condition при min_scale
    min_px = int(solver.min_scale * 8)
    cond_down = F.interpolate(cond_pixels, size=min_px, mode='area')
    cond_latent = vae_encode_normalize(cond_down, vae, bn_mean, bn_std)  # (B,C,min_scale,min_scale)

    # 2. Initial latent — structured noise от condition
    latent = generate_structured_noise_batch_vectorized(cond_latent, cutoff_radius=structural_noise_radius)

    # 3. Sampler loop (аналог sampling() в FlowMatchingSolver)
    sigmas    = solver.noise_scheduler.sigmas[solver.boundary_idx]
    timesteps = solver.noise_scheduler.timesteps[solver.boundary_start_idx]
    k = 0

    for idx_start, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        noise_pred = forward_klein(model, latent, cond_latent, prompt_embeds, text_ids,
                                    timesteps[idx_start], guidance=guidance_scale)
        latent = latent - noise_pred * sigma  # flow matching step (Euler)

        # Boundary: upscale оба, добавить шум
        if idx_start + 1 < len(solver.scales):
            next_scale = int(solver.scales[idx_start + 1])
            latent      = F.interpolate(latent,      size=next_scale, mode='bicubic')
            # condition тоже поднимаем до нового масштаба (из оригинала, не из latent)
            next_px     = next_scale * 8
            cond_down   = F.interpolate(cond_pixels, size=next_px, mode='area')
            cond_latent = vae_encode_normalize(cond_down, vae, bn_mean, bn_std)
            # добавляем шум на boundary
            noise = generate_structured_noise_batch_vectorized(cond_latent, ...)
            latent = sigma_next * noise + (1 - sigma_next) * latent
        k += 1

    return latent  # финальный латент (unpatchify + VAE decode снаружи)
```

---

## Фаза 2 — Основной тренировочный файл

### Шаг 7. `train_klein.py`

На основе `train.py`, заменяем:
- `prepare_models(args, accelerator)` → `prepare_models_klein(args, accelerator)`
- `get_loader(args.train_batch_size, root_dir)` → `get_loader_klein(args.train_batch_size, root_dir)`
- `sample_batch(...)` → `sample_batch_klein(...)`
- `fake_diffusion_loss(...)` → `fake_diffusion_loss_klein(...)`
- `generator_loss(...)` → `generator_loss_klein(...)`
- Убираем `uncond_prompt_embeds` / compute at start
- В валидации: `log_validation` → `log_validation_klein` со triplet картинками

**Evaluation loop** (каждые `evaluation_steps`):
- Убираем `distributed_sampling` (FID/pick/clip для text2img — нерелевантно)
- `saving(transformer, ...)` и `saving(transformer_fake, ...)` оставляем без изменений (работают с PEFT-моделями)

**Validation loop** (каждые `validation_steps`):
- `sampling_klein_img2img()` на 4 случайных примерах из датасета
- VAE decode результат → денормализовать в [0,1]
- Log в tensorboard: триплеты `[condition_orig | generated | target_orig]`
- Конкатенировать горизонтально как одну картинку на строку: `torchvision.utils.make_grid([cond, gen, tgt], nrow=3)`

---

## Фаза 3 — Точки входа

### Шаг 8. `main.py` — минимальные изменения

```diff
- choices=["medium", "large"]
+ choices=["medium", "large", "klein"]
```

```python
if __name__ == "__main__":
    args = parse_args()
    if args.model_name == "klein":
        from train_klein import train as train_klein
        train_klein(args)
    else:
        train(args)
```

### Шаг 9. `train_klein_img2img_single_gpu.sh`

```bash
#!/bin/bash
source /code/auto_remaster/sandbox/swd/.venv_diff/bin/activate

accelerate launch --num_processes=1 --mixed_precision bf16 main.py \
    --model_name "klein" \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --learning_rate_cls 5e-6 \
    --num_boundaries 4 \
    --num_timesteps 28 \
    --scales 32 40 48 64 \
    --boundaries 0 7 14 21 28 \
    --do_dmd_loss \
    --do_gan_loss \
    --do_mmd_loss \
    --cfg_teacher 3.5 \
    --cls_blocks 8 \
    --mmd_blocks 8 \
    --apply_lora_to_attn_projections \
    --lora_rank 64 \
    --seed 42 \
    --gradient_checkpointing \
    --resume_from_checkpoint "latest" \
    --validation_steps 250 \
    --evaluation_steps 3000 \
    --max_train_steps 3000
```

**Scales explained:** `scales = [32, 40, 48, 64]` — в единицах латентного пространства Klein (VAE stride=8):
- boundary 0 (t=0→7, высокий шум, структура): 32 → 256×256 px — учим глобальный tone-mapping
- boundary 1 (t=7→14): 40 → 320×320 px
- boundary 2 (t=14→21): 48 → 384×384 px
- boundary 3 (t=21→28, малый шум, детали): 64 → 512×512 px — учим текстуры

---

## Верификация

```bash
python main.py \
    --model_name "klein" \
    --max_train_steps 4 \
    --validation_steps 1 \
    --evaluation_steps 4 \
    --max_eval_samples 1 \
    --train_batch_size 1
```

Критерии:
1. Loss не NaN на первом шаге
2. Все три лосса (fake_loss, dmd_loss, mmd_loss) ненулевые
3. Validation запускается, triplet картинки сохраняются

---

## Решения к известным проблемам

- **Teacher загрузка**: Klein 4B скачивается при первом запуске в `/code/models` через `cache_dir`. `FLUX.2-klein-9B` уже есть, но нам нужна **4B версия**.
- **`disable_adapter()` убран**: В SD3 SWD teacher = student (один объект), `disable_adapter()` открывает base weights. Для Klein teacher — отдельный merged объект, это чище и надёжнее.
- **CFG без двойного forward**: Klein `guidance` параметр встроен в трансформер. Для `cfg_teacher=3.5` просто передаём `guidance=3.5` в teacher forward.
- **`uncond_prompt_embeds` убраны**: Полностью ненужны для Klein.
- **`distributed_sampling` убран**: FID-метрики для text2img датасетов (COCO, MJHQ) нерелевантны для img2img реастера.
- **Scale-wise для img2img**: Condition image даунсэмплится синхронно с target при каждом boundary. Ранние шаги = коррекция экспозиции/тона на 256px, поздние = текстурный синтез на 512px.
- **Structured noise**: Заменяет `torch.randn_like` в начальном шуме — ключевая идея `train_img2img.sh` (параметр `--structural_noise_radius`).
- **Датасет (241 пример)**: Достаточно для дистилляции — лоссы про distribution matching, не про запоминание пар. При необходимости добавить `nfs_pix2pix_v5/v6` датасеты.

---

## Открытые вопросы

1. **Klein 4B блоки**: `inner_dim` известен — `num_attention_heads * attention_head_dim`. Для 4B дефолт конфига: `num_layers=8` dvojí-stream блоков + `num_single_layers=48` single-stream блоков. `--cls_blocks 8` (и `--mmd_blocks 8`) индексируют в `single_transformer_blocks`. Уточнить реальное число слоёв в 4B через `len(transformer.single_transformer_blocks)` после загрузки.
2. **`inner_dim`**: `transformer.inner_dim` есть напрямую — это `num_attention_heads * attention_head_dim`, не нужно идти через `.config`.
3. **`structural_noise_radius` как аргумент**: Добавить в `main.py` как `--structural_noise_radius` с default=100 (как в `train_img2img.sh`).
4. **Resolution 512 vs 768**: `train_img2img.sh` использует 768px, LoRA обучена на 512px. Начинаем с 512px в SWD for consistency, масштабируем позже.

---

## Создаваемые файлы

| Новый файл | Шаблон | Роль |
|---|---|---|
| `src/dataset_klein.py` | `src/dataset.py` | Paired HuggingFace датасет |
| `src/transformer_with_discriminator_klein.py` | `src/transformer_with_discriminator.py` | Klein-style forward + discriminator |
| `src/utils/prepare_utils_klein.py` | `src/utils/prepare_utils.py` | Загрузка Klein моделей, merge LoRA, encode prompt |
| `src/utils/train_utils_klein.py` | `src/utils/train_utils.py` | `sample_batch_klein` с paired data + scale-wise |
| `src/losses_klein.py` | `src/losses.py` | Три лосса с Klein forward signature |
| `src/flow_matching_sampler_klein.py` | `src/flow_matching_sampler.py` | `sampling_klein_img2img()` для валидации |
| `train_klein.py` | `train.py` | Основной тренировочный скрипт |
| `train_klein_img2img_single_gpu.sh` | `train_sd35_medium_single_gpu.sh` | Launch script |

**Изменяемые файлы:**
| Файл | Изменение |
|---|---|
| `main.py` | +`"klein"` в choices + dispatch |
