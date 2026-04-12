# План исправлений SWD Klein img2img дистилляции

## Проблема

После 1250 шагов тренировки результат дистилляции хуже, чем базовая модель за 4 шага без дистилляции.

## Анализ

Проведён детальный аудит всех Klein-адаптированных файлов и сравнение с оригинальной SWD-реализацией для SD3.5.

### Архитектура (корректна)

- **Три лосса** (DMD + GAN + MMD) — перенесены корректно из `losses.py` → `losses_klein.py`
- **call_klein_transformer** — обёртка forward pass: packing concat [noisy_target | cond], передача guidance, unpacking. Корректно.
- **TransformerClsKlein** — дискриминатор с feature extraction из single_transformer_blocks. Корректно.
- **pack_klein_input / unpack_klein_output** — корректная работа с Klein patchified latents.
- **sample_batch_klein** — scale-wise downsampling, VAE encode+normalize, boundary cycling. Корректно.
- **Structured noise** — используется generate_structured_noise_batch_vectorized вместо randn_like. По дизайну.
- **Teacher offloading** — `_teacher_on_device` перемещает teacher на GPU только для forward pass. Корректно.

### Обнаруженные баги

#### Баг 1 (КРИТИЧЕСКИЙ): guidance scale не используется в инференс-сэмплере

**Файл:** `src/flow_matching_sampler_klein.py`, функция `sampling_klein_img2img`

Функция принимает параметр `guidance_scale=1.0`, но **никогда его не использует**. Модель вызывается с `guidance=None`:

```python
out = model(
    hidden_states=hidden_states.to(dtype),
    encoder_hidden_states=prompt_embeds,
    timestep=T / 1000,
    img_ids=img_ids,
    txt_ids=text_ids,
    guidance=None,           # ← БАГ: должен быть guidance_scale
    return_dict=False,
)
```

**Во время тренировки** через `call_klein_transformer`:
```python
guidance = torch.full([B], guidance_scale, device=device, dtype=dtype)  # guidance_scale=3.5
```

Klein — guidance-distilled модель: guidance embedding — часть conditioning. Student обучается предсказывать при guidance=3.5, но на инференсе видит guidance=None (нулевой embedding). Это значительное расхождение train/test, которое объясняет деградацию.

#### Баг 2 (ВАЖНЫЙ): validation вызывается с guidance_scale=1.0

**Файл:** `train_klein.py`, функция `log_validation_klein`

```python
sampled_latents = sampling_klein_img2img(
    ...,
    guidance_scale=1.0,  # ← должен быть args.cfg_teacher (3.5)
    ...
)
```

Даже после исправления Бага 1, validation будет использовать guidance=1.0 вместо 3.5.

### Что НЕ является проблемой (проверено)

| Компонент | Статус |
|---|---|
| Формула Euler step: x₀ = xₜ - σ·v | ✓ Одинакова в losses и sampler |
| Sigma/timestep индексация | ✓ Boundary indices корректны |
| Scale-wise прогрессия в sampler | ✓ Matches training boundaries |
| pack/unpack Klein input/output | ✓ Корректная работа с concatenated tokens |
| Teacher forward pass | ✓ CPU-offloading и no_grad работают |
| Feature extraction в discriminator | ✓ single_transformer_blocks features |
| Structured noise | ✓ Одинаков в training и inference |
| LoRA через PeftModel | ✓ Атрибуты и forward проброс работают |
| DMD gradient формула | ✓ Идентична оригиналу |
| Dataset / batch sampling | ✓ Корректные scale-wise пары |

## Исправления

### Fix 1: Использовать guidance_scale в sampler

В `src/flow_matching_sampler_klein.py`, создать guidance tensor и передать в модель:

```python
guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)
# ...
out = model(
    ...
    guidance=guidance,
    ...
)
```

### Fix 2: Передавать cfg_teacher в validation

В `train_klein.py`:

```python
sampling_klein_img2img(
    ...,
    guidance_scale=args.cfg_teacher,  # вместо 1.0
    ...
)
```

### Fix 3: Передавать args в log_validation_klein (уже передаётся)

Параметр `args` уже доступен в `log_validation_klein`, нужно только использовать `args.cfg_teacher`.
