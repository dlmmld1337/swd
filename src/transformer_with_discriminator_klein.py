import torch
import torch.nn as nn

from accelerate.logging import get_logger

logger = get_logger(__name__)


########################################################################################################################
#                   TRANSFORMER WITH CLASSIFICATION HEAD FOR GAN DISTILLATION — FLUX.2 Klein                         #
########################################################################################################################


def FeedForward(dim, outdim=None):
    if outdim is None:
        outdim = dim
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.GELU(),
        nn.Linear(dim, outdim),
    )


class TransformerClsKlein(nn.Module):
    """
    Wraps a Flux2Transformer2DModel (with forward_with_feature_extraction_klein injected)
    and adds a linear discriminator head.  Mirrors TransformerCls from transformer_with_discriminator.py
    so that setup_utils.saving() / load_if_exist() work without modification.
    """

    def __init__(self, args, teacher_transformer):
        super().__init__()
        self.teacher_transformer = teacher_transformer

        dimensions = torch.linspace(
            teacher_transformer.inner_dim,
            1,
            args.num_discriminator_layers + 1,
            dtype=int,
        )
        layers = []
        for j, dim in enumerate(dimensions[:-1]):
            layers.append(FeedForward(dim.item(), dimensions[j + 1].item()))
        self.cls_pred_branch = nn.Sequential(*layers)
        self.cls_pred_branch.requires_grad_(True)

    @property
    def module(self):
        # In single-GPU mode this mirrors DDP's .module unwrap.
        return self

    def forward(self, *args, **kwargs):
        return self.teacher_transformer(*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
def forward_with_feature_extraction_klein(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.Tensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs=None,
    return_dict: bool = False,
    classify_index_block: list = [],
    return_only_features: bool = False,
    return_features: bool = True,
):
    """
    Drop-in replacement for Flux2Transformer2DModel.forward that also collects
    intermediate features from single_transformer_blocks for the GAN/MMD discriminator losses.

    Convention (mirrors forward_with_feature_extraction for SD3):
      - Returns ((output,), features)  when return_only_features=False
      - Returns  features              when return_only_features=True
    where:
      output   : (B, orig_noisy_len, C)  token predictions — trimmed to target tokens
      features : list[Tensor(B, orig_noisy_len, inner_dim)]

    IMPORTANT:
      - hidden_states must be the concatenated packed tensor [packed_noisy | packed_cond]
        with length 2 * noisy_seq_len (both must have the same spatial size)
      - timestep must be in [0, 1] — the model internally multiplies by 1000
      - guidance is the raw guidance scale (model internally multiplies by 1000)
    """
    # hidden_states layout: first half = noisy target, second half = condition
    orig_noisy_len = hidden_states.shape[1] // 2

    hidden_states_collect = []
    num_txt_tokens = encoder_hidden_states.shape[1]

    # ------------------------------------------------------------------
    # Replicate Flux2Transformer2DModel.forward step-by-step
    # ------------------------------------------------------------------

    # 1. Timestep and guidance embeddings
    #    The model's __call__ normally applies PEFT lora_scale here via @apply_lora_scale
    #    — we skip that decorator since we're calling the raw blocks directly.
    ts = timestep.to(hidden_states.dtype) * 1000
    gd = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None
    temb = self.time_guidance_embed(ts, gd)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    # 2. Input projections
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 3. Rotary positional embeddings
    _img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
    _txt_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
    image_rotary_emb = self.pos_embed(_img_ids)
    text_rotary_emb = self.pos_embed(_txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # 4. Dual-stream transformer blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    # 5. Merge txt + img for single-stream blocks
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 6. Single-stream blocks — collect features here
    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,  # encoder_hidden_states
                single_stream_mod,
                concat_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if classify_index_block and index_block in classify_index_block:
            # Slice: skip txt tokens, keep only noisy-target tokens (not cond tokens)
            img_features = hidden_states[
                :, num_txt_tokens : num_txt_tokens + orig_noisy_len, :
            ]
            hidden_states_collect.append(img_features)
            if return_only_features and index_block == classify_index_block[-1]:
                return hidden_states_collect

    # 7. Remove txt tokens; output projection
    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # 8. Trim output to noisy-target tokens only
    output = output[:, :orig_noisy_len, :]

    if return_only_features:
        return hidden_states_collect

    if return_features:
        return (output,), hidden_states_collect

    return (output,)
