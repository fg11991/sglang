# SPDX-License-Identifier: Apache-2.0
"""Inference-only Mistral-Small-4 (`text_config.model_type == "mistral4"`) model.

Mistral-Small-4-119B-2603 is a multimodal ``Mistral3ForConditionalGeneration``
whose text stack is DeepSeek-V2-shaped: MLA (``q_lora_rank`` + ``kv_lora_rank``),
fine-grained MoE (128 routed + 1 shared, top-4), and YaRN. The delta from the
DeepSeek-V2/V3 backbone SGLang already ships is entirely at the *config surface*
plus one weight-name difference, so this file is a thin subclass -- the same
shape as ``mistral_large_3.py``. Everything numerically hard (MLA, MoE routing,
the interleaved+YaRN+llama-4 RoPE, and the DFlash/DSpark aux-hidden-state capture
hook) is inherited unchanged from ``DeepseekV2ForCausalLM``.

This class loads the TEXT STACK ONLY. Offline DSpark capture is text-only and
needs no vision tower, so rather than teaching the multimodal wrapper
(LlavaForConditionalGeneration hardcodes LlamaForCausalLM) to route MLA+MoE, the
capture side forces this architecture and we read the nested ``text_config`` and
drop the ``vision_tower`` / ``multi_modal_projector`` weights. Select it with::

    --json-model-override-args '{"architectures": ["Mistral4ForCausalLM"]}'

(SpecForge exposes this as ``--sglang-json-model-override-args`` on
``scripts/prepare_hidden_states.py``.)

Why the inherited RoPE is already correct for mistral4:
  * ``rope_interleave: True``   -> deepseek_v2 reads it (is_neox_style = not it)
  * ``rope_parameters`` (yarn)  -> deepseek_v2 reads config.rope_parameters
  * ``llama_4_scaling_beta``    -> deepseek_v2 applies config.llama_4_scaling
These three were added to this branch's deepseek_v2 "for supporting
Mistral-Large-3"; mistral4 reuses them. The config translation below only
reshapes them into the attribute names deepseek_v2 reads.

======================================================================
ON-DEVICE (NPU) VERIFICATION REQUIRED -- cannot be checked without the
hardware and the transformers>=5.3 mistral4 reference:
  1. Logit parity: lm_head(last_hidden) for one prompt must match the HF
     mistral4 reference within fp tolerance. This is the gate that proves the
     RoPE variant (interleave + yarn mscale + llama-4 scaling) and the MoE
     routing match -- an offline DSpark capture is only as correct as this.
  2. FP8 fused-expert loading (see load_weights): the checkpoint stores experts
     stacked as ``mlp.experts.gate_up_proj`` / ``down_proj`` with block
     ``weight_scale_inv`` and static ``activation_scale``. The per-expert split
     of the block scales is the highest-risk part of this file. If this branch
     already grew a stacked ``experts.gate_up_proj`` loader, prefer routing to
     it over this manual split.
======================================================================
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

logger = logging.getLogger(__name__)

# Weight-name prefix the multimodal checkpoint puts the text stack behind.
_TEXT_PREFIX = "language_model."
# Sub-modules that belong to the vision path; dropped for a text-only load.
_VISION_PREFIXES = (
    "vision_tower.",
    "multi_modal_projector.",
    "model.vision_tower.",
    "model.multi_modal_projector.",
)


def _translate_mistral4_config(text_config: PretrainedConfig) -> PretrainedConfig:
    """Fill the DeepSeek-V2 config attributes mistral4 leaves implicit.

    mistral4 already uses DeepSeek-compatible names for MLA
    (``q_lora_rank`` / ``kv_lora_rank`` / ``qk_nope_head_dim`` /
    ``qk_rope_head_dim`` / ``v_head_dim``) and for MoE
    (``n_routed_experts`` / ``n_shared_experts`` / ``num_experts_per_tok`` /
    ``moe_intermediate_size`` / ``first_k_dense_replace`` / ``n_group`` /
    ``topk_group`` / ``norm_topk_prob`` / ``routed_scaling_factor``), so those
    pass through untouched. Only three things are missing:

    * ``topk_method`` / ``scoring_func`` -- mistral4's router is plain softmax
      top-k with renormalization, NOT DeepSeek-V3's auxiliary-loss-free
      correction-bias gate. ``topk_method="greedy"`` keeps deepseek_v2 from
      allocating an ``e_score_correction_bias`` the checkpoint does not have
      (deepseek_v2 only creates it when ``topk_method == "noaux_tc"``).
    * ``llama_4_scaling`` -- mistral4 nests ``llama_4_scaling_beta`` and
      ``original_max_position_embeddings`` inside ``rope_parameters``;
      deepseek_v2 reads a top-level ``{original_max_position_embeddings, beta}``.
    """

    def _set_default(name: str, value) -> None:
        if getattr(text_config, name, None) is None:
            setattr(text_config, name, value)

    _set_default("topk_method", "greedy")
    _set_default("scoring_func", "softmax")

    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    beta = rope_parameters.get("llama_4_scaling_beta")
    if beta is not None and getattr(text_config, "llama_4_scaling", None) is None:
        original_max = rope_parameters.get(
            "original_max_position_embeddings",
            getattr(text_config, "max_position_embeddings", None),
        )
        text_config.llama_4_scaling = {
            "original_max_position_embeddings": original_max,
            "beta": beta,
        }

    return text_config


class Mistral4ForCausalLM(DeepseekV3ForCausalLM):
    """Text-only Mistral-Small-4 backbone (MLA + fine-grained MoE + YaRN).

    Inherits MLA, MoE, RoPE (interleave/yarn/llama-4) and, crucially for the
    offline DSpark pipeline, ``set_dflash_layers_to_capture`` /
    ``set_eagle3_layers_to_capture`` + the aux-hidden-state return path from
    ``DeepseekV2ForCausalLM``. SpecForge's offline capture backend falls back
    from ``set_dspark_layers_to_capture`` to ``set_dflash_layers_to_capture``,
    so the inherited hook is sufficient for DSpark data preparation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # The forced architecture keeps the multimodal top-level config; operate
        # on the nested text_config so DeepseekV3 sees MLA/MoE/rope attributes at
        # the top level, matching a standalone causal-LM config.
        text_config = getattr(config, "text_config", config)
        _translate_mistral4_config(text_config)
        super().__init__(config=text_config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Keep only the text stack, strip its prefix, unstack fused experts.

        The multimodal checkpoint lays the text stack behind
        ``language_model.`` (``language_model.model.layers.*``,
        ``language_model.model.embed_tokens.weight``,
        ``language_model.lm_head.weight``). Stripping that prefix yields the
        standard DeepSeek HF layout deepseek_v2 expects. Vision / projector
        weights are dropped.

        Routed experts arrive transformers-v5 fused
        (``mlp.experts.gate_up_proj`` [E, 2*I, H] and ``down_proj`` [E, H, I]);
        deepseek_v2's FusedMoE loader consumes per-expert names, so unstack them
        (gate_up additionally halves along the intermediate dim). The static fp8
        activation scale is renamed to the name deepseek_v2 expects.

        !!! ON-DEVICE VERIFICATION REQUIRED for the per-expert block-scale split
        (see module docstring). !!!
        """

        return super().load_weights(self._prepare_text_weights(weights))

    def _prepare_text_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            if name.startswith(_VISION_PREFIXES) or name.startswith(
                tuple("model." + p for p in ("vision_tower.", "multi_modal_projector."))
            ):
                continue
            if name.startswith(_TEXT_PREFIX):
                name = name[len(_TEXT_PREFIX) :]
            elif name.startswith("model." + _TEXT_PREFIX):
                # transformers-v5 sometimes nests as model.language_model.*
                name = name[len("model." + _TEXT_PREFIX) :]
            else:
                # A bare model.* / lm_head.* key already in text layout is fine;
                # anything else that is not vision is unexpected -- let it flow to
                # super().load_weights, which raises on genuinely unknown names.
                pass

            yield from self._maybe_unstack_expert(name, weight)

    def _maybe_unstack_expert(self, name, weight):
        # Static fp8 activation scale -> deepseek_v2's input_scale name.
        if name.endswith(".activation_scale"):
            name = name[: -len(".activation_scale")] + ".input_scale"

        if ".mlp.experts." not in name:
            yield name, weight
            return

        prefix, tail = name.split(".mlp.experts.", 1)
        base = f"{prefix}.mlp.experts"

        if tail.startswith("gate_up_proj"):
            suffix = tail[len("gate_up_proj") :]  # "" | ".weight_scale_inv" | ...
            num_experts = weight.shape[0]
            half = weight.shape[1] // 2
            for e in range(num_experts):
                per_expert = weight[e]
                yield f"{base}.{e}.gate_proj{suffix}", per_expert[:half]
                yield f"{base}.{e}.up_proj{suffix}", per_expert[half:]
            return

        if tail.startswith("down_proj"):
            suffix = tail[len("down_proj") :]
            for e in range(weight.shape[0]):
                yield f"{base}.{e}.down_proj{suffix}", weight[e]
            return

        yield name, weight


EntryClass = Mistral4ForCausalLM
