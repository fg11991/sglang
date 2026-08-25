# SPDX-License-Identifier: Apache-2.0
"""Inference-only Mistral-Small-4 (`text_config.model_type == "mistral4"`) model.

Mistral-Small-4-119B-2603 is a multimodal ``Mistral3ForConditionalGeneration``
whose text stack is DeepSeek-V2-shaped: MLA (``q_lora_rank`` + ``kv_lora_rank``),
fine-grained MoE (128 routed + 1 shared, top-4), and YaRN. The delta from the
DeepSeek-V2/V3 backbone SGLang already ships is entirely at the *config surface*
plus the transformers-v5 weight layout, so this is a thin subclass. Everything
numerically hard (MLA, MoE routing, the interleaved+YaRN+llama-4 RoPE, and the
DFlash/DSpark aux-hidden-state capture hook) is inherited unchanged from
``DeepseekV2ForCausalLM``.

RoPE inherited from deepseek_v2 already covers mistral4:
  * ``rope_interleave: True``   -> is_neox_style = not it
  * ``rope_parameters`` (yarn)  -> read directly
  * ``llama_4_scaling_beta``    -> applied via config.llama_4_scaling
(added to this branch's deepseek_v2 "for supporting Mistral-Large-3").
NOTE: model_config._derive_model_shapes() only routes to the MLA branch for a
whitelist of architectures; "Mistral4ForCausalLM" was added there, otherwise the
override below silently falls through to MHA (use_mla=False on Ascend).

Loads the TEXT STACK ONLY. Offline DSpark capture is text-only, so instead of
routing MLA+MoE through the multimodal wrapper, the capture side forces this
architecture and we drop the vision weights::

    --json-model-override-args '{"architectures": ["Mistral4ForCausalLM"]}'

(SpecForge exposes this as ``--sglang-json-model-override-args`` on
``scripts/prepare_hidden_states.py``.)

======================================================================
ON-DEVICE (NPU) VERIFICATION REQUIRED -- cannot be checked without the
hardware and the transformers>=5.3 mistral4 reference:
  1. Logit parity vs the HF mistral4 reference for one prompt -- proves the
     RoPE variant + MoE routing. An offline DSpark capture is only as correct
     as this.
  2. FP8 static-activation numerics. The plumbing itself is settled, not open:
     mistral4 is ``activation_scheme: static`` with ``weight_block_size: null``,
     and Fp8MoEMethod.create_weights registers ``w13_input_scale`` /
     ``w2_input_scale`` (shape [num_experts]) for exactly that scheme, which the
     ordinary expert mapping reaches via
     ``experts.{e}.gate_proj.`` -> ``experts.w13_``. What still needs the device
     is whether the resulting fp8 path is numerically right on NPU -- covered by
     the logit-parity check above.
======================================================================
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM

logger = logging.getLogger(__name__)

# The multimodal checkpoint puts the text stack behind ``language_model.`` and
# the vision path behind ``vision_tower.`` / ``multi_modal_projector.``. Both
# layouts (``language_model.model.*`` and the v5-unified ``model.language_model.*``)
# are normalized to the standard DeepSeek layout ``model.*`` / ``lm_head.*``.
_VISION_MARKERS = ("vision_tower.", "multi_modal_projector.")

# transformers-v5 stacks routed experts across the expert dim; deepseek_v2's
# loader (make_expert_params_mapping) consumes per-expert dotted names. Map each
# checkpoint suffix to the per-expert suffix deepseek_v2 expects.
_EXPERT_SUFFIX_MAP = {
    "": ".weight",
    "_scale_inv": ".weight_scale_inv",
    "_activation_scale": ".input_scale",
}

# Matches one unstacked routed-expert weight, for load-coverage accounting.
_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)


def _translate_mistral4_config(text_config: PretrainedConfig) -> PretrainedConfig:
    """Fill the DeepSeek-V2 config attributes mistral4 leaves implicit.

    mistral4 already uses DeepSeek-compatible names for MLA and MoE, but
    ``Mistral4Config.__init__`` defines none of the following, and we hand the
    *nested text config* to DeepseekV3 -- so every one of these is missing and
    the parent would crash during construction, before any weight is loaded:

    * ``architectures`` -- ``PretrainedConfig`` defaults it to ``None`` and
      ``_patch_text_config`` propagates only pad/bos/eos/tie_word_embeddings, so
      the JSON architecture override never reaches the text config.
      ``determine_num_fused_shared_experts`` does ``self.config.architectures[0]``
      -> ``TypeError: 'NoneType' object is not subscriptable``. Naming it
      ``Mistral4ForCausalLM`` also correctly *disables* shared-expert fusion
      (that path is validated only for DeepSeek-V3-shaped 256/384-expert
      checkpoints; mistral4 stores its shared expert loose).
    * ``moe_layer_freq`` -- ``DeepseekV2DecoderLayer._is_layer_sparse`` evaluates
      ``layer_id % self.config.moe_layer_freq``; with ``n_routed_experts=128``
      and ``first_k_dense_replace=0`` the guard always falls through to it, so a
      missing attribute is an ``AttributeError`` on layer 0. All 36 layers of
      2603 carry routed experts (verified against the published index: 36/36
      have ``mlp.experts``, zero dense MLP layers), hence freq 1.
    * ``topk_method`` / ``scoring_func`` -- mistral4's router is plain softmax
      top-k with renormalization, NOT DeepSeek-V3's correction-bias gate.
      ``topk_method="greedy"`` keeps deepseek_v2 from allocating an
      ``e_score_correction_bias`` the checkpoint does not have.
    * ``llama_4_scaling`` -- mistral4 nests ``llama_4_scaling_beta`` and
      ``original_max_position_embeddings`` inside ``rope_parameters``;
      deepseek_v2 reads a top-level ``{original_max_position_embeddings, beta}``.
    """

    def _set_default(name: str, value) -> None:
        if getattr(text_config, name, None) is None:
            setattr(text_config, name, value)

    _set_default("architectures", ["Mistral4ForCausalLM"])
    _set_default("moe_layer_freq", 1)
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


def _split_gate_up(tensor: torch.Tensor, *, kind: str) -> tuple:
    """Split a stacked gate_up tensor's output dim into (gate, up).

    ``gate_up_proj`` fuses gate and up along the OUTPUT dim (dim 0 of a
    per-expert 2D weight ``[2*I, H]`` or its block scale ``[2*I/blk, H/blk]``).
    The activation (input) scale is per the INPUT, shared by gate and up, so it
    is never split -- callers handle that before reaching here.
    """
    out_dim = tensor.shape[0]
    if out_dim % 2 != 0:
        raise ValueError(
            f"cannot split fused gate_up {kind}: leading dim {out_dim} is odd "
            f"(shape {tuple(tensor.shape)})"
        )
    half = out_dim // 2
    return tensor[:half], tensor[half:]


def _to_text_name(name: str) -> Optional[str]:
    """Map a checkpoint key to the standard DeepSeek text layout, or None.

    ``language_model.model.X``   -> ``model.X``
    ``language_model.lm_head.X`` -> ``lm_head.X``
    ``model.language_model.X``   -> ``model.X`` (v5-unified layout)
    vision / projector keys      -> None
    """
    if any(marker in name for marker in _VISION_MARKERS):
        return None
    for prefix in ("language_model.", "model.language_model."):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            # ``language_model.model.*`` / ``language_model.lm_head.*`` already
            # carry the segment DeepSeek wants; the v5-unified
            # ``model.language_model.*`` layout needs the ``model.`` restored.
            if rest.startswith(("model.", "lm_head.")):
                return rest
            return "model." + rest
    return name


def _unstack_stacked_expert(base: str, tail: str, weight: torch.Tensor):
    """Turn one stacked expert tensor into per-expert deepseek_v2 names.

    ``tail`` is e.g. ``gate_up_proj`` / ``gate_up_proj_scale_inv`` /
    ``gate_up_proj_activation_scale`` / ``down_proj`` / ``down_proj_scale_inv``
    / ``down_proj_activation_scale``. The leading (expert) dim indexes experts;
    gate_up additionally fuses gate and up on the output dim, split for weight
    and block scale but NOT for the activation (input) scale, which is shared by
    gate and up and therefore copied.
    """
    for stem, targets in (
        ("gate_up_proj", ("gate_proj", "up_proj")),
        ("down_proj", ("down_proj",)),
    ):
        if not tail.startswith(stem):
            continue
        suffix = tail[len(stem) :]
        if suffix not in _EXPERT_SUFFIX_MAP:
            break  # unknown suffix -> passthrough below
        target_suffix = _EXPERT_SUFFIX_MAP[suffix]
        is_activation = suffix == "_activation_scale"
        fused_gate_up = stem == "gate_up_proj"
        for e in range(weight.shape[0]):
            per_expert = weight[e]
            if fused_gate_up and not is_activation:
                gate, up = _split_gate_up(per_expert, kind=suffix or "weight")
                yield f"{base}.{e}.gate_proj{target_suffix}", gate
                yield f"{base}.{e}.up_proj{target_suffix}", up
            else:
                # down_proj (all suffixes), or the gate_up activation scale which
                # is shared by gate and up -> copy to each, never split.
                for target in targets:
                    yield f"{base}.{e}.{target}{target_suffix}", per_expert
        return

    # Unknown expert tail: pass through so deepseek_v2 reports it rather than
    # silently dropping it.
    yield f"{base}.{tail}", weight


def _expert_weight_slot(name: str) -> Optional[tuple]:
    """``model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight`` -> (L, E, shard).

    Only the *weight* itself counts as a slot: scales are optional per quant
    scheme, so requiring them would make coverage depend on the checkpoint's
    quantization rather than on its structural completeness.
    """
    match = _EXPERT_WEIGHT_RE.match(name)
    if match is None:
        return None
    layer_id, expert_id, shard = match.groups()
    return int(layer_id), int(expert_id), shard


def _adapt_weight_name(name: str, weight: torch.Tensor):
    """Adapt one already-text-layout weight to deepseek_v2's expectations."""
    # transformers-v5 static-fp8 uses ``.activation_scale`` for the input scale
    # on non-fused projections (attention, shared experts); deepseek_v2 expects
    # ``.input_scale``. Fused routed-expert scales use the underscore
    # ``_activation_scale`` form and are handled by _unstack_stacked_expert.
    if name.endswith(".activation_scale"):
        name = name[: -len(".activation_scale")] + ".input_scale"

    if ".mlp.experts." not in name:
        yield name, weight
        return

    prefix, tail = name.split(".mlp.experts.", 1)
    yield from _unstack_stacked_expert(f"{prefix}.mlp.experts", tail, weight)


class Mistral4ForCausalLM(DeepseekV3ForCausalLM):
    """Text-only Mistral-Small-4 backbone (MLA + fine-grained MoE + YaRN).

    Inherits MLA, MoE, RoPE and -- crucially for offline DSpark --
    ``set_dflash_layers_to_capture`` / ``set_eagle3_layers_to_capture`` plus the
    aux-hidden-state return path from ``DeepseekV2ForCausalLM``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        text_config = getattr(config, "text_config", config)
        _translate_mistral4_config(text_config)
        super().__init__(config=text_config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Filled as super() consumes the generator, so it is complete by the
        # time load_weights returns. See _assert_expert_coverage for why the
        # loader's own matched-parameter set cannot carry this.
        expert_coverage: set = set()
        loaded = super().load_weights(
            self._prepare_text_weights(weights, expert_coverage)
        )
        self._assert_key_weights_loaded(loaded)
        self._assert_expert_coverage(expert_coverage)
        return loaded

    # ---- weight-name adaptation -------------------------------------------

    def _prepare_text_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        expert_coverage: set,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            text_name = _to_text_name(name)
            if text_name is None:
                continue  # vision / projector weight, dropped for text-only
            for adapted_name, adapted_weight in _adapt_weight_name(text_name, weight):
                slot = _expert_weight_slot(adapted_name)
                if slot is not None:
                    expert_coverage.add(slot)
                yield adapted_name, adapted_weight

    # ---- post-load validation ---------------------------------------------

    def _assert_key_weights_loaded(self, loaded) -> None:
        """Fail loudly if a structural parameter never received a checkpoint tensor.

        ``loaded`` is the set of parameter names deepseek_v2's loader actually
        resolved an incoming tensor to (``do_load_weights`` returns it) -- NOT
        the raw checkpoint keys, which it records before matching. A silent
        prefix / expert-naming drift leaves those params at their init value
        while the loader only *warns*; capture would then proceed on an
        uninitialized target. Verify the real load happened.

        Patterns are the loader's *post-fusion* parameter names: with
        ``q_lora_rank`` set, q_a/kv_a fuse into ``fused_qkv_a_proj_with_mqa``
        (a separate loader branch from the plain projections, so it needs its own
        entry), and routed experts fuse into ``experts.w13_weight`` /
        ``experts.w2_weight``. Match a family across any layer so the check is
        robust to layer sharding; per-(layer, expert, shard) completeness is
        checked separately in _assert_expert_coverage, because the fused names
        cannot express it. Assumes the offline-capture pp_size=1 (embed /
        lm_head / final norm live on the single pipeline rank), which the
        capture backend guarantees.
        """
        if loaded is None:
            # Verification is the point of this subclass; a base loader that does
            # not report what it matched makes "the target is initialized"
            # unprovable, and capture on an uninitialized target fails silently.
            raise RuntimeError(
                "Mistral4ForCausalLM: the weight loader returned None instead of "
                "its matched-parameter set, so the load cannot be verified. "
                "DeepseekV2ForCausalLM.load_weights must return "
                "do_load_weights()'s matched_params."
            )

        required = [
            "embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "self_attn.fused_qkv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate.weight",
            "mlp.experts.w13_weight",
            "mlp.experts.w2_weight",
        ]
        missing = [
            key for key in required if not any(key in name for name in loaded)
        ]
        if missing:
            raise RuntimeError(
                "Mistral4ForCausalLM: required target parameters were never fed a "
                f"checkpoint tensor: {missing}. This usually means a checkpoint "
                "prefix / expert-naming mismatch -- capture would otherwise "
                f"proceed on an uninitialized target. Matched {len(loaded)} "
                "parameters."
            )

    def _assert_expert_coverage(self, expert_coverage: set) -> None:
        """Require every (layer, expert, shard) routed-expert weight to be present.

        The loader's matched-parameter set CANNOT express this: its expert branch
        rewrites ``experts.{e}.gate_proj.`` -> ``experts.w13_``, so all 128
        experts and both gate/up shards of a layer collapse onto the single name
        ``...experts.w13_weight``. Half the experts could be missing from the
        checkpoint and the name-level check would still pass.

        So account on the emitting side instead: every unstacked
        ``experts.{e}.{gate,up,down}_proj.weight`` this adapter produced is
        recorded, and compared against what the config says must exist. That
        proves the checkpoint carried a complete, correctly named expert grid;
        placing each slot inside the fused parameter is then the loader's own
        ``weight_loader(param, w, name, shard_id=..., expert_id=...)`` contract.

        Counts the checkpoint stream, not this rank's shard, so it is unaffected
        by EP/TP/PP (the loader filters after we emit). Assumes the standard
        one-shot load: a partial/incremental weight-update call would trip it.
        """
        num_layers = int(self.config.num_hidden_layers)
        num_experts = int(self.config.n_routed_experts)
        first_moe_layer = int(getattr(self.config, "first_k_dense_replace", 0) or 0)
        moe_layer_freq = int(getattr(self.config, "moe_layer_freq", 1) or 1)

        expected = {
            (layer_id, expert_id, shard)
            for layer_id in range(first_moe_layer, num_layers)
            if layer_id % moe_layer_freq == 0
            for expert_id in range(num_experts)
            for shard in ("gate", "up", "down")
        }
        missing = expected - expert_coverage
        if missing:
            sample = sorted(missing)[:8]
            raise RuntimeError(
                "Mistral4ForCausalLM: the checkpoint did not provide every routed "
                f"expert weight -- {len(missing)} of {len(expected)} "
                "(layer, expert, gate|up|down) slots are absent, e.g. "
                f"{sample}. Those experts would keep their uninitialized values. "
                f"Saw {len(expert_coverage)} slots."
            )
        unexpected = expert_coverage - expected
        if unexpected:
            logger.warning(
                "Mistral4ForCausalLM: checkpoint carried %d routed-expert slots "
                "outside the configured grid (e.g. %s); they were forwarded to "
                "the loader, which drops unknown names.",
                len(unexpected),
                sorted(unexpected)[:4],
            )


EntryClass = Mistral4ForCausalLM
