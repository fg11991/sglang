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
  2. FP8 static per-tensor numerics. The naming is settled (see
     _scale_param_name: the checkpoint spells its scales ``..._scale_inv`` but
     the tensors are per-tensor, so the parameters are ``weight_scale`` /
     ``w13_weight_scale`` / ``w2_weight_scale`` plus ``input_scale``), and
     _assert_key_weights_loaded fails loudly if any emitted scale finds no
     parameter. What still needs the device is whether the resulting fp8 path is
     numerically right on NPU -- covered by the logit-parity check above.
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
# Stacked routed-expert tails this adapter understands. The emitted *name* for a
# weight scale depends on the tensor, not just the tail -- see _scale_param_name.
_EXPERT_TAILS = ("", "_scale_inv", "_activation_scale")

# Matches one unstacked routed-expert tensor, for load accounting.
_EXPERT_PARAM_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(\w+)$"
)
# Scale parameters whose family must be seen among the loader's matches.
_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".input_scale")


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
    / ``down_proj_activation_scale``. The leading dim always indexes experts.

    Whether the gate/up halves are SPLIT or COPIED depends on what the tensor
    describes, not on the tail:

    * the weight itself fuses gate and up along the output dim -> split;
    * a *block* scale has one entry per output block -> split;
    * a *per-tensor* weight scale is one scalar for the whole expert -> copy it
      to gate and up. 2603 ships exactly this (``[128, 1, 1]``), and splitting
      would take the leading dim of a ``[1, 1]`` slice and raise on it being odd;
    * an activation (input) scale describes the shared input -> copy.
    """
    for stem, targets in (
        ("gate_up_proj", ("gate_proj", "up_proj")),
        ("down_proj", ("down_proj",)),
    ):
        if not tail.startswith(stem):
            continue
        suffix = tail[len(stem) :]
        if suffix not in _EXPERT_TAILS:
            break  # unknown suffix -> passthrough below
        fused_gate_up = stem == "gate_up_proj"
        for e in range(weight.shape[0]):
            per_expert = weight[e]
            if suffix == "":
                param, splittable = "weight", True
            elif suffix == "_activation_scale":
                param, splittable = "input_scale", False
            else:  # "_scale_inv"
                param = _scale_param_name(per_expert)
                splittable = not _is_per_tensor_scale(per_expert)

            if fused_gate_up and splittable:
                gate, up = _split_gate_up(per_expert, kind=suffix or "weight")
                yield f"{base}.{e}.gate_proj.{param}", gate
                yield f"{base}.{e}.up_proj.{param}", up
            else:
                # down_proj (single target), or a gate_up scalar shared by gate
                # and up -> copy to each target, never split.
                for target in targets:
                    yield f"{base}.{e}.{target}.{param}", per_expert
        return

    # Unknown expert tail: pass through so deepseek_v2 reports it rather than
    # silently dropping it.
    yield f"{base}.{tail}", weight


def _is_per_tensor_scale(scale: torch.Tensor) -> bool:
    """A single scalar covering the whole (per-expert) tensor, not a block grid."""
    return scale.numel() == 1


def _scale_param_name(scale: torch.Tensor) -> str:
    """Parameter name SGLang registers for this weight scale.

    fp8 has two layouts and SGLang names them differently:

    * block-quantised (``weight_block_size`` set) -> ``weight_scale_inv``
      (``w13_weight_scale_inv`` / ``w2_weight_scale_inv`` on a FusedMoE), and
      that path additionally asserts ``activation_scheme == "dynamic"``;
    * per-tensor (``weight_block_size: null``) -> ``weight_scale``
      (``w13_weight_scale`` / ``w2_weight_scale``).

    Mistral-Small-4-119B-2603 is the second kind -- ``activation_scheme: static``
    with ``weight_block_size: null`` -- yet its checkpoint *keys* are still
    spelled ``..._scale_inv``. Measured from the published safetensors header:
    ``experts.gate_up_proj_scale_inv`` and ``experts.down_proj_scale_inv`` are
    BF16 ``[128, 1, 1]`` (one scalar per expert) and every linear
    ``weight_scale_inv`` is a BF16 0-d scalar. Taking the checkpoint spelling at
    face value would target parameters that do not exist: the expert scales
    would be silently dropped (``if name not in params_dict: continue``) and the
    fused-qkv / shared-expert branches -- which index ``params_dict[name]``
    without a guard -- would raise KeyError. So decide by the tensor.
    """
    return "weight_scale" if _is_per_tensor_scale(scale) else "weight_scale_inv"


def _expert_weight_slot(name: str) -> Optional[tuple]:
    """``model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight`` -> (L, E, shard).

    Only the *weight* itself counts as a slot: scales are optional per quant
    scheme, so requiring them would make coverage depend on the checkpoint's
    quantization rather than on its structural completeness.
    """
    match = _EXPERT_PARAM_RE.match(name)
    if match is None or match.group(4) != "weight":
        return None
    layer_id, expert_id, shard, _ = match.groups()
    return int(layer_id), int(expert_id), shard


def _expected_param_family(name: str) -> Optional[str]:
    """The post-mapping family a scale must show up under, or None.

    The loader rewrites ``experts.{e}.gate_proj.`` to ``experts.w13_``, so an
    emitted ``experts.3.gate_proj.weight_scale`` must land on some
    ``...experts.w13_weight_scale``. Plain projections keep their trailing
    parameter name through the fused-qkv / gate_up renames, so a suffix match is
    enough for them. Only scales are tracked: a wrong scale spelling is the
    failure that either KeyErrors or silently skips.
    """
    match = _EXPERT_PARAM_RE.match(name)
    if match is not None:
        shard, param = match.group(3), match.group(4)
        if param == "weight":
            return None  # covered by the (layer, expert, shard) grid instead
        prefix = "w2" if shard == "down" else "w13"
        return f"experts.{prefix}_{param}"
    for suffix in _SCALE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def _adapt_weight_name(name: str, weight: torch.Tensor):
    """Adapt one already-text-layout weight to deepseek_v2's expectations."""
    # Plain projections (attention, shared experts). transformers-v5 static-fp8
    # spells the input scale ``.activation_scale`` and the weight scale
    # ``.weight_scale_inv`` even when the quantisation is per-tensor; SGLang
    # registers ``.input_scale`` and (for per-tensor) ``.weight_scale``. Getting
    # this wrong is not silent here: the fused-qkv and shared-expert branches
    # index params_dict without a guard and raise KeyError.
    # Stacked routed-expert scales use the underscore spellings and are handled
    # by _unstack_stacked_expert.
    if name.endswith(".activation_scale"):
        name = name[: -len(".activation_scale")] + ".input_scale"
    elif name.endswith(".weight_scale_inv"):
        name = name[: -len(".weight_scale_inv")] + "." + _scale_param_name(weight)

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
        # Both are filled as super() consumes the generator, so they are complete
        # by the time load_weights returns. See _assert_expert_coverage for why
        # the loader's own matched-parameter set cannot carry the grid.
        expert_coverage: set = set()
        scale_families: set = set()
        loaded = super().load_weights(
            self._prepare_text_weights(weights, expert_coverage, scale_families)
        )
        self._assert_key_weights_loaded(loaded, scale_families)
        self._assert_expert_coverage(expert_coverage)
        return loaded

    # ---- weight-name adaptation -------------------------------------------

    def _prepare_text_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        expert_coverage: set,
        scale_families: set,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            text_name = _to_text_name(name)
            if text_name is None:
                continue  # vision / projector weight, dropped for text-only
            for adapted_name, adapted_weight in _adapt_weight_name(text_name, weight):
                slot = _expert_weight_slot(adapted_name)
                if slot is not None:
                    expert_coverage.add(slot)
                family = _expected_param_family(adapted_name)
                if family is not None:
                    scale_families.add(family)
                yield adapted_name, adapted_weight

    # ---- post-load validation ---------------------------------------------

    def _assert_key_weights_loaded(self, loaded, scale_families: set = frozenset()) -> None:
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

        # Every scale spelling this adapter emitted must have reached a real
        # parameter. A wrong fp8 scale name is the quiet failure mode: routed
        # expert scales are dropped by the loader's `not in params_dict` guard
        # and the model then runs fp8 weights against default scales.
        unmatched = [
            family
            for family in sorted(scale_families)
            if not any(name.endswith(family) for name in loaded)
        ]
        if unmatched:
            raise RuntimeError(
                "Mistral4ForCausalLM: quantization scales were emitted under "
                f"names no parameter accepted: {unmatched}. The checkpoint's "
                "fp8 layout and the registered parameter names disagree (e.g. "
                "per-tensor 'weight_scale' vs block 'weight_scale_inv'); the "
                "model would run fp8 weights against default scales."
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
