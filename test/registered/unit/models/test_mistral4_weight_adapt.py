"""CPU unit tests for Mistral-Small-4 (mistral4) offline-capture weight adapt.

Exercises the pure weight-name adaptation helpers of ``mistral4.py`` with
synthetic tensors shaped like the published checkpoint -- no model
instantiation, no GPU.

Guarded bugs (all observed against the real 2603 safetensors header):

  * ``gate_up_proj_scale_inv`` / ``down_proj_scale_inv`` are BF16 ``[128, 1, 1]``
    -- one scalar per expert, NOT a block grid. Splitting them along the output
    dim takes the leading dim of a ``[1, 1]`` slice and raises "leading dim 1 is
    odd", so every load died on the first expert scale. They must be COPIED to
    gate and up; only a genuine block scale is split.
  * The checkpoint spells its scales ``..._scale_inv`` even though the
    quantisation is per-tensor (``activation_scheme: static``,
    ``weight_block_size: null``). SGLang registers ``weight_scale`` /
    ``w13_weight_scale`` / ``w2_weight_scale`` for that scheme and
    ``*_scale_inv`` only for block quant, so passing the checkpoint spelling
    through made 360 scale keys target parameters that do not exist: silently
    dropped for routed experts, KeyError in the fused-qkv and shared-expert
    branches (they index params_dict unguarded).
  * ``_to_text_name``: the multimodal ``language_model.`` prefix must be
    stripped to the DeepSeek text layout, and vision weights dropped.

Run: python3 test/registered/unit/models/test_mistral4_weight_adapt.py
"""

import unittest

import torch

from sglang.srt.models.mistral4 import (
    _adapt_weight_name,
    _expected_param_family,
    _split_gate_up,
    _to_text_name,
    _unstack_stacked_expert,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

BASE = "model.layers.0.mlp.experts"


class TestMistral4ToTextName(CustomTestCase):
    def test_language_model_prefix_stripped_to_deepseek_layout(self):
        self.assertEqual(
            _to_text_name("language_model.model.embed_tokens.weight"),
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            _to_text_name("language_model.lm_head.weight"), "lm_head.weight"
        )
        self.assertEqual(
            _to_text_name("language_model.model.layers.0.self_attn.q_a_proj.weight"),
            "model.layers.0.self_attn.q_a_proj.weight",
        )

    def test_v5_unified_layout_restores_model_segment(self):
        self.assertEqual(
            _to_text_name("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight",
        )

    def test_vision_and_projector_dropped(self):
        self.assertIsNone(
            _to_text_name("vision_tower.transformer.layers.0.attention.q_proj.weight")
        )
        self.assertIsNone(_to_text_name("multi_modal_projector.norm.weight"))


class TestMistral4SplitGateUp(CustomTestCase):
    def test_splits_output_dim(self):
        t = torch.arange(2 * 8 * 6).reshape(2 * 8, 6).float()
        gate, up = _split_gate_up(t, kind="weight")
        self.assertEqual(gate.shape, (8, 6))
        self.assertTrue(torch.equal(gate, t[:8]))
        self.assertTrue(torch.equal(up, t[8:]))

    def test_odd_leading_dim_raises(self):
        with self.assertRaises(ValueError):
            _split_gate_up(torch.zeros(7, 3), kind="weight")


class TestMistral4UnstackExpert(CustomTestCase):
    E, I, H, BLK = 4, 8, 6, 2

    def test_gate_up_weight_split_per_expert(self):
        w = (
            torch.arange(self.E * 2 * self.I * self.H)
            .reshape(self.E, 2 * self.I, self.H)
            .float()
        )
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj", w))
        self.assertEqual(len(out), self.E * 2)
        self.assertEqual(out[f"{BASE}.0.gate_proj.weight"].shape, (self.I, self.H))
        self.assertTrue(torch.equal(out[f"{BASE}.0.gate_proj.weight"], w[0][: self.I]))
        self.assertTrue(torch.equal(out[f"{BASE}.0.up_proj.weight"], w[0][self.I :]))

    def test_per_tensor_weight_scale_is_copied_and_renamed(self):
        """The published layout: [E, 1, 1]. Pre-fix this raised on the odd dim."""
        s = torch.arange(self.E).reshape(self.E, 1, 1).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_scale_inv", s))

        # Per-tensor -> the parameter SGLang registers is weight_scale, not _inv.
        self.assertIn(f"{BASE}.0.gate_proj.weight_scale", out)
        self.assertIn(f"{BASE}.0.up_proj.weight_scale", out)
        self.assertFalse(any("weight_scale_inv" in key for key in out))
        # Copied whole, not halved: shape preserved and gate == up.
        self.assertEqual(out[f"{BASE}.0.gate_proj.weight_scale"].shape, (1, 1))
        self.assertTrue(
            torch.equal(
                out[f"{BASE}.2.gate_proj.weight_scale"],
                out[f"{BASE}.2.up_proj.weight_scale"],
            )
        )

    def test_per_tensor_down_proj_scale_renamed(self):
        s = torch.ones(self.E, 1, 1)
        out = dict(_unstack_stacked_expert(BASE, "down_proj_scale_inv", s))
        self.assertIn(f"{BASE}.0.down_proj.weight_scale", out)
        self.assertFalse(any("weight_scale_inv" in key for key in out))

    def test_genuine_block_scale_is_still_split_and_keeps_inv(self):
        """A real block grid (numel > 1 per expert) keeps the block behaviour."""
        s = (
            torch.arange(self.E * (2 * self.I // self.BLK) * (self.H // self.BLK))
            .reshape(self.E, 2 * self.I // self.BLK, self.H // self.BLK)
            .float()
        )
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_scale_inv", s))
        self.assertEqual(
            out[f"{BASE}.0.gate_proj.weight_scale_inv"].shape,
            (self.I // self.BLK, self.H // self.BLK),
        )

    def test_activation_scale_1d_copied_as_input_scale(self):
        """The published layout: [E] -> a 0-d scalar per expert."""
        a = torch.arange(self.E).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_activation_scale", a))
        self.assertEqual(out[f"{BASE}.0.gate_proj.input_scale"].ndim, 0)
        # The input scale describes the shared input: gate and up get the same.
        self.assertTrue(
            torch.equal(
                out[f"{BASE}.1.gate_proj.input_scale"],
                out[f"{BASE}.1.up_proj.input_scale"],
            )
        )

    def test_activation_scale_3d_copied_as_input_scale(self):
        a = torch.arange(self.E).reshape(self.E, 1, 1).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_activation_scale", a))
        self.assertEqual(out[f"{BASE}.2.gate_proj.input_scale"].shape, (1, 1))

    def test_down_proj_weight_and_activation(self):
        dw = (
            torch.arange(self.E * self.H * self.I)
            .reshape(self.E, self.H, self.I)
            .float()
        )
        out = dict(_unstack_stacked_expert(BASE, "down_proj", dw))
        self.assertEqual(out[f"{BASE}.0.down_proj.weight"].shape, (self.H, self.I))

        a = torch.arange(self.E).float()
        out = dict(_unstack_stacked_expert(BASE, "down_proj_activation_scale", a))
        self.assertIn(f"{BASE}.0.down_proj.input_scale", out)


class TestMistral4AdaptWeightName(CustomTestCase):
    """Plain projections: attention (fused qkv) and shared experts."""

    def test_scalar_weight_scale_inv_renamed_to_weight_scale(self):
        # 2603 stores these as 0-d BF16 scalars; the registered parameter for a
        # per-tensor fp8 linear is weight_scale. Pre-fix these passed through as
        # weight_scale_inv and the loader raised KeyError.
        for name in (
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            "model.layers.0.mlp.shared_experts.gate_proj.weight_scale_inv",
        ):
            out = dict(_adapt_weight_name(name, torch.tensor(1.0)))
            adapted = next(iter(out))
            self.assertTrue(adapted.endswith(".weight_scale"), adapted)
            self.assertNotIn("weight_scale_inv", adapted)

    def test_block_weight_scale_inv_kept(self):
        out = dict(
            _adapt_weight_name(
                "model.layers.0.self_attn.o_proj.weight_scale_inv",
                torch.zeros(4, 4),
            )
        )
        self.assertIn("model.layers.0.self_attn.o_proj.weight_scale_inv", out)

    def test_activation_scale_renamed(self):
        out = dict(
            _adapt_weight_name(
                "model.layers.0.self_attn.q_a_proj.activation_scale",
                torch.tensor(1.0),
            )
        )
        self.assertIn("model.layers.0.self_attn.q_a_proj.input_scale", out)

    def test_shared_experts_not_treated_as_routed(self):
        name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        out = dict(_adapt_weight_name(name, torch.zeros(3, 4)))
        # shared_experts pass straight through, not unstacked per-expert.
        self.assertIn(name, out)


class TestMistral4ExpectedParamFamily(CustomTestCase):
    """The families _assert_key_weights_loaded requires among the matches."""

    def test_expert_scales_map_to_fused_families(self):
        self.assertEqual(
            _expected_param_family(f"{BASE}.0.gate_proj.weight_scale"),
            "experts.w13_weight_scale",
        )
        self.assertEqual(
            _expected_param_family(f"{BASE}.7.up_proj.input_scale"),
            "experts.w13_input_scale",
        )
        self.assertEqual(
            _expected_param_family(f"{BASE}.0.down_proj.weight_scale"),
            "experts.w2_weight_scale",
        )

    def test_expert_weight_has_no_scale_family(self):
        # Covered by the (layer, expert, shard) grid instead.
        self.assertIsNone(_expected_param_family(f"{BASE}.0.gate_proj.weight"))

    def test_plain_linear_scale_family_is_the_suffix(self):
        self.assertEqual(
            _expected_param_family("model.layers.0.self_attn.q_a_proj.weight_scale"),
            ".weight_scale",
        )
        self.assertIsNone(
            _expected_param_family("model.layers.0.self_attn.q_a_proj.weight")
        )


class TestMistral4ExpertScaleMappingIntegration(CustomTestCase):
    """The adapter's per-expert names must resolve through the REAL deepseek_v2
    expert mapping to the FusedMoE parameters a static per-tensor fp8 checkpoint
    actually registers.

    Uses ``FusedMoE.make_expert_params_mapping`` (the exact mapping
    deepseek_weight_loader builds) and replays the loader's
    ``name.replace(weight_name, param_name)`` + ``if name not in params_dict``
    step, like existing MoE loader tests (test_inkling_per_expert_sync) do.

    The params_dict is not arbitrary: for ``activation_scheme: "static"`` with
    ``weight_block_size: null`` -- 2603's scheme -- ``Fp8MoEMethod.create_weights``
    registers ``w13_weight_scale`` / ``w2_weight_scale`` and ``w13_input_scale``
    / ``w2_input_scale``, and reserves the ``*_scale_inv`` names for block quant.
    Scope: this pins the NAME resolution. Building a real quantized FusedMoE
    needs a device, so end-to-end fp8 behaviour is covered by the on-device
    logit-parity run, not here.
    """

    NUM_EXPERTS = 2
    # Exactly what Fp8MoEMethod registers for static per-tensor fp8.
    STATIC_FP8_PARAMS = (
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
        "model.layers.0.mlp.experts.w13_weight_scale",
        "model.layers.0.mlp.experts.w2_weight_scale",
        "model.layers.0.mlp.experts.w13_input_scale",
        "model.layers.0.mlp.experts.w2_input_scale",
    )

    def _mapping(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.NUM_EXPERTS,
        )

    def _resolve(self, checkpoint_name, params_dict, mapping):
        for param_name, weight_name, expert_id, shard_id in mapping:
            if weight_name in checkpoint_name:
                resolved = checkpoint_name.replace(weight_name, param_name)
                if resolved not in params_dict:
                    continue
                return resolved, shard_id, expert_id
        return None

    def test_per_tensor_weight_scale_resolves_to_w13_weight_scale(self):
        adapted = dict(
            _unstack_stacked_expert(
                BASE,
                "gate_up_proj_scale_inv",
                torch.ones(self.NUM_EXPERTS, 1, 1),
            )
        )
        params = {name: object() for name in self.STATIC_FP8_PARAMS}
        mapping = self._mapping()

        gate = self._resolve(f"{BASE}.0.gate_proj.weight_scale", params, mapping)
        self.assertIsNotNone(gate, "per-tensor weight scale must reach a parameter")
        self.assertEqual(gate[0], "model.layers.0.mlp.experts.w13_weight_scale")
        self.assertEqual((gate[1], gate[2]), ("w1", 0))

        up = self._resolve(f"{BASE}.1.up_proj.weight_scale", params, mapping)
        self.assertEqual(up[0], "model.layers.0.mlp.experts.w13_weight_scale")
        self.assertEqual((up[1], up[2]), ("w3", 1))

        down = self._resolve(f"{BASE}.0.down_proj.weight_scale", params, mapping)
        self.assertEqual(down[0], "model.layers.0.mlp.experts.w2_weight_scale")
        self.assertEqual((down[1], down[2]), ("w2", 0))

    def test_checkpoint_spelling_would_not_resolve(self):
        """Why the rename is required: _scale_inv finds no static-fp8 parameter."""
        params = {name: object() for name in self.STATIC_FP8_PARAMS}
        self.assertIsNone(
            self._resolve(
                f"{BASE}.0.gate_proj.weight_scale_inv", params, self._mapping()
            )
        )

    def test_activation_scale_resolves_to_input_scale(self):
        adapted = dict(
            _unstack_stacked_expert(
                BASE,
                "gate_up_proj_activation_scale",
                torch.ones(self.NUM_EXPERTS),
            )
        )
        self.assertIn(f"{BASE}.0.gate_proj.input_scale", adapted)

        params = {name: object() for name in self.STATIC_FP8_PARAMS}
        resolved = self._resolve(
            f"{BASE}.0.gate_proj.input_scale", params, self._mapping()
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved[0], "model.layers.0.mlp.experts.w13_input_scale")
        self.assertEqual((resolved[1], resolved[2]), ("w1", 0))


if __name__ == "__main__":
    unittest.main()
