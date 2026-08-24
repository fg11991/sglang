"""CPU unit tests for Mistral-Small-4 (mistral4) offline-capture weight adapt.

Exercises the pure weight-name adaptation helpers of ``mistral4.py`` with
synthetic tensors -- no model instantiation, no GPU:

  - ``_to_text_name``: strip the multimodal ``language_model.`` prefix (both
    layouts) to the DeepSeek text layout; drop vision / projector weights.
  - ``_split_gate_up``: split a fused gate_up tensor's OUTPUT dim; reject odd.
  - ``_unstack_stacked_expert``: turn transformers-v5 stacked experts into the
    per-expert dotted names deepseek_v2's loader consumes. Weight and block
    scale split on the output dim; the activation (input) scale is shared by
    gate and up and copied, never split.
  - ``_adapt_weight_name``: ``.activation_scale`` -> ``.input_scale`` on
    non-fused projections; shared_experts are NOT mistaken for routed experts.

Run: python3 test/registered/unit/models/test_mistral4_weight_adapt.py
"""

import unittest

import torch

from sglang.srt.models.mistral4 import (
    _adapt_weight_name,
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
        w = torch.arange(self.E * 2 * self.I * self.H).reshape(
            self.E, 2 * self.I, self.H
        ).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj", w))
        self.assertEqual(len(out), self.E * 2)
        self.assertEqual(out[f"{BASE}.0.gate_proj.weight"].shape, (self.I, self.H))
        self.assertTrue(torch.equal(out[f"{BASE}.0.gate_proj.weight"], w[0][: self.I]))
        self.assertTrue(torch.equal(out[f"{BASE}.0.up_proj.weight"], w[0][self.I :]))

    def test_gate_up_block_scale_split(self):
        s = torch.arange(
            self.E * (2 * self.I // self.BLK) * (self.H // self.BLK)
        ).reshape(self.E, 2 * self.I // self.BLK, self.H // self.BLK).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_scale_inv", s))
        self.assertEqual(
            out[f"{BASE}.0.gate_proj.weight_scale_inv"].shape,
            (self.I // self.BLK, self.H // self.BLK),
        )

    def test_gate_up_activation_scale_copied_not_split_1d(self):
        # Per-expert scalar input scale, shape [E].
        a = torch.arange(self.E).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_activation_scale", a))
        self.assertEqual(out[f"{BASE}.0.gate_proj.input_scale"].ndim, 0)
        # gate and up share the SAME input scale (copied, not halved).
        self.assertTrue(
            torch.equal(
                out[f"{BASE}.1.gate_proj.input_scale"],
                out[f"{BASE}.1.up_proj.input_scale"],
            )
        )

    def test_gate_up_activation_scale_copied_not_split_3d(self):
        a = torch.arange(self.E).reshape(self.E, 1, 1).float()
        out = dict(_unstack_stacked_expert(BASE, "gate_up_proj_activation_scale", a))
        self.assertEqual(out[f"{BASE}.2.gate_proj.input_scale"].shape, (1, 1))

    def test_down_proj_weight_scale_and_activation(self):
        dw = torch.arange(self.E * self.H * self.I).reshape(
            self.E, self.H, self.I
        ).float()
        out = dict(_unstack_stacked_expert(BASE, "down_proj", dw))
        self.assertEqual(out[f"{BASE}.0.down_proj.weight"].shape, (self.H, self.I))

        s = torch.zeros(self.E, self.H // self.BLK, self.I // self.BLK)
        out = dict(_unstack_stacked_expert(BASE, "down_proj_scale_inv", s))
        self.assertIn(f"{BASE}.0.down_proj.weight_scale_inv", out)

        a = torch.arange(self.E).float()
        out = dict(_unstack_stacked_expert(BASE, "down_proj_activation_scale", a))
        self.assertIn(f"{BASE}.0.down_proj.input_scale", out)


class TestMistral4AdaptWeightName(CustomTestCase):
    def test_non_expert_activation_scale_renamed(self):
        out = dict(
            _adapt_weight_name(
                "model.layers.0.self_attn.q_a_proj.activation_scale",
                torch.zeros(1),
            )
        )
        self.assertIn("model.layers.0.self_attn.q_a_proj.input_scale", out)

    def test_shared_experts_not_treated_as_routed(self):
        name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        out = dict(_adapt_weight_name(name, torch.zeros(3, 4)))
        # shared_experts pass straight through, not unstacked per-expert.
        self.assertIn(name, out)


class TestMistral4ExpertScaleMappingIntegration(CustomTestCase):
    """The adapter's per-expert names must resolve through the REAL deepseek_v2
    expert mapping to the FusedMoE parameter, proving the activation (input)
    scale is consumed -- not just that the helper produces a string.

    Uses ``FusedMoE.make_expert_params_mapping`` (the exact mapping
    deepseek_weight_loader builds) and replays the loader's
    ``name.replace(weight_name, param_name)`` + ``if name not in params_dict``
    step against a fake params_dict, exactly like existing MoE loader tests
    (test_inkling_per_expert_sync) do -- no real quantized layer, CPU only.

    NOTE: whether a block-fp8 *static-activation* FusedMoE actually creates the
    ``experts.w13_input_scale`` / ``experts.w2_input_scale`` params is the
    remaining on-device gate (make_expert_input_scale_params_mapping is wired
    only for W4A8/W4A16). This test proves the NAME threads through so that,
    when those params exist, the value lands; the fake params_dict includes them.
    """

    def _resolve(self, checkpoint_name, params_dict, mapping):
        # Mirror deepseek_weight_loader's expert branch resolution.
        for param_name, weight_name, expert_id, shard_id in mapping:
            if weight_name in checkpoint_name:
                resolved = checkpoint_name.replace(weight_name, param_name)
                if resolved not in params_dict:
                    continue
                return resolved, shard_id, expert_id
        return None

    def test_gate_up_activation_scale_resolves_to_w13_input_scale(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        num_experts = 2
        # 1) adapter turns the stacked activation scale into per-expert .input_scale
        adapted = dict(
            _adapt_weight_name(
                "model.layers.0.mlp.experts.gate_up_proj_activation_scale",
                torch.arange(num_experts).float(),
            )
        )
        gate_key = "model.layers.0.mlp.experts.0.gate_proj.input_scale"
        up_key = "model.layers.0.mlp.experts.1.up_proj.input_scale"
        self.assertIn(gate_key, adapted)
        self.assertIn(up_key, adapted)

        # 2) a FusedMoE exposing the fp8 input-scale params
        params_dict = {
            "model.layers.0.mlp.experts.w13_input_scale": object(),
            "model.layers.0.mlp.experts.w2_input_scale": object(),
        }
        mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        # 3) gate_proj -> w13 (shard w1), up_proj -> w13 (shard w3)
        gate_resolved = self._resolve(gate_key, params_dict, mapping)
        self.assertIsNotNone(gate_resolved)
        self.assertEqual(
            gate_resolved[0], "model.layers.0.mlp.experts.w13_input_scale"
        )
        self.assertEqual((gate_resolved[1], gate_resolved[2]), ("w1", 0))

        up_resolved = self._resolve(up_key, params_dict, mapping)
        self.assertEqual(
            up_resolved[0], "model.layers.0.mlp.experts.w13_input_scale"
        )
        self.assertEqual((up_resolved[1], up_resolved[2]), ("w3", 1))

    def test_down_proj_scale_resolves_to_w2(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        adapted = dict(
            _adapt_weight_name(
                "model.layers.0.mlp.experts.down_proj_scale_inv",
                torch.zeros(2, 2, 2),
            )
        )
        down_key = "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"
        self.assertIn(down_key, adapted)

        params_dict = {"model.layers.0.mlp.experts.w2_weight_scale_inv": object()}
        mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=2,
        )
        resolved = self._resolve(down_key, params_dict, mapping)
        self.assertIsNotNone(resolved)
        self.assertEqual(
            resolved[0], "model.layers.0.mlp.experts.w2_weight_scale_inv"
        )
        self.assertEqual((resolved[1], resolved[2]), ("w2", 0))


if __name__ == "__main__":
    unittest.main()
