"""Direct coverage for Mistral4ForCausalLM.load_weights().

Guards the load-verification contract without building a real 119B model:
  - the adapted (text-layout, unstacked) weights reach super().load_weights;
  - the returned matched-parameter set is passed through;
  - a matched set missing a structural family raises -- including
    ``fused_qkv_a_proj_with_mqa``, which the loader fills through its own
    cached_a_proj branch and so can fail independently of o_proj/kv_b_proj;
  - a None return RAISES: verification is this subclass's whole job, and a
    loader that will not say what it matched makes an initialized target
    unprovable (regression for the earlier ``loaded is None`` TypeError, now
    an explicit error rather than a silent skip);
  - every (layer, expert, gate|up|down) slot must be present. The loader's own
    matched set cannot show this -- it rewrites ``experts.{e}.gate_proj.`` to
    ``experts.w13_``, collapsing all experts and both shards of a layer onto
    one name -- so coverage is accounted on the emitting side.

super().load_weights is monkeypatched (it would otherwise need a constructed
model), so this is a CPU unit test.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
from sglang.srt.models.mistral4 import Mistral4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# A tiny but structurally complete stand-in for the 36x128 grid.
_LAYERS = 1
_EXPERTS = 2


def _fake_checkpoint(num_experts=_EXPERTS):
    """Multimodal ``language_model.*`` layout, v5 stacked experts, static fp8.

    Scale shapes mirror the published header: per-expert ``[E, 1, 1]`` weight
    scales, ``[E]`` activation scales, 0-d scalars on plain projections.
    """
    L = "language_model.model.layers.0"
    yield "language_model.model.embed_tokens.weight", torch.zeros(4, 4)
    yield "language_model.lm_head.weight", torch.zeros(4, 4)
    yield "language_model.model.norm.weight", torch.zeros(4)
    for proj in ("q_a_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        yield f"{L}.self_attn.{proj}.weight", torch.zeros(4, 4)
        yield f"{L}.self_attn.{proj}.weight_scale_inv", torch.tensor(1.0)
        yield f"{L}.self_attn.{proj}.activation_scale", torch.tensor(1.0)
    yield f"{L}.mlp.gate.weight", torch.zeros(num_experts, 4)
    yield f"{L}.mlp.experts.gate_up_proj", torch.zeros(num_experts, 4, 4)
    yield f"{L}.mlp.experts.gate_up_proj_scale_inv", torch.ones(num_experts, 1, 1)
    yield f"{L}.mlp.experts.gate_up_proj_activation_scale", torch.ones(num_experts)
    yield f"{L}.mlp.experts.down_proj", torch.zeros(num_experts, 4, 2)
    yield f"{L}.mlp.experts.down_proj_scale_inv", torch.ones(num_experts, 1, 1)
    yield f"{L}.mlp.experts.down_proj_activation_scale", torch.ones(num_experts)
    # a vision weight that must be dropped
    yield "vision_tower.transformer.layers.0.attention.q_proj.weight", torch.zeros(4, 4)


# What the loader reports matched on a healthy load: post-fusion parameter names,
# with the scale names a static per-tensor fp8 checkpoint really registers.
_HEALTHY_MATCHED = {
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale",
    "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.input_scale",
    "model.layers.0.self_attn.kv_b_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.w13_weight",
    "model.layers.0.mlp.experts.w2_weight",
    "model.layers.0.mlp.experts.w13_weight_scale",
    "model.layers.0.mlp.experts.w2_weight_scale",
    "model.layers.0.mlp.experts.w13_input_scale",
    "model.layers.0.mlp.experts.w2_input_scale",
}


class TestMistral4LoadWeights(CustomTestCase):
    def setUp(self):
        self._orig = DeepseekV3ForCausalLM.load_weights

    def tearDown(self):
        DeepseekV3ForCausalLM.load_weights = self._orig

    def _instance(self, num_experts=_EXPERTS):
        # Skip the real __init__ (which builds the whole model).
        inst = object.__new__(Mistral4ForCausalLM)
        inst.config = SimpleNamespace(
            num_hidden_layers=_LAYERS,
            n_routed_experts=num_experts,
            first_k_dense_replace=0,
            moe_layer_freq=1,
        )
        return inst

    @staticmethod
    def _patch_super(matched, seen=None):
        def fake_super(self, weights, *args, **kwargs):
            names = [name for name, _ in weights]
            if seen is not None:
                seen["names"] = names
            return matched

        DeepseekV3ForCausalLM.load_weights = fake_super

    def test_adapts_weights_and_passes_through_matched_set(self):
        seen = {}
        self._patch_super(set(_HEALTHY_MATCHED), seen)

        result = self._instance().load_weights(_fake_checkpoint())

        self.assertIn("model.embed_tokens.weight", seen["names"])
        self.assertIn("model.layers.0.mlp.experts.0.gate_proj.weight", seen["names"])
        self.assertIn("model.layers.0.mlp.experts.1.down_proj.weight", seen["names"])
        self.assertFalse(
            any(n.startswith("vision_tower") for n in seen["names"]),
            "vision weights must be dropped",
        )
        self.assertFalse(
            any("language_model." in n for n in seen["names"]),
            "language_model. prefix must be stripped",
        )
        self.assertEqual(result, _HEALTHY_MATCHED)

    def test_missing_structural_family_raises(self):
        self._patch_super(_HEALTHY_MATCHED - {"model.layers.0.mlp.experts.w13_weight"})
        with self.assertRaises(RuntimeError) as ctx:
            self._instance().load_weights(_fake_checkpoint())
        self.assertIn("experts.w13_weight", str(ctx.exception))

    def test_missing_fused_qkv_a_proj_raises(self):
        # Its own loader branch (cached_a_proj), so it can fail on its own.
        self._patch_super(
            _HEALTHY_MATCHED
            - {"model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"}
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._instance().load_weights(_fake_checkpoint())
        self.assertIn("fused_qkv_a_proj_with_mqa", str(ctx.exception))

    def test_none_matched_set_raises(self):
        self._patch_super(None)
        with self.assertRaises(RuntimeError) as ctx:
            self._instance().load_weights(_fake_checkpoint())
        self.assertIn("None", str(ctx.exception))

    def test_full_expert_grid_passes(self):
        self._patch_super(set(_HEALTHY_MATCHED))
        # 1 layer x 2 experts x {gate, up, down} all present -> no raise.
        self._instance().load_weights(_fake_checkpoint())

    def test_incomplete_expert_grid_raises(self):
        # Model expects 4 experts; the checkpoint only carries 2.
        self._patch_super(set(_HEALTHY_MATCHED))
        with self.assertRaises(RuntimeError) as ctx:
            self._instance(num_experts=4).load_weights(_fake_checkpoint(num_experts=2))
        message = str(ctx.exception)
        self.assertIn("routed expert", message)
        # 2 missing experts x 3 shards
        self.assertIn("6 of 12", message)

    def test_unmatched_scale_family_raises(self):
        """A scale spelling no parameter accepted must not pass silently.

        This is the 360-key failure: emitting ``weight_scale_inv`` against a
        static per-tensor fp8 model, whose parameters are ``weight_scale``.
        Routed-expert scales are dropped by the loader's `not in params_dict`
        guard, so without this check the model runs fp8 weights against default
        scales and only the numerics are wrong.
        """
        self._patch_super(
            _HEALTHY_MATCHED - {"model.layers.0.mlp.experts.w13_weight_scale"}
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._instance().load_weights(_fake_checkpoint())
        message = str(ctx.exception)
        self.assertIn("experts.w13_weight_scale", message)
        self.assertIn("scales", message)


if __name__ == "__main__":
    unittest.main()
