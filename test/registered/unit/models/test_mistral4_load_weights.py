"""Direct coverage for Mistral4ForCausalLM.load_weights().

Guards the load-verification contract without building a real 119B model:
  - the adapted (text-layout, unstacked) weights reach super().load_weights;
  - the returned matched-parameter set is passed through;
  - a matched set missing a structural family raises (silent prefix / expert
    drift caught, not trained on an uninitialized target);
  - a None return (a base loader that does not report matches) does NOT crash
    -- this is the regression for the original ``loaded is None`` TypeError.

super().load_weights is monkeypatched (it would otherwise need a constructed
model), so this is a CPU unit test.
"""

import unittest

import torch

from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
from sglang.srt.models.mistral4 import Mistral4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# A minimal real-shaped checkpoint stream (multimodal language_model.* layout,
# transformers-v5 stacked experts) covering every structural family.
_E = 2


def _fake_checkpoint():
    yield "language_model.model.embed_tokens.weight", torch.zeros(4, 4)
    yield "language_model.lm_head.weight", torch.zeros(4, 4)
    yield "language_model.model.norm.weight", torch.zeros(4)
    yield "language_model.model.layers.0.self_attn.q_a_proj.weight", torch.zeros(4, 4)
    yield "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight", torch.zeros(4, 4)
    yield "language_model.model.layers.0.self_attn.kv_b_proj.weight", torch.zeros(4, 4)
    yield "language_model.model.layers.0.self_attn.o_proj.weight", torch.zeros(4, 4)
    yield "language_model.model.layers.0.mlp.gate.weight", torch.zeros(_E, 4)
    yield "language_model.model.layers.0.mlp.experts.gate_up_proj", torch.zeros(_E, 4, 4)
    yield "language_model.model.layers.0.mlp.experts.down_proj", torch.zeros(_E, 4, 2)
    # a vision weight that must be dropped
    yield "vision_tower.transformer.layers.0.attention.q_proj.weight", torch.zeros(4, 4)


# What the loader would report matched on a healthy load (post-fusion names).
_HEALTHY_MATCHED = {
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.kv_b_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.w13_weight",
    "model.layers.0.mlp.experts.w2_weight",
}


class TestMistral4LoadWeights(CustomTestCase):
    def setUp(self):
        self._orig = DeepseekV3ForCausalLM.load_weights

    def tearDown(self):
        DeepseekV3ForCausalLM.load_weights = self._orig

    def _instance(self):
        # Skip the real __init__ (which builds the whole model).
        return object.__new__(Mistral4ForCausalLM)

    def test_adapts_weights_and_passes_through_matched_set(self):
        seen = {}

        def fake_super(self, weights, *args, **kwargs):
            seen["names"] = [name for name, _ in weights]
            return set(_HEALTHY_MATCHED)

        DeepseekV3ForCausalLM.load_weights = fake_super
        result = self._instance().load_weights(_fake_checkpoint())

        # Prefix stripped to DeepSeek layout, vision dropped, experts unstacked.
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
        # The matched set is returned unchanged.
        self.assertEqual(result, _HEALTHY_MATCHED)

    def test_missing_structural_family_raises(self):
        def fake_super(self, weights, *args, **kwargs):
            list(weights)  # consume
            return _HEALTHY_MATCHED - {"model.layers.0.mlp.experts.w13_weight"}

        DeepseekV3ForCausalLM.load_weights = fake_super
        with self.assertRaises(RuntimeError) as ctx:
            self._instance().load_weights(_fake_checkpoint())
        self.assertIn("experts.w13_weight", str(ctx.exception))

    def test_none_matched_set_does_not_crash(self):
        # Regression: `loaded is None` must not raise TypeError.
        def fake_super(self, weights, *args, **kwargs):
            list(weights)
            return None

        DeepseekV3ForCausalLM.load_weights = fake_super
        # Should complete without raising.
        self.assertIsNone(self._instance().load_weights(_fake_checkpoint()))


if __name__ == "__main__":
    unittest.main()
