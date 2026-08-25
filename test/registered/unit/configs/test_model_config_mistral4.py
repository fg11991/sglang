"""Mistral-Small-4 (mistral4): MLA shapes + the construction-time config gaps.

Two separate things are guarded here, both using a REAL nested config pair --
an outer multimodal Mistral3Config carrying the forced architecture, and a
distinct inner text config. They must not be collapsed into one object: the
json-model-override only rewrites the OUTER architectures, and
``_patch_text_config`` propagates just pad/bos/eos/tie_word_embeddings, so the
inner config genuinely has ``architectures = None``. A test that shares one
object would hide exactly the bug this file exists for.

  1. ``ModelConfig._derive_model_shapes`` must route mistral4 to the MLA branch
     (reading the arch off the outer config, the KV dims off the inner one).
  2. ``_translate_mistral4_config`` must fill the attributes DeepseekV3 touches
     during CONSTRUCTION -- before any weight is loaded -- which neither
     Mistral4Config nor the checkpoint's config.json defines:
       * ``architectures``   -> determine_num_fused_shared_experts does
                                ``self.config.architectures[0]`` (None[0] -> TypeError)
       * ``moe_layer_freq``  -> _is_layer_sparse does ``layer_id % ...`` (AttributeError)
"""

import unittest

from transformers import PretrainedConfig

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.models.mistral4 import _translate_mistral4_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# The real Mistral-Small-4-119B-2603 text_config values.
_TEXT_FIELDS = dict(
    model_type="mistral4",
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
    num_hidden_layers=36,
    vocab_size=131072,
    head_dim=128,
    kv_lora_rank=256,
    q_lora_rank=1024,
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=128,
    n_shared_experts=1,
    num_experts_per_tok=4,
    first_k_dense_replace=0,
    moe_intermediate_size=2048,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
    n_group=1,
    topk_group=1,
    rope_interleave=True,
    rope_parameters={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 128.0,
        "llama_4_scaling_beta": 0.1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_theta": 10000.0,
        "rope_type": "yarn",
        "type": "yarn",
    },
    max_position_embeddings=1048576,
    rms_norm_eps=1e-06,
)


def _make_inner_text_config():
    """The inner mistral4 config as SGLang really sees it.

    Prefers the real ``Mistral4Config`` (transformers>=5.3). The fallback is a
    bare ``PretrainedConfig`` carrying the same fields, which reproduces the two
    properties under test on any transformers: ``architectures`` defaults to
    None and ``moe_layer_freq`` is absent (verified: Mistral4Config.__init__
    defines neither).
    """
    try:
        from transformers import Mistral4Config

        return Mistral4Config(**_TEXT_FIELDS)
    except (ImportError, AttributeError):
        return PretrainedConfig(**_TEXT_FIELDS)


def _make_outer_config(inner):
    """Outer multimodal config with the capture-time architecture override.

    Built empty and then given the inner object directly, rather than through
    ``Mistral3Config(text_config=...)``: that constructor resolves the sub-config
    through CONFIG_MAPPING, which cannot build a "mistral4" entry on transformers
    versions that predate it. What matters here is only that outer and inner are
    two distinct objects and that the architecture override lands on the outer.
    """
    try:
        from transformers import Mistral3Config

        outer = Mistral3Config()
    except Exception:  # noqa: BLE001 - any transformers-version issue
        outer = PretrainedConfig()
    # What --json-model-override-args '{"architectures": [...]}' rewrites.
    outer.architectures = ["Mistral4ForCausalLM"]
    outer.text_config = inner
    return outer


class TestMistral4ModelConfigShapes(CustomTestCase):
    def _derive(self, outer, inner):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = outer
        model_config.hf_text_config = inner
        model_config._derive_model_shapes()
        return model_config

    def test_mistral4_is_mla_with_compressed_kv_dims(self):
        inner = _make_inner_text_config()
        outer = _make_outer_config(inner)
        # Precondition: outer and inner are distinct, and only outer got the arch.
        self.assertIsNot(outer, inner)
        self.assertIsNone(getattr(inner, "architectures", None))

        model_config = self._derive(outer, inner)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.kv_lora_rank, 256)
        self.assertEqual(model_config.qk_nope_head_dim, 64)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.v_head_dim, 128)

    def test_unlisted_arch_is_not_mla(self):
        inner = _make_inner_text_config()
        outer = _make_outer_config(inner)
        outer.architectures = ["SomeUnlistedForCausalLM"]
        model_config = self._derive(outer, inner)
        self.assertNotEqual(model_config.attention_arch, AttentionArch.MLA)


class TestMistral4ConfigTranslation(CustomTestCase):
    """The construction-time gaps: architectures and moe_layer_freq."""

    def test_inner_config_starts_without_the_attributes(self):
        inner = _make_inner_text_config()
        self.assertIsNone(getattr(inner, "architectures", None))
        self.assertIsNone(getattr(inner, "moe_layer_freq", None))

    def test_translation_fills_construction_attributes(self):
        inner = _translate_mistral4_config(_make_inner_text_config())

        self.assertEqual(inner.architectures, ["Mistral4ForCausalLM"])
        self.assertEqual(inner.moe_layer_freq, 1)
        self.assertEqual(inner.topk_method, "greedy")
        self.assertEqual(inner.scoring_func, "softmax")
        self.assertEqual(
            inner.llama_4_scaling,
            {"original_max_position_embeddings": 8192, "beta": 0.1},
        )

    def test_shared_expert_fusion_check_no_longer_crashes(self):
        """Replays determine_num_fused_shared_experts' architecture probe."""
        raw = _make_inner_text_config()
        with self.assertRaises(TypeError):
            raw.architectures[0]  # None[0] -- the pre-fix blocker

        inner = _translate_mistral4_config(_make_inner_text_config())
        # Does not raise, and differs from the fusion-eligible architecture, so
        # shared-expert fusion stays disabled (mistral4 is not pre-fused).
        self.assertNotEqual(inner.architectures[0], "DeepseekV3ForCausalLM")

    def test_layer_sparsity_check_no_longer_crashes(self):
        """Replays DeepseekV2DecoderLayer._is_layer_sparse for every layer."""
        raw = _make_inner_text_config()
        with self.assertRaises(AttributeError):
            0 % raw.moe_layer_freq  # the pre-fix blocker

        inner = _translate_mistral4_config(_make_inner_text_config())
        for layer_id in range(inner.num_hidden_layers):
            is_sparse = (
                inner.n_routed_experts is not None
                and layer_id >= inner.first_k_dense_replace
                and layer_id % inner.moe_layer_freq == 0
            )
            # All 36 layers of 2603 carry routed experts (published index).
            self.assertTrue(is_sparse, f"layer {layer_id} should be MoE")


class TestMistral4ExpertLocationConfig(CustomTestCase):
    """EPLB metadata is built from the TOP-LEVEL config, so it must unwrap too.

    ModelConfigForExpertLocation.from_model_config passes model_config.hf_config
    -- the multimodal wrapper -- not hf_text_config. Reading n_routed_experts off
    that wrapper resolves through transformers' composite-config delegation and
    dies as "'PixtralVisionConfig' object has no attribute 'n_routed_experts'"
    during ModelRunner init, before a single weight is loaded.
    """

    def test_unwraps_text_config(self):
        from sglang.srt.models.mistral4 import Mistral4ForCausalLM

        inner = _make_inner_text_config()
        outer = _make_outer_config(inner)

        result = Mistral4ForCausalLM.get_model_config_for_expert_location(outer)

        self.assertEqual(result.num_layers, 36)
        self.assertEqual(result.num_logical_experts, 128)
        self.assertEqual(result.num_groups, 1)

    def test_flat_config_still_works(self):
        """A config without a text_config (a hypothetical flat export) passes through."""
        from sglang.srt.models.mistral4 import Mistral4ForCausalLM

        flat = _make_inner_text_config()
        result = Mistral4ForCausalLM.get_model_config_for_expert_location(flat)
        self.assertEqual(result.num_logical_experts, 128)


if __name__ == "__main__":
    unittest.main()
