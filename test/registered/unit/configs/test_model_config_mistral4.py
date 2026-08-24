"""Mistral-Small-4 (mistral4) must derive an MLA attention arch.

Regression for the model_config MLA whitelist: when the offline capture path
forces ``architectures=["Mistral4ForCausalLM"]`` (text-only load of the
multimodal 2603 checkpoint), ``_derive_model_shapes`` must route to the MLA
branch and read the compressed-KV dims from the (nested) text config. Without
the whitelist entry it falls through to MHA and Ascend gets use_mla=False.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _mistral4_text_config(**overrides):
    # The mistral4 text_config shape (Mistral-Small-4-119B-2603), with the
    # forced text-only architecture that the capture path passes via
    # json_model_override_args.
    defaults = dict(
        architectures=["Mistral4ForCausalLM"],
        model_type="mistral4",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=2,
        vocab_size=131072,
        head_dim=128,
        kv_lora_rank=256,
        q_lora_rank=1024,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_scaling=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMistral4ModelConfig(CustomTestCase):
    def _derive(self, text_config):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = text_config
        model_config.hf_text_config = text_config
        model_config._derive_model_shapes()
        return model_config

    def test_mistral4_is_mla_with_compressed_kv_dims(self):
        model_config = self._derive(_mistral4_text_config())

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.kv_lora_rank, 256)
        self.assertEqual(model_config.qk_nope_head_dim, 64)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.v_head_dim, 128)

    def test_unknown_arch_without_whitelist_is_not_mla(self):
        # Guard the regression itself: a same-shaped config under a name that is
        # NOT whitelisted must not be silently treated as MLA.
        model_config = self._derive(
            _mistral4_text_config(architectures=["SomeUnlistedForCausalLM"])
        )
        self.assertNotEqual(model_config.attention_arch, AttentionArch.MLA)


if __name__ == "__main__":
    unittest.main()
