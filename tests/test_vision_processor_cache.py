"""Real pinned-vLLM processor/cache regression tests (no model weights loaded).

Run in the deployment image with DSV41_TEST_MODEL_PATH pointing to the local
checkpoint, whose tokenizer and config are used for every processing call.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get("DSV41_TEST_MODEL_PATH"), "requires the pinned vLLM image and checkpoint tokenizer")
class VisionProcessorCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoTokenizer
        from vllm.config.multimodal import MultiModalConfig
        from vllm.models.deepseek_v4_1.common.mm_preprocess import (
            DeepseekV4VLDummyInputsBuilder,
            DeepseekV4VLMultiModalProcessor,
            DeepseekV4VLProcessingInfo,
        )
        from vllm.multimodal.cache import MultiModalProcessorOnlyCache
        from vllm.multimodal.processing import InputProcessingContext
        from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

        model_path = Path(os.environ["DSV41_TEST_MODEL_PATH"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        config = DeepseekV41Config(**json.loads((model_path / "config.json").read_text()))
        mm_config = MultiModalConfig(mm_processor_cache_gb=0.25)
        cls.model_config = SimpleNamespace(
            model=str(model_path), hf_config=config, dtype=torch.bfloat16,
            multimodal_config=mm_config,
            get_multimodal_config=lambda: mm_config,
        )
        cls.info = DeepseekV4VLProcessingInfo(InputProcessingContext(cls.model_config, tokenizer))
        cls.processor_type = DeepseekV4VLMultiModalProcessor
        cls.dummy = DeepseekV4VLDummyInputsBuilder(cls.info)
        cls.cache_type = MultiModalProcessorOnlyCache
        cls.tokenizer = tokenizer
        red = np.full((384, 384, 3), [255, 0, 0], dtype=np.uint8)
        blue = np.full((384, 384, 3), [0, 0, 255], dtype=np.uint8)
        cls.media = {
            "video-a": (np.stack([red]), {"frames_indices": [0], "fps": 1.0, "duration": 1.0}),
            "video-b": (np.stack([blue, red]), {"frames_indices": [0, 1], "fps": 1.0, "duration": 2.0}),
            "image-a": Image.fromarray(red),
            "image-b": Image.fromarray(blue),
        }

    def make_processor(self, cached):
        return self.processor_type(
            self.info, self.dummy,
            cache=self.cache_type(self.model_config) if cached else None,
        )

    def process(self, processor, order):
        from vllm.multimodal.processing import ProcessorInputs, TimingContext
        from vllm.tokenizers.video_prompt import IMAGE_PLACEHOLDER, render_video_prompt

        data, ids, text = {}, {}, ["Describe the media in order."]
        video_idx = 0
        for name in order:
            modality = name.split("-")[0]
            data.setdefault(modality, []).append(self.media[name])
            ids.setdefault(modality, []).append(name)
            if modality == "video":
                text.append(render_video_prompt({"video_id": name}, video_idx))
                video_idx += 1
            else:
                text.append(IMAGE_PLACEHOLDER)
        text.append("Give the answer.")
        prompt = self.tokenizer.encode("\n\n".join(text), add_special_tokens=False)
        items = self.info.parse_mm_data(data, validate=False)
        return processor.apply(ProcessorInputs(prompt, items, ids), TimingContext(enabled=False))

    def assert_same_processed(self, actual, expected):
        import torch

        self.assertEqual(actual["prompt_token_ids"], expected["prompt_token_ids"])
        self.assertEqual(actual["mm_hashes"], expected["mm_hashes"])
        actual_ranges = actual["mm_placeholders"]
        expected_ranges = expected["mm_placeholders"]
        self.assertEqual(set(actual_ranges), set(expected_ranges))
        for modality in expected_ranges:
            self.assertEqual(
                [(x.offset, x.length) for x in actual_ranges[modality]],
                [(x.offset, x.length) for x in expected_ranges[modality]],
            )
        for modality, items in expected["mm_kwargs"].items():
            for index, item in enumerate(items):
                for name, field in item.items():
                    torch.testing.assert_close(actual["mm_kwargs"][modality][index][name].data, field.data, rtol=0, atol=0)

    def assert_cached_matches(self, warmup, order):
        cached = self.make_processor(True)
        if warmup:
            self.process(cached, warmup)
            for name in warmup:
                self.assertTrue(cached.cache.is_cached_item(name))
        expected = self.process(self.make_processor(False), order)
        self.assert_same_processed(self.process(cached, order), expected)
        self.assert_same_processed(self.process(cached, order), expected)

    def test_first_video_hit_second_video_miss(self):
        self.assert_cached_matches(["video-a"], ["video-a", "video-b"])

    def test_first_video_miss_second_video_hit(self):
        self.assert_cached_matches(["video-b"], ["video-a", "video-b"])

    def test_cached_videos_can_change_occurrence_order(self):
        self.assert_cached_matches(["video-a", "video-b"], ["video-b", "video-a"])

    def test_duplicate_video_can_appear_twice(self):
        self.assert_cached_matches([], ["video-a", "video-a"])

    def test_image_video_interleaving_with_partial_cache(self):
        self.assert_cached_matches(["video-a", "image-a"], ["image-a", "video-a", "video-b", "image-b"])


if __name__ == "__main__":
    unittest.main()
