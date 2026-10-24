# CPU test support; original notices in NOTICE.
"""Reference-encoder parity and renderer media-order checks; no GPU needed."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.source_extract import load_class, load_function

ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = ROOT / "adaptation"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encoder = load(
    "encoder", ADAPTATION / "files/vllm/tokenizers/deepseek_v41_encoding.py"
)
reference = load("reference", ADAPTATION / "reference/encoder.py")
installer = load("installer", ADAPTATION / "install.py")
video_prompt = load(
    "video_prompt", ADAPTATION / "files/vllm/tokenizers/video_prompt.py"
)


def calls():
    return [
        {
            "id": "first",
            "type": "function",
            "function": {"name": "inspect", "arguments": '{"label":"FIRST"}'},
        },
        {
            "id": "second",
            "type": "function",
            "function": {"name": "inspect", "arguments": '{"label":"SECOND"}'},
        },
    ]


def history(with_images=False):
    def content(label):
        parts = [{"type": "text", "text": label}]
        if with_images:
            parts.append({"type": "image_url", "image_url": {"url": label + ".png"}})
        return parts

    return [
        {"role": "user", "content": "Inspect the results."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "keep this thought",
            "tool_calls": calls(),
        },
        {"role": "tool", "tool_call_id": "second", "content": content("SECOND")},
        {"role": "tool", "tool_call_id": "first", "content": content("FIRST")},
    ]


def video_message(url="clip.mp4", *, video_id=None):
    """Build one video content block without depending on vLLM protocol types."""
    video = {"url": url}
    if video_id is not None:
        video["video_id"] = video_id
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video."},
            {"type": "video_url", "video_url": video},
        ],
    }


def video_ids(content):
    """Extract the stable IDs emitted by the video-level text wrapper."""
    return re.findall(r"\bvid_[A-Za-z0-9_-]{1,32}\b", content)


def normalizer():
    """Execute the serving wrapper function without transformers/vLLM."""
    namespace = {
        "copy": copy,
        "Any": object,
        "ChatCompletionMessageParam": dict,
        "IMAGE_PLACEHOLDER": encoder.IMAGE_PLACEHOLDER,
        "render_video_prompt": video_prompt.render_video_prompt,
    }
    return load_function(
        ADAPTATION / "files/vllm/tokenizers/deepseek_v41.py",
        "_normalize_messages",
        namespace,
        annotations=False,
        filename="reference/tokenizer_wrapper.py",
    )


normalize = normalizer()


class OfficialParityTests(unittest.TestCase):
    def test_video_content_uses_stable_wrapper_and_one_placeholder(self):
        messages = [video_message()]
        normalized = normalize(messages)
        content = normalized[0]["content"]
        ids = video_ids(content)
        self.assertEqual(len(set(ids)), 1)
        self.assertRegex(content, r"Video ID:\s*vid_[A-Za-z0-9_-]{1,32}")
        self.assertEqual(content.count(encoder.IMAGE_PLACEHOLDER), 1)
        self.assertIn("chronological", content)
        self.assertIn("End of video", content)

        # Normalizing a replayed request must produce byte-identical wrapper
        # text so the prompt and multimodal cache key remain stable.
        self.assertEqual(normalize(copy.deepcopy(messages)), normalized)

    def test_video_id_is_stable_across_history_replay_and_distinct_by_source(self):
        first = normalize([video_message("first.mp4")])[0]["content"]
        replay = normalize([video_message("first.mp4")])[0]["content"]
        other = normalize([video_message("second.mp4")])[0]["content"]
        self.assertEqual(video_ids(first), video_ids(replay))
        self.assertNotEqual(video_ids(first), video_ids(other))

        alias = normalize(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_video", "input_video": {"url": "first.mp4"}}
                    ],
                }
            ]
        )[0]["content"]
        self.assertEqual(video_ids(first), video_ids(alias))

        video_id = video_ids(first)[0]
        history_messages = [
            video_message("first.mp4"),
            {"role": "assistant", "content": "I found the key event."},
            {"role": "user", "content": f"Continue with {video_id}."},
        ]
        normalized_history = normalize(history_messages)
        self.assertIn(video_id, normalized_history[0]["content"])
        self.assertEqual(normalized_history[2]["content"], f"Continue with {video_id}.")
        encoded = encoder.encode_messages(normalized_history, thinking_mode="chat")
        self.assertIn(video_id, encoded)
        self.assertIn(f"Continue with {video_id}.", encoded)

    def test_explicit_video_id_is_preserved(self):
        content = normalize(
            [video_message("renamed.mp4", video_id="vid_abcdef123456")]
        )[0]["content"]
        self.assertIn("Video ID: vid_abcdef123456.", content)
        self.assertEqual(content.count(encoder.IMAGE_PLACEHOLDER), 1)

    def test_video_uuid_attachment_is_stable_and_non_mutating(self):
        messages = [video_message("clip.mp4")]
        annotated = video_prompt.attach_video_uuids(messages)
        block = annotated[0]["content"][1]
        self.assertEqual(block["uuid"], video_prompt.stable_video_id(messages[0]["content"][1]))
        self.assertNotIn("uuid", messages[0]["content"][1])
        replayed = video_prompt.attach_video_uuids(copy.deepcopy(messages))
        self.assertEqual(replayed[0]["content"][1]["uuid"], block["uuid"])

    def test_video_wrapper_maps_media_timestamps_to_frame_order(self):
        messages = [video_message("long.mp4")]
        media = [
            (
                [object()] * 32,
                {"frames_indices": list(range(0, 64, 2)), "fps": 1.0},
            )
        ]
        annotated = video_prompt.attach_video_prompt_timestamps(
            messages, {"video": media}
        )
        content = normalize(annotated)[0]["content"]
        self.assertEqual(content.count("Frame "), 32)
        self.assertIn("Frame 1 at 0.000s.", content)
        self.assertIn("Frame 32 at 62.000s.", content)
        self.assertEqual(content.count(encoder.IMAGE_PLACEHOLDER), 1)

    def test_multiple_videos_get_independent_wrappers(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these videos."},
                    {"type": "video_url", "video_url": {"url": "first.mp4"}},
                    {"type": "video_url", "video_url": {"url": "second.mp4"}},
                ],
            }
        ]
        content = normalize(messages)[0]["content"]
        ids = video_ids(content)
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(content.count(encoder.IMAGE_PLACEHOLDER), 2)
        unique_ids = list(dict.fromkeys(ids))
        self.assertLess(content.index(unique_ids[0]), content.index(unique_ids[1]))
        for video_id in unique_ids:
            self.assertGreaterEqual(content.count(video_id), 2)

    def test_image_and_plain_video_id_text_do_not_get_video_wrapper(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Refer to vid_abcdef123456."},
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                ],
            }
        ]
        content = normalize(messages)[0]["content"]
        self.assertNotIn("Video ID:", content)
        self.assertNotIn("End of video", content)
        self.assertEqual(content.count(encoder.IMAGE_PLACEHOLDER), 1)

    def test_checkpoint_cases_through_tokenizer_normalization(self):
        manifest = json.loads((ROOT / "manifests/model-files.json").read_text())
        fixtures = sorted(
            (ROOT / "tests/fixtures/encoding_cases").glob("test_input_*.json")
        )
        self.assertEqual(len(fixtures), 5)
        for fixture in fixtures:
            expected_hash = next(
                row["sha256"]
                for row in manifest["files"]
                if row["path"] == "encoding/tests/" + fixture.name
            )
            self.assertEqual(
                hashlib.sha256(fixture.read_bytes()).hexdigest(), expected_hash
            )
            for case in reference.load_cases(str(fixture)):
                for mode in ("chat", "thinking"):
                    for effort in ("low", "high", "max", 42):
                        with self.subTest(file=fixture.name, mode=mode, effort=effort):
                            # The serving wrapper turns media into placeholders before
                            # the text encoder. The independent reference accepts raw media.
                            actual = encoder.encode_messages(
                                normalize(case["messages"]),
                                thinking_mode=mode,
                                reasoning_effort=effort,
                            )
                            expected = reference.encode_messages(
                                case["messages"],
                                thinking_mode=mode,
                                reasoning_effort=effort,
                            )
                            self.assertEqual(actual, expected)

    def test_long_tool_loop_scans_the_user_boundary_once(self):
        messages = [{"role": "user", "content": "Continue checking until finished."}]
        for i in range(80):
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": f"Check {i}.",
                        "tool_calls": [
                            {
                                "id": str(i),
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"key":"alpha"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": str(i), "content": f"Value {i}"},
                ]
            )
        before = copy.deepcopy(messages)
        for mode in ("chat", "thinking"):
            with patch.object(
                encoder, "find_last_user_index", wraps=encoder.find_last_user_index
            ) as scan:
                actual = encoder.encode_messages(messages, thinking_mode=mode)
            self.assertLessEqual(scan.call_count, 3)
            self.assertEqual(
                actual, reference.encode_messages(messages, thinking_mode=mode)
            )
        self.assertEqual(messages, before)

    def test_context_and_drop_modes_keep_reference_encoding(self):
        messages = history() + [
            {
                "role": "assistant",
                "content": "Verified",
                "reasoning_content": "Remember",
            },
            {"role": "system", "content": "Apply this new instruction."},
            {"role": "assistant", "content": "Done", "reasoning_content": "Checked"},
            {"role": "user", "content": "Continue"},
        ]
        for split in (0, 2, 4, len(messages)):
            for thinking in ("chat", "thinking"):
                for drop in (False, True):
                    for bos in (False, True):
                        kwargs = dict(
                            thinking_mode=thinking,
                            context=messages[:split],
                            drop_thinking=drop,
                            add_default_bos_token=bos,
                        )
                        self.assertEqual(
                            encoder.encode_messages(messages[split:], **kwargs),
                            reference.encode_messages(messages[split:], **kwargs),
                        )

    def test_owned_message_path_matches_copying_path(self):
        messages = history(with_images=True)
        owned = copy.deepcopy(messages)
        expected = encoder.encode_messages(messages, thinking_mode="thinking")
        actual = encoder.encode_messages(
            owned, thinking_mode="thinking", copy_messages=False
        )
        self.assertEqual(actual, expected)

    def test_normalizer_returns_independent_messages_without_redundant_shallow_copy(
        self,
    ):
        messages = history(with_images=True)
        normalized = normalize(messages)
        normalized[0]["content"] = "changed"
        self.assertNotEqual(normalized[0]["content"], messages[0]["content"])
        self.assertEqual(messages, history(with_images=True))

    def test_reference_is_pinned(self):
        manifest = json.loads((ADAPTATION / "manifest.json").read_text())
        self.assertEqual(
            hashlib.sha256((ADAPTATION / "reference/encoder.py").read_bytes()).hexdigest(),
            manifest["encoder_reference"]["sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                (ADAPTATION / "files/vllm/tokenizers/deepseek_v41.py").read_bytes()
            ).hexdigest(),
            "5f87bc88b0aa24e25d773ed341f296d4173993b977778ed94db98c3577144a34",
        )

    def test_text_and_tool_histories_match_reference(self):
        examples = [
            [{"role": "user", "content": "Hello"}],
            [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "2+2"},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "alpha"},
                        {"type": "text", "text": "beta"},
                    ],
                }
            ],
            history(),
            history()
            + [{"role": "assistant", "content": "Done", "reasoning_content": "Result"}],
            history() + [{"role": "latest_reminder", "content": "Check carefully."}],
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "OK"},
                {"role": "system", "content": "A new instruction"},
            ],
            [
                {
                    "role": "tool",
                    "content_blocks": [{"type": "text", "text": "Structured"}],
                }
            ],
        ]
        for index, messages in enumerate(examples):
            before = copy.deepcopy(messages)
            for mode in ("chat", "thinking"):
                for effort in ("low", "high", "max", 61):
                    with self.subTest(example=index, mode=mode, effort=effort):
                        kwargs = {"thinking_mode": mode, "reasoning_effort": effort}
                        self.assertEqual(
                            encoder.encode_messages(messages, **kwargs),
                            reference.encode_messages(messages, **kwargs),
                        )
            self.assertEqual(messages, before)

    def test_tool_continuation_preserves_reasoning(self):
        prompt = encoder.encode_messages(history(), thinking_mode="thinking")
        self.assertIn("keep this thought</think>", prompt)
        self.assertLess(
            prompt.index("<tool_result>FIRST"), prompt.index("<tool_result>SECOND")
        )

    def test_context_is_merged_before_sorted_slice(self):
        full = history()
        context = full[:3]
        messages = full[3:]
        for mode in ("chat", "thinking"):
            self.assertEqual(
                encoder.encode_messages(messages, thinking_mode=mode, context=context),
                reference.encode_messages(
                    messages, thinking_mode=mode, context=context
                ),
            )

    def test_legal_wrapper_extensions_remain(self):
        messages = [
            {"role": "developer", "content": [{"type": "input_text", "text": "Guard"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
            {"role": "latest_reminder", "content": "Recall"},
        ]
        normalized = normalize(messages)
        self.assertEqual(normalized[0], {"role": "system", "content": "Guard"})
        self.assertIn(
            "<｜latest_reminder｜>Recall",
            encoder.encode_messages(normalized, thinking_mode="chat"),
        )
        tools = [
            {
                "type": "function",
                "namespace": {"name": "repo", "description": "Local"},
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ]
        encoded = encoder.tools_from_openai_format(tools)
        self.assertEqual(encoded[0]["name"], "repo::read")
        self.assertIn("Local", encoded[0]["description"])

    def test_stable_unknown_ids_and_non_tool_boundaries(self):
        messages = history()[:2] + [
            {"role": "tool", "tool_call_id": "unknown-A", "content": "A"},
            {"role": "tool", "tool_call_id": "second", "content": "SECOND"},
            {"role": "tool", "tool_call_id": "unknown-B", "content": "B"},
            {"role": "user", "content": "Boundary"},
            {"role": "tool", "tool_call_id": "first", "content": "FIRST"},
        ]
        ordered = encoder.sort_tool_messages_for_media(messages)
        self.assertEqual(
            [m.get("tool_call_id") for m in ordered[2:5]],
            ["unknown-A", "unknown-B", "second"],
        )
        self.assertEqual(ordered[5:], messages[5:])
        self.assertEqual(
            [m["tool_call_id"] for m in messages if m["role"] == "tool"],
            ["unknown-A", "second", "unknown-B", "first"],
        )

    def test_media_sort_does_not_leak_order_across_a_user_boundary(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "first",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    },
                    {
                        "id": "second",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "second", "content": "old-second"},
            {"role": "tool", "tool_call_id": "first", "content": "old-first"},
            {"role": "user", "content": "New turn"},
            {"role": "tool", "tool_call_id": "later-b", "content": "later-b"},
            {"role": "tool", "tool_call_id": "later-a", "content": "later-a"},
        ]
        ordered = encoder.sort_tool_messages_for_media(messages)
        self.assertEqual(
            [
                message.get("tool_call_id")
                for message in ordered
                if message.get("role") == "tool"
            ],
            ["first", "second", "later-b", "later-a"],
        )


class RendererOrderTests(unittest.TestCase):
    def renderer(self, model_type="deepseek_v41"):
        trace = {}

        def parse(messages, *args, **kwargs):
            trace["parsed"] = [
                m.get("tool_call_id") for m in messages if m["role"] == "tool"
            ]
            images = [
                p["image_url"]["url"]
                for m in messages
                for p in m.get("content", [])
                if isinstance(p, dict) and p.get("type") == "image_url"
            ]
            return messages, {"image": images}, {"image": images}

        async def parse_async(*args, **kwargs):
            return parse(*args, **kwargs)

        namespace = {
            "VllmConfig": object,
            "DeepseekV4Tokenizer": object,
            "ChatCompletionMessageParam": dict,
            "ConversationMessage": dict,
            "DictPrompt": dict,
            "ChatParams": object,
            "parse_chat_messages": parse,
            "parse_chat_messages_async": parse_async,
            "parse_dec_only_prompt": lambda text: {"prompt": text},
        "attach_video_uuids": video_prompt.attach_video_uuids,
        "attach_video_prompt_timestamps": video_prompt.attach_video_prompt_timestamps,
        }
        renderer_type = load_class(
            ADAPTATION / "files/vllm/renderers/deepseek_v4.py",
            namespace,
            bases=["object"],
            filename="patched-renderer.py",
        )
        instance = renderer_type.__new__(renderer_type)
        instance.model_config = types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=model_type)
        )

        def apply_template(**kwargs):
            trace["rendered"] = [
                m.get("tool_call_id") for m in kwargs["messages"] if m["role"] == "tool"
            ]
            return encoder.encode_messages(
                normalize(kwargs["messages"]), thinking_mode="thinking"
            )

        instance.get_tokenizer = lambda: types.SimpleNamespace(
            apply_chat_template=apply_template
        )

        async def apply_async(**kwargs):
            return instance._apply_chat_template(**kwargs)

        instance._apply_chat_template_async = apply_async
        return instance, trace

    def exercise(self, asynchronous, model_type):
        renderer, trace = self.renderer(model_type)
        params = types.SimpleNamespace(
            media_io_kwargs=None,
            mm_processor_kwargs=None,
            get_apply_chat_template_kwargs=lambda: {},
        )
        modules = {"vllm.tokenizers.deepseek_v41_encoding": encoder}
        messages = history(with_images=True)
        before = copy.deepcopy(messages)
        with patch.dict(sys.modules, modules):
            if asynchronous:
                _, prompt = asyncio.run(
                    renderer.render_messages_async(messages, params)
                )
            else:
                _, prompt = renderer.render_messages(messages, params)
        wanted = (
            ["first", "second"] if model_type == "deepseek_v41" else ["second", "first"]
        )
        self.assertEqual(trace["parsed"], wanted)
        self.assertEqual(trace["rendered"], wanted)
        self.assertEqual(
            prompt["multi_modal_data"]["image"], [x.upper() + ".png" for x in wanted]
        )
        if model_type == "deepseek_v41":
            self.assertLess(
                prompt["prompt"].index("<tool_result>FIRST"),
                prompt["prompt"].index("<tool_result>SECOND"),
            )
            self.assertEqual(prompt["prompt"].count(encoder.IMAGE_PLACEHOLDER), 2)
            reference_prompt, reference_media = reference.encode_messages(
                messages, thinking_mode="thinking", return_multi_modal_data=True
            )
            self.assertEqual(prompt["prompt"], reference_prompt)
            self.assertEqual(
                prompt["multi_modal_data"]["image"],
                [item["url"] for item in reference_media["images"]],
            )
        self.assertEqual(messages, before)

    def test_sync_v41_collects_images_in_encoded_tool_order(self):
        self.exercise(False, "deepseek_v41")

    def test_async_v41_collects_images_in_encoded_tool_order(self):
        self.exercise(True, "deepseek_v41")

    def test_sync_other_model_preserves_input_order(self):
        self.exercise(False, "deepseek_v4")

    def test_async_other_model_preserves_input_order(self):
        self.exercise(True, "deepseek_v4")


class InstallerTests(unittest.TestCase):
    def test_installer_rejects_everything_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            package = folder / "package"
            shutil.copytree(ADAPTATION, package)
            site = folder / "site"
            rows = json.loads((package / "manifest.json").read_text())["files"]
            for row in rows:
                target = site / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(package / "files" / row["path"], target)
            installer.install(site, package)
            installer.install(site, package)
            first = site / rows[0]["path"]
            second = site / rows[1]["path"]
            expected = first.read_bytes()
            second.write_text("unexpected upstream modification\n")
            with self.assertRaisesRegex(RuntimeError, "Unexpected installed base"):
                installer.install(site, package)
            self.assertEqual(first.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
