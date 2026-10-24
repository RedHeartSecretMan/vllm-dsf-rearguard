from __future__ import annotations

import json
import unittest
from unittest.mock import Mock


class ResponsesCustomToolHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from vllm.entrypoints.generate.base.protocol import (
                DeltaFunctionCall,
                DeltaMessage,
                DeltaToolCall,
                FunctionCall,
            )
            from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        except ModuleNotFoundError as error:
            raise unittest.SkipTest("Responses tests require the pinned vLLM runtime") from error
        from openai.types.responses import CustomTool
        from vllm.entrypoints.openai.responses.streaming_events import (
            SimpleStreamingEventProcessor,
            _StateType,
            split_delta,
        )
        from vllm.entrypoints.openai.responses.utils import (
            build_response_output_items,
            construct_chat_messages_with_tool_call,
            construct_input_messages,
        )

        cls.ResponsesRequest = ResponsesRequest
        cls.DeltaFunctionCall = DeltaFunctionCall
        cls.DeltaMessage = DeltaMessage
        cls.DeltaToolCall = DeltaToolCall
        cls.SimpleStreamingEventProcessor = SimpleStreamingEventProcessor
        cls.StateType = _StateType
        cls.split_delta = staticmethod(split_delta)
        cls.FunctionCall = FunctionCall
        cls.CustomTool = CustomTool
        cls.build_output = staticmethod(build_response_output_items)
        cls.construct_messages = staticmethod(construct_chat_messages_with_tool_call)
        cls.construct_input = staticmethod(construct_input_messages)

    def test_responses_video_extension_preserves_mixed_content_and_history(self):
        content = [
            {"type": "input_text", "text": "Compare the still image and video."},
            {"type": "input_image", "image_url": "https://example.com/still.png", "detail": "auto"},
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4", "video_id": "vid_clip"}, "uuid": "vid_clip"},
            {"type": "input_text", "text": "Keep their original order."},
        ]
        request = self.ResponsesRequest.model_validate({
            "model": "deepseek-flash", "input": [{"role": "user", "content": content}],
        })
        messages = self.construct_messages(request.input)
        self.assertEqual(messages, [{"role": "user", "content": content}])
        replay = self.ResponsesRequest.model_validate(request.model_dump())
        self.assertEqual(self.construct_messages(replay.input), messages)
        followup = self.construct_input(request_input="Continue video vid_clip.", prev_msg=messages)
        self.assertEqual(followup[0], messages[0])
        self.assertEqual(followup[-1], {"role": "user", "content": "Continue video vid_clip."})
        from vllm.entrypoints.chat_utils import _parse_chat_message_content_part

        media_parser = Mock()
        _parse_chat_message_content_part(
            messages[0]["content"][2], media_parser,
            wrap_dicts=False, interleave_strings=True,
        )
        media_parser.parse_video.assert_called_once_with("https://example.com/clip.mp4", "vid_clip")

    def test_responses_video_extension_rejects_malformed_inputs(self):
        from pydantic import ValidationError

        malformed = [
            {"type": "video_url", "video_url": "https://example.com/clip.mp4"},
            {"type": "video_url", "video_url": {}},
            {"type": "video_url", "video_url": {"url": ""}},
            {"type": "video_url", "video_url": {"url": 42}},
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}, "uuid": ["wrong"]},
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4", "unknown_option": True}},
        ]
        for part in malformed:
            with self.subTest(part=part), self.assertRaises(ValidationError):
                self.ResponsesRequest.model_validate({
                    "model": "deepseek-flash", "input": [{"role": "user", "content": [part]}],
                })

    def test_responses_video_extension_does_not_relax_official_image_schema(self):
        from pydantic import ValidationError

        video = {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
        for image in (
            {"type": "input_image", "image_url": 123, "detail": "auto"},
            {"type": "input_image", "image_url": "https://example.com/still.png", "detail": "invalid"},
        ):
            with self.subTest(image=image), self.assertRaises(ValidationError):
                self.ResponsesRequest.model_validate({
                    "model": "deepseek-flash", "input": [{"role": "user", "content": [image, video]}],
                })

    def test_deepseek_named_tools_parse_bare_json_through_real_parser(self):
        from types import SimpleNamespace

        from vllm.parser.deepseek_v4 import DeepSeekV4Parser
        from vllm.parser.deepseek_v41 import DeepSeekV41Parser

        tokenizer = SimpleNamespace(get_vocab=lambda: {}, all_special_tokens=[], all_special_ids=[])
        arguments = json.dumps({"input": 'line one\nline "two" \\ end <think>literal</think>'})
        for parser_class in (DeepSeekV4Parser, DeepSeekV41Parser):
            for kind in ("custom", "function"):
                for thinking in (False, True):
                    tool = {"type": kind, "name": "record_text"}
                    if kind == "function":
                        tool["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
                    request = self.ResponsesRequest.model_validate({
                        "model": "deepseek-flash", "input": "Call record_text.",
                        "tools": [tool], "tool_choice": {"type": kind, "name": "record_text"},
                        "chat_template_kwargs": {"thinking": thinking},
                    })
                    parser = parser_class(tokenizer, tools=request.tools, chat_template_kwargs={"thinking": thinking})
                    text = ("Considering the call.</think>" if thinking else "") + arguments
                    with self.subTest(parser=parser_class.__name__, kind=kind, thinking=thinking):
                        reasoning, content, calls = parser.parse(text, request)
                        self.assertEqual(reasoning, "Considering the call." if thinking else None)
                        self.assertIsNone(content)
                        self.assertEqual(len(calls or []), 1)
                        self.assertEqual(calls[0].name, "record_text")
                        self.assertEqual(calls[0].arguments, arguments)

    def test_deepseek_named_tool_json_stream_reaches_tool_events(self):
        from types import SimpleNamespace

        from vllm.parser.deepseek_v41 import DeepSeekV41Parser

        tokenizer = SimpleNamespace(get_vocab=lambda: {}, all_special_tokens=[], all_special_ids=[])
        value = 'line one\nline "two" \\ end <think>literal</think> <｜DSML｜ calls>'
        arguments = json.dumps({"input": value}, ensure_ascii=False)
        for kind in ("custom", "function"):
            for thinking in (False, True):
                tool = {"type": kind, "name": "record_text"}
                if kind == "function":
                    tool["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
                request = self.ResponsesRequest.model_validate({
                    "model": "deepseek-flash", "input": "Call record_text.",
                    "tools": [tool], "tool_choice": {"type": kind, "name": "record_text"},
                    "chat_template_kwargs": {"thinking": thinking}, "stream": True,
                })
                parser = DeepSeekV41Parser(tokenizer, tools=request.tools, chat_template_kwargs={"thinking": thinking})
                text = ("Considering the call.</think>" if thinking else "") + arguments
                chunks = [text[index:index + 3] for index in range(0, len(text), 3)]
                deltas = [parser.parse_delta(chunk, [], request, finished=index == len(chunks) - 1)
                          for index, chunk in enumerate(chunks)]
                with self.subTest(kind=kind, thinking=thinking):
                    self.assertFalse(any(delta.content for delta in deltas if delta))
                    calls = [call for delta in deltas if delta for call in delta.tool_calls or []]
                    self.assertTrue(calls)
                    self.assertEqual("".join(call.function.arguments or "" for call in calls), arguments)
                    self.assertEqual([call.function.name for call in calls if call.function.name], ["record_text"])
                    ids = [call.id for call in calls if call.id]
                    self.assertEqual(len(ids), 1)
                    self.assertTrue(any(delta and delta.tool_calls for delta in deltas[:-1]))
                    # Responses finalization reparses with the same parser.
                    final_calls = parser.parse(text, request)[2]
                    self.assertEqual(final_calls[0].id, ids[0])
                    processor = self.SimpleStreamingEventProcessor(tools=request.tools)
                    events = []
                    for delta in deltas:
                        if delta is None:
                            continue
                        for part in self.split_delta(delta):
                            target, call = processor.resolve_target_state(part)
                            if target == self.StateType.NONE:
                                continue
                            if processor.needs_transition(target, call):
                                events.extend(processor.close_current())
                            if processor.state.current_state != target:
                                events.extend(processor.open(target, call))
                            events.extend(processor.emit_delta(part, None))
                    events.extend(processor.close_current())
                    event_name = "response.custom_tool_call_input.done" if kind == "custom" else "response.function_call_arguments.done"
                    done = [event for event in events if event.type == event_name]
                    self.assertEqual(len(done), 1)
                    self.assertEqual(done[0].input if kind == "custom" else done[0].arguments,
                                     value if kind == "custom" else arguments)

    def test_deepseek_auto_json_remains_text_and_native_dsml_still_parses(self):
        from types import SimpleNamespace

        from vllm.parser.deepseek_v41 import DeepSeekV41Parser

        tokenizer = SimpleNamespace(get_vocab=lambda: {}, all_special_tokens=[], all_special_ids=[])
        request = self.ResponsesRequest.model_validate({
            "model": "deepseek-flash", "input": "Call record_text.",
            "tools": [{"type": "custom", "name": "record_text"}], "tool_choice": "auto",
            "chat_template_kwargs": {"thinking": False},
        })
        parser = DeepSeekV41Parser(tokenizer, tools=request.tools, chat_template_kwargs={"thinking": False})
        self.assertEqual(parser.parse('{"input":"ordinary text"}', request)[1:],
                         ('{"input":"ordinary text"}', None))
        native = ('<｜DSML｜ calls><｜DSML｜ invoke name="record_text">'
                  '<｜DSML｜ parameter name="input" string="true">native'
                  '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>')
        reasoning, content, calls = parser.parse(native, request)
        self.assertIsNone(reasoning)
        self.assertIsNone(content)
        self.assertEqual(calls[0].name, "record_text")
        self.assertEqual(json.loads(calls[0].arguments), {"input": "native"})

    def test_stream_completed_and_stored_history_reuse_emitted_items(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from vllm.entrypoints.openai.responses.context import SimpleContext
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
        from vllm.parser.deepseek_v41 import DeepSeekV41Parser
        from vllm.sampling_params import SamplingParams

        async def exercise(kind):
            tokenizer = SimpleNamespace(get_vocab=lambda: {}, all_special_tokens=[], all_special_ids=[])
            tools = [] if kind == "text" else [{"type": kind, "name": "record_text"}]
            if kind == "function":
                tools[0]["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
            request = self.ResponsesRequest.model_validate({
                "model": "deepseek-flash", "input": "Test streaming.", "tools": tools,
                "tool_choice": {"type": kind, "name": "record_text"} if tools else "none",
                "chat_template_kwargs": {"thinking": False}, "stream": True, "store": True,
            })
            parser = DeepSeekV41Parser(tokenizer, tools=request.tools, chat_template_kwargs={"thinking": False})
            text = "ACCEPTANCE_OK" if kind == "text" else json.dumps({"input": "streamed value"})
            context = Mock(spec=SimpleContext)
            context.response_parser = parser
            context.num_prompt_tokens = 20
            context.num_output_tokens = 10
            context.num_cached_tokens = context.num_reasoning_tokens = 0
            context._accumulated_token_ids = []
            context.all_turn_metrics = []
            context.kv_transfer_params = context.ec_transfer_params = None
            context.final_output = SimpleNamespace(
                outputs=[SimpleNamespace(text=text, finish_reason="stop")], prompt_token_ids=[])
            server = object.__new__(OpenAIServingResponses)
            server.use_harmony = False
            server.parser = parser
            server._initialize_tool_sessions = AsyncMock()
            server.response_store = {}
            server.response_store_lock = asyncio.Lock()
            server._make_response_output_items = Mock(side_effect=AssertionError("stream must not reparse"))

            async def chunks():
                for index in range(0, len(text), 3):
                    context.last_output = SimpleNamespace(
                        outputs=[SimpleNamespace(text=text[index:index + 3], token_ids=[],
                                                 finish_reason="stop" if index + 3 >= len(text) else None)],
                        prompt_token_ids=[],
                    )
                    yield context

            events = [event async for event in server.responses_stream_generator(
                request, SamplingParams(max_tokens=256), chunks(), context,
                "deepseek-flash", tokenizer, SimpleNamespace(),
            )]
            items = [event.item.model_dump() for event in events if event.type == "response.output_item.done"]
            completed = next(event.response for event in events if event.type == "response.completed")
            self.assertTrue(items)
            self.assertEqual([item.model_dump() for item in completed.output], items)
            stored = server.response_store[request.request_id]
            self.assertEqual([item.model_dump() for item in stored.output], items)
            server._make_response_output_items.assert_not_called()

        for kind in ("text", "custom", "function"):
            with self.subTest(kind=kind):
                asyncio.run(exercise(kind))

    def test_custom_tool_call_and_output_round_trip(self):
        request = self.ResponsesRequest.model_validate(
            {
                "model": "deepseek-flash",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_shell",
                        "name": "shell",
                        "input": "printf 'ok'",
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call_shell",
                        "output": "ok",
                    },
                ],
            }
        )

        messages = self.construct_messages(request.input)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "assistant")
        tool_call = messages[0]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "call_shell")
        self.assertEqual(tool_call["function"]["name"], "shell")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"input": "printf 'ok'"},
        )
        self.assertEqual(
            messages[1],
            {
                "role": "tool",
                "tool_call_id": "call_shell",
                "content": "ok",
            },
        )

    def test_custom_tool_output_item_with_optional_response_fields(self):
        request = self.ResponsesRequest.model_validate(
            {
                "model": "deepseek-flash",
                "input": [
                    {
                        "id": "ctc_1",
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "apply_patch",
                        "input": "*** Begin Patch\n*** End Patch",
                    },
                    {
                        "id": "cto_1",
                        "type": "custom_tool_call_output",
                        "status": "completed",
                        "call_id": "call_1",
                        "output": "applied",
                    },
                ],
            }
        )

        messages = self.construct_messages(request.input)

        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[1]["content"], "applied")

    def test_previous_response_output_replays_tool_calls_and_outputs(self):
        request = self.ResponsesRequest.model_validate(
            {
                "model": "deepseek-flash",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_history",
                        "name": "apply_patch",
                        "input": "*** Begin Patch\n*** End Patch",
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call_history",
                        "output": "applied",
                    },
                ],
            }
        )

        messages = self.construct_input(
            request_input="Continue.",
            prev_response_output=request.input,
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_history")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_history")
        self.assertEqual(messages[1]["content"], "applied")
        self.assertEqual(messages[2], {"role": "user", "content": "Continue."})

    def test_custom_tool_output_keeps_freeform_input(self):
        output = self.build_output(
            reasoning=None,
            content=None,
            tool_calls=[
                self.FunctionCall(
                    id="call_1",
                    name="apply_patch",
                    arguments=json.dumps({"input": "*** Begin Patch\n*** End Patch"}),
                )
            ],
            tools=[self.CustomTool(type="custom", name="apply_patch")],
        )
        self.assertEqual(output[0].type, "custom_tool_call")
        self.assertEqual(output[0].call_id, "call_1")
        self.assertEqual(output[0].input, "*** Begin Patch\n*** End Patch")

    def test_streaming_custom_tool_events_preserve_freeform_input(self):
        processor = self.SimpleStreamingEventProcessor(
            tools=[self.CustomTool(type="custom", name="apply_patch")]
        )

        def emit(delta):
            events = []
            for part in self.split_delta(delta):
                target, call = processor.resolve_target_state(part)
                if target == self.StateType.NONE:
                    continue
                if processor.needs_transition(target, call):
                    events.extend(processor.close_current())
                if processor.state.current_state != target:
                    events.extend(processor.open(target, call))
                events.extend(processor.emit_delta(part, None))
            return events

        events = emit(self.DeltaMessage(tool_calls=[
            self.DeltaToolCall(
                index=0,
                function=self.DeltaFunctionCall(
                    name="apply_patch", arguments='{"input":"ls '
                ),
            )
        ]))
        events.extend(emit(self.DeltaMessage(tool_calls=[
            self.DeltaToolCall(
                index=0,
                function=self.DeltaFunctionCall(arguments='-la"}')
            )
        ])))
        events.extend(processor.close_current())

        event_types = [event.type for event in events]
        self.assertIn("response.custom_tool_call_input.delta", event_types)
        self.assertIn("response.custom_tool_call_input.done", event_types)
        self.assertEqual(events[-1].item.type, "custom_tool_call")
        self.assertEqual(events[-1].item.input, "ls -la")


if __name__ == "__main__":
    unittest.main()
