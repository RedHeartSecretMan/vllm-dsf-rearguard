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

    def production_parser(self, request):
        from types import SimpleNamespace

        from vllm.parser import ParserManager

        vocab = {text: index for index, text in enumerate(
            ("<think>", "</think>", "<｜DSML｜ calls>", "</｜DSML｜ calls>"), 1)}
        tokenizer = SimpleNamespace(get_vocab=lambda: vocab, all_special_tokens=list(vocab),
                                    all_special_ids=list(vocab.values()))
        parser_class = ParserManager.get_parser(
            tool_parser_name="deepseek_v41", reasoning_parser_name="deepseek_v41",
            enable_auto_tools=True, model_name="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        )
        parser = parser_class(tokenizer, tools=request.tools,
                              chat_template_kwargs=request.chat_template_kwargs)
        self.assertFalse(parser.tool_parser.supports_required_and_named)
        parser.adjust_request(request)
        return parser, tokenizer

    def test_production_parser_enables_named_custom_dsml_grammar(self):
        for kind in ("custom", "function"):
            for choice in ({"type": kind, "name": "record_text"}, "required"):
                for thinking in (False, True):
                    tool = {"type": kind, "name": "record_text"}
                    if kind == "function":
                        tool["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
                    request = self.ResponsesRequest.model_validate({
                        "model": "deepseek-flash", "input": "Call record_text.", "tools": [tool],
                        "tool_choice": choice, "chat_template_kwargs": {"thinking": thinking},
                    })
                    parser, _ = self.production_parser(request)
                    with self.subTest(kind=kind, choice=choice, thinking=thinking):
                        self.assertIsNotNone(request.structured_outputs)
                        grammar = request.structured_outputs.structural_tag
                        self.assertIn("record_text", grammar)
                        self.assertIn("DSML", grammar)
                        value = 'line one\nline "two" \\ end'
                        native = ('<｜DSML｜ calls><｜DSML｜ invoke name="record_text">'
                                  '<｜DSML｜ parameter name="input" string="true">' + value +
                                  '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>')
                        text = ("Considering the call.</think>" if thinking else "") + native
                        reasoning, content, calls = parser.parse(text, request, enable_auto_tools=True)
                        self.assertEqual(reasoning, "Considering the call." if thinking else None)
                        self.assertIsNone(content)
                        self.assertEqual(len(calls or []), 1)
                        self.assertEqual(calls[0].name, "record_text")
                        self.assertEqual(json.loads(calls[0].arguments), {"input": value})

    def test_production_parser_streams_custom_and_function_events(self):
        value = 'line one\nline "two" \\ end'
        for kind in ("custom", "function"):
            for thinking in (False, True):
                tool = {"type": kind, "name": "record_text"}
                if kind == "function":
                    tool["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
                request = self.ResponsesRequest.model_validate({
                    "model": "deepseek-flash", "input": "Call record_text.", "tools": [tool],
                    "tool_choice": {"type": kind, "name": "record_text"},
                    "chat_template_kwargs": {"thinking": thinking}, "stream": True,
                })
                parser, _ = self.production_parser(request)
                native = ('<｜DSML｜ calls><｜DSML｜ invoke name="record_text">'
                          '<｜DSML｜ parameter name="input" string="true">' + value +
                          '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>')
                text = ("Considering the call.</think>" if thinking else "") + native
                chunks = [text[index:index + 3] for index in range(0, len(text), 3)]
                deltas = [parser.parse_delta(chunk, [], request, prompt_token_ids=[], finished=index == len(chunks) - 1)
                          for index, chunk in enumerate(chunks)]
                with self.subTest(kind=kind, thinking=thinking):
                    self.assertIsNotNone(request.structured_outputs)
                    self.assertFalse(any(delta.content for delta in deltas if delta))
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
                    self.assertEqual(done[0].input if kind == "custom" else json.loads(done[0].arguments)["input"], value)

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
        from vllm.sampling_params import SamplingParams

        async def exercise(kind, with_logprobs=False, with_logging=False):
            from openai.types.responses import response_text_delta_event

            tools = [] if kind == "text" else [{"type": kind, "name": "record_text"}]
            if kind == "function":
                tools[0]["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
            request = self.ResponsesRequest.model_validate({
                "model": "deepseek-flash", "input": "Test streaming.", "tools": tools,
                "tool_choice": {"type": kind, "name": "record_text"} if tools else "none",
                "chat_template_kwargs": {"thinking": False}, "stream": True, "store": True,
                "include": ["message.output_text.logprobs"] if with_logprobs else [],
                "top_logprobs": 1 if with_logprobs else 0,
            })
            parser, tokenizer = self.production_parser(request)
            text = "ACCEPTANCE_OK" if kind == "text" else (
                '<｜DSML｜ calls><｜DSML｜ invoke name="record_text">'
                '<｜DSML｜ parameter name="input" string="true">streamed value'
                '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>'
            )
            context = Mock(spec=SimpleContext)
            context.response_parser = parser
            context.num_prompt_tokens = 20
            context.num_output_tokens = 10
            context.num_cached_tokens = context.num_reasoning_tokens = 0
            context._accumulated_token_ids = []
            context.all_turn_metrics = []
            context.kv_transfer_params = context.ec_transfer_params = None
            context.final_output = SimpleNamespace(
                outputs=[SimpleNamespace(text=text, token_ids=[], finish_reason="stop")], prompt_token_ids=[])
            server = object.__new__(OpenAIServingResponses)
            server.use_harmony = False
            server.parser = parser
            server.enable_log_outputs = with_logging
            server.request_logger = Mock() if with_logging else None
            server._initialize_tool_sessions = AsyncMock()
            server.response_store = {}
            server.response_store_lock = asyncio.Lock()
            server._make_response_output_items = Mock(side_effect=AssertionError("stream must not reparse"))
            logprob_batches = [[response_text_delta_event.Logprob(
                token=text[index:index + 3], logprob=-0.25,
                bytes=list(text[index:index + 3].encode()),
                top_logprobs=[response_text_delta_event.LogprobTopLogprob(
                    token="alternative", logprob=-2.0, bytes=list(b"alternative"),
                )],
            )] for index in range(0, len(text), 3)]
            server._create_stream_response_logprobs = Mock(side_effect=logprob_batches)

            async def chunks():
                for index in range(0, len(text), 3):
                    context.last_output = SimpleNamespace(
                        outputs=[SimpleNamespace(text=text[index:index + 3], token_ids=[],
                                                 logprobs=[{}] if with_logprobs else None,
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
            if with_logprobs:
                expected = [item.model_dump() for batch in logprob_batches for item in batch]
                text_done = next(event for event in events if event.type == "response.output_text.done")
                self.assertEqual([item.model_dump() for item in text_done.logprobs], expected)
                self.assertEqual(items[0]["content"][0]["logprobs"], expected)
                self.assertEqual([item.model_dump() for item in stored.output[0].content[0].logprobs], expected)
            if with_logging:
                server.request_logger.log_outputs.assert_called_once_with(
                    request_id=request.request_id, outputs=text, output_token_ids=[],
                    finish_reason="stop", is_streaming=False, delta=False,
                )

        for kind in ("text", "custom", "function"):
            with self.subTest(kind=kind):
                asyncio.run(exercise(kind))
        with self.subTest(feature="logprobs"):
            asyncio.run(exercise("text", with_logprobs=True))
        with self.subTest(feature="final output logging"):
            asyncio.run(exercise("text", with_logging=True))

    def test_content_logprobs_survive_done_and_reset_for_next_message(self):
        from openai.types.responses import response_text_delta_event

        logs = [response_text_delta_event.Logprob(
            token=token, logprob=-0.1, bytes=list(token.encode()),
            top_logprobs=[response_text_delta_event.LogprobTopLogprob(
                token=token.upper(), logprob=-1.5, bytes=list(token.upper().encode()),
            )],
        ) for token in ("a", "b")]
        processor = self.SimpleStreamingEventProcessor()
        processor.open(self.StateType.CONTENT)
        for logprob in logs:
            processor.emit_delta(self.DeltaMessage(content=logprob.token), None,
                                 get_logprobs=lambda _, value=logprob: [value])
        events = processor.close_current()
        expected = [item.model_dump() for item in logs]
        self.assertEqual([item.model_dump() for item in events[0].logprobs], expected)
        self.assertEqual([item.model_dump() for item in events[1].part.logprobs], expected)
        self.assertEqual([item.model_dump() for item in events[2].item.content[0].logprobs], expected)
        processor.open(self.StateType.CONTENT)
        processor.emit_delta(self.DeltaMessage(content="no probabilities requested"), None)
        following = processor.close_current()
        self.assertEqual(following[0].logprobs, [])
        self.assertIsNone(following[2].item.content[0].logprobs)

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
