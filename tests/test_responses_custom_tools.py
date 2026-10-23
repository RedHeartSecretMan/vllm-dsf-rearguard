from __future__ import annotations

import json
import unittest


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

    def test_custom_tool_output_keeps_freeform_input(self):
        output = self.build_output(
            reasoning=None,
            content=None,
            tool_calls=[
                self.FunctionCall(id="call_1", name="apply_patch", arguments='{"input": "*** Begin Patch\n*** End Patch"}')
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
