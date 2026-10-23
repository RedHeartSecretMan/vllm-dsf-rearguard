from __future__ import annotations

import json
import unittest


class ResponsesCustomToolHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.entrypoints.openai.responses.utils import (
            construct_chat_messages_with_tool_call,
        )

        cls.ResponsesRequest = ResponsesRequest
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


if __name__ == "__main__":
    unittest.main()
