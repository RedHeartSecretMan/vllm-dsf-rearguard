"""CPU-only contracts for deterministic acceptance gating."""

import unittest

from scripts.acceptance.core import response_digest
from scripts.acceptance.quality_gate import compare


def receipt(digest="a" * 64, status="PASS"):
    return {
        "status": status,
        "cases": [
            {
                "name": "arithmetic-thinking-False",
                "status": "PASS",
                "output": {"sha256": digest, "chars": 12},
            }
        ],
    }


class QualityGateTests(unittest.TestCase):
    def test_structured_tool_output_is_part_of_digest(self):
        first = response_digest(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {"name": "lookup", "arguments": '{"key":"a"}'},
                    }
                ],
            }
        )
        second = response_digest(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "function": {"name": "lookup", "arguments": '{"key":"b"}'},
                    }
                ],
            }
        )
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_identical_receipts_pass(self):
        result = compare(receipt(), receipt())
        self.assertEqual(result["status"], "PASS")

    def test_changed_output_fails(self):
        result = compare(receipt(), receipt("b" * 64))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(len(result["digest_mismatches"]), 1)

    def test_missing_digest_fails_closed(self):
        candidate = receipt()
        del candidate["cases"][0]["output"]
        result = compare(receipt(), candidate)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("missing assistant output digest", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
