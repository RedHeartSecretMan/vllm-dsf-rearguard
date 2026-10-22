"""CPU-only contracts for quality and speculative metrics."""

import unittest

from scripts.quality_metrics import (
    aggregate_speculative,
    speculative_metrics,
    text_sha256,
)


class QualityMetricTests(unittest.TestCase):
    def test_text_digest_is_stable_and_exact(self):
        self.assertEqual(text_sha256("answer"), text_sha256("answer"))
        self.assertNotEqual(text_sha256("answer"), text_sha256("answer "))

    def test_speculative_metadata_is_optional(self):
        self.assertEqual(speculative_metrics({})["available"], False)
        self.assertEqual(
            speculative_metrics({"draft_tokens": 8}, {"accepted_tokens": 6}),
            {
                "available": True,
                "draft_tokens": 8,
                "accepted_tokens": 6,
                "acceptance_rate": 0.75,
            },
        )

    def test_invalid_counters_are_not_reported_as_zero_acceptance(self):
        self.assertFalse(
            speculative_metrics({"draft_tokens": 3, "accepted_tokens": 4})[
                "available"
            ]
        )

    def test_nested_vllm_per_request_metrics_are_supported(self):
        metrics = speculative_metrics(
            {
                "metrics": {
                    "speculative_decoding": {
                        "num_draft_tokens": 12,
                        "num_accepted_draft_tokens": 9,
                    }
                }
            }
        )
        self.assertEqual(metrics["acceptance_rate"], 0.75)

    def test_aggregate_uses_only_rows_with_evidence(self):
        rows = [
            {
                "speculative": speculative_metrics(
                    {"draft_tokens": 10, "accepted_tokens": 8}
                )
            },
            {"speculative": speculative_metrics({})},
        ]
        self.assertEqual(
            aggregate_speculative(rows),
            {
                "available": True,
                "requests_with_evidence": 1,
                "draft_tokens": 10,
                "accepted_tokens": 8,
                "acceptance_rate": 0.8,
            },
        )


if __name__ == "__main__":
    unittest.main()

