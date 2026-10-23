"""CPU tests for fail-closed PPL aggregation."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import ppl_score, ppl_wikitext


class PplTests(unittest.TestCase):
    def test_target_token_logprobs_skip_first_position(self):
        response = {
            "choices": [
                {
                    "prompt_token_ids": [10, 11, 12],
                    "prompt_logprobs": [None, {"11": {"logprob": -1.0}}, {12: {"logprob": -2.0}}],
                }
            ]
        }
        self.assertEqual(ppl_score.target_logprobs(response), [-1.0, -2.0])

    def test_target_tokens_can_come_from_the_input_when_api_omits_ids(self):
        response = {
            "choices": [
                {
                    "prompt_token_ids": None,
                    "prompt_logprobs": [None, {"11": {"logprob": -1.0}}],
                }
            ]
        }
        self.assertEqual(ppl_score.target_logprobs(response, [10, 11]), [-1.0])

    def test_missing_target_is_rejected_instead_of_approximated(self):
        response = {
            "choices": [
                {
                    "prompt_token_ids": [10, 11],
                    "prompt_logprobs": [None, {"99": {"logprob": -1.0}}],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "absent"):
            ppl_score.target_logprobs(response)

    def test_evaluate_aggregates_nll_and_perplexity(self):
        responses = iter(
            [
                {
                    "choices": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "prompt_logprobs": [
                                None,
                                {"2": {"logprob": -1}},
                                {"3": {"logprob": -2}},
                            ],
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "prompt_token_ids": [4, 5],
                            "prompt_logprobs": [None, {"5": {"logprob": -3}}],
                        }
                    ]
                },
            ]
        )
        with patch.object(ppl_score, "json_request", side_effect=lambda *args, **kwargs: next(responses)):
            result = ppl_score.evaluate("http://example", [[1, 2, 3], [4, 5]], 1, 10)
        self.assertEqual(result["scored_tokens"], 3)
        self.assertAlmostEqual(result["mean_nll"], 2.0)
        self.assertAlmostEqual(result["perplexity"], math.exp(2.0))

    def test_request_form_selects_the_server_fields_and_is_recorded(self):
        sent = []

        def fake(base, path, body, timeout):
            sent.append(body)
            return {"choices": [{"prompt_token_ids": [4, 5], "prompt_logprobs": [None, {"5": {"logprob": -1}}]}]}

        with patch.object(ppl_score, "json_request", side_effect=fake):
            echo = ppl_score.evaluate("http://x", [[4, 5]], 1, 10, "echo")
            plain = ppl_score.evaluate("http://x", [[4, 5]], 1, 10)
        self.assertEqual((sent[0]["logprobs"], sent[0]["echo"]), (1, True))
        self.assertNotIn("prompt_logprobs", sent[0])
        self.assertEqual(sent[1]["prompt_logprobs"], 1)
        self.assertEqual((echo["request_form"], plain["request_form"]), ("echo", "prompt_logprobs"))
        with self.assertRaisesRegex(ValueError, "request_form"):
            ppl_score.evaluate("http://x", [[4, 5]], 1, 10, "other")

    def test_windows_are_non_overlapping_and_each_skips_its_first_token(self):
        windows = ppl_score.token_windows("x", [{"tokens": list(range(5000))}], 2048, 2048, 1)
        self.assertEqual([len(w) for w in windows], [2048, 2048, 904])
        self.assertEqual(sum(len(w) - 1 for w in windows), 5000 - len(windows))

    def test_word_and_byte_perplexity_need_one_whole_corpus_record(self):
        report = {"total_nll": 8.0}
        ppl_score.add_unit_perplexities(report, [{"tokens": [1, 2], "word_count": 4, "byte_count": 8}])
        self.assertAlmostEqual(report["word_perplexity"], math.exp(2.0))
        self.assertAlmostEqual(report["byte_perplexity"], math.exp(1.0))
        partial = {"total_nll": 8.0}
        ppl_score.add_unit_perplexities(partial, [{"tokens": [1], "word_count": 4}, {"tokens": [2]}])
        self.assertNotIn("word_perplexity", partial)

    def test_read_records_requires_text_or_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            path.write_text(json.dumps({"other": 1}) + "\n")
            with self.assertRaisesRegex(ValueError, "tokens or text"):
                ppl_score.read_records(path)


class WikitextPreparationTests(unittest.TestCase):
    def test_text_is_detokenized_but_counts_come_from_the_raw_rows(self):
        rows = ["", "a , b", "", "c d e"]
        text, words, size = ppl_wikitext.normalize_documents(rows, lambda r: r.replace(" , ", ", "))
        self.assertEqual(text, "a, b\n\nc d e")
        raw = "a , b\nc d e"
        self.assertEqual(size, len(raw.encode()))
        self.assertEqual(words, 6)


if __name__ == "__main__":
    unittest.main()
