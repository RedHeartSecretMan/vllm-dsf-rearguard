"""CPU tests for fail-closed PPL aggregation."""

import gzip
import hashlib
import io
import json
import math
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts.acceptance import ppl_score, ppl_wikitext


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

    @staticmethod
    def cold_response(tokens, cached=0):
        return {
            "choices": [{
                "prompt_token_ids": tokens,
                "prompt_logprobs": [None] + [{str(token): {"logprob": -1.0}} for token in tokens[1:]],
            }],
            "usage": {"prompt_tokens": len(tokens), "completion_tokens": 1,
                      "prompt_tokens_details": {"cached_tokens": cached}},
        }

    def test_evidence_preserves_exact_windows_requests_and_responses(self):
        windows = [[1, 2, 3], [4, 5]]
        responses = [self.cold_response(tokens) for tokens in windows]
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(ppl_score, "json_request", side_effect=responses) as request:
                report = ppl_score.evaluate(
                    "http://example", windows, 1, 10, "echo", evidence_dir=Path(folder),
                    cache_salt="ppl-comparison", seed=1017,
                )
            self.assertEqual((report["status"], report["successful_windows"], report["failed_windows"]), ("PASS", 2, 0))
            self.assertEqual(report["scored_tokens"], report["expected_scored_tokens"])
            self.assertAlmostEqual(report["perplexity"], math.e)
            for index, row in enumerate(report["rows"]):
                sent = request.call_args_list[index].args[2]
                self.assertEqual(sent["prompt"], windows[index])
                self.assertEqual((sent["seed"], sent["cache_salt"]), (1017, "ppl-comparison"))
                raw = Path(row["raw_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["raw_sha256"])
                self.assertEqual(len(raw), row["raw_bytes"])
                self.assertEqual(json.loads(gzip.decompress(raw)), {
                    "request": sent, "response": responses[index], "error": None,
                })
                self.assertEqual(row["input_tokens"], row["observed_input_tokens"])
                self.assertEqual(row["scored_tokens"], len(windows[index]) - 1)
                self.assertEqual(row["cached_tokens"], 0)

    def test_missing_and_warm_cache_cannot_pass_but_keep_the_response(self):
        for cached in (None, 1):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as folder:
                response = self.cold_response([1, 2], cached)
                if cached is None:
                    response["usage"].pop("prompt_tokens_details")
                with patch.object(ppl_score, "json_request", return_value=response):
                    report = ppl_score.evaluate("http://x", [[1, 2]], 1, 10, evidence_dir=Path(folder))
                self.assertEqual(report["status"], "FAIL")
                self.assertIsNone(report["perplexity"])
                self.assertIn("cold cache", report["rows"][0]["error"])
                raw = json.loads(gzip.decompress(Path(report["rows"][0]["raw_path"]).read_bytes()))
                self.assertEqual(raw["response"], response)

    def test_failed_window_keeps_http_evidence_and_does_not_hide_later_windows(self):
        error = urllib.error.HTTPError("http://x", 500, "failed", {}, io.BytesIO(b'{"error":"injected"}'))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(ppl_score, "json_request", side_effect=[error, self.cold_response([3, 4])]):
                report = ppl_score.evaluate("http://x", [[1, 2], [3, 4]], 1, 10, evidence_dir=Path(folder))
            self.assertEqual((report["status"], report["successful_windows"], report["failed_windows"]), ("FAIL", 1, 1))
            self.assertEqual((report["scored_tokens"], report["expected_scored_tokens"]), (1, 2))
            self.assertIsNone(report["mean_nll"])
            self.assertIsNone(report["perplexity"])
            raw = json.loads(gzip.decompress(Path(report["rows"][0]["raw_path"]).read_bytes()))
            self.assertEqual(raw["request"]["prompt"], [1, 2])
            self.assertEqual(raw["response"], {"http_status": 500, "body": '{"error":"injected"}'})
            self.assertIn("HTTPError", raw["error"])
            ppl_score.add_unit_perplexities(report, [{"word_count": 4, "byte_count": 8}])
            self.assertNotIn("word_perplexity", report)
            self.assertNotIn("byte_perplexity", report)

    def test_evidence_rejects_misaligned_or_nonfinite_scores(self):
        for problem in ("usage", "token_ids", "logprobs"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as folder:
                response = self.cold_response([1, 2])
                if problem == "usage":
                    response["usage"]["prompt_tokens"] = 3
                elif problem == "token_ids":
                    response["choices"][0]["prompt_token_ids"] = [1, 3]
                else:
                    response["choices"][0]["prompt_logprobs"][1]["2"]["logprob"] = float("nan")
                with patch.object(ppl_score, "json_request", return_value=response):
                    report = ppl_score.evaluate("http://x", [[1, 2]], 1, 10, evidence_dir=Path(folder))
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["scored_tokens"], 0)
                self.assertTrue(Path(report["rows"][0]["raw_path"]).is_file())

    def test_existing_evidence_is_not_overwritten_or_remeasured(self):
        with tempfile.TemporaryDirectory() as folder:
            prior = Path(folder) / "window-0000.json.gz"
            prior.write_bytes(b"preserve")
            with patch.object(ppl_score, "json_request") as request:
                with self.assertRaisesRegex(ValueError, "fresh evidence"):
                    ppl_score.evaluate("http://x", [[1, 2]], 1, 10, evidence_dir=Path(folder))
                request.assert_not_called()
            self.assertEqual(prior.read_bytes(), b"preserve")

    def test_cli_forwards_frozen_request_options_and_writes_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            dataset = path / "input.jsonl"
            dataset.write_text(json.dumps({"tokens": [1, 2], "word_count": 2, "byte_count": 4}) + "\n")
            argv = ["ppl_score", str(dataset), "--output", str(path / "report.json"),
                    "--base-url", "http://127.0.0.1:6688", "--request-form", "echo", "--top-k", "1",
                    "--evidence-dir", str(path / "raw"), "--cache-salt", "shared-ppl", "--seed", "1017"]
            with patch("sys.argv", argv), patch("sys.stdout", new=io.StringIO()), patch.object(
                ppl_score, "json_request", return_value=self.cold_response([1, 2]),
            ):
                self.assertEqual(ppl_score.main(), 0)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual((report["seed"], report["cache_salt"]), (1017, "shared-ppl"))
            self.assertEqual(report["dataset_sha256"], hashlib.sha256(dataset.read_bytes()).hexdigest())
            self.assertEqual((report["seq_len"], report["stride"]), (2048, 2048))


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
