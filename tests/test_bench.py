"""CPU contracts for the four benchmark configurations and their runners."""

import io
import unittest
from unittest.mock import patch

import launch
from scripts import bench_ab, bench_excluded, bench_long_tail, bench_regular
from scripts.bench_common import (
    CONFIG_DIR,
    build_prompts,
    chat_body,
    launch_args,
    load_config,
    repeated_text,
    select_cases,
    stream_metrics,
    summarize,
    verify_cache,
)

NAMES = ("regular", "long_tail", "ab", "excluded")


def parse_launch(args):
    """Parse launch arguments with the real launcher, so a config cannot drift."""
    return launch.parser().parse_args(["/model", *args])


class SummaryTests(unittest.TestCase):
    ROWS = [
        {
            "status": "PASS",
            "input_tokens": 100,
            "output_tokens": 10,
            "ttft_s": 2.0,
            "tpot_s": 0.1,
            "decode_span_s": 0.9,
            "itl_s": [0.1, 0.2],
        },
        {
            "status": "PASS",
            "input_tokens": 100,
            "output_tokens": 10,
            "ttft_s": 4.0,
            "tpot_s": 0.2,
            "decode_span_s": 1.8,
            "itl_s": [0.2, 0.3],
        },
    ]

    def test_three_throughputs_share_the_wall_clock(self):
        report = summarize(self.ROWS, 10.0, case="x")
        self.assertEqual(report["observed_input_tokens"], 200)
        self.assertEqual(report["observed_output_tokens"], 20)
        self.assertEqual(report["input_tps"], 20.0)
        self.assertEqual(report["output_tps"], 2.0)
        self.assertEqual(report["total_tps"], 22.0)
        self.assertEqual(report["ttft_p50_s"], 2.0)
        self.assertEqual(report["tpot_p50_s"], 0.1)

    def test_failed_request_is_excluded_but_fails_the_run(self):
        rows = [*self.ROWS, {"status": "FAIL", "error": "boom"}]
        report = summarize(rows, 10.0)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["success"], 2)
        self.assertEqual(report["observed_input_tokens"], 200)
        self.assertEqual(report["errors"], ["boom"])

    def test_cache_state_is_verified_not_assumed(self):
        cold = [{"status": "PASS", "cached_tokens": 0}]
        warm = [{"status": "PASS", "cached_tokens": 8}]
        unknown = [{"status": "PASS", "cached_tokens": None}]
        self.assertTrue(verify_cache(cold, "cold"))
        self.assertFalse(verify_cache(warm, "cold"))
        self.assertTrue(verify_cache(warm, "warm"))
        self.assertFalse(verify_cache(unknown, "cold"))
        self.assertFalse(verify_cache([], "cold"))

    def test_chat_stream_reads_deltas_and_enforces_output_count(self):
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"length"}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":5,"prompt_tokens":7}}\n\n'
            b"data: [DONE]\n\n"
        )
        row = stream_metrics(
            stream, 0, True, expected_output=5, clock=iter(range(1, 30)).__next__
        )
        self.assertEqual((row["input_tokens"], row["output_tokens"]), (7, 5))
        stream.seek(0)
        with self.assertRaisesRegex(RuntimeError, "expected 9"):
            stream_metrics(stream, 0, True, expected_output=9, clock=lambda: 1)

    def test_prompt_builders_are_deterministic_and_exact(self):
        with patch(
            "scripts.bench_common.token_pool", return_value=list(range(10, 400))
        ):
            first = build_prompts("x", "random", 64, 3, 7)
            again = build_prompts("x", "random", 64, 3, 7)
        self.assertEqual(first, again)
        self.assertEqual({len(p) for p in first}, {64})
        self.assertEqual(len({tuple(p) for p in first}), 3)
        text = repeated_text("hello", 3, "prefill_1k")
        self.assertEqual(text, "unique_prefill_1k hello hello hello \nReturn one token.")
        body = chat_body("hi", 128)
        self.assertEqual(body["chat_template_kwargs"], {"thinking": False})
        self.assertEqual(body["max_tokens"], 128)

    def test_cyclic_probe_prompt_matches_the_recorded_formula(self):
        prompt = bench_excluded.cyclic_prompt(260016, 1000, 30000)
        self.assertEqual(len(prompt), 260016)
        self.assertEqual(prompt[0], 1000)
        self.assertEqual(prompt[30000], 1000)
        self.assertEqual(prompt[29999], 1000 + 29999)


class ConfigContractTests(unittest.TestCase):
    def test_every_config_loads_and_names_the_served_model(self):
        for name in NAMES:
            with self.subTest(name=name):
                config = load_config(name)
                self.assertEqual(config["model"], launch.MODEL_NAME)
                self.assertTrue((CONFIG_DIR / f"{name}.json").is_file())

    def test_every_launch_is_accepted_by_the_launcher(self):
        for name in NAMES:
            config = load_config(name)
            for key, args in config["launch"].items():
                with self.subTest(config=name, launch=key):
                    parse_launch(args)
        regular = load_config("regular")
        for case in regular["cases"]:
            with self.subTest(case=case["id"]):
                parse_launch(launch_args(regular, case))
        ab = load_config("ab")
        for case in ab["cases"]:
            for variant in case["variants"]:
                with self.subTest(case=case["id"], variant=variant["id"]):
                    parse_launch(bench_ab.variant_plan(ab, variant)["launch_args"])
        excluded = load_config("excluded")
        for case in excluded["cases"]:
            with self.subTest(case=case["id"]):
                parse_launch(bench_excluded.case_plan(excluded, case)["launch_args"])

    def test_two_launch_configurations_are_defined_by_context(self):
        launches = load_config("regular")["launch"]
        self.assertEqual(set(launches), {"262k", "1m"})
        small, large = parse_launch(launches["262k"]), parse_launch(launches["1m"])
        self.assertEqual((small.context, large.context), (262144, 1048576))
        self.assertEqual(small.experimental, ["stacked-kv"])
        self.assertEqual(large.experimental, ["stacked-kv", "bounded-prefill"])

    def test_long_request_capacity_adds_prefill_throttle_to_the_1m_launch(self):
        regular = load_config("regular")
        (case,) = select_cases(regular, "long_request_capacity")
        args = parse_launch(launch_args(regular, case))
        self.assertEqual(args.context, 1048576)
        self.assertEqual(args.experimental, ["stacked-kv", "bounded-prefill", "prefill-throttle"])

    def test_every_case_references_a_declared_launch(self):
        for name in ("regular", "long_tail"):
            config = load_config(name)
            for case in config["cases"]:
                with self.subTest(config=name, case=case["id"]):
                    self.assertIn(case["launch"], config["launch"])
        ab = load_config("ab")
        for case in ab["cases"]:
            for variant in case["variants"]:
                with self.subTest(case=case["id"], variant=variant["id"]):
                    self.assertIn(variant["launch"], ab["launch"])

    def test_case_ids_are_unique_within_a_config(self):
        for name in ("regular", "long_tail", "ab"):
            ids = [case["id"] for case in load_config(name)["cases"]]
            with self.subTest(config=name):
                self.assertEqual(len(ids), len(set(ids)))

    def test_ab_cases_each_vary_one_thing(self):
        ab = load_config("ab")
        for case in ab["cases"]:
            if case["control"] == "one_option_per_variant":
                for variant in case["variants"][1:]:
                    changes = [
                        key
                        for key in ("env", "drop_args", "extra_launch")
                        if variant.get(key)
                    ]
                    with self.subTest(variant=variant["id"]):
                        self.assertEqual(len(changes), 1)


class RecordedParameterTests(unittest.TestCase):
    """The configs must keep the parameters the published numbers were measured with."""

    def test_matrix_grid(self):
        cases = {c["id"]: c for c in load_config("regular")["cases"]}
        single, grid = cases["matrix_single"], cases["matrix_concurrent"]
        self.assertEqual(single["isl"], [4096, 16384, 65536, 262144, 1047000])
        self.assertEqual(single["concurrency"], [1])
        self.assertEqual(grid["isl"], [256, 1024, 4096, 16384])
        self.assertEqual(grid["concurrency"], [4, 16, 64, 256])
        for case in (single, grid):
            self.assertEqual(case["datasets"], ["random", "nature"])
            self.assertEqual((case["osl"], case["seed"]), (1000, 20261021))
        # input plus output must fit the 1M context
        self.assertLessEqual(max(single["isl"]) + single["osl"], 1048576)

    def test_short_requests_use_osl_128_on_the_262k_launch(self):
        cases = {c["id"]: c for c in load_config("regular")["cases"]}
        burst = cases["short_concurrency"]
        self.assertEqual(burst["launch"], "262k")
        self.assertEqual(burst["osl"], 128)
        self.assertEqual(
            [(r["max_seqs"], r["concurrency"]) for r in burst["runs"]],
            [(128, 64), (128, 128), (256, 128), (256, 256)],
        )
        prefill = cases["single_prefill"]
        self.assertEqual(prefill["repeats"], [1000, 32000, 128000, 250000, 260000])
        self.assertEqual((prefill["osl"], prefill["word"]), (1, "hello"))

    def test_long_tail_cases_keep_their_fixed_shapes(self):
        cases = {c["id"]: c for c in load_config("long_tail")["cases"]}
        self.assertEqual(
            set(cases),
            {
                "tool_history_16384_c16",
                "vision_mix_4096_c4",
                "long_soak_1024_c64",
                "mixed_prefill_decode_same_service",
            },
        )
        for case_id in (
            "tool_history_16384_c16",
            "vision_mix_4096_c4",
            "long_soak_1024_c64",
        ):
            self.assertEqual(cases[case_id]["osl"], 1000)
            self.assertEqual(cases[case_id]["seed"], 1017)
        mixed = cases["mixed_prefill_decode_same_service"]
        self.assertEqual(
            (mixed["decode_isl"], mixed["decode_osl"], mixed["decode_concurrency"]),
            (1024, 1000, 64),
        )
        self.assertEqual((mixed["prefill_isl"], mixed["prefill_osl"]), (262144, 1000))
        self.assertEqual((mixed["prefill_interval_s"], mixed["duration_s"]), (20, 300))

    def test_bounded_workspace_and_memory_seeds(self):
        cases = {c["id"]: c for c in load_config("ab")["cases"]}
        burst = cases["bounded_workspace"]["workload"]["burst"]
        self.assertEqual(
            (burst["isl"], burst["osl"], burst["seed"], burst["concurrency"]),
            (1024, 128, 8080, [128, 256]),
        )
        seeds = {v["id"]: v["seed"] for v in cases["memory_utilization"]["variants"]}
        self.assertEqual(seeds, {"0.90": 8080, "0.95": 9090})

    def test_comm_scheduling_baseline_and_variants(self):
        case = {c["id"]: c for c in load_config("ab")["cases"]}["comm_scheduling"]
        burst = case["workload"]["burst"]
        self.assertEqual((burst["prompt_tokens"], burst["osl"]), (10, 128))
        self.assertEqual(burst["concurrency"], [128, 256])
        self.assertEqual(
            [v["id"] for v in case["variants"]],
            ["baseline", "no_eplb", "no_custom_ag_rs", "omp1", "decode_metadata_ragged_reuse"],
        )

    def test_excluded_probe_runs_do_not_share_prefix_cache(self):
        config = load_config("excluded")
        probe = config["prefill_probe"]
        self.assertEqual((probe["isl"], probe["osl"], probe["stream"]), (260016, 1, False))
        probes = [c for c in config["cases"] if c["kind"] == "cyclic_prefill"]
        offsets = [c["offset"] for c in probes]
        seeds = [c["seed"] for c in probes]
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertEqual(len(seeds), len(set(seeds)))
        # all offsets stay clear of each other's token ranges
        ordered = sorted(offsets)
        self.assertTrue(all(b - a >= 1000 for a, b in zip(ordered, ordered[1:], strict=False)))
        self.assertEqual(
            {c["id"] for c in probes},
            {"baseline_chunk16384", "chunk32768", "logits_row_chunk_512", "dummy_q_empty", "tp8"},
        )

    def test_compact_mqa_marks_its_unrecovered_workload(self):
        compact = {c["id"]: c for c in load_config("excluded")["cases"]}["compact_candidate_mqa"]
        self.assertEqual(compact["env"], {"VLLM_SM80_CANDIDATE_MQA_COMPACT": "1"})
        recorded, unrecovered = compact["workloads"]
        self.assertTrue(recorded["source"].startswith("recorded"))
        self.assertTrue(unrecovered["source"].startswith("unverified"))
        self.assertNotIn("osl", unrecovered)


class CapacityAndKvTransferTests(unittest.TestCase):
    def test_long_request_capacity_uses_one_seed_per_concurrency(self):
        cases = {c["id"]: c for c in load_config("regular")["cases"]}
        case = cases["long_request_capacity"]
        self.assertEqual(case["concurrency"], [4, 16])
        self.assertEqual((case["isl"], case["osl"]), ([1047000], 1000))
        # the recorded run shared one seed; the switch gives each cell its own
        recorded = {c: bench_regular.cell_seed(case, c) for c in case["concurrency"]}
        self.assertEqual(set(recorded.values()), {case["seed"]})
        independent = {**case, "seed_per_concurrency": True}
        seeds = {c: bench_regular.cell_seed(independent, c) for c in case["concurrency"]}
        self.assertEqual(len(set(seeds.values())), 2)
        # a shared seed would make C16 reuse the first four prompts of C4
        shared = bench_regular.cell_seed({"seed": 7}, 4), bench_regular.cell_seed({"seed": 7}, 16)
        self.assertEqual(shared, (7, 7))
        with patch("scripts.bench_common.token_pool", return_value=list(range(10, 400))):
            c4 = build_prompts("x", "random", 32, 4, seeds[4])
            c16 = build_prompts("x", "random", 32, 16, seeds[16])
            same = build_prompts("x", "random", 32, 16, seeds[4])
        self.assertNotEqual(c4[0], c16[0])
        self.assertEqual(c4[0], same[0])

    def test_bounded_workspace_measures_both_workloads_on_both_sides(self):
        case = {c["id"]: c for c in load_config("ab")["cases"]}["bounded_workspace"]
        for variant in case["variants"]:
            self.assertEqual(variant["workloads"], ["burst", "prefill"])
        self.assertEqual({v["id"] for v in case["variants"]}, {"off", "on"})

    def test_kv_transfer_connectors_are_excluded_cases_the_launcher_accepts(self):
        excluded = load_config("excluded")
        cases = {c["id"]: c for c in excluded["cases"]}
        self.assertNotIn("kv_transfer", {c["id"] for c in load_config("ab")["cases"]})
        for name in ("simple_cpu_offload", "offloading"):
            with self.subTest(case=name):
                self.assertEqual(cases[name]["kind"], "request_smoke")
                args = parse_launch(launch_args(excluded, cases[name]))
                config = launch.load_kv_transfer_config(args.kv_transfer_config)
                self.assertIn(config["kv_connector"], launch.KV_TRANSFER_CONNECTORS)

    def test_request_smoke_sends_the_same_prompt_twice_and_records_a_failure(self):
        config = load_config("excluded")
        (case,) = select_cases(config, "simple_cpu_offload")
        sent = []

        def respond(base, path, body, timeout):
            sent.append(body)
            if len(sent) == 2:
                raise OSError("HTTP Error 500")
            return {
                "choices": [{"text": "ok", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1024, "completion_tokens": 128},
            }

        with (
            patch("scripts.bench_excluded.token_pool", return_value=list(range(10, 400))),
            patch("scripts.bench_excluded.json_request", side_effect=respond),
        ):
            report = bench_excluded.run_request_smoke("http://x", config, case, 1)
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(len(sent[0]["prompt"]), 1024)
        self.assertEqual((sent[0]["max_tokens"], sent[0]["stream"]), (128, False))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([r["status"] for r in report["rows"]], ["PASS", "FAIL"])

    def test_only_startup_rejected_and_digest_cases_are_not_sent_by_the_script(self):
        config = load_config("excluded")
        for case in config["cases"]:
            if case["kind"] not in bench_excluded.RUNNERS:
                with self.subTest(case=case["id"]):
                    self.assertIn("run", case)


class RunnerTests(unittest.TestCase):
    def test_unknown_case_is_rejected_with_the_valid_ids(self):
        with self.assertRaisesRegex(SystemExit, "choose from"):
            select_cases(load_config("regular"), "nope")

    def test_runners_list_cases_without_a_service(self):
        for module, argv in (
            (bench_regular, []),
            (bench_long_tail, []),
            (bench_excluded, []),
            (bench_ab, ["--plan"]),
        ):
            with (
                self.subTest(module=module.__name__),
                patch("sys.argv", [module.__name__, *argv]),
                patch("sys.stdout", new=io.StringIO()) as out,
            ):
                self.assertEqual(module.main(), 0)
                self.assertTrue(out.getvalue().strip())

    def test_ab_plan_translates_options_launch_does_not_have(self):
        ab = load_config("ab")
        case = {c["id"]: c for c in ab["cases"]}["comm_scheduling"]
        plans = {v["id"]: bench_ab.variant_plan(ab, v) for v in case["variants"]}
        self.assertEqual(plans["no_eplb"]["drop_args"], ["--enable-eplb", "--eplb-config"])
        self.assertEqual(plans["omp1"]["env"], {"OMP_NUM_THREADS": "1"})
        self.assertEqual(plans["no_custom_ag_rs"]["env"], {"VLLM_SM80_EP8_CUSTOM_AG_RS": "0"})
        self.assertEqual(plans["baseline"]["env"], {})

    def test_throttle_scenario_is_delegated_to_the_long_tail_runner(self):
        ab = load_config("ab")
        case = {c["id"]: c for c in ab["cases"]}["prefill_throttle"]
        target = case["workload"]["scenario"]["long_tail_case"]
        self.assertIn(target, {c["id"] for c in load_config("long_tail")["cases"]})
        with self.assertRaisesRegex(SystemExit, "bench_long_tail"):
            bench_ab.measure("http://x", case, case["variants"][0], 1)


if __name__ == "__main__":
    unittest.main()
