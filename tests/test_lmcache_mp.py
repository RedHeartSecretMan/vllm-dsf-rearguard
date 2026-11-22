import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import launch
from scripts.lmcache_mp import DEFAULT_PROFILE, LMCacheMP


class LMCacheMPPlanTests(unittest.TestCase):
    def plan(self, *options):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text('{"text_config":{"dspark_block_size":5}}')
            expected = next(
                row["sha256"] for row in json.loads(launch.MANIFEST.read_text())["files"]
                if row["path"] == "config.json"
            )
            args = launch.parser().parse_args([str(model), *options])
            with patch.object(launch, "sha256", return_value=expected):
                return launch.make_plan(args)

    def test_mp_binds_both_services_to_loopback_with_shared_ipc(self):
        plan = self.plan("--lmcache-mp", "--port", "16688")
        for command in (plan["command"], plan["lmcache_server_command"]):
            for option in ("--network=host", "--ipc=host", "--init"):
                self.assertIn(option, command)
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertNotIn("-p", command)
            self.assertNotIn("--privileged", command)
        sidecar = plan["lmcache_server_command"]
        self.assertEqual(sidecar[sidecar.index("--http-host") + 1], "127.0.0.1")
        self.assertEqual(sidecar[sidecar.index("--l1-size-gb") + 1], "512")
        self.assertEqual(sidecar[sidecar.index("--l1-init-size-gb") + 1], "16")
        self.assertIn("--l1-use-lazy", sidecar)
        self.assertIn("--separate-object-groups", sidecar)
        self.assertEqual(plan["gpu_validation"], "NOT_RUN")
        config = plan["kv_transfer_config"]
        self.assertEqual(config["kv_connector_module_path"], "lmcache.integration.vllm.lmcache_mp_connector")
        self.assertEqual(config["kv_load_failure_policy"], "recompute")
        self.assertEqual(config["kv_connector_extra_config"]["lmcache.mp.port"], 15555)
        self.assertEqual(config["kv_connector_extra_config"]["lmcache.mp.mp_transfer_mode"], "lmcache_driven")

    def test_host_control_uses_same_docker_and_serve_options(self):
        baseline = self.plan("--host-network-ipc", "--port", "16688")
        experiment = self.plan("--lmcache-mp", "--port", "16688")
        self.assertIsNone(baseline["lmcache_server_command"])
        self.assertNotIn("--kv-transfer-config", baseline["command"])
        self.assertIn("VLLM_WORKER_MULTIPROC_METHOD=spawn", baseline["command"])
        command = experiment["command"].copy()
        offset = command.index("--kv-transfer-config")
        del command[offset:offset + 2]
        # Only the test's temporary model source differs.
        def normalize(argv):
            return [
                arg.split(",dst=", 1)[1] if arg.startswith("type=bind,src=") else arg
                for arg in argv
            ]

        self.assertEqual(normalize(command), normalize(baseline["command"]))

    def test_ports_and_l1_allocation_are_checked_before_start(self):
        with self.assertRaisesRegex(ValueError, "different ports"):
            self.plan("--lmcache-mp", "--port", "19180")
        profile = json.loads(DEFAULT_PROFILE.read_text())
        profile["l1_init_size_gb"] = 513
        with tempfile.TemporaryDirectory() as folder:
            invalid = Path(folder) / "invalid.json"
            invalid.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "initial allocation"):
                LMCacheMP.load(invalid)

    def test_failed_health_stops_only_new_sidecar_and_leaves_receipt(self):
        plan = self.plan("--lmcache-mp", "--port", "16688")
        args = SimpleNamespace(image="vllm-dsf-rearguard:dev", model=Path("model"),
                               port=16688, receipt=Path("weights.json"), name="vllm-dsf-rearguard")
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            patchers = {
                "ROOT": Path(folder),
                "require_idle_gpus": Mock(),
                "inspect_runtime": Mock(return_value={"status": "PASS"}),
                "require_matching_report": Mock(return_value="sources"),
                "require_receipt": Mock(return_value={
                    "revision": "fixture-revision",
                    "manifest_sha256": "fixture-manifest",
                }),
                "require_l1_capacity": Mock(),
                "require_free_ports": Mock(),
                "wait_for_health": Mock(side_effect=TimeoutError("dead sidecar")),
                "stop_sidecar": Mock(),
            }
            for name, value in patchers.items():
                stack.enter_context(patch.object(launch, name, value))
            docker = stack.enter_context(patch.object(launch.subprocess, "check_output", side_effect=[
                b'[{"Id":"sha256:fixture"}]', "created-sidecar\n",
            ]))
            with self.assertRaisesRegex(TimeoutError, "dead sidecar"):
                launch.start_plan(args, plan)
            patchers["stop_sidecar"].assert_called_once_with("created-sidecar")
            self.assertEqual(docker.call_count, 2)
            (receipt,) = (Path(folder) / "runtime").glob("*/launch.json")
            record = json.loads(receipt.read_text())
            self.assertEqual(record["lmcache_container_id"], "created-sidecar")
            self.assertNotIn("container_id", record)
            self.assertEqual(record["model_revision"], "fixture-revision")
            self.assertEqual(record["model_manifest_sha256"], "fixture-manifest")
            sidecar_command = record["lmcache_server_command"]
            self.assertEqual(sidecar_command[sidecar_command.index("server") - 1], "sha256:fixture")
