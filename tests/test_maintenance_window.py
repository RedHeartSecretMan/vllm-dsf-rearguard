import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmarks import maintenance_window as window


class MaintenanceWindowTests(unittest.TestCase):
    def test_control_cache_requires_exact_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = window.control_identity(
                experiment="kv-only-short",
                baseline_image="sha256:base",
                launch={"chunk": 16384, "context": 1048576},
            )
            result = {"status": "PASS", "rows": [1, 2]}
            path = window.save_control_cache(root, "kv-only-short", identity, result)
            self.assertTrue(path.is_file())
            self.assertEqual(
                window.load_control_cache(root, "kv-only-short", identity), result
            )
            changed = {**identity, "launch_sha256": "changed"}
            self.assertIsNone(window.load_control_cache(root, "kv-only-short", changed))

    def test_open_window_stops_running_production_and_records_identity(self):
        inspect = iter(
            [
                "prod-id sha256:prod running\n",
                "prod-id sha256:prod exited\n",
            ]
        )
        def fake_run(args, **kwargs):
            stdout = next(inspect) if args[1] == "inspect" else ""
            return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

        with tempfile.TemporaryDirectory() as directory, patch.object(
            window.subprocess, "run", side_effect=fake_run
        ) as run:
            state = window.open_window(
                production_container="dsf-main",
                state_path=Path(directory) / "window.json",
            )
            self.assertEqual(state["status"], "open")
            self.assertEqual(state["production"]["id"], "prod-id")
            self.assertEqual(run.call_args_list[0].args[0][:3], ["docker", "inspect", "-f"])
            self.assertEqual(run.call_args_list[1].args[0], ["docker", "stop", "--time", "90", "dsf-main"])

    def test_close_window_starts_recorded_container_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "window.json"
            path.write_text(
                json.dumps(
                    {
                        "window_version": 1,
                        "status": "open",
                        "health_url": "http://127.0.0.1:6789",
                        "production": {
                            "name": "dsf-main",
                            "id": "prod-id",
                            "image": "sha256:prod",
                        },
                    }
                )
            )
            current = {
                "name": "dsf-main",
                "id": "prod-id",
                "image": "sha256:prod",
                "status": "exited",
            }
            with patch.object(window, "_inspect_container", return_value=current), patch.object(
                window, "_wait_http_health"
            ) as health, patch.object(window.subprocess, "run") as run:
                result = window.close_window(state_path=path)
            self.assertEqual(result["status"], "closed")
            run.assert_called_once_with(["docker", "start", "prod-id"], check=True)
            health.assert_called_once_with("http://127.0.0.1:6789", 900)

    def test_close_window_rejects_changed_production_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "window.json"
            path.write_text(
                json.dumps(
                    {
                        "window_version": 1,
                        "status": "open",
                        "health_url": "http://127.0.0.1:6789",
                        "production": {
                            "name": "dsf-main",
                            "id": "prod-id",
                            "image": "sha256:prod",
                        },
                    }
                )
            )
            with patch.object(
                window,
                "_inspect_container",
                return_value={
                    "name": "dsf-main",
                    "id": "different-id",
                    "image": "sha256:prod",
                    "status": "exited",
                },
            ):
                with self.assertRaisesRegex(window.MaintenanceWindowError, "identity changed"):
                    window.close_window(state_path=path)


if __name__ == "__main__":
    unittest.main()
