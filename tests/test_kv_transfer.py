"""CPU contract tests for the optional KV transfer launch configuration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch


class KvTransferConfigTests(unittest.TestCase):
    def test_supported_configs_are_normalized_into_the_serve_command(self):
        for connector in sorted(launch.KV_TRANSFER_CONNECTORS):
            with self.subTest(connector=connector):
                config = {
                    "kv_connector": connector,
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"cpu_bytes_to_use": 64 * 1024**3},
                }
                with tempfile.TemporaryDirectory() as folder:
                    model = Path(folder)
                    (model / "config.json").write_text(
                        json.dumps({"text_config": {"dspark_block_size": 5}})
                    )
                    expected = next(
                        row["sha256"]
                        for row in json.loads(launch.MANIFEST.read_text())["files"]
                        if row["path"] == "config.json"
                    )
                    with patch.object(launch, "sha256", return_value=expected):
                        plan = launch.make_plan(
                            launch.parser().parse_args(
                                [
                                    str(model),
                                    "--kv-transfer-config",
                                    json.dumps(config),
                                ]
                            )
                        )
                command = plan["command"]
                self.assertEqual(plan["kv_transfer_config"], config)
                self.assertEqual(
                    command[command.index("--kv-transfer-config") + 1],
                    json.dumps(config, separators=(",", ":")),
                )
                self.assertEqual(len(plan["kv_transfer_config_sha256"]), 64)

    def test_at_file_and_malformed_configs_fail_closed(self):
        config = {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": 64 * 1024**3,
                "lazy_offload": False,
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kv.json"
            path.write_text(json.dumps(config))
            self.assertEqual(launch.load_kv_transfer_config("@" + str(path)), config)
        for value, message in (
            ("[]", "JSON object"),
            ('{"kv_connector":"Simple","kv_role":"kv_both"}', "Unsupported"),
            ('{"kv_connector":"OffloadingConnector","kv_role":"both"}', "kv_role"),
            (
                '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",'
                '"kv_connector_extra_config":{"cpu_bytes_to_use":0}}',
                "positive",
            ),
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, message),
            ):
                launch.load_kv_transfer_config(value)


if __name__ == "__main__":
    unittest.main()
