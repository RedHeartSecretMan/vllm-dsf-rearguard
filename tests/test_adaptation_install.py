"""Fail-closed installation contracts for optional adaptation layer."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = ROOT / "adaptation"
spec = importlib.util.spec_from_file_location(
    "adaptation_installer", ADAPTATION / "install.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def test_corrupt_later_destination_prevents_every_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, site = root / "adaptation", root / "site"
            shutil.copytree(ADAPTATION, package)
            rows = json.loads((package / "manifest.json").read_text())["files"]
            for row in rows:
                target = site / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(package / "files" / row["path"], target)
            self.assertEqual(
                installer.install(site, package)["installed_files"], len(rows)
            )
            installer.install(site, package)
            first, last = site / rows[0]["path"], site / rows[-1]["path"]
            before = first.stat().st_mtime_ns
            last.write_text("unexpected local changes")
            with self.assertRaisesRegex(RuntimeError, "Unexpected installed base"):
                installer.install(site, package)
            self.assertEqual(first.stat().st_mtime_ns, before)

    def test_corrupt_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ADAPTATION, root / "adaptation")
            package = root / "adaptation"
            row = json.loads((package / "manifest.json").read_text())["files"][0]
            (package / "files" / row["path"]).write_text("damaged payload")
            with self.assertRaisesRegex(RuntimeError, "Adaptation payload hash mismatch"):
                installer.install(root / "site", package)


if __name__ == "__main__":
    unittest.main()
