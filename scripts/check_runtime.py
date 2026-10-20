"""Check installed source bytes after the unified adaptation overlay."""

import hashlib
import json
import sysconfig
from pathlib import Path


def expected_sources(root):
    base = root / "adaptation"
    manifest = json.loads((base / "manifest.json").read_text())
    expected = {}
    for row in manifest["files"]:
        path = base / "files" / row["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["patched_sha256"]:
            raise ValueError("Adaptation payload differs from manifest: " + row["path"])
        expected[row["path"]] = row["patched_sha256"]
    return expected


def fingerprint(expected):
    return hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_matching_report(report, root):
    wanted = fingerprint(expected_sources(root))
    if report.get("status") != "PASS" or report.get("source_manifest_sha256") != wanted:
        raise ValueError("Image sources differ from this checkout; rebuild before A/B")
    return wanted


def main():
    root = Path("/opt/flash1017")
    site = Path(sysconfig.get_paths()["purelib"])
    expected = expected_sources(root)
    for name, digest in expected.items():
        path = site / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Installed source differs: " + name)
    print(
        json.dumps(
            {
                "installed_source_files": len(expected),
                "status": "PASS",
                "source_manifest_sha256": fingerprint(expected),
                "gpu_validation": "NOT_RUN",
            }
        )
    )


if __name__ == "__main__":
    main()
