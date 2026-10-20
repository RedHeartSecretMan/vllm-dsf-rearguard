"""Read-only model audit. Only a complete SHA256 pass permits a model launch."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/model-files.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def model_files(manifest):
    # ModelScope's downloader may omit Git-only metadata.
    return [r for r in manifest["files"] if r["path"] != ".gitattributes"]


def audit(model, full=False, manifest_path=MANIFEST):
    model = model.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for row in model_files(manifest):
        path = model / row["path"]
        if not path.resolve().is_relative_to(model):
            raise ValueError("Model path escapes snapshot")
        result = {"path": row["path"], "expected_bytes": row["size"]}
        if not path.is_file():
            result["status"] = "MISSING"
        else:
            before = signature(path)
            result["signature"] = before
            if before[2] != row["size"]:
                result["status"] = "SIZE_MISMATCH"
            elif full or not path.name.endswith(".safetensors"):
                result["sha256"] = sha256(path)
                result["status"] = (
                    "PASS"
                    if result["sha256"] == row["sha256"] and signature(path) == before
                    else "HASH_OR_CHANGE_MISMATCH"
                )
            else:
                result["status"] = "SIZE_ONLY"
        rows.append(result)
    passed = all(r["status"] == "PASS" for r in rows)
    return {
        "schema": 1,
        "model": str(model),
        "revision": manifest["modelscope_revision"],
        "manifest_sha256": sha256(manifest_path),
        "full_hash": full,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed and full else "NOT_VERIFIED",
        "complete_shards": sum(
            r["path"].endswith(".safetensors") and r["status"] in ("PASS", "SIZE_ONLY")
            for r in rows
        ),
        "total_shards": sum(r["path"].endswith(".safetensors") for r in rows),
        "files": rows,
    }


def require_receipt(model, receipt_path, manifest_path=MANIFEST):
    model = model.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        receipt.get("status") != "PASS"
        or not receipt.get("full_hash")
        or receipt.get("model") != str(model)
        or receipt.get("manifest_sha256") != sha256(manifest_path)
        or receipt.get("revision") != manifest["modelscope_revision"]
    ):
        raise ValueError("A complete SHA256 model receipt is required")
    rows = {r["path"]: r for r in receipt["files"]}
    for expected in model_files(manifest):
        row = rows.get(expected["path"], {})
        path = model / expected["path"]
        if (
            row.get("status") != "PASS"
            or row.get("sha256") != expected["sha256"]
            or not path.is_file()
            or not path.resolve().is_relative_to(model)
            or row.get("signature") != signature(path)
        ):
            raise ValueError(
                "Model changed or missing since verification: " + expected["path"]
            )
        # Some filesystems preserve size/mtime/ctime for two writes in one tick.
        # A stat-only receipt therefore cannot establish checkpoint identity.
        if sha256(path) != expected["sha256"] or row["signature"] != signature(path):
            raise ValueError("Model changed since verification: " + expected["path"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument(
        "--full", action="store_true", help="Hash all ~510 GB after download completes"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    receipt = audit(args.model, args.full)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
