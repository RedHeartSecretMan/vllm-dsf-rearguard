"""Install the unified SM80 adaptation layer into the pinned vLLM runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sysconfig


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(site: Path, root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    pending = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid adaptation path: " + row["path"])
        source, target = root / "files" / relative, site / relative
        if digest(source) != row["patched_sha256"]:
            raise RuntimeError("Adaptation payload hash mismatch: " + row["path"])
        actual = digest(target) if target.is_file() else None
        if actual not in {row.get("base_sha256"), row["patched_sha256"], None}:
            raise RuntimeError("Unexpected installed base: " + row["path"])
        compile(source.read_bytes(), str(target), "exec")
        pending.append((target, source.read_bytes()))

    for target, content in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".adaptation.tmp")
        try:
            temporary.write_bytes(content)
            temporary.chmod(target.stat().st_mode & 0o777 if target.exists() else 0o644)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return {"id": manifest["id"], "installed_files": len(pending),
            "components": manifest["components"], "gpu_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    print(json.dumps(install(parser.parse_args().site)))
