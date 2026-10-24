"""Deterministic, locally generated numbered media for endpoint acceptance."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

COLORS = {1: ("red", (224, 32, 32)), 2: ("green", (32, 176, 48)), 3: ("blue", (32, 80, 224))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number_png(number: int, path: Path) -> None:
    """Draw a large seven-segment digit without font or imaging dependencies."""
    size = 384
    rgb = COLORS[number][1]
    pixels = bytearray(bytes(rgb) * (size * size))

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(y0, y1):
            start = (y * size + x0) * 3
            pixels[start : start + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

    rect(48, 24, 336, 312, (255, 255, 255))
    segments = {
        "a": (136, 48, 248, 72), "b": (236, 60, 260, 168),
        "c": (236, 156, 260, 264), "d": (136, 252, 248, 276),
        "e": (124, 156, 148, 264), "f": (124, 60, 148, 168),
        "g": (136, 156, 248, 180),
    }
    for segment in {1: "bc", 2: "abged", 3: "abgcd"}[number]:
        rect(*segments[segment], (0, 0, 0))

    def chunk(name: bytes, value: bytes) -> bytes:
        return struct.pack(">I", len(value)) + name + value + struct.pack(">I", zlib.crc32(name + value))

    rows = b"".join(b"\x00" + pixels[y * size * 3 : (y + 1) * size * 3] for y in range(size))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def prepare_fixtures(directory: Path) -> dict[str, Any]:
    """Generate three images and three H.264 videos; only ffmpeg is required."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "images": {}, "videos": {}}
    for number, (color, _) in COLORS.items():
        path = directory / f"{number:02}.png"
        _number_png(number, path)
        manifest["images"][str(number)] = {
            "path": path.name, "sha256": sha256(path), "number": number, "color": color,
        }
    for name, duration, numbers in (
        ("short", 6, [1, 2, 3]), ("long", 96, [1, 2, 3]), ("second", 6, [3, 1, 2]),
    ):
        source = directory / f"{name}.ffconcat"
        source.write_text(
            "ffconcat version 1.0\n"
            + "".join(f"file '{number:02}.png'\nduration {duration / 3}\n" for number in numbers)
            + f"file '{numbers[-1]:02}.png'\n"
        )
        path = directory / f"{name}.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(source), "-vf", "fps=10,tpad=stop_mode=clone:stop_duration=1", "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        subprocess.run(command, check=True)
        manifest["videos"][name] = {
            "path": path.name, "sha256": sha256(path), "duration_seconds": duration,
            "source_fps": 10, "numbers": numbers,
            "colors": [COLORS[number][0] for number in numbers],
            "stage_start_seconds": [index * duration / 3 for index in range(3)],
            "generation_command": command,
        }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_fixtures(directory)


def load_fixtures(directory: Path) -> dict[str, Any]:
    """Read a fixture manifest and verify the exact media bytes before use."""
    manifest = json.loads((directory / "manifest.json").read_text())
    for group in ("images", "videos"):
        for name, item in manifest[group].items():
            path = Path(item["path"])
            if not path.is_absolute():
                path = directory / path
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise ValueError(f"fixture content differs from manifest: {group}.{name}")
            item["path"] = str(path.resolve())
    return manifest
