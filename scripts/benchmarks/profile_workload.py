"""Freeze shared tokenizer inputs and build reproducible profiling requests."""

from __future__ import annotations

import copy
import gzip
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from scripts.acceptance import core, visual
from scripts.acceptance.catalog import load as acceptance_catalog
from scripts.acceptance.media import load_fixtures, sha256
from scripts.bench_common import (
    MODEL,
    NATURE_CORPUS,
    build_prompts,
    chat_body,
    completion_body,
    json_request,
    load_config,
    repeated_text,
    token_pool,
    workload_digest,
)
from scripts.benchmarks.bench_long_tail import _tool_body, _tool_messages, _vision_body

BASE_URL = "http://127.0.0.1:6688"


def write_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"), ensure_ascii=False)


def read_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def source_identity() -> dict[str, Any]:
    from scripts.bench_common import ROOT
    from scripts.check_runtime import expected_sources, fingerprint

    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_manifest_sha256": fingerprint(expected_sources(ROOT)),
        "catalogs": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "config").glob("*/*.json"))},
    }


def _probe(base: str, body: dict[str, Any], name: str, output: Path) -> int:
    probe = copy.deepcopy(body)
    probe.update(stream=False, max_tokens=1, ignore_eos=True,
                 cache_salt=f"profile-prepare:{name}")
    probe.pop("stream_options", None)
    result = json_request(base, "/v1/chat/completions", probe, 3600)
    write_gzip(output / f"probe-{name}.json.gz", {"request": probe, "response": result})
    usage = result["usage"]
    if (usage.get("prompt_tokens_details") or {}).get("cached_tokens") != 0:
        raise ValueError(f"preparation probe cache state is not cold: {name}")
    return int(usage["prompt_tokens"])


def prepare(base: str, fixtures_dir: Path, output: Path, run_id: str) -> dict[str, Any]:
    if base != BASE_URL:
        raise ValueError("profiling preparation must use port 6688")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    probes = output.parent / "preparation"
    probes.mkdir(exist_ok=False)
    fixtures = load_fixtures(fixtures_dir)
    pool = token_pool(base)
    longest = max(max(c.get("isl", [0])) for c in load_config("regular")["cases"] if c["kind"] == "matrix")
    nature = build_prompts(base, "nature", longest, 1, 0)[0]
    # build_prompts changes token 1 to identify request 0. Restore the corpus
    # token so each cell can apply the original seed/request identity itself.
    first = json_request(base, "/tokenize", {"model": MODEL, "prompt": NATURE_CORPUS})["tokens"]
    nature[1] = first[1]
    bodies: dict[str, dict[str, Any]] = {}
    lengths: dict[str, int] = {}
    for case in load_config("regular")["cases"]:
        if case["kind"] == "chat_prefill":
            for count in case["repeats"]:
                name = f"{case['id']}-{count}"
                body = chat_body(repeated_text(case["word"], count, f"prefill_{count // 1000}k"),
                                 case["osl"], 1017, case["thinking"])
                body["ignore_eos"] = True
                bodies[name] = body
                lengths[name] = _probe(base, body, name, probes)
        elif case["kind"] == "chat_burst":
            for run in case["runs"]:
                name = f"{case['id']}-{run['concurrency']}"
                if name in bodies:
                    continue  # Same C128 payload appears in two historical max-seqs variants.
                body = chat_body(case["prompt"].format(n=run["concurrency"]), case["osl"], 1017, case["thinking"])
                body["ignore_eos"] = True
                bodies[name] = body
                lengths[name] = _probe(base, body, name, probes)
    for case in load_config("long_tail")["cases"]:
        if case["kind"] == "tool_history":
            body = _tool_body(_tool_messages(case["tool_rounds"]), case["reasoning_effort"], case["osl"], case["seed"])
            bodies[case["id"]] = body
            lengths[case["id"]] = _probe(base, body, case["id"], probes)
        elif case["kind"] == "vision_mix":
            for count in case["image_counts"]:
                name = f"{case['id']}-{count}"
                body = _vision_body(base, count, case["isl"], case["osl"], case["seed"])
                observed = _probe(base, body, name, probes)
                # Calibrate only the filler text, never the media, requested
                # output, concurrency or actual measurement. Preserve probes.
                for attempt in range(3):
                    if abs(observed - case["isl"]) <= case["isl"] * case["input_tolerance"]:
                        break
                    text = body["messages"][0]["content"][0]["text"]
                    filler, instruction = text.rsplit(" Read every supplied image", 1)
                    tokens = json_request(base, "/tokenize", {"model": MODEL, "prompt": filler})["tokens"]
                    wanted = len(tokens) + case["isl"] - observed
                    if wanted <= 0:
                        raise ValueError("images exceed the vision-mix input budget")
                    while len(tokens) < wanted:
                        tokens += first
                    adjusted = json_request(base, "/detokenize", {"model": MODEL, "tokens": tokens[:wanted]})["prompt"]
                    body["messages"][0]["content"][0]["text"] = adjusted + " Read every supplied image" + instruction
                    observed = _probe(base, body, f"{name}-calibration-{attempt + 1}", probes)
                if abs(observed - case["isl"]) > case["isl"] * case["input_tolerance"]:
                    raise ValueError("vision-mix input could not satisfy its declared tolerance")
                bodies[name] = body
                lengths[name] = observed
    for target in acceptance_catalog("core")["needle_tokens"]:
        item = core.needle_case(base, target)
        bodies[item["name"]] = item
        probe_body = {"model": MODEL, "messages": item["messages"], "max_tokens": 512,
                      "temperature": 0, "seed": 1017,
                      "chat_template_kwargs": {"thinking": False, "reasoning_effort": "high"}}
        actual = _probe(base, probe_body, item["name"], probes)
        if not .90 * target <= actual <= 1.05 * target:
            raise ValueError(f"prepared needle input {actual} does not satisfy target {target}")
        lengths[item["name"]] = actual
    artifact = {
        "schema_version": 1, "run_id": run_id, "source": source_identity(),
        "token_pool": pool, "nature_tokens": nature, "fixtures": fixtures,
        "fixture_directory": str(fixtures_dir.resolve()),
        "core_media": {str(p): sha256(p) for p in sorted((Path(__file__).resolve().parents[2] / "tests/fixtures/video_frames").glob("*.png"))},
        "bodies": bodies, "observed_input_tokens": lengths,
        "cache_policy": "cold KV with per-cell/repetition/request salts; encoder reused after DP warmup",
        "sampling": {"temperature": 0, "seed": 1017, "ignore_eos_performance": True},
    }
    highres_video_case(artifact)
    highres = fixtures_dir / "long-1200.mp4"
    artifact["highres_video"] = {"path": str(highres.resolve()), "sha256": sha256(highres),
                                 "frames": 32, "visual_tokens_reference": 27904}
    artifact["sha256"] = workload_digest(artifact)
    write_gzip(output, artifact)
    return artifact


def load(path: Path) -> dict[str, Any]:
    result = read_gzip(path)
    expected = result.pop("sha256")
    if workload_digest(result) != expected:
        raise ValueError("frozen input digest mismatch")
    result["sha256"] = expected
    for group in ("images", "videos"):
        for item in result.get("fixtures", {}).get(group, {}).values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("frozen media bytes changed")
    for name, digest in result.get("core_media", {}).items():
        if sha256(Path(name)) != digest:
            raise ValueError("core media bytes changed")
    if result.get("highres_video"):
        media = result["highres_video"]
        if sha256(Path(media["path"])) != media["sha256"]:
            raise ValueError("high-resolution media bytes changed")
    return result


def request_body(artifact: dict[str, Any], cell: dict[str, Any], label: str, index: int) -> dict[str, Any]:
    """Same cell/label/index yields identical bytes for every deployment."""
    source = cell["source_case"]
    kind = cell["kind"]
    cell_id = cell["cell_id"]
    seed = source.get("seed", 1017)
    if kind == "matrix" or cell_id.startswith("capacity_fixed_profile"):
        isl = cell["isl"]
        if cell.get("dataset", "random") == "random":
            # Independent of profile and cell order; salt handles cache isolation.
            rng = random.Random(workload_digest([seed, cell_id, label, index]))
            prompt = [rng.choice(artifact["token_pool"]) for _ in range(isl)]
        else:
            prompt = list(artifact["nature_tokens"][:isl])
            prompt[1] = ((seed + index) % 127000) + 1000
        body = completion_body(prompt, cell["osl"], seed)
    elif kind == "chat_prefill":
        body = copy.deepcopy(artifact["bodies"][f"{source['id']}-{cell['prompt_repetitions']}"])
    elif kind == "chat_burst":
        body = copy.deepcopy(artifact["bodies"][f"{source['id']}-{cell['concurrency']}"])
    elif kind == "tool_history":
        body = copy.deepcopy(artifact["bodies"][source["id"]])
    elif kind == "vision_mix":
        count = source["image_counts"][index % len(source["image_counts"])]
        body = copy.deepcopy(artifact["bodies"][f"{source['id']}-{count}"])
    else:
        raise ValueError(f"unsupported performance kind: {kind}")
    body["cache_salt"] = "profile:" + workload_digest([artifact["run_id"], cell_id, label, index])
    return body


def observed_input(artifact: dict[str, Any], cell: dict[str, Any], index: int = 0) -> int | None:
    source = cell["source_case"]
    if cell["kind"] == "chat_prefill":
        key = f"{source['id']}-{cell['prompt_repetitions']}"
    elif cell["kind"] == "chat_burst":
        key = f"{source['id']}-{cell['concurrency']}"
    elif cell["kind"] == "tool_history":
        key = source["id"]
    elif cell["kind"] == "vision_mix":
        key = f"{source['id']}-{source['image_counts'][index % len(source['image_counts'])]}"
    else:
        return cell.get("isl")
    return artifact["observed_input_tokens"][key]


def highres_video_case(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(artifact["fixture_directory"])
    path = directory / "long-1200.mp4"
    if artifact.get("highres_video") and sha256(path) != artifact["highres_video"]["sha256"]:
        raise ValueError("high-resolution video differs from frozen input")
    if not path.is_file():
        source = artifact["fixtures"]["videos"]["long"]["path"]
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source,
                        "-vf", "scale=1200:1200", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(path)], check=True)
    fixtures = copy.deepcopy(artifact["fixtures"])
    fixtures["videos"]["long"].update(path=str(path), sha256=sha256(path))
    body, expected, _ = visual.build_case(
        {"id": "visual-video-highres-semantic", "kind": "video", "fixtures": ["long"]},
        MODEL, fixtures,
    )
    return body, expected
