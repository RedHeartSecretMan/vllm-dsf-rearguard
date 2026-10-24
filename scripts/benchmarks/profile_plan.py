"""Expand source catalogs into an auditable, offline fixed-profile test plan.

The profile overrides historical context limits, never truncates inputs, and
keeps independent A/B experiments visible without executing their variants on
the fixed service. Unknown chat lengths must be resolved during preparation.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CATALOGS = {
    "benchmarks": ("regular", "long_tail", "ab", "excluded"),
    "acceptance": ("core", "visual", "responses", "quality"),
}


def _launch(config: dict, case: dict) -> list[str]:
    return list(config.get("launch", {}).get(case.get("launch"), [])) + list(case.get("extra_launch", []))


def _cell(catalog: str, config: dict, case: dict, suffix: str = "", **values) -> dict:
    case_id = case["id"]
    row = {
        "catalog": catalog, "group": catalog.split("/")[1],
        "case": case_id, "case_id": case_id,
        "cell_id": case_id + ("/" + suffix if suffix else ""),
        "kind": case.get("kind", "ab"), "runner": case.get("runner", config["runner"]),
        "source_case": copy.deepcopy(case), "source_launch": _launch(config, case),
        "dataset": case.get("dataset"), "isl": case.get("isl"), "osl": case.get("osl"),
        "concurrency": case.get("concurrency", 1), "max_seqs": None,
        "seed": case.get("seed"), "requires": list(case.get("requires", [])),
        "once": False, "once_only": False, "derived": False,
        "repeats": 3 if catalog.startswith("benchmarks/") else 1,
        "actual_expected_input": None, "needs_tokenization": False,
        "status": "UNPLANNED", "reason": None,
    }
    row.update(values)
    row["actual_expected_input"] = None if row["needs_tokenization"] else row["isl"]
    if row["needs_tokenization"] and "tokenize_before_run" not in row["requires"]:
        row["requires"].append("tokenize_before_run")
    return row


def _matrix(catalog: str, config: dict, case: dict) -> list[dict]:
    return [
        _cell(catalog, config, case, f"{dataset}/i{isl}-o{case['osl']}-c{concurrency}",
              dataset=dataset, isl=isl, concurrency=concurrency,
              repeats=1 if case["id"] == "long_request_capacity" else 3)
        for dataset, isl, concurrency in product(case["datasets"], case["isl"], case["concurrency"])
    ]


def _regular(catalog: str, config: dict, case: dict) -> list[dict]:
    if case["kind"] == "matrix":
        return _matrix(catalog, config, case)
    if case["kind"] == "chat_prefill":
        return [_cell(catalog, config, case, f"repeat-word-{length}", isl=None,
                      prompt_repetitions=length, needs_tokenization=True)
                for length in case["repeats"]]
    if case["kind"] == "chat_burst":
        return [_cell(catalog, config, case, f"maxseq{run['max_seqs']}/c{run['concurrency']}",
                      isl=None, concurrency=run["concurrency"], max_seqs=run["max_seqs"],
                      needs_tokenization=True)
                for run in case["runs"]]
    raise ValueError(f"unhandled regular case kind: {case['kind']}")


def _long_tail(catalog: str, config: dict, case: dict) -> list[dict]:
    kind = case["kind"]
    if kind in ("tool_history", "vision_mix"):
        return [_cell(catalog, config, case, needs_tokenization=True)]
    if kind == "soak":
        return [_cell(catalog, config, case, repeats=1, once=True, once_only=True,
                      duration_s=case["duration_s"])]
    if kind == "mixed":
        flows = {
            "decode": {"isl": case["decode_isl"], "osl": case["decode_osl"], "concurrency": case["decode_concurrency"]},
            "prefill": {"isl": case["prefill_isl"], "osl": case["prefill_osl"], "concurrency": case["prefill_max_inflight"]},
        }
        return [_cell(catalog, config, case, repeats=1, flows=flows,
                      isl=max(v["isl"] for v in flows.values()),
                      osl=max(v["osl"] for v in flows.values()),
                      admission_tokens=max(v["isl"] + v["osl"] for v in flows.values()),
                      concurrency=None, duration_s=case["duration_s"])]
    raise ValueError(f"unhandled long-tail case kind: {kind}")


def _ab(catalog: str, config: dict, case: dict) -> list[dict]:
    rows = []
    for variant in case["variants"]:
        common = {"variant": variant["id"], "source_variant": copy.deepcopy(variant),
                  "source_launch": _launch(config, variant),
                  "requires": ["independent_ab_service"]}
        workloads = case["workload"]
        if "cells" in workloads:
            for spec in workloads["cells"]:
                rows.append(_cell(catalog, config, case, variant["id"] + "/" + spec["id"],
                                  **common, **{k: v for k, v in spec.items() if k != "id"},
                                  workload_cell_id=spec["id"], needs_tokenization=spec["kind"] != "text",
                                  repeats=workloads["repeats"]))
            continue
        for name in variant.get("workloads", workloads):
            spec = workloads[name]
            options = {**common, "workload": name, "source_workload": copy.deepcopy(spec),
                       "kind": spec.get("kind", "scenario_reference"), "osl": spec.get("osl"),
                       "seed": variant.get("seed", spec.get("seed"))}
            prefix = variant["id"] + "/" + name
            if "long_tail_case" in spec:
                rows.append(_cell(catalog, config, case, prefix, **options,
                                  referenced_case=spec["long_tail_case"], repeats=1))
            elif spec["kind"] == "chat_prefill":
                rows.append(_cell(catalog, config, case, prefix, **options,
                                  prompt_repetitions=spec["repeats"], needs_tokenization=True))
            elif spec["kind"] == "random_prefill":
                for isl, seed in zip(spec["isl"], spec["seed"], strict=True):
                    rows.append(_cell(catalog, config, case, f"{prefix}/i{isl}",
                                      **{**options, "seed": seed}, isl=isl))
            elif spec["kind"] in ("random_burst", "token_burst", "chat_burst"):
                for concurrency in spec["concurrency"]:
                    rows.append(_cell(catalog, config, case, f"{prefix}/c{concurrency}", **options,
                                      isl=spec.get("isl", spec.get("prompt_tokens")),
                                      concurrency=concurrency, needs_tokenization=spec["kind"] == "chat_burst"))
            else:
                raise ValueError(f"unhandled A/B workload: {spec}")
    return rows


def _excluded(catalog: str, config: dict, case: dict) -> list[dict]:
    common = {"requires": ["independent_excluded_experiment"], "repeats": 1,
              "variant": case["id"]}
    if case["kind"] == "matrix_digest":
        return [_cell(catalog, config, case, f"workload-{index + 1}", **common,
                      source_workload=copy.deepcopy(spec), isl=spec.get("isl"), osl=spec.get("osl"),
                      concurrency=spec["concurrency"], dataset=spec.get("dataset"), seed=spec.get("seed"))
                for index, spec in enumerate(case["workloads"])]
    probe = config.get("prefill_probe" if case["kind"] == "cyclic_prefill" else "smoke_probe", {})
    return [_cell(catalog, config, case, **common,
                  isl=probe.get("isl") if case["kind"] != "startup_rejected" else None,
                  osl=probe.get("osl") if case["kind"] != "startup_rejected" else None,
                  probe=copy.deepcopy(probe), source_change=case.get("change", case.get("image_variant")))]


def _acceptance(catalog: str, config: dict, case: dict) -> list[dict]:
    group = catalog.split("/")[1]
    if group == "quality":
        if case["kind"] == "ppl":
            row = _cell(catalog, config, case, isl=case["window_tokens"], osl=1,
                        requires=["prepared_quality_dataset", "all_ppl_windows", "request_form:" + case["request_form"]])
            # A PPL case contains many windows; its final window can be shorter.
            row.update(actual_expected_input=None, input_length_kind="per_window",
                       input_upper_bound=case["window_tokens"])
            return [row]
        if case["kind"] == "gpqa":
            return [_cell(catalog, config, case, once=True, once_only=True,
                          requires=["external_gpqa_evaluator", f"all_{case['questions']}_questions"],
                          questions=case["questions"], needs_tokenization=True)]
        raise ValueError(f"unhandled quality kind: {case['kind']}")
    requires = list(case.get("requires", [])) + list(case.get("requires_launch", []))
    if group == "visual":
        requires.append("prepared_visual_fixtures")
    if case.get("fixtures"):
        requires.append("prepared_media")
    return [_cell(catalog, config, case, isl=None,
                  osl=2048 if group in ("core", "visual") else None,
                  requires=requires, needs_tokenization=True)]


def catalog_inventory() -> dict[str, Any]:
    """Return all source cases, expanded dimensions, variants and explicit derivatives."""
    catalogs, configs = {}, {}
    for family, names in CATALOGS.items():
        for name in names:
            catalog = family + "/" + name
            config = json.loads((CONFIG_DIR / family / f"{name}.json").read_text())
            configs[catalog] = config
            expander = _acceptance if family == "acceptance" else {
                "regular": _regular, "long_tail": _long_tail, "ab": _ab, "excluded": _excluded,
            }[name]
            rows = [row for case in config["cases"] for row in expander(catalog, config, case)]
            if name == "core" and family == "acceptance":
                for target in config["needle_tokens"]:
                    source = {"id": f"needle-target-{target}", "kind": "needle", "target_tokens": target,
                              "source_field": "needle_tokens"}
                    rows.append(_cell(catalog, config, source, isl=target, osl=512,
                                      target_tokens=target, needs_tokenization=True,
                                      admission_tokens=target + 512))
            catalogs[catalog] = {"source": f"config/{catalog}.json", "runner": config["runner"],
                                 "case_count": len(config["cases"]), "cells": rows}
    capacity = next(case for case in configs["benchmarks/regular"]["cases"] if case["id"] == "long_request_capacity")
    for original in _matrix("benchmarks/regular", configs["benchmarks/regular"], capacity):
        row = copy.deepcopy(original)
        ident = f"capacity_fixed_profile_c{row['concurrency']}"
        row.update(case=ident, case_id=ident, cell_id=ident, derived=True, source_launch=[],
                   requires=["context:1048576", "prefill_throttle:false"],
                   derivation={"from_case": capacity["id"], "changes": {"prefill_throttle": False},
                               "preserved": ["dataset", "isl", "osl", "concurrency", "seed"]})
        catalogs["benchmarks/regular"]["cells"].append(row)
    visual = configs["acceptance/visual"]
    source = next(case for case in visual["cases"] if case["id"] == "visual-video-long")
    highres = _cell("acceptance/visual", visual, source, isl=None, osl=2048,
                    needs_tokenization=True, derived=True,
                    requires=["prepared_visual_fixtures", "verified_highres_video_token_proof"],
                    visual_tokens=27904, frame_count=32, frame_width=1200, frame_height=1200,
                    derivation={"from_case": source["id"], "changes": {"frame_width": 1200, "frame_height": 1200},
                                "assertions": ["numbers:[1,2,3]", "colors:[red,green,blue]"]})
    highres.update(case="visual-video-highres-semantic", case_id="visual-video-highres-semantic",
                   cell_id="visual-video-highres-semantic")
    catalogs["acceptance/visual"]["cells"].append(highres)
    all_cells = [row for catalog in catalogs.values() for row in catalog["cells"]]
    ids = [row["cell_id"] for row in all_cells]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded catalog cell IDs must be globally unique")
    for catalog in catalogs.values():
        catalog["cell_count"] = len(catalog["cells"])
        catalog["derived_count"] = sum(c["derived"] for c in catalog["cells"])
    return {
        "catalogs": catalogs,
        "counts": {
            "catalogs": len(catalogs), "source_cases": sum(v["case_count"] for v in catalogs.values()),
            "audit_cells": len(all_cells), "derived_cells": sum(c["derived"] for c in all_cells),
            "ab_variants": sum(len(c["variants"]) for c in configs["benchmarks/ab"]["cases"]),
            "ab_cells": catalogs["benchmarks/ab"]["cell_count"],
            "excluded_variants": len(configs["benchmarks/excluded"]["cases"]),
            "excluded_cells": catalogs["benchmarks/excluded"]["cell_count"],
        },
    }


def group_plan(context: int, chunk: int = 16384, memory: float = .90,
               max_seqs: int = 256, throttle: bool = False) -> dict[str, Any]:
    """Bind the audit inventory to one fixed profile; pending lengths never imply PASS."""
    if context <= 0 or chunk <= 0 or max_seqs <= 0 or max_seqs % 2 or not 0 < memory <= 1:
        raise ValueError("context/chunk/max_seqs must be positive, max_seqs even and 0 < memory <= 1")
    inventory = catalog_inventory()
    cells = []
    for catalog in inventory["catalogs"].values():
        for source_row in catalog["cells"]:
            row = copy.deepcopy(source_row)
            reason = None
            if row["group"] in ("ab", "excluded"):
                reason = "Independent controlled experiment; its full variants are audited separately from this fixed profile."
            elif row["max_seqs"] is not None and row["max_seqs"] != max_seqs:
                reason = f"Historical secondary max-seqs={row['max_seqs']} differs from fixed max-seqs={max_seqs}."
            elif "prefill-throttle" in row["source_launch"] and not throttle:
                reason = "Source case requires prefill-throttle; this profile disables it."
            elif row["derived"] and row["case_id"].startswith("capacity_fixed_profile_") and (context != 1048576 or throttle):
                reason = "Derived fixed-profile capacity case requires context=1048576 and prefill-throttle=false."
            elif row["source_case"].get("launch_arg") == "--disable-chunked-mm-input":
                reason = "Source case requires the opposite multimodal mode; fixed profiles enable MM chunking."
            elif row["case_id"] == "visual-video-chunked-frame-boundary" and chunk >= 16384:
                reason = "384px fixture does not cross this chunk budget; use the dedicated boundary acceptance evidence."
                row["requires"].append("dedicated_boundary_evidence:docs/evidence/gpu-acceptance-20261024.json")
            required_tokens = row.get("admission_tokens")
            if required_tokens is None and row["isl"] is not None and row["osl"] is not None:
                required_tokens = row["isl"] + row["osl"]
            row["admission_tokens"] = required_tokens
            if reason is None and required_tokens is not None and required_tokens > context:
                reason = f"Input plus output budget {required_tokens} exceeds context {context}; do not truncate."
            row["requested_repeats"] = row["repeats"]
            row["length_check"] = "PENDING" if row["needs_tokenization"] else "KNOWN"
            if reason is not None:
                row.update(status="NOT_APPLICABLE", reason=reason, repeats=0)
            elif row["once"]:
                row.update(status="ONCE_ONLY", reason="Required once for the comparison campaign, not once per profile.")
            elif row["needs_tokenization"]:
                row.update(status="PENDING_PREPARATION", reason="Tokenize the complete request and resolve output limit before execution; reject overflow without truncation.")
            else:
                row.update(status="APPLICABLE", reason=None)
            cells.append(row)
    eligible = [c for c in cells if c["status"] in ("APPLICABLE", "PENDING_PREPARATION")]
    return {
        "profile": {"context": context, "chunk": chunk, "memory": memory, "max_seqs": max_seqs,
                    "throttle": throttle, "disable_chunked_mm_input": False, "responses_store": True,
                    "api_server_count": 1, "test_port": 6688},
        "context_policy": "Override historical launch context; admit full input plus output without truncation.",
        "cells": cells,
        "counts": {**inventory["counts"], "total": len(cells), "applicable": len(eligible),
                   "pending_preparation": sum(c["status"] == "PENDING_PREPARATION" for c in cells),
                   "not_applicable": sum(c["status"] == "NOT_APPLICABLE" for c in cells),
                   "once_only": sum(c["status"] == "ONCE_ONLY" for c in cells),
                   "repeats": sum(c["repeats"] for c in eligible),
                   "once_repeats": sum(c["repeats"] for c in cells if c["status"] == "ONCE_ONLY"),
                   "matrix_cells": sum(c["case_id"] in ("matrix_single", "matrix_concurrent") for c in cells),
                   "applicable_matrix_cells": sum(c["case_id"] in ("matrix_single", "matrix_concurrent") for c in eligible)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--memory", type=float, default=.90)
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--throttle", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(json.dumps(group_plan(args.context, args.chunk, args.memory, args.max_seqs, args.throttle), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
