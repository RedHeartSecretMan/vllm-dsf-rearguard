# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable text wrappers for DeepSeek-V4.1 video items.

The checkpoint has image-span delimiters, but no trained video delimiters.  A
video therefore keeps one application-level image placeholder and receives a
small ordinary-text wrapper.  The wrapper is deliberately independent of the
decoded frames so a replayed conversation renders the same prompt before the
multimodal processor runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
VIDEO_PART_TYPES = ("video_url", "input_video", "video")
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,32}\Z")


def video_frame_marker(video_index: int) -> str:
    """Return the ordinary-text marker used to bind one video occurrence."""
    return f"Video frame sequence {video_index + 1}."


def _explicit_video_id(block: Mapping[str, Any]) -> str | None:
    """Return a caller-provided stable id when one is present."""
    candidates: list[Any] = [
        block.get("video_id"),
        block.get("media_id"),
        block.get("uuid"),
    ]
    for key in ("video_url", "input_video", "video"):
        value = block.get(key)
        if isinstance(value, Mapping):
            candidates.extend(
                value.get(name)
                for name in ("video_id", "media_id", "uuid", "id", "file_id")
            )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _canonical_block(block: Mapping[str, Any]) -> str:
    """Build a stable identity payload without protocol alias differences."""
    media_keys = ("video_url", "input_video", "video")
    media_value: Any = None
    for key in media_keys:
        if key in block:
            media_value = block[key]
            break
    if isinstance(media_value, Mapping):
        media_value = dict(media_value)
        for alias in ("video_id", "media_id", "uuid", "id", "file_id"):
            media_value.pop(alias, None)
    options = {
        key: value
        for key, value in block.items()
        if key not in (*media_keys, "type", "video_id", "media_id", "uuid")
    }
    payload = {"media": media_value, "options": options}
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        # API requests normally contain JSON-compatible media references.  Keep
        # direct engine callers deterministic when they pass an unusual object.
        return repr(payload)


def stable_video_id(block: Mapping[str, Any]) -> str:
    """Return a short, prompt-safe id stable across history replays.

    Explicit ids are retained when they are already safe prompt tokens.  For
    ordinary API media blocks, the id is derived from the canonical source
    descriptor.  Crop/segment fields remain part of that descriptor, so two
    different views of one URL do not silently share an id.
    """
    explicit = _explicit_video_id(block)
    if explicit is not None and _SAFE_ID.fullmatch(explicit):
        return explicit if explicit.startswith("vid_") else f"vid_{explicit}"
    identity = _canonical_block(block)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"vid_{digest}"


def render_video_prompt(block: Mapping[str, Any], video_index: int = 0) -> str:
    """Render one video wrapper while preserving exactly one media placeholder."""
    video_id = stable_video_id(block)
    timestamps = block.get("video_prompt_timestamps")
    if isinstance(timestamps, (list, tuple)) and timestamps:
        frame_lines = "\n".join(
            f"Frame {index} at {float(timestamp):.3f}s."
            for index, timestamp in enumerate(timestamps, start=1)
        )
        frame_context = (
            f"The following {len(timestamps)} image spans are these frames in "
            "the same order:\n"
            f"{frame_lines}"
        )
    else:
        frame_context = "The following image spans are chronological frames from this video."
    return (
        f"Video ID: {video_id}.\n"
        f"{frame_context}\n"
        "Treat them as one video sequence.\n\n"
        f"{video_frame_marker(video_index)}\n"
        f"{IMAGE_PLACEHOLDER}\n\n"
        f"End of video {video_id}."
    )


def attach_video_uuids(messages: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Attach stable vLLM media UUIDs without mutating caller messages.

    vLLM uses a content-part ``uuid`` as the multimodal encoder/prefix-cache
    identity. Keeping it equal to the prompt id makes history replay stable
    while leaving non-video content untouched.
    """
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        updated_content = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") in VIDEO_PART_TYPES:
                block = dict(block)
                if not block.get("uuid"):
                    block["uuid"] = stable_video_id(block)
            updated_content.append(block)
        message["content"] = updated_content
    return result


def _media_timestamps(video: object) -> list[float] | None:
    """Extract actual sampled timestamps from one vLLM media item."""
    if hasattr(video, "media"):
        video = video.media
    if not (isinstance(video, tuple) and len(video) == 2):
        return None
    frames, metadata = video
    if not isinstance(metadata, Mapping):
        return None
    frame_count = len(frames) if hasattr(frames, "__len__") else 0
    indices = metadata.get("frames_indices")
    fps = metadata.get("fps")
    if (
        isinstance(indices, (list, tuple))
        and len(indices) == frame_count
        and isinstance(fps, (int, float))
        and fps > 0
    ):
        return [float(index) / float(fps) for index in indices]
    duration = metadata.get("duration")
    if isinstance(duration, (int, float)) and duration > 0 and frame_count > 1:
        step = float(duration) / float(frame_count - 1)
        return [step * index for index in range(frame_count)]
    return None


def attach_video_prompt_timestamps(
    messages: list[Mapping[str, Any]],
    mm_data: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    """Attach media-IO timestamps to video blocks without mutating callers."""
    result = copy.deepcopy(messages)
    videos = (mm_data or {}).get("video", [])
    video_index = 0
    for message in result:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        updated_content = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") in VIDEO_PART_TYPES:
                block = dict(block)
                if video_index < len(videos):
                    timestamps = _media_timestamps(videos[video_index])
                    if timestamps:
                        block["video_prompt_timestamps"] = timestamps
                video_index += 1
            updated_content.append(block)
        message["content"] = updated_content
    return result


def render_dummy_video_prompt(index: int) -> str:
    """Render a deterministic profiling wrapper for one dummy video."""
    return render_video_prompt({"video_id": f"vid_{index:016x}"}, index)
