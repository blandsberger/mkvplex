"""ffprobe/ffmpeg media inspection, splitting, presentation, and content fingerprints."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import difflib
import errno
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional
from .models import AggregateSplitPlan, BlackInterval, Episode, MKVPlexError, Match, SplitBoundary, TrackAnalysis, TvRipGroup
from .common import _disc_key, _median, episode_expected_seconds, format_duration
from .discovery import discovery_db
from .naming import episode_basename, episode_title_similarity, epl_number, movie_basename, normalize_for_match, track_number
from .fsops import recursive_chmod

def probe_video_presentation(path: Path) -> dict[str, Any]:
    """Return encoded video raster + field-order metadata for filename tagging.

    This is intentionally a stream-metadata probe, not a content analysis pass.
    In particular, a stream reported as progressive is named progressive; we do
    not run idet here to second-guess telecine or field structure hidden inside
    progressive-coded frames.
    """
    db = discovery_db()
    if db is not None and path.exists():
        cached = db.get_video_profile(path)
        if cached is not None:
            return cached
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise MKVPlexError(
            "ffprobe was not found in PATH. Technical presentation suffixes require ffprobe."
        )
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,field_order,codec_name,pix_fmt",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MKVPlexError(f"ffprobe timed out while reading video presentation for {path}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise MKVPlexError(f"ffprobe failed for {path}: {detail[:300]}")
    try:
        streams = json.loads(proc.stdout or "{}").get("streams") or []
        row = streams[0]
        profile = {
            "width": int(row.get("width") or 0),
            "height": int(row.get("height") or 0),
            "field_order": str(row.get("field_order") or "unknown").lower(),
            "codec_name": str(row.get("codec_name") or ""),
            "pix_fmt": str(row.get("pix_fmt") or ""),
        }
    except Exception as exc:
        raise MKVPlexError(f"ffprobe returned invalid video presentation metadata for {path}") from exc
    if profile["width"] <= 0 or profile["height"] <= 0:
        raise MKVPlexError(f"ffprobe found no usable video raster for {path}")
    if db is not None and path.exists():
        db.put_video_profile(path, profile)
    return profile


def _presentation_scan_component(field_order: str) -> str:
    """Normalize FFmpeg field_order into a compact filename component."""
    order = (field_order or "unknown").lower()
    if order == "progressive":
        return "p"
    if order == "tt":
        return "i-TFF"
    if order == "bb":
        return "i-BFF"
    # FFmpeg distinguishes coded-first from displayed-first for these rarer
    # forms. Preserve that distinction rather than collapsing it incorrectly.
    if order == "tb":
        return "i-TB"
    if order == "bt":
        return "i-BT"
    return ""


def _presentation_resolution_component(width: int, height: int) -> tuple[str, Optional[str]]:
    """Return (resolution, raster-family) without guessing from path names."""
    w, h = int(width), int(height)
    if w == 3840 and h == 2160:
        return "2160", "UHD"
    if w == 4096 and h == 2160:
        return "2160", "DCI4K"
    # Common full-raster broadcast/disc formats. 1088 is retained as 1080
    # because MPEG-family coded height can include padding beyond active 1080.
    if h in {1080, 1088}:
        return "1080", None
    if h == 720:
        return "720", None
    if h in {480, 486}:
        return "480", None
    if h == 576:
        return "576", None
    # Do not pretend a cropped/re-encoded raster is a standard full-raster
    # presentation. Its encoded dimensions are the truthful fallback.
    return f"{w}x{h}", None


def technical_presentation_tag(path: Path) -> str:
    """Return canonical technical descriptor, e.g. 1080p or 2160p-UHD."""
    profile = probe_video_presentation(path)
    resolution, family = _presentation_resolution_component(
        int(profile["width"]), int(profile["height"])
    )
    scan = _presentation_scan_component(str(profile.get("field_order") or "unknown"))
    tag = f"{resolution}{scan}"
    if family:
        tag += f"-{family}"
    return tag


def technical_presentation_suffix(path: Path) -> str:
    return f" [{technical_presentation_tag(path)}]"


def episode_filename(match: Match, episode: Episode, source: Path) -> str:
    return episode_basename(match, episode) + technical_presentation_suffix(source) + ".mkv"


def movie_filename(match: Match, source: Path) -> str:
    return movie_basename(match) + technical_presentation_suffix(source) + ".mkv"


def probe_duration_seconds(path: Path) -> float:
    """Read media duration with ffprobe without decoding the file."""
    db = discovery_db()
    if db is not None and path.exists():
        cached = db.get_duration(path)
        if cached is not None:
            return cached
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise MKVPlexError(
            "ffprobe was not found in PATH. TV auto-selection needs ffprobe "
            "to distinguish episodes from extras."
        )
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MKVPlexError(f"ffprobe timed out on {path}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise MKVPlexError(f"ffprobe failed for {path}: {detail[:300]}")
    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise MKVPlexError(f"ffprobe returned an invalid duration for {path}: {proc.stdout!r}") from exc
    if duration <= 0:
        raise MKVPlexError(f"ffprobe returned a non-positive duration for {path}: {duration}")
    if db is not None and path.exists():
        db.put_duration(path, duration)
    return duration



def probe_container_title(path: Path) -> Optional[str]:
    """Return the authored container title, when present.

    This is intentionally a tiny metadata-only probe used as a *fallback* for
    provider discovery.  Filesystem names remain the primary identity source;
    a container title is consulted only when the parsed title produced no
    metadata candidates.  Failures return ``None`` rather than masking the
    original provider lookup error.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format_tags=title",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
        title = str(((data.get("format") or {}).get("tags") or {}).get("title") or "").strip()
    except Exception:
        return None
    return title or None

def _rotational_flags_for_device(device: str) -> list[int]:
    """Return lsblk rotational flags for DEVICE and its backing devices."""
    lsblk = shutil.which("lsblk")
    if not lsblk or not device.startswith("/dev/"):
        return []
    try:
        proc = subprocess.run(
            [lsblk, "-ndo", "ROTA", "-s", device],
            text=True, capture_output=True, timeout=5,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    flags: list[int] = []
    for line in proc.stdout.splitlines():
        value = line.strip()
        if value in {"0", "1"}:
            flags.append(int(value))
    return flags


def detect_storage_class(path: Path) -> str:
    """Best-effort Linux storage classification: hdd | ssd | unknown.

    ZFS hides leaf devices from findmnt, so for a ZFS dataset we inspect the
    pool's /dev leaves. Any rotational data member makes the conservative
    answer HDD/mixed; an all-nonrotational pool is treated as SSD/NVMe.
    """
    target = path.expanduser().resolve()
    findmnt = shutil.which("findmnt")
    source = ""
    fstype = ""
    if findmnt:
        try:
            proc = subprocess.run(
                [findmnt, "-no", "SOURCE,FSTYPE", "-T", str(target)],
                text=True, capture_output=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parts = proc.stdout.strip().split()
                if parts:
                    source = parts[0]
                if len(parts) > 1:
                    fstype = parts[1].lower()
        except Exception:
            pass

    flags: list[int] = []
    if fstype == "zfs" and source:
        pool = source.split("/", 1)[0]
        zpool = shutil.which("zpool")
        if zpool:
            try:
                proc = subprocess.run(
                    [zpool, "status", "-P", pool],
                    text=True, capture_output=True, timeout=8,
                )
                if proc.returncode == 0:
                    devices = sorted(set(re.findall(r"(/dev/[^\s]+)", proc.stdout)))
                    for device in devices:
                        flags.extend(_rotational_flags_for_device(device))
            except Exception:
                pass
    elif source.startswith("/dev/"):
        flags.extend(_rotational_flags_for_device(source))

    if any(flag == 1 for flag in flags):
        return "hdd"
    if flags and all(flag == 0 for flag in flags):
        return "ssd"
    return "unknown"


def resolve_media_workers(path: Path, requested: int, storage_mode: str = "auto") -> tuple[int, str]:
    storage = storage_mode if storage_mode in {"hdd", "ssd"} else detect_storage_class(path)
    if requested > 0:
        return max(1, min(int(requested), 32)), storage
    if storage == "hdd":
        return 2, storage
    if storage == "ssd":
        return min(12, max(4, os.cpu_count() or 4)), storage
    return 4, storage


def probe_durations(paths: Iterable[Path], workers: int = 4) -> dict[Path, float]:
    """Probe several files concurrently; ffprobe only reads container metadata."""
    unique = list(dict.fromkeys(paths))
    if not unique:
        return {}
    workers = max(1, min(workers, len(unique)))
    result: dict[Path, float] = {}
    if workers == 1:
        return {p: probe_duration_seconds(p) for p in unique}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(probe_duration_seconds, p): p for p in unique}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            result[path] = future.result()
    return result


def probe_chapter_boundaries(path: Path, *, source_duration: Optional[float] = None) -> list[float]:
    """Return authored chapter boundary timestamps from an MKV.

    MakeMKV normally preserves chapter metadata from the selected physical-media
    title.  We collect both chapter starts and ends, de-duplicate adjacent values,
    and exclude source start/EOF because those are already known anchors.
    """
    db = discovery_db()
    if db is not None and path.exists():
        cached = db.get_chapters(path)
        if cached is not None:
            return cached

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    cmd = [
        ffprobe, "-v", "error", "-show_chapters", "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception:
        return []

    values: list[float] = []
    for chapter in payload.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        for key in ("start_time", "end_time"):
            try:
                value = float(chapter.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                values.append(value)

    duration = float(source_duration) if source_duration is not None else None
    values.sort()
    deduped: list[float] = []
    for value in values:
        if value <= 0.75:
            continue
        if duration is not None and value >= duration - 0.75:
            continue
        if deduped and abs(value - deduped[-1]) <= 0.50:
            continue
        deduped.append(value)
    if db is not None and path.exists():
        db.put_chapters(path, deduped)
    return deduped


def _chapter_confidence(delta: float) -> str:
    if delta <= 20.0:
        return "very-high:chapter"
    if delta <= 50.0:
        return "high:chapter"
    if delta <= 100.0:
        return "medium:chapter"
    return "low:chapter"


def select_chapter_snaps(
    predicted_cuts: list[float],
    chapter_points: list[float],
    *,
    max_delta: float = 100.0,
) -> dict[int, SplitBoundary]:
    """Choose a globally monotonic subset of chapter points near expected cuts.

    A play-all title may have several chapters per episode, so chapter count is
    not treated as episode count.  Dynamic programming chooses at most one
    authored marker for each expected episode boundary while preserving chapter
    order.  Runtime fallback has a fixed cost; a chapter wins only when it lies
    close enough to the expected cut to improve the solution.
    """
    if not predicted_cuts or not chapter_points:
        return {}
    points = sorted(float(v) for v in chapter_points)
    fallback_cost = 1.75
    # state: last chapter index -> (cost, tuple(assignments)); None is encoded -1
    states: dict[int, tuple[float, tuple[Optional[int], ...]]] = {-1: (0.0, tuple())}
    for predicted in predicted_cuts:
        next_states: dict[int, tuple[float, tuple[Optional[int], ...]]] = {}
        for last_idx, (cost, assignments) in states.items():
            fallback = (cost + fallback_cost, assignments + (None,))
            current = next_states.get(last_idx)
            if current is None or fallback[0] < current[0]:
                next_states[last_idx] = fallback
            for idx in range(last_idx + 1, len(points)):
                delta = abs(points[idx] - predicted)
                if points[idx] > predicted + max_delta:
                    break
                if delta > max_delta:
                    continue
                # Authored markers get a small preference over pure runtime only
                # when they are genuinely close to the expected boundary.
                chapter_cost = max(0.0, delta / 60.0 - 0.15)
                candidate = (cost + chapter_cost, assignments + (idx,))
                current = next_states.get(idx)
                if current is None or candidate[0] < current[0]:
                    next_states[idx] = candidate
        # Bound state growth without changing the best practical solutions.
        states = dict(sorted(next_states.items(), key=lambda item: item[1][0])[:256])

    if not states:
        return {}
    _last, (_cost, assignments) = min(states.items(), key=lambda item: item[1][0])
    selected: dict[int, SplitBoundary] = {}
    for pos, (predicted, chapter_idx) in enumerate(zip(predicted_cuts, assignments), 1):
        if chapter_idx is None:
            continue
        timestamp = points[chapter_idx]
        delta = abs(timestamp - predicted)
        confidence = _chapter_confidence(delta)
        if confidence.startswith("low"):
            continue
        selected[pos] = SplitBoundary(
            predicted=float(predicted), selected=float(timestamp),
            black_start=None, black_end=None, black_duration=None,
            delta=float(delta), confidence=confidence, cached=False,
        )
    return selected


def probe_black_intervals(
    path: Path,
    start: float,
    end: float,
    *,
    min_duration: float = 0.30,
) -> list[BlackInterval]:
    """Decode only a small search window and return sustained black intervals.

    ffmpeg's blackdetect timestamps are relative to the seeked window when -ss
    precedes -i, so START is added back before returning absolute source times.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MKVPlexError(
            "ffmpeg was not found in PATH. Aggregate episode splitting needs "
            "ffmpeg's blackdetect filter."
        )
    start = max(0.0, float(start))
    end = max(start, float(end))
    length = end - start
    if length <= 0:
        return []
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info",
        "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{length:.3f}",
        "-vf", f"blackdetect=d={max(0.05, min_duration):.3f}:pix_th=0.10:pic_th=0.98",
        "-an", "-sn", "-dn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise MKVPlexError(f"ffmpeg blackdetect failed for {path}: {detail[-500:]}")
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    pattern = re.compile(
        r"black_start:(?P<start>[0-9.]+)\s+black_end:(?P<end>[0-9.]+)\s+black_duration:(?P<duration>[0-9.]+)"
    )
    found: list[BlackInterval] = []
    for m in pattern.finditer(text):
        local_start = float(m.group("start"))
        local_end = float(m.group("end"))
        duration = float(m.group("duration"))
        found.append(BlackInterval(start + local_start, start + local_end, duration))
    return found


def _boundary_confidence(delta: float, black_duration: float) -> str:
    if delta <= 20.0 and black_duration >= 0.45:
        return "very-high:black"
    if delta <= 50.0 and black_duration >= 0.30:
        return "high:black"
    if delta <= 100.0 and black_duration >= 0.20:
        return "medium:black"
    return "low"


def detect_fade_boundary(
    path: Path,
    *,
    current_start: float,
    expected_duration: float,
    source_duration: float,
    search_window: float = 180.0,
    min_black_duration: float = 0.30,
) -> SplitBoundary:
    """Find the likely episode-ending fade-to-black near an expected end.

    The expected runtime only defines a search neighborhood.  The selected
    timestamp is the *end* of the sustained black interval so the preceding
    episode keeps its fade/black and the next episode begins with picture.
    """
    db = discovery_db()
    if db is not None and path.exists():
        cached = db.get_black_boundary(
            path, current_start=current_start, expected_duration=expected_duration,
            source_duration=source_duration, search_window=search_window,
            min_black=min_black_duration,
        )
        if cached is not None:
            return cached

    def remember(boundary: SplitBoundary) -> SplitBoundary:
        if db is not None and path.exists():
            db.put_black_boundary(
                path, boundary, current_start=current_start,
                expected_duration=expected_duration, source_duration=source_duration,
                search_window=search_window, min_black=min_black_duration,
            )
        return boundary

    predicted = min(source_duration, current_start + expected_duration)
    earliest = current_start + max(60.0, expected_duration * 0.60)
    window_start = max(earliest, predicted - max(30.0, search_window))
    window_end = min(source_duration, predicted + max(30.0, search_window))
    intervals = probe_black_intervals(
        path, window_start, window_end, min_duration=min_black_duration
    )
    usable = [i for i in intervals if i.end > current_start + 60.0]
    if usable:
        # Prefer temporal proximity but reward a sustained black interval.
        def score(i: BlackInterval) -> tuple[float, float]:
            delta = abs(i.end - predicted)
            reward = min(i.duration, 2.5) * 12.0
            return (delta - reward, delta)
        chosen = min(usable, key=score)
        selected = chosen.end
        if source_duration - selected <= 2.0:
            selected = source_duration
        delta = abs(selected - predicted)
        return remember(SplitBoundary(
            predicted=predicted, selected=selected,
            black_start=chosen.start, black_end=chosen.end,
            black_duration=chosen.duration, delta=delta,
            confidence=_boundary_confidence(delta, chosen.duration),
        ))

    # End-of-file is a reasonable final boundary when metadata predicts it
    # closely, but it is weaker than seeing the fade itself.
    eof_delta = abs(source_duration - predicted)
    if eof_delta <= min(90.0, max(30.0, search_window)):
        return remember(SplitBoundary(
            predicted=predicted, selected=source_duration,
            black_start=None, black_end=None, black_duration=None,
            delta=eof_delta, confidence="medium:eof",
        ))
    return remember(SplitBoundary(
        predicted=predicted, selected=None, black_start=None, black_end=None,
        black_duration=None, delta=None, confidence="unresolved",
    ))


def _aggregate_complete_series_prefix_hint(
    groups: list[TvRipGroup],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
    workers: int = 4,
) -> Optional[dict[str, Any]]:
    """Detect a complete-series corpus stored as a few giant play-all MKVs.

    The ordinary complete-series heuristic uses source-track counts, which
    cannot recognize layouts such as one 8-10 hour MKV per physical disc.
    Here we compare ordered giant-source *durations* against the provider's
    expected runtime for all regular episodes and look for a plausible prefix.

    Returning a prefix rather than requiring every giant file to fit is
    intentional: box sets can append Gaiden/OVA/movie/bonus discs after the
    primary show's regular-series discs. Those residual giants remain deferred
    until another identity explains them.
    """
    if not groups or not episodes:
        return None
    tracks_with_group = [(track, group) for group in groups for track in group.tracks]
    if not tracks_with_group:
        return None

    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    typical = _median(expected) or 0.0
    total_expected = sum(expected)
    if typical <= 0 or total_expected <= 0:
        return None

    # This path is specifically for aggregate-media layouts: far fewer source
    # titles than logical episodes, with most source titles much longer than an
    # episode. Avoid doing extra probes for ordinary per-episode rips.
    if len(tracks_with_group) >= max(16, int(len(episodes) * 0.50)):
        return None

    durations = probe_durations((p for p, _g in tracks_with_group), workers=workers)
    ordered: list[tuple[Path, TvRipGroup, float]] = []
    for path, group in tracks_with_group:
        duration = float(durations.get(path, 0.0) or 0.0)
        if duration >= max(typical * 2.25, 45.0 * 60.0):
            ordered.append((path, group, duration))
    ordered.sort(key=lambda row: (_disc_key(row[1]), track_number(row[0]), str(row[0]).casefold()))

    if len(ordered) < 2:
        return None
    # Require aggregate/long-form titles to dominate the physical corpus.
    if len(ordered) < max(2, int(math.ceil(len(tracks_with_group) * 0.75))):
        return None

    best_n: Optional[int] = None
    best_ratio: Optional[float] = None
    best_score = float("inf")
    running = 0.0
    for n, (_path, _group, duration) in enumerate(ordered, 1):
        running += duration
        ratio = running / total_expected
        # Same broad acceptance philosophy as select_aggregate_source_subset,
        # but favor a prefix close to 100% and penalize under-coverage harder.
        score = abs(math.log(max(ratio, 1e-6)))
        if ratio < 0.82:
            score += (0.82 - ratio) * 8.0
        if ratio > 1.45:
            score += (ratio - 1.45) * 8.0
        if score < best_score:
            best_score = score
            best_n = n
            best_ratio = ratio

    if best_n is None or best_ratio is None or not (0.82 <= best_ratio <= 1.45):
        return None

    return {
        "prefix_count": best_n,
        "giant_count": len(ordered),
        "source_count": len(tracks_with_group),
        "ratio": best_ratio,
        "prefix_duration": sum(row[2] for row in ordered[:best_n]),
        "deferred_count": len(ordered) - best_n,
    }


def _aggregate_episode_sources(
    analyses: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
) -> list[TrackAnalysis]:
    """Return giant titles that could contain several consecutive episodes.

    This intentionally returns *candidates*, not necessarily the whole source
    set.  A physical TV collection may have giant movie/bonus titles after the
    TV discs, so selection happens separately against the target's expected
    total runtime.
    """
    if not episodes:
        return []
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    typical = _median(expected) or 0.0
    if typical <= 0:
        return []
    rows = [
        r for r in analyses
        if not r.aggregate_of
        and r.duration >= max(typical * 2.25, 45.0 * 60.0)
    ]
    rows.sort(key=lambda r: (_disc_key(r.group), track_number(r.path), str(r.path).casefold()))
    return rows


def select_aggregate_source_subset(
    sources: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
) -> tuple[list[TrackAnalysis], list[TrackAnalysis]]:
    """Choose the ordered giant-source prefix that best explains the TV target.

    MakeMKV collections can contain giant non-TV titles (movies, bonus programs)
    beside giant TV-disc titles.  Treating every giant file as part of the TV
    season forces the episode allocator to spread episodes across unrelated
    sources.  Compare ordered prefixes against the provider's total expected
    target runtime and choose the closest plausible coverage.

    Sources after the selected prefix are deferred for manual/other-media
    handling and are not moved by TV mode.
    """
    if not sources or not episodes:
        return [], list(sources)
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    total_expected = sum(expected)
    if total_expected <= 0:
        return list(sources), []

    best_n: Optional[int] = None
    best_score = float("inf")
    running = 0.0
    for n, row in enumerate(sources, 1):
        running += row.duration
        ratio = running / total_expected
        if ratio < 0.88:
            score = (0.88 - ratio) * 1000.0 + abs(math.log(max(ratio, 1e-6))) * 100.0
        else:
            score = abs(math.log(ratio)) * 100.0
            if ratio < 0.97:
                score += (0.97 - ratio) * 220.0
            if ratio > 1.18:
                score += (ratio - 1.18) * 240.0
        if score < best_score:
            best_score = score
            best_n = n

    if best_n is None:
        return list(sources), []

    selected = list(sources[:best_n])
    deferred = list(sources[best_n:])
    selected_total = sum(r.duration for r in selected)
    ratio = selected_total / total_expected
    if ratio < 0.82 or ratio > 1.45:
        return [], list(sources)
    return selected, deferred


def aggregate_presentation_runtime_scale(
    sources: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
) -> float:
    """Calibrate provider runtimes to the physical presentation.

    TMDb runtimes are metadata-scale measurements (often rounded broadcast
    runtimes), not frame-accurate mastering durations.  A physical release can
    be systematically shorter or longer because of mastering, repeated OP/ED
    handling, PAL speed-up, distributor edits, or other presentation choices.

    For an aggregate source set that is already hypothesized to cover the full
    episode corpus exactly once, total physical duration / total provider
    duration is an excellent collection-level calibration factor.  Keep this
    deliberately bounded: a large discrepancy is evidence that the aggregate
    hypothesis itself is wrong and must not be normalized away.
    """
    if not sources or not episodes:
        return 1.0
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    expected_total = sum(expected)
    actual_total = sum(max(0.0, float(row.duration)) for row in sources)
    if expected_total <= 0 or actual_total <= 0:
        return 1.0
    scale = actual_total / expected_total
    # A +/-20% systematic mastering difference is already very large. Beyond
    # that, retain literal provider runtimes so the normal mismatch safeguards
    # expose a likely wrong corpus/allocation instead of concealing it.
    if 0.80 <= scale <= 1.20:
        return scale
    return 1.0


def allocate_episodes_to_aggregate_sources(
    sources: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
) -> Optional[list[tuple[TrackAnalysis, list[Episode]]]]:
    """Partition consecutive season episodes across aggregate source files.

    Dynamic programming chooses episode counts whose summed provider runtimes
    best fit each source duration. Provider runtimes are first calibrated to the
    physical presentation as a collection-wide scale; exact cut points are
    located later from fade-to-black evidence plus those calibrated weights.
    """
    if not sources or len(episodes) < len(sources):
        return None
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    runtime_scale = aggregate_presentation_runtime_scale(
        sources, episodes, show_runtime_minutes=show_runtime_minutes
    )
    n, m = len(sources), len(episodes)
    prefix = [0.0]
    for value in expected:
        prefix.append(prefix[-1] + value)
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    prev: list[list[Optional[int]]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n):
        remaining_sources = n - i - 1
        for used in range(m + 1):
            if not math.isfinite(dp[i][used]):
                continue
            actual = sources[i].duration
            # Earlier versions capped a giant source at 12 episodes. That was
            # sufficient for Evangelion-style two-hour masters but rejects
            # physical-disc play-all files that contain 20-30 episodes. Size
            # the ceiling from the source duration while retaining 12 as the
            # ordinary conservative floor.
            typical = (_median(expected) or 1.0) * runtime_scale
            duration_based_cap = int(math.ceil(actual / max(typical, 1.0) * 1.35)) + 2
            max_k = min(max(12, duration_based_cap), m - used - remaining_sources)
            for k in range(1, max_k + 1):
                end = used + k
                expected_sum = (prefix[end] - prefix[used]) * runtime_scale
                ratio = max(actual, 1.0) / max(expected_sum, 1.0)
                # Symmetric relative-duration cost.  A modest source tail is
                # tolerated; grossly implausible partitions become expensive.
                cost = (math.log(ratio) ** 2) * 100.0
                if ratio < 0.75:
                    cost += (0.75 - ratio) * 150.0
                candidate = dp[i][used] + cost
                if candidate < dp[i + 1][end]:
                    dp[i + 1][end] = candidate
                    prev[i + 1][end] = used
    if not math.isfinite(dp[n][m]):
        return None
    cuts: list[tuple[int, int]] = []
    i, end = n, m
    while i > 0:
        start = prev[i][end]
        if start is None:
            return None
        cuts.append((start, end))
        i -= 1
        end = start
    cuts.reverse()
    return [(src, list(episodes[a:b])) for src, (a, b) in zip(sources, cuts)]


def _runtime_resolve_aggregate_boundaries(
    episodes: list[Episode],
    expected: list[float],
    raw_boundaries: list[SplitBoundary],
    *,
    source_duration: float,
) -> tuple[list[SplitBoundary], int]:
    """Reconcile authored/visual boundary observations with TMDb runtime constraints.

    Chapter markers and black frames are evidence, not commandments.  A long play-all source can
    contain many legitimate intra-episode black frames, so blindly treating
    every medium/high black hit as a hard anchor can make otherwise coherent
    runtime intervals look impossible.  Build a globally consistent subset of
    black anchors, then interpolate the gaps from provider runtimes.

    The anchor subset is chosen with a small dynamic program.  Consecutive
    anchors are compatible only when their observed time span is within 15% of
    the summed TMDb runtimes for the episodes between them.  Among compatible
    paths from source start to EOF, prefer the strongest/most numerous black
    observations.  Black hits rejected by this consistency test are replaced by
    runtime-derived endpoints rather than poisoning the whole source plan.

    If even source-start -> EOF differs from the provider runtime model by more
    than 15%, the fallback remains low confidence and therefore non-executable.
    """
    if not episodes or len(episodes) != len(expected) or len(raw_boundaries) != len(episodes):
        return list(raw_boundaries), 0

    count = len(episodes)
    total_expected = sum(expected)
    if total_expected <= 0 or source_duration <= 0:
        return list(raw_boundaries), 0

    cumulative = [0.0]
    for value in expected:
        cumulative.append(cumulative[-1] + float(value))

    def span_error(left_pos: int, left_t: float, right_pos: int, right_t: float) -> float:
        span_expected = cumulative[right_pos] - cumulative[left_pos]
        span_actual = right_t - left_t
        if span_expected <= 0 or span_actual <= 0:
            return float("inf")
        return abs((span_actual / span_expected) - 1.0)

    def anchor_weight(boundary: SplitBoundary) -> float:
        c = boundary.confidence
        if c.startswith("very-high:chapter"):
            return 5.0
        if c.startswith("high:chapter"):
            return 4.0
        if c.startswith("medium:chapter"):
            return 3.0
        if c.startswith("very-high:black"):
            return 4.0
        if c.startswith("high:black"):
            return 3.0
        if c.startswith("medium:black"):
            return 2.0
        if c.startswith("medium:eof"):
            return 0.5
        return 1.0

    # (boundary-position, timestamp, raw-boundary-index or None, weight)
    nodes: list[tuple[int, float, Optional[int], float]] = [(0, 0.0, None, 0.0)]
    for idx, boundary in enumerate(raw_boundaries[:-1], 1):
        if (
            boundary.selected is not None
            and boundary.confidence not in {"unresolved", "low"}
            and not boundary.confidence.startswith("low:")
        ):
            nodes.append((idx, float(boundary.selected), idx - 1, anchor_weight(boundary)))
    nodes.append((count, float(source_duration), None, 0.0))
    nodes.sort(key=lambda row: (row[0], row[1]))

    # Longest/strongest compatible path from source start to EOF.  Score is
    # primarily confidence weight, secondarily anchor count, and finally lower
    # cumulative runtime mismatch.
    n = len(nodes)
    best: list[Optional[tuple[float, int, float]]] = [None] * n
    prev: list[Optional[int]] = [None] * n
    best[0] = (0.0, 0, 0.0)
    max_error = 0.15
    for j in range(1, n):
        rpos, rt, _ridx, rweight = nodes[j]
        for i in range(j):
            state = best[i]
            if state is None:
                continue
            lpos, lt, _lidx, _lweight = nodes[i]
            if rpos <= lpos or rt <= lt:
                continue
            err = span_error(lpos, lt, rpos, rt)
            if err > max_error:
                continue
            cand = (state[0] + rweight, state[1] + (1 if _ridx is not None else 0), state[2] - err)
            if best[j] is None or cand > best[j]:
                best[j] = cand
                prev[j] = i

    end_index = n - 1
    if best[end_index] is None:
        # No globally compatible black-anchor path exists because even the full
        # source duration disagrees too much with provider runtimes.  Still
        # produce deterministic numeric cuts for review, but mark them low.
        resolved: list[SplitBoundary] = []
        for pos, raw in enumerate(raw_boundaries, 1):
            selected = source_duration * (cumulative[pos] / total_expected)
            if pos == count:
                selected = float(source_duration)
            resolved.append(SplitBoundary(
                predicted=float(raw.predicted), selected=selected,
                black_start=None, black_end=None, black_duration=None,
                delta=abs(selected - float(raw.predicted)),
                confidence="low:runtime-mismatch", cached=raw.cached,
            ))
        return resolved, count

    path_indices: list[int] = []
    cursor: Optional[int] = end_index
    while cursor is not None:
        path_indices.append(cursor)
        cursor = prev[cursor]
    path_indices.reverse()
    selected_nodes = [nodes[i] for i in path_indices]
    selected_raw_indexes = {raw_idx for _p, _t, raw_idx, _w in selected_nodes if raw_idx is not None}

    resolved = list(raw_boundaries)
    fallback_count = 0
    for (left_pos, left_t, _li, _lw), (right_pos, right_t, _ri, _rw) in zip(selected_nodes, selected_nodes[1:]):
        span_expected = cumulative[right_pos] - cumulative[left_pos]
        span_actual = right_t - left_t
        if span_expected <= 0 or span_actual <= 0:
            continue
        relative_error = abs((span_actual / span_expected) - 1.0)
        confidence = "medium:runtime-anchored" if relative_error <= 0.06 else "medium:runtime-scaled"

        for boundary_pos in range(left_pos + 1, right_pos + 1):
            raw_index = boundary_pos - 1
            raw = raw_boundaries[raw_index]
            # Preserve only black observations selected by the global consistency
            # path.  All other black hits are treated as plausible intra-episode
            # fades and replaced by the runtime model.
            if raw_index in selected_raw_indexes and boundary_pos != count:
                continue

            elapsed_expected = cumulative[boundary_pos] - cumulative[left_pos]
            selected = left_t + span_actual * (elapsed_expected / span_expected)
            if boundary_pos == count:
                selected = float(source_duration)
            resolved[raw_index] = SplitBoundary(
                predicted=float(raw.predicted), selected=selected,
                black_start=None, black_end=None, black_duration=None,
                delta=abs(selected - float(raw.predicted)),
                confidence=(
                    "medium:eof-runtime" if boundary_pos == count
                    else ("medium:runtime-over-black" if raw.selected is not None else confidence)
                ),
                cached=raw.cached,
            )
            fallback_count += 1

    return resolved, fallback_count


def build_aggregate_split_plans(
    allocations: list[tuple[TrackAnalysis, list[Episode]]],
    *,
    show_runtime_minutes: Optional[float] = None,
    search_window: float = 180.0,
    min_black_duration: float = 0.30,
) -> list[AggregateSplitPlan]:
    plans: list[AggregateSplitPlan] = []
    total = sum(len(eps) for _row, eps in allocations)
    all_sources = [row for row, _eps in allocations]
    all_episodes = [ep for _row, eps in allocations for ep in eps]
    runtime_scale = aggregate_presentation_runtime_scale(
        all_sources, all_episodes, show_runtime_minutes=show_runtime_minutes
    )
    if abs(runtime_scale - 1.0) >= 0.02:
        print(
            f"      presentation runtime calibration: {runtime_scale:.3f}x provider runtimes "
            f"(TMDb used as relative episode weights)",
            flush=True,
        )
    completed = 0
    print(
        "      boundary model: authored chapters snap compatible TMDb-weighted cuts; "
        "black/fade is secondary; runtime-derived timestamps are the fallback",
        flush=True,
    )
    for source_index, (row, eps) in enumerate(allocations, 1):
        # Once episode ownership has been allocated to a physical source, the
        # source's exact EOF is stronger timing evidence than a collection-wide
        # mastering scale.  Normalize TMDb's per-episode runtimes to this source
        # so they define a complete, drift-free local timeline.  A black/fade
        # observation may refine an individual boundary, but it never moves the
        # prediction origin for any later boundary.
        provider_expected = episode_expected_seconds(eps, show_runtime_minutes)
        provider_total = sum(provider_expected)
        if provider_total > 0.0 and row.duration > 0.0:
            local_scale = row.duration / provider_total
            expected = [value * local_scale for value in provider_expected]
        else:
            local_scale = runtime_scale
            expected = [value * local_scale for value in provider_expected]

        boundaries: list[SplitBoundary] = []
        cumulative_expected: list[float] = []
        running = 0.0
        for value in expected[:-1]:
            running += value
            cumulative_expected.append(min(float(row.duration), running))
        chapter_points = probe_chapter_boundaries(row.path, source_duration=row.duration)
        chapter_snaps = select_chapter_snaps(cumulative_expected, chapter_points)
        print(
            f"      boundary scan source {source_index}/{len(allocations)}: "
            f"{row.path.name} ({len(eps)} episode boundary search(es); "
            f"local runtime scale {local_scale:.3f}x; "
            f"{len(chapter_points)} authored chapter marker(s), "
            f"{len(chapter_snaps)} compatible snap(s))",
            flush=True,
        )
        predicted_start = 0.0
        for boundary_index, (ep, exp) in enumerate(zip(eps, expected), 1):
            completed += 1
            predicted = min(row.duration, predicted_start + exp)
            # Make the final boundary exact EOF.  There is no value in spending
            # an ffmpeg blackdetect pass to rediscover a timestamp we already
            # know exactly.
            if boundary_index == len(eps):
                boundary = SplitBoundary(
                    predicted=float(row.duration), selected=float(row.duration),
                    black_start=None, black_end=None, black_duration=None,
                    delta=0.0, confidence="medium:eof-runtime", cached=False,
                )
                print(
                    f"        [{completed}/{total}] {ep.season}x{ep.number:02d}: "
                    f"predicted ~{format_duration(row.duration)}; "
                    f"boundary lookup ... {format_duration(row.duration)} "
                    f"[medium:eof-runtime, delta=0.0s]",
                    flush=True,
                )
                boundaries.append(boundary)
                predicted_start = predicted
                continue

            print(
                f"        [{completed}/{total}] {ep.season}x{ep.number:02d}: "
                f"predicted ~{format_duration(predicted)}; boundary lookup ... ",
                end="", flush=True,
            )
            chapter_boundary = chapter_snaps.get(boundary_index)
            if chapter_boundary is not None:
                boundary = chapter_boundary
                delta = f"{boundary.delta:.1f}s" if boundary.delta is not None else "?"
                print(
                    f"{format_duration(boundary.selected or predicted)} "
                    f"[{boundary.confidence}, delta={delta}]",
                    flush=True,
                )
            else:
                boundary = detect_fade_boundary(
                    row.path, current_start=predicted_start, expected_duration=exp,
                    source_duration=row.duration, search_window=search_window,
                    min_black_duration=min_black_duration,
                )
                if boundary.selected is None:
                    print("no authored chapter/fade; runtime fallback" +
                          (" [db-cache]" if boundary.cached else ""), flush=True)
                else:
                    delta = f"{boundary.delta:.1f}s" if boundary.delta is not None else "?"
                    cache_note = ", db-cache" if boundary.cached else ""
                    print(
                        f"{format_duration(boundary.selected)} "
                        f"[{boundary.confidence}, delta={delta}{cache_note}]",
                        flush=True,
                    )
            boundaries.append(boundary)

            # Crucial: predictions are anchored to the independent local runtime
            # model, never to the black timestamp selected above.  This prevents
            # one questionable fade from shifting all subsequent searches.
            predicted_start = predicted

        resolved_boundaries, runtime_fallbacks = _runtime_resolve_aggregate_boundaries(
            eps, expected, boundaries, source_duration=row.duration
        )
        if runtime_fallbacks:
            executable_runtime = sum(
                1 for b in resolved_boundaries
                if b.confidence.startswith("medium:runtime") or b.confidence == "medium:eof-runtime"
            )
            low_runtime = sum(
                1 for b in resolved_boundaries if b.confidence.startswith("low:runtime")
            )
            print(
                f"      runtime fallback: resolved {runtime_fallbacks} boundary/boundaries "
                f"from TMDb runtimes + neighboring anchors "
                f"({executable_runtime} medium, {low_runtime} low)",
                flush=True,
            )
        plans.append(AggregateSplitPlan(
            source=row.path, group=row.group, episodes=tuple(eps),
            source_duration=row.duration, boundaries=tuple(resolved_boundaries),
        ))
    return plans


def _split_segment_rows(plan: AggregateSplitPlan) -> list[tuple[Episode, float, float, SplitBoundary]]:
    rows: list[tuple[Episode, float, float, SplitBoundary]] = []
    start = 0.0
    for ep, boundary in zip(plan.episodes, plan.boundaries):
        if boundary.selected is None:
            break
        end = boundary.selected
        rows.append((ep, start, end, boundary))
        start = end
    return rows


def execute_aggregate_split_plans(
    plans: list[AggregateSplitPlan],
    match: Match,
    destination_dir: Path,
    *,
    mode: int,
) -> int:
    """Stream-copy aggregate MKVs into canonical per-episode MKVs.

    All segments for one source are created and duration-checked in a temporary
    directory before any final destination is committed.  The original source
    is left in place here; the ordinary Extras transfer archives it afterward.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MKVPlexError("ffmpeg was not found in PATH; cannot execute aggregate splits")
    destination_dir.mkdir(parents=True, exist_ok=True, mode=mode)
    created = 0
    for plan in plans:
        if not plan.executable:
            raise MKVPlexError(
                f"Refusing to split {plan.source.name}: one or more fade boundaries "
                "are low-confidence/unresolved. Re-run --dry-run and review them."
            )
        segments = _split_segment_rows(plan)
        if len(segments) != len(plan.episodes):
            raise MKVPlexError(f"Incomplete split plan for {plan.source}")
        with tempfile.TemporaryDirectory(prefix=".mkvplex-split-", dir=destination_dir) as td:
            tempdir = Path(td)
            staged: list[tuple[Path, Path, float]] = []
            for index, (ep, start, end, _boundary) in enumerate(segments, start=1):
                destination = destination_dir / episode_filename(match, ep, plan.source)
                if destination.exists():
                    raise MKVPlexError(f"Refusing to overwrite existing file: {destination}")
                tmp = tempdir / f"segment-{index:03d}.mkv"
                duration = end - start
                cmd = [
                    ffmpeg, "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-i", str(plan.source),
                    "-t", f"{duration:.3f}",
                    "-map", "0", "-map_chapters", "-1", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", str(tmp),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0 or not tmp.exists():
                    detail = (proc.stderr or proc.stdout).strip()
                    raise MKVPlexError(
                        f"ffmpeg split failed for {plan.source.name} episode "
                        f"{ep.season}x{ep.number:02d}: {detail[-500:]}"
                    )
                actual = probe_duration_seconds(tmp)
                if abs(actual - duration) > 15.0:
                    raise MKVPlexError(
                        f"Split verification failed for {ep.season}x{ep.number:02d}: "
                        f"planned {format_duration(duration)}, got {format_duration(actual)}"
                    )
                staged.append((tmp, destination, actual))
            for tmp, destination, _actual in staged:
                os.replace(tmp, destination)
                destination.chmod(mode)
                created += 1
    recursive_chmod(destination_dir, mode)
    return created


def select_episode_tracks_by_epl(
    tracks_with_group: list[tuple[Path, TvRipGroup]],
    episodes: list[Episode],
    durations: dict[Path, float],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> Optional[tuple[list[tuple[Path, TvRipGroup]], list[tuple[Path, float]]]]:
    """Select episode tracks by EPL ordinal when a complete 1..N set exists.

    Runtime remains the discriminator when more than one track has the same
    EPL ordinal.  Returning None means the hints are incomplete and callers
    should fall back to order-preserving runtime matching.
    """
    if not episodes:
        return ([], [])

    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance = max(1.0, tolerance_minutes) * 60.0
    by_epl: dict[int, list[tuple[Path, TvRipGroup]]] = {}
    for item in tracks_with_group:
        idx = epl_number(item[0])
        if idx is not None:
            by_epl.setdefault(idx, []).append(item)

    needed = range(1, len(episodes) + 1)
    if any(i not in by_epl for i in needed):
        return None

    selected: list[tuple[Path, TvRipGroup]] = []
    used: set[Path] = set()
    for i, ep in enumerate(episodes, start=1):
        candidates = [item for item in by_epl[i] if item[0] not in used]
        if not candidates:
            return None
        best = min(candidates, key=lambda item: abs(durations[item[0]] - expected[i - 1]))
        delta = abs(durations[best[0]] - expected[i - 1])
        if delta > tolerance:
            return None
        selected.append(best)
        used.add(best[0])

    skipped = [(src, durations[src]) for src, _group in tracks_with_group if src not in used]
    return selected, skipped


def contiguous_epl_count(tracks_with_group: list[tuple[Path, TvRipGroup]]) -> Optional[int]:
    """Return N when EPL hints contain a contiguous 1..N sequence."""
    values = {epl_number(src) for src, _group in tracks_with_group}
    values.discard(None)
    if not values or 1 not in values:
        return None
    n = max(values)
    if values.issuperset(range(1, n + 1)):
        return n
    return None


def select_episode_tracks(
    tracks_with_group: list[tuple[Path, TvRipGroup]],
    episodes: list[Episode],
    durations: dict[Path, float],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> tuple[list[tuple[Path, TvRipGroup]], list[tuple[Path, float]]]:
    """Choose the best order-preserving subset of tracks for EPISODES.

    Blu-ray TV discs commonly contain many extras.  We use episode runtimes as
    the signal but never reorder tracks: the selected files remain in physical
    MakeMKV/disc order. Dynamic programming finds the minimum total runtime
    error while allowing arbitrary non-episode tracks to be skipped.
    """
    if not episodes:
        return [], []
    if len(tracks_with_group) < len(episodes):
        raise MKVPlexError(
            f"Only {len(tracks_with_group)} MKV track(s) were found for "
            f"{len(episodes)} episode(s)."
        )

    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance = max(1.0, tolerance_minutes) * 60.0
    n = len(tracks_with_group)
    m = len(episodes)
    inf = float("inf")

    # dp[e][t] = best cost after considering first t tracks and selecting e episodes.
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    prev: list[list[Optional[tuple[int, int, bool]]]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for t in range(n):
        path, _group = tracks_with_group[t]
        actual = durations[path]
        for e in range(m + 1):
            base = dp[e][t]
            if base == inf:
                continue
            # Skip this track. Tiny cost prefers earlier matching tracks when ties occur.
            skip_cost = base + 0.000001
            if skip_cost < dp[e][t + 1]:
                dp[e][t + 1] = skip_cost
                prev[e][t + 1] = (e, t, False)
            if e < m:
                delta = abs(actual - expected[e])
                if delta <= tolerance:
                    # Runtime is the baseline; source filenames sometimes
                    # contain the actual episode title, which is powerful
                    # evidence when runtimes are otherwise indistinguishable.
                    title_sim = episode_title_similarity(path, episodes[e].title)
                    cost = base + (delta / 60.0) ** 2 - (title_sim * 25.0)
                    if cost < dp[e + 1][t + 1]:
                        dp[e + 1][t + 1] = cost
                        prev[e + 1][t + 1] = (e, t, True)

    # We may finish after any number of trailing skipped extras. dp[m][n]
    # already includes those skip transitions.
    if dp[m][n] == inf:
        expected_text = ", ".join(
            f"{ep.number}:{(sec/60):.0f}m" for ep, sec in zip(episodes, expected)
        )
        raise MKVPlexError(
            "Could not find an ordered set of MKV tracks close enough to the "
            f"episode runtimes (tolerance ±{tolerance_minutes:g} min). "
            f"Expected: {expected_text}. Try --runtime-tolerance 20, or use --all-tracks."
        )

    selected_indices: list[int] = []
    e, t = m, n
    while e or t:
        step = prev[e][t]
        if step is None:
            raise MKVPlexError("internal error reconstructing TV runtime match")
        pe, pt, took = step
        if took:
            selected_indices.append(t - 1)
        e, t = pe, pt
    selected_indices.reverse()
    selected_set = set(selected_indices)
    selected = [tracks_with_group[i] for i in selected_indices]
    skipped = [(tracks_with_group[i][0], durations[tracks_with_group[i][0]]) for i in range(n) if i not in selected_set]
    return selected, skipped


def probe_video_packet_fingerprint(path: Path, duration: float) -> tuple[str, list[str]]:
    """Fingerprint compressed video payloads at several points in a title.

    This intentionally hashes video packet payloads rather than the MKV file.
    MakeMKV titles that present the same video with different audio/commentary
    tracks should therefore collapse to the same visual presentation without
    decoding the program or reading it end-to-end.
    """
    db = discovery_db()
    if db is not None:
        cached = db.get_video_fingerprint(path)
        if cached is not None:
            return cached

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise MKVPlexError("ffprobe was not found in PATH; visual presentation clustering requires ffprobe")
    if duration <= 0:
        raise MKVPlexError(f"Cannot visually fingerprint non-positive duration: {path}")

    samples: list[str] = []
    # Avoid openings/end credits because authored titles often share them. Six
    # compressed packets at each position are enough to identify an exact
    # reused video presentation while keeping the probe cheap.
    for ratio in (0.17, 0.53, 0.83):
        start = max(1.0, min(duration - 1.0, duration * ratio))
        cmd = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-read_intervals", f"{start:.3f}%+#6",
            "-show_packets", "-show_entries", "packet=size,data_hash",
            "-show_data_hash", "md5", "-of", "compact=p=0:nk=0", str(path),
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=45)
        except subprocess.TimeoutExpired as exc:
            raise MKVPlexError(f"visual fingerprint probe timed out on {path}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "unknown ffprobe error").strip()
            raise MKVPlexError(f"visual fingerprint probe failed for {path}: {detail[:300]}")
        packet_rows: list[str] = []
        for line in proc.stdout.splitlines():
            m_hash = re.search(r"data_hash=MD5:([0-9a-fA-F]{32})", line)
            m_size = re.search(r"(?:^|\|)size=(\d+)", line)
            if m_hash:
                packet_rows.append(f"{m_size.group(1) if m_size else '?'}:{m_hash.group(1).lower()}")
        if packet_rows:
            samples.append(";".join(packet_rows))

    if not samples:
        raise MKVPlexError(f"ffprobe returned no video packet hashes for {path}")
    digest = hashlib.sha256("|".join(samples).encode("ascii", errors="ignore")).hexdigest()
    if db is not None:
        db.put_video_fingerprint(path, digest, samples)
    return digest, samples


def probe_video_packet_fingerprints(
    rows: list[TrackAnalysis], *, workers: int = 4,
) -> dict[Path, str]:
    """Fingerprint episode-like rows concurrently and return path -> digest."""
    unique: dict[Path, TrackAnalysis] = {row.path: row for row in rows}
    if not unique:
        return {}
    workers = max(1, min(int(workers), 32, len(unique)))
    results: dict[Path, str] = {}
    print(f"  Visual-content fingerprinting {len(unique)} episode/long-form candidate title(s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(probe_video_packet_fingerprint, row.path, row.duration): row.path
            for row in unique.values()
        }
        done = 0
        total = len(future_map)
        for future in concurrent.futures.as_completed(future_map):
            path = future_map[future]
            digest, _samples = future.result()
            results[path] = digest
            done += 1
            if done == total or done % 10 == 0:
                print(f"    visual fingerprints: {done}/{total}")
    return results


def _presentation_stream_profile(path: Path) -> dict[str, int]:
    """Cheaply describe audio/subtitle richness for equivalent-video titles."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"audio": 0, "non_commentary_audio": 0, "commentary_audio": 0, "subtitles": 0}
    cmd = [
        ffprobe, "-v", "error", "-show_entries",
        "stream=codec_type:stream_tags=title,language", "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            raise ValueError(proc.stderr)
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return {"audio": 0, "non_commentary_audio": 0, "commentary_audio": 0, "subtitles": 0}
    audio = commentary = subtitles = 0
    for row in data.get("streams", []) or []:
        kind = str(row.get("codec_type") or "")
        if kind == "audio":
            audio += 1
            tags = row.get("tags") or {}
            title = normalize_for_match(str(tags.get("title") or ""))
            if "commentary" in title:
                commentary += 1
        elif kind == "subtitle":
            subtitles += 1
    return {
        "audio": audio,
        "non_commentary_audio": max(0, audio - commentary),
        "commentary_audio": commentary,
        "subtitles": subtitles,
    }


def _packet_sample_hash_groups(samples: list[str]) -> list[list[str]]:
    """Extract the per-seek video packet MD5 groups stored by the visual cache."""
    groups: list[list[str]] = []
    for sample in samples:
        hashes: list[str] = []
        for item in str(sample).split(";"):
            match = re.search(r"(?:^|:)([0-9a-fA-F]{32})$", item.strip())
            if match:
                hashes.append(match.group(1).lower())
        if hashes:
            groups.append(hashes)
    return groups


def _scan_master_packet_hash_positions(
    master: Path, target_hashes: set[str], *, timeout: int = 600
) -> dict[str, list[float]]:
    """Locate selected compressed-video packet hashes inside a play-all master.

    MakeMKV remuxes DVD/Blu-ray elementary video without re-encoding it.  Packet
    payload hashes sampled from an individual authored title can therefore be
    searched inside the disc's play-all title.  Only hashes requested by the
    caller are retained, even though ffprobe streams through the master once.
    """
    if not target_hashes:
        return {}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise MKVPlexError(
            "ffprobe was not found in PATH; numbered-volume play-all ordering "
            "requires compressed-video packet matching"
        )
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_packets", "-show_entries", "packet=pts_time,data_hash",
        "-show_data_hash", "md5", "-of", "compact=p=0:nk=0", str(master),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MKVPlexError(f"play-all content-order scan timed out on {master}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown ffprobe error").strip()
        raise MKVPlexError(
            f"play-all content-order scan failed for {master}: {detail[:400]}"
        )
    hits: dict[str, list[float]] = {value: [] for value in target_hashes}
    for line in proc.stdout.splitlines():
        hash_match = re.search(r"data_hash=MD5:([0-9a-fA-F]{32})", line)
        if not hash_match:
            continue
        digest = hash_match.group(1).lower()
        if digest not in hits:
            continue
        pts_match = re.search(r"(?:^|\|)pts_time=([-+0-9.eE]+)", line)
        if not pts_match:
            continue
        try:
            hits[digest].append(float(pts_match.group(1)))
        except ValueError:
            continue
    return {key: values for key, values in hits.items() if values}


def _component_master_anchor_times(
    component_groups: dict[Path, list[list[str]]],
    hits: dict[str, list[float]],
) -> dict[Path, float]:
    """Convert packet-hit evidence into one robust master position per title.

    A sample group contains adjacent packets from one seek point.  We accept a
    group only when at least two of its unique packet hashes land in one tight
    region of the master.  At least two distinct seek regions must agree and
    increase monotonically within each component.  Repeated video or accidental
    hash matches therefore fail closed instead of manufacturing an order.
    """
    ownership: dict[str, set[Path]] = {}
    for path, groups in component_groups.items():
        for group in groups:
            for digest in group:
                ownership.setdefault(digest, set()).add(path)

    anchors: dict[Path, float] = {}
    for path, groups in component_groups.items():
        region_times: list[float] = []
        for group in groups:
            values: list[float] = []
            for digest in group:
                if ownership.get(digest) != {path}:
                    continue
                values.extend(hits.get(digest, []))
            if len(values) < 2:
                continue
            values.sort()
            # Six adjacent packets normally span far less than a second.  A
            # generous 15-second envelope tolerates odd timestamps while still
            # rejecting the same visual material appearing in distant places.
            if values[-1] - values[0] > 15.0:
                continue
            region_times.append(float(_median(values) or values[len(values)//2]))
        if len(region_times) < 2:
            raise MKVPlexError(
                f"Could not locate at least two unique video regions from {path.name} "
                "inside its play-all master; refusing tNN-based program ordering"
            )
        if any(b <= a for a, b in zip(region_times, region_times[1:])):
            raise MKVPlexError(
                f"Video-region matches for {path.name} are not monotonic inside the play-all master; "
                "repeated/ambiguous content prevents safe program ordering"
            )
        anchors[path] = float(_median(region_times) or region_times[len(region_times)//2])
    return anchors


def playall_component_order_by_video(
    master: TrackAnalysis, components: list[TrackAnalysis]
) -> list[TrackAnalysis]:
    """Order disc components by where their video actually occurs in the master.

    tNN is only a MakeMKV title number and is not an authored playback-order
    guarantee.  For numbered retail volumes we have a much stronger oracle: the
    verified disc play-all title itself.  Sample packet payloads from every
    component, scan the master once for those exact payloads, and sort by their
    observed master positions.  Any ambiguity is fatal rather than falling back
    to tNN order.
    """
    if len(components) < 2:
        return list(components)
    paths = [row.path.resolve() for row in components]
    db = discovery_db()
    if db is not None:
        cached = db.get_master_component_order(master.path, paths)
        if cached is not None:
            by_path = {row.path.resolve(): row for row in components}
            try:
                ordered = [by_path[path.resolve()] for path in cached]
            except KeyError:
                ordered = []
            if len(ordered) == len(components):
                print(f"      play-all content order: {master.path.name} [cached]")
                return ordered

    component_groups: dict[Path, list[list[str]]] = {}
    for row in components:
        _digest, samples = probe_video_packet_fingerprint(row.path, row.duration)
        groups = _packet_sample_hash_groups(samples)
        if len(groups) < 2:
            raise MKVPlexError(
                f"Too little compressed-video fingerprint evidence for {row.path}; "
                "refusing tNN-based numbered-volume ordering"
            )
        component_groups[row.path.resolve()] = groups

    # Only packet hashes unique to one component are useful ordering anchors.
    ownership: dict[str, set[Path]] = {}
    for path, groups in component_groups.items():
        for group in groups:
            for digest in group:
                ownership.setdefault(digest, set()).add(path)
    targets = {digest for digest, owners in ownership.items() if len(owners) == 1}
    if not targets:
        raise MKVPlexError(
            f"Play-all master {master.path.name} has no component-unique packet fingerprints; "
            "refusing tNN-based program ordering"
        )
    print(
        f"      play-all content-order scan: {master.path.name} -> "
        f"{len(components)} component title(s)"
    )
    hits = _scan_master_packet_hash_positions(master.path, targets)
    anchors = _component_master_anchor_times(component_groups, hits)
    ordered = sorted(components, key=lambda row: (anchors[row.path.resolve()], str(row.path).casefold()))

    # Adjacent component anchors should be meaningfully separated inside a
    # multi-program master.  Very close anchors indicate duplicated/repeated
    # content and are not strong enough to establish authored order.
    ordered_anchor = [anchors[row.path.resolve()] for row in ordered]
    minimum_duration = min(row.duration for row in components if row.duration > 0)
    minimum_gap = max(30.0, minimum_duration * 0.15)
    for left, right in zip(ordered_anchor, ordered_anchor[1:]):
        if right - left < minimum_gap:
            raise MKVPlexError(
                f"Play-all master {master.path.name} produced overlapping component anchors "
                f"({right-left:.1f}s apart); refusing ambiguous program order"
            )

    if db is not None:
        db.put_master_component_order(
            master.path, paths, [row.path.resolve() for row in ordered]
        )
    print(
        "      authored program order from play-all video: "
        + " -> ".join(row.path.name for row in ordered)
    )
    return ordered


__all__ = ['probe_video_presentation', '_presentation_scan_component', '_presentation_resolution_component', 'technical_presentation_tag', 'technical_presentation_suffix', 'episode_filename', 'movie_filename', 'probe_duration_seconds', 'probe_container_title', '_rotational_flags_for_device', 'detect_storage_class', 'resolve_media_workers', 'probe_durations', 'probe_chapter_boundaries', '_chapter_confidence', 'select_chapter_snaps', 'probe_black_intervals', '_boundary_confidence', 'detect_fade_boundary', '_aggregate_complete_series_prefix_hint', '_aggregate_episode_sources', 'select_aggregate_source_subset', 'aggregate_presentation_runtime_scale', 'allocate_episodes_to_aggregate_sources', '_runtime_resolve_aggregate_boundaries', 'build_aggregate_split_plans', '_split_segment_rows', 'execute_aggregate_split_plans', 'select_episode_tracks_by_epl', 'contiguous_epl_count', 'select_episode_tracks', 'probe_video_packet_fingerprint', 'probe_video_packet_fingerprints', '_presentation_stream_profile', '_packet_sample_hash_groups', '_scan_master_packet_hash_positions', '_component_master_anchor_times', 'playall_component_order_by_video']
