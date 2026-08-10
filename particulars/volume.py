"""Numbered retail-volume inference and chapter-topology refinement."""
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
from .models import AggregateSplitPlan, COMPLETE_SERIES_SENTINEL, Episode, EpisodeAssignment, JUNK_TOKENS, MKVPlexError, Match, SkippedTrack, SplitBoundary, TrackAnalysis, TvRipGroup
from .common import _group_sort_key, _median, episode_expected_seconds
from .naming import canonical_spaces, normalize_for_match, similarity
from .media import _split_segment_rows, build_aggregate_split_plans, playall_component_order_by_video, probe_chapter_boundaries, select_chapter_snaps
from .discs import _EPISODE_ORDINAL_RANGE_RE
from .tmdb import TMDbClient, _volume_order_sequences

SERIES_LABEL_JUNK = {
    "complete", "collection", "box", "set", "edition", "dvd", "disc", "disk",
    "bluray", "blu-ray", "uhd", "season", "seasons",
}


GENERIC_COLLECTION_DIRS = {
    "big", "extras", "extra", "bonus", "bonus material", "special features",
    "special feature", "supplements", "supplement",
    # Generic staging labels left behind by manual ripping workflows.  These
    # are only treated as generic when the *entire* residual component is the
    # token (e.g. New_Disc1 -> "New").  Names such as "New Blood" remain
    # meaningful series-title components.
    "new",
}


def _series_title_component(name: str) -> str:
    """Return only the show-name portion of one rip-directory component.

    Unlike parse_source_name(), this deliberately returns an empty string when
    a component is only ``Season 2``/``Disc 1`` rather than falling back to the
    raw component name.  That makes nested layouts such as Show/Season 2/Disc 1
    safe for mixed-series detection.
    """
    work = name.translate(str.maketrans({"™": "", "®": "", "©": ""}))
    # Packaging spans such as "Rose of Versailles 11-20" describe episode
    # ordinals, not a distinct series title.  Strip them before mixed-series
    # title inference.
    work = _EPISODE_ORDINAL_RANGE_RE.sub(" ", work)
    work = re.sub(r"(?i)\((?:19|20|21)\d{2}\)|\[(?:19|20|21)\d{2}\]", " ", work)
    work = re.sub(r"(?i)(?<!\d)(?:19|20|21)\d{2}(?!\d)", " ", work)
    work = re.sub(r"(?i)(?<![A-Za-z0-9])final[ ._-]*season(?![A-Za-z0-9])", " ", work)
    work = re.sub(r"(?i)(?<![A-Za-z0-9])(?:S|Season[ ._-]*)(?:0*\d{1,2})(?!\d)", " ", work)
    # Retail TV sets often use numbered packaging volumes that are unrelated
    # to the provider's season numbering, e.g. ``PINKY AND THE BRAIN VOL 2
    # DISC 1``.  Treat only an explicit numbered Vol/Volume marker as packaging
    # metadata here; the physical disc number is still parsed separately.
    work = re.sub(
        r"(?i)(?<![A-Za-z0-9])vol(?:ume)?\.?[ ._-]*0*\d{1,3}(?!\d)",
        " ", work,
    )
    work = re.sub(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"(?:blu[ ._-]*ray|bluray)[ ._-]*(?:disc|disk)?|"
        r"bd|br|dvd|disc|disk|d"
        r")[ ._-]*0*\d{1,2}(?!\d)",
        " ", work,
    )
    work = re.sub(r"[._]+", " ", work)
    work = re.sub(r"\s*-\s*", " ", work)
    work = re.sub(r"[,;]+", " ", work)

    kept: list[str] = []
    for word in work.split():
        cleaned = re.sub(r"[^A-Za-z0-9-]", "", word).lower()
        if not cleaned:
            continue
        if cleaned in JUNK_TOKENS or cleaned in SERIES_LABEL_JUNK:
            continue
        if re.fullmatch(r"(?i)(?:x|h)26[45]", cleaned):
            continue
        kept.append(word)
    return canonical_spaces(" ".join(kept))


def _group_series_query(input_dir: Path, root_query: str, group: TvRipGroup) -> str:
    """Infer the logical show title represented by one physical rip directory."""
    rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
    candidates: list[str] = []
    if rel != Path("."):
        for component in reversed(rel.parts):
            title = _series_title_component(component)
            if title:
                candidates.append(title)
    else:
        title = _series_title_component(group.directory.name)
        if title:
            candidates.append(title)

    if not candidates:
        return root_query

    candidate = candidates[0]
    if normalize_for_match(candidate) in GENERIC_COLLECTION_DIRS:
        return root_query
    if similarity(candidate, root_query) >= 0.88:
        return root_query

    # A nested subtree may be named merely "New Blood" beneath a root named
    # "Dexter".  Prefixing the root produces the useful TMDb query
    # "Dexter New Blood" without affecting self-contained names such as
    # DEXTER_NEW_BLOOD_DISC_1, which already contain the root title.
    root_norm = normalize_for_match(root_query)
    cand_norm = normalize_for_match(candidate)
    root_tokens = set(root_norm.split())
    cand_tokens = set(cand_norm.split())
    if root_tokens and root_tokens.isdisjoint(cand_tokens):
        combined = canonical_spaces(f"{root_query} {candidate}")
        return combined
    return candidate


_NUMBERED_VOLUME_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])vol(?:ume)?\.?[ ._-]*0*(\d{1,3})(?!\d)"
)


def numbered_volume_from_name(name: str) -> Optional[int]:
    """Return a retail packaging volume number such as Vol 2 / Volume 02."""
    m = _NUMBERED_VOLUME_RE.search(name)
    if not m:
        return None
    value = int(m.group(1))
    return value if value > 0 else None


def common_numbered_volume(input_dir: Path, groups: list[TvRipGroup]) -> Optional[int]:
    """Return one volume number only when every physical rip group agrees.

    This deliberately does not reinterpret a volume as a TV season.  It is
    merely a signal that the input may be a partial retail slice spanning
    provider seasons.
    """
    values: list[int] = []
    for group in groups:
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(group.directory.name)
        local = [numbered_volume_from_name(part) for part in rel.parts]
        local = [value for value in local if value is not None]
        if not local:
            root_value = numbered_volume_from_name(input_dir.name)
            if root_value is not None:
                local = [root_value]
        if len(set(local)) != 1:
            return None
        values.append(local[0])
    return values[0] if values and len(set(values)) == 1 else None


def _program_expected_seconds(
    episodes: list[Episode], show_runtime_minutes: Optional[float]
) -> float:
    return sum(episode_expected_seconds(episodes, show_runtime_minutes))


def _partition_same_date_rows(
    rows: list[Episode], target_seconds: float, show_runtime_minutes: Optional[float]
) -> list[list[Episode]]:
    """Split an unusually large same-air-date run into program-sized chunks."""
    if not rows:
        return []
    if len(rows) == 1 or target_seconds <= 0:
        return [list(rows)]
    values = episode_expected_seconds(rows, show_runtime_minutes)
    if sum(values) <= target_seconds * 1.55:
        return [list(rows)]
    out: list[list[Episode]] = []
    current: list[Episode] = []
    current_total = 0.0
    for ep, value in zip(rows, values):
        if current:
            before = abs(current_total - target_seconds)
            after = abs((current_total + value) - target_seconds)
            if current_total >= target_seconds * 0.72 and after > before:
                out.append(current)
                current = []
                current_total = 0.0
        current.append(ep)
        current_total += value
    if current:
        out.append(current)
    return out


def episodes_to_program_units(
    episodes: list[Episode], *, target_seconds: float,
    show_runtime_minutes: Optional[float] = None,
) -> list[list[Episode]]:
    """Collapse segment-level metadata into broadcast/DVD program units.

    Consecutive TMDb episodes sharing the same exact air date are treated as
    one physical program when their combined runtime fits the observed DVD
    program envelope.  Canonical Episode identities are retained inside each
    unit so a multi-segment physical title can later be split back to Plex SxE
    files.
    """
    if not episodes:
        return []
    groups: list[list[Episode]] = []
    current: list[Episode] = []
    current_date: Optional[str] = None
    for ep in episodes:
        if ep.air_date and current and ep.air_date == current_date:
            current.append(ep)
            continue
        if current:
            groups.extend(_partition_same_date_rows(current, target_seconds, show_runtime_minutes))
        current = [ep]
        current_date = ep.air_date
    if current:
        groups.extend(_partition_same_date_rows(current, target_seconds, show_runtime_minutes))
    return groups


def _ordered_volume_program_rows(
    groups: list[TvRipGroup], analyses: list[TrackAnalysis]
) -> tuple[list[TrackAnalysis], dict[Path, Path]]:
    """Recover individual authored program titles from disc-local play-all masters.

    Returns physical program rows in disc/play-all-authored order plus program->master map.
    Every disc must expose one aggregate whose component set is the authored
    program inventory; otherwise the numbered-volume hypothesis is not strong
    enough to execute automatically.
    """
    by_dir: dict[Path, list[TrackAnalysis]] = {}
    for row in analyses:
        by_dir.setdefault(row.group.directory, []).append(row)
    ordered: list[TrackAnalysis] = []
    owner_master: dict[Path, Path] = {}
    for group in sorted(groups, key=lambda g: _group_sort_key(g, COMPLETE_SERIES_SENTINEL)):
        rows = by_dir.get(group.directory, [])
        masters = [row for row in rows if len(row.aggregate_of) >= 2]
        if not masters:
            raise MKVPlexError(
                f"Numbered volume {group.directory.name!r} has no structurally verified play-all title; "
                "refusing to guess program membership from tNN order alone."
            )
        master = max(masters, key=lambda row: (len(row.aggregate_of), row.duration, row.size_bytes))
        component_paths = set(master.aggregate_of)
        components = [row for row in rows if row.path in component_paths]
        if len(components) != len(component_paths):
            raise MKVPlexError(f"Internal error resolving play-all components for {master.path}")
        components = playall_component_order_by_video(master, components)
        for row in components:
            owner_master[row.path] = master.path
        ordered.extend(components)
    return ordered, owner_master


def _volume_alignment_cost(
    sources: list[TrackAnalysis], units: list[list[Episode]],
    *, show_runtime_minutes: Optional[float]
) -> tuple[float, float]:
    if len(sources) != len(units) or not sources:
        return float("inf"), float("inf")
    expected = [_program_expected_seconds(unit, show_runtime_minutes) for unit in units]
    ratios = [src.duration / max(exp, 1.0) for src, exp in zip(sources, expected) if src.duration > 0 and exp > 0]
    scale = _median(ratios) or 1.0
    errors = [abs(math.log(max(src.duration, 1.0) / max(exp * scale, 1.0))) for src, exp in zip(sources, expected)]
    return sum(error * error for error in errors) / len(errors), scale


def _program_chapter_topology_fit(
    source: TrackAnalysis, unit: list[Episode],
    *, show_runtime_minutes: Optional[float],
) -> dict[str, Any]:
    """Score how well authored MKV chapters support a multi-segment program.

    Extra chapters are deliberately *not* evidence against a single-segment
    episode: DVDs commonly carry scene chapters inside an ordinary episode.
    The useful asymmetric signal is the opposite one.  If provider metadata
    says a physical title contains two or more canonical segments, there should
    be an authored boundary near each runtime-weighted segment cut.  MakeMKV
    preserves those DVD chapter timestamps even when it does not preserve human
    episode names.

    Missing expected cuts are expensive.  Close chapter snaps are cheap.  This
    score is used only to refine a small disc-local ordering ambiguity after the
    retail-volume window itself has already been established.
    """
    required = max(0, len(unit) - 1)
    if required == 0:
        return {
            "required": 0, "matched": 0, "missing": 0, "cost": 0.0,
            "predicted": [], "snaps": {}, "chapters": [],
        }

    expected = episode_expected_seconds(unit, show_runtime_minutes)
    total = sum(expected)
    if total <= 0.0 or source.duration <= 0.0:
        return {
            "required": required, "matched": 0, "missing": required,
            "cost": float(required), "predicted": [], "snaps": {}, "chapters": [],
        }

    predicted: list[float] = []
    running = 0.0
    for value in expected[:-1]:
        running += value
        predicted.append(float(source.duration) * (running / total))

    chapters = probe_chapter_boundaries(source.path, source_duration=source.duration)
    snaps = select_chapter_snaps(predicted, chapters, max_delta=100.0)
    matched = len(snaps)
    missing = required - matched
    # One genuinely missing authored segment boundary must dominate tiny runtime
    # differences between neighboring ~21-minute TV programs.  Matched cuts add
    # only a small delta cost, preserving preference for frame-near authored
    # markers without overfitting ordinary scene chapters.
    cost = float(missing)
    for snap in snaps.values():
        cost += 0.25 * (min(100.0, float(snap.delta)) / 100.0) ** 2
    return {
        "required": required, "matched": matched, "missing": missing,
        "cost": cost, "predicted": predicted, "snaps": snaps,
        "chapters": chapters,
    }


def _unit_identity_text(unit: list[Episode]) -> str:
    if not unit:
        return "empty unit"
    if len(unit) == 1:
        ep = unit[0]
        return f"{ep.season}x{ep.number:02d} {ep.title}"
    first, last = unit[0], unit[-1]
    return (
        f"{first.season}x{first.number:02d} {first.title} / "
        f"{last.season}x{last.number:02d} {last.title}"
    )


def _refine_volume_units_by_chapter_topology(
    programs: list[TrackAnalysis], units: list[list[Episode]],
    *, show_runtime_minutes: Optional[float],
) -> tuple[list[list[Episode]], list[dict[str, Any]]]:
    """Apply only decisive adjacent disc-local swaps from authored chapters.

    The play-all packet scan has already proven physical program order.  This
    function therefore never reorders source files.  It asks a narrower
    question: when provider numeric order assigns two adjacent metadata program
    units to those physical titles, do the MKV-authored segment boundaries prove
    that the two metadata identities are reversed?

    A swap requires strictly more matched expected cuts, strictly fewer missing
    expected cuts, a large topology-cost improvement, and no material runtime
    contradiction.  Single-vs-single ambiguity is untouched.  This keeps the
    correction local and fail-closed instead of turning repetitive TV runtimes
    into a general permutation engine.
    """
    if len(programs) != len(units):
        raise MKVPlexError("Internal error: volume program/unit cardinality mismatch")
    refined = [list(unit) for unit in units]
    changes: list[dict[str, Any]] = []

    # One forward pass is intentional.  A metadata unit may move by at most one
    # physical program position per plan, which is enough to correct a local DVD
    # ordering discrepancy while preventing long-range chapter-driven shuffles.
    for i in range(len(refined) - 1):
        left_src, right_src = programs[i], programs[i + 1]
        if left_src.group.directory != right_src.group.directory:
            continue
        left_unit, right_unit = refined[i], refined[i + 1]
        if len(left_unit) <= 1 and len(right_unit) <= 1:
            continue
        if len(left_unit) == len(right_unit):
            # Equal segment cardinality is usually too ambiguous for a generic
            # local swap; retain provider order unless a future evidence layer
            # can distinguish the actual segment identities directly.
            continue

        current_left = _program_chapter_topology_fit(
            left_src, left_unit, show_runtime_minutes=show_runtime_minutes
        )
        current_right = _program_chapter_topology_fit(
            right_src, right_unit, show_runtime_minutes=show_runtime_minutes
        )
        swapped_left = _program_chapter_topology_fit(
            left_src, right_unit, show_runtime_minutes=show_runtime_minutes
        )
        swapped_right = _program_chapter_topology_fit(
            right_src, left_unit, show_runtime_minutes=show_runtime_minutes
        )
        current_cost = float(current_left["cost"]) + float(current_right["cost"])
        swapped_cost = float(swapped_left["cost"]) + float(swapped_right["cost"])
        current_matched = int(current_left["matched"]) + int(current_right["matched"])
        swapped_matched = int(swapped_left["matched"]) + int(swapped_right["matched"])
        current_missing = int(current_left["missing"]) + int(current_right["missing"])
        swapped_missing = int(swapped_left["missing"]) + int(swapped_right["missing"])

        current_runtime, _ = _volume_alignment_cost(
            [left_src, right_src], [left_unit, right_unit],
            show_runtime_minutes=show_runtime_minutes,
        )
        swapped_runtime, _ = _volume_alignment_cost(
            [left_src, right_src], [right_unit, left_unit],
            show_runtime_minutes=show_runtime_minutes,
        )
        decisive = (
            swapped_matched > current_matched
            and swapped_missing < current_missing
            and current_cost - swapped_cost >= 0.75
            and swapped_runtime <= current_runtime + 0.01
        )
        if not decisive:
            continue

        refined[i], refined[i + 1] = right_unit, left_unit
        best_snap = None
        for fit in (swapped_left, swapped_right):
            for snap in fit["snaps"].values():
                if best_snap is None or float(snap.delta) < float(best_snap.delta):
                    best_snap = snap
        changes.append({
            "left_source": left_src.path, "right_source": right_src.path,
            "before_left": _unit_identity_text(left_unit),
            "before_right": _unit_identity_text(right_unit),
            "after_left": _unit_identity_text(right_unit),
            "after_right": _unit_identity_text(left_unit),
            "topology_improvement": current_cost - swapped_cost,
            "chapter_delta": float(best_snap.delta) if best_snap is not None else None,
        })
    return refined, changes


def _known_volume_balanced_start(
    *, total: int, count: int, volume_number: int, volume_total: int
) -> Optional[int]:
    """Balanced start for a known-width numbered retail volume.

    The selected volume's observed physical width is evidence; do not compute
    its boundary as though every volume had the same fractional width.  Spread
    the remaining provider programs across the other volumes as evenly as
    possible, assigning indivisible remainder programs to earlier retail
    volumes first.  This is deterministic and preserves source order.

    Example: 66 provider program units with an observed 21-program Volume 2
    across 3 volumes leaves 45 units for Volumes 1 and 3 -> 23/22, so Volume 2
    starts at zero-based program 23 rather than round(66/3) == 22.
    """
    if total <= 0 or count <= 0 or volume_total <= 0:
        return None
    if not (1 <= volume_number <= volume_total) or count > total:
        return None
    if volume_total == 1:
        return 0 if volume_number == 1 and count == total else None
    remaining = total - count
    other_slots = volume_total - 1
    base, extra = divmod(remaining, other_slots)
    start = 0
    other_index = 0
    for number in range(1, volume_total + 1):
        if number == volume_number:
            return start
        width = base + (1 if other_index < extra else 0)
        if number < volume_number:
            start += width
        other_index += 1
    return None


def infer_numbered_volume_plan(
    client: TMDbClient,
    match: Match,
    *,
    volume_number: int,
    groups: list[TvRipGroup],
    analyses: list[TrackAnalysis],
    show_runtime_minutes: Optional[float] = None,
) -> dict[str, Any]:
    """Infer a partial multi-season retail volume without hard-coding a title.

    The physical side must be strong: every disc needs a verified play-all
    master whose component set identifies the individual program titles.  The
    metadata side is converted from segment-level SxE rows into program units
    using exact air dates.  Volume number supplies a balanced-partition prior;
    runtime fit chooses the best nearby contiguous slice.  Grossly incompatible
    or weakly positioned results fail closed.
    """
    programs, owner_master = _ordered_volume_program_rows(groups, analyses)
    if len(programs) < 2:
        raise MKVPlexError("Numbered-volume analysis found too few authored program titles")
    median_program = _median([row.duration for row in programs]) or 0.0
    if median_program <= 0:
        raise MKVPlexError("Numbered-volume analysis could not determine a physical program runtime")

    options = _volume_order_sequences(client, match)
    # A numbered retail DVD volume follows the provider's DVD ordering whenever
    # a high-coverage DVD group exists.  Do not let a tiny runtime-fit advantage
    # from repetitive ~21-minute programs outvote explicit disc-authored order.
    if "dvd" in options:
        order_keys = ["dvd"]
    elif "chronological" in options:
        # For a physical retail volume, complete provider air chronology is a
        # stronger fallback than numeric SxE order.  If it cannot produce a
        # safe contiguous-volume alignment, fail closed rather than silently
        # reverting to a contradictory numeric ordering.
        order_keys = ["chronological"]
    else:
        order_keys = [key for key in ("regular", "production") if key in options]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for order_key in order_keys:
        order_label, sequence, group_id = options[order_key]
        units = episodes_to_program_units(
            sequence, target_seconds=median_program,
            show_runtime_minutes=show_runtime_minutes,
        )
        total = len(units)
        count = len(programs)
        if total < count or volume_number < 1:
            continue
        approx_volumes = max(volume_number, int(round(total / max(count, 1))))
        volume_counts = range(
            max(volume_number, approx_volumes - 1),
            max(volume_number, approx_volumes - 1) + 3,
        )
        for volume_total in volume_counts:
            average = total / volume_total
            # A numbered retail volume should be close to the average volume
            # size implied by the complete program inventory.  Keep this
            # fairly tight so "Volume 2" cannot be reinterpreted as one of
            # four much smaller slices merely because 21-minute runtimes are
            # repetitive.
            if abs(average - count) > max(3.0, count * 0.15):
                continue
            expected_start = _known_volume_balanced_start(
                total=total, count=count, volume_number=volume_number, volume_total=volume_total
            )
            if expected_start is None:
                continue
            radius = max(3, int(math.ceil(abs(average - count))) + 2)
            for start in range(max(0, expected_start - radius), min(total - count, expected_start + radius) + 1):
                selected = units[start:start + count]
                runtime_cost, scale = _volume_alignment_cost(
                    programs, selected, show_runtime_minutes=show_runtime_minutes
                )
                if not math.isfinite(runtime_cost):
                    continue
                position_penalty = (abs(start - expected_start) / max(count, 1)) ** 2 * 1.50
                count_penalty = (abs(average - count) / max(count, 1)) ** 2 * 0.20
                score = runtime_cost + position_penalty + count_penalty
                candidates.append((score, {
                    "volume": volume_number,
                    "volume_total": volume_total,
                    "order_key": order_key,
                    "order_label": order_label,
                    "order_group_id": group_id,
                    "programs": programs,
                    "owner_master": owner_master,
                    "all_units": units,
                    "units": selected,
                    "start": start,
                    "end": start + count,
                    "expected_start": expected_start,
                    "runtime_cost": runtime_cost,
                    "runtime_scale": scale,
                }))
    if not candidates:
        raise MKVPlexError(
            f"Could not align retail Volume {volume_number} to a contiguous provider program slice"
        )
    candidates.sort(key=lambda item: (item[0], item[1]["order_key"] != "dvd", item[1]["start"]))
    best_score, best = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else float("inf")
    # Runtime cost is mean squared log-ratio after one mastering-scale fit.  A
    # value above ~0.04 corresponds to a broad ~20% per-program mismatch.
    if best["runtime_cost"] > 0.04:
        raise MKVPlexError(
            f"Retail Volume {volume_number} runtime alignment is contradictory "
            f"(cost={best['runtime_cost']:.3f}); refusing to guess the episode window"
        )
    if not (0.70 <= float(best["runtime_scale"]) <= 1.35):
        raise MKVPlexError(
            f"Retail Volume {volume_number} metadata/program runtime scale "
            f"{best['runtime_scale']:.3f}x is implausible. Provider segment rows may not have "
            "collapsed into physical program units; refusing automatic mapping."
        )
    # Position comes from the volume-number partition prior.  If the best fit
    # drifts too far from that prior, the packaging label is not enough evidence.
    if abs(best["start"] - best["expected_start"]) > 4:
        raise MKVPlexError(
            f"Retail Volume {volume_number} best runtime window is too far from its balanced-volume position; "
            "refusing automatic cross-season mapping"
        )
    best["score"] = best_score
    best["margin"] = second_score - best_score if math.isfinite(second_score) else float("inf")
    if best["start"] != best["expected_start"] and best["margin"] < 0.005:
        raise MKVPlexError(
            f"Retail Volume {volume_number} has two nearly-equivalent contiguous program windows "
            "away from the balanced packaging boundary; refusing to shift the volume on weak runtime evidence."
        )

    refined_units, topology_refinements = _refine_volume_units_by_chapter_topology(
        best["programs"], best["units"], show_runtime_minutes=show_runtime_minutes
    )
    if topology_refinements:
        best["units"] = refined_units
        refined_runtime_cost, refined_scale = _volume_alignment_cost(
            best["programs"], best["units"], show_runtime_minutes=show_runtime_minutes
        )
        if refined_runtime_cost > 0.04 or not (0.70 <= float(refined_scale) <= 1.35):
            raise MKVPlexError(
                f"Retail Volume {volume_number} authored chapter-topology refinement contradicts "
                "provider runtimes; refusing the local program permutation"
            )
        best["runtime_cost"] = refined_runtime_cost
        best["runtime_scale"] = refined_scale
        best["order_label"] = best["order_label"] + " + authored MKV chapter-topology refinement"
    best["topology_refinements"] = topology_refinements
    best["allocations"] = [
        (row, list(unit)) for row, unit in zip(best["programs"], best["units"])
    ]
    best["episodes"] = [ep for unit in best["units"] for ep in unit]
    return best


def assign_numbered_volume_plan(
    plan: dict[str, Any],
    analyses: list[TrackAnalysis],
    *,
    show_runtime_minutes: Optional[float],
    split_search_window: float,
    split_black_min: float,
) -> tuple[
    list[EpisodeAssignment], list[SkippedTrack], list[AggregateSplitPlan],
    list[tuple[AggregateSplitPlan, Episode, float, float, SplitBoundary]],
]:
    assignments: list[EpisodeAssignment] = []
    multi: list[tuple[TrackAnalysis, list[Episode]]] = []
    used: set[Path] = set()
    for row, eps in plan["allocations"]:
        used.add(row.path)
        if len(eps) == 1:
            assignments.append(EpisodeAssignment(row.path, row.group, eps[0], row.duration))
        else:
            multi.append((row, eps))
    split_plans = build_aggregate_split_plans(
        multi,
        show_runtime_minutes=show_runtime_minutes,
        search_window=max(30.0, split_search_window),
        min_black_duration=max(0.05, split_black_min),
    ) if multi else []
    aggregate_rows: list[tuple[AggregateSplitPlan, Episode, float, float, SplitBoundary]] = []
    for split_plan in split_plans:
        aggregate_rows.extend(
            (split_plan, ep, start, end, boundary)
            for ep, start, end, boundary in _split_segment_rows(split_plan)
        )
    skipped: list[SkippedTrack] = []
    for row in analyses:
        if row.path in used:
            if any(plan.source == row.path for plan in split_plans):
                skipped.append(SkippedTrack(row.path, row.group, row.duration, "multi-segment source (split master)"))
            continue
        if row.aggregate_of:
            reason = "disc play-all master"
        elif row.duration > (_median([r.duration for r in plan["programs"]]) or 0.0) * 1.25:
            reason = "volume bonus/long-form non-program"
        else:
            reason = "non-program/alternate title"
        skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))
    assignments.sort(key=lambda row: (row.episode.season, row.episode.number))
    return assignments, skipped, split_plans, aggregate_rows


__all__ = ['SERIES_LABEL_JUNK', 'GENERIC_COLLECTION_DIRS', '_series_title_component', '_group_series_query', '_NUMBERED_VOLUME_RE', 'numbered_volume_from_name', 'common_numbered_volume', '_program_expected_seconds', '_partition_same_date_rows', 'episodes_to_program_units', '_ordered_volume_program_rows', '_volume_alignment_cost', '_program_chapter_topology_fit', '_unit_identity_text', '_refine_volume_units_by_chapter_topology', '_known_volume_balanced_start', 'infer_numbered_volume_plan', 'assign_numbered_volume_plan']
