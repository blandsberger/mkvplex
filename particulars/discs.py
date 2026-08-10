"""TV disc/rip grouping, structural analysis, and episode assignment."""
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
from .models import DiscHypothesis, Episode, EpisodeAssignment, MKVPlexError, SkippedTrack, TrackAnalysis, TvRipGroup, VIDEO_EXTENSIONS
from .common import _disc_key, _median, _relative_delta, episode_expected_seconds
from .naming import episode_title_similarity, epl_number, parse_source_name, track_number

def analyze_tv_tracks(
    tracks_with_group: list[tuple[Path, TvRipGroup]],
    episodes: list[Episode],
    durations: dict[Path, float],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> list[TrackAnalysis]:
    """Classify TV tracks using duration, byte size and disc-local bitrate.

    v0.7 deliberately treats these as structural hints rather than provider
    truth.  The strongest classification is aggregate/play-all detection: a
    title is an aggregate when both its duration and byte size closely equal
    the sum of two or more other long titles on the same disc.

    Episode-length bonus material is harder.  On authored Blu-rays, episode
    titles on one disc often form a tight apparent-bitrate family.  When at
    least two peers form a dominant family, a radically different long title
    is marked as a probable bitrate outlier and excluded from automatic
    episode assignment (but still preserved in Extras).
    """
    if not tracks_with_group:
        return []
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance = max(1.0, tolerance_minutes) * 60.0
    low = max(10.0 * 60.0, min(expected) - tolerance)
    high = max(expected) + tolerance

    rows: list[TrackAnalysis] = []
    for path, group in tracks_with_group:
        duration = durations[path]
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        bitrate = (size * 8.0 / duration / 1_000_000.0) if duration > 0 and size > 0 else 0.0
        rows.append(TrackAnalysis(path, group, duration, size, bitrate))

    # Detect disc-local aggregate/play-all titles.  Restrict components to
    # long-form material so random collections of short featurettes do not
    # accidentally add up to an episode.
    by_dir: dict[Path, list[TrackAnalysis]] = {}
    for row in rows:
        by_dir.setdefault(row.group.directory, []).append(row)

    aggregate_map: dict[Path, tuple[Path, ...]] = {}
    min_component = max(20.0 * 60.0, low * 0.45)
    for group_rows in by_dir.values():
        long_rows = [r for r in group_rows if r.duration >= min_component and r.size_bytes > 0]
        for target in long_rows:
            others = [r for r in long_rows if r.path != target.path and r.duration < target.duration]
            best: Optional[tuple[float, tuple[TrackAnalysis, ...]]] = None
            # Small DVD/BD discs can expose six or more individual program
            # titles plus one play-all title.  Earlier versions stopped at five
            # components, which missed perfectly exact 6-program masters (for
            # example a 4-disc retail volume with 6/5/5/5 programs).  Search up
            # to eight components only when the local candidate set is small;
            # keep the old cap for dense supplement discs to avoid combinatorial
            # growth.
            combo_cap = 8 if len(others) <= 10 else 5
            for k in range(2, min(combo_cap, len(others)) + 1):
                for combo in itertools.combinations(others, k):
                    dur_sum = sum(r.duration for r in combo)
                    size_sum = sum(r.size_bytes for r in combo)
                    dur_tol = max(12.0, min(120.0, target.duration * 0.008))
                    size_tol = max(64 * 1024 * 1024, target.size_bytes * 0.008)
                    if abs(dur_sum - target.duration) > dur_tol:
                        continue
                    if abs(size_sum - target.size_bytes) > size_tol:
                        continue
                    error = (abs(dur_sum - target.duration) / dur_tol) + (abs(size_sum - target.size_bytes) / size_tol)
                    if best is None or error < best[0]:
                        best = (error, combo)
            if best is not None:
                aggregate_map[target.path] = tuple(r.path for r in best[1])

    rows = [replace(r, aggregate_of=aggregate_map.get(r.path, ())) for r in rows]

    # Disc-local apparent-bitrate family.  This is intentionally conservative:
    # it only activates with at least three episode-length non-aggregate titles
    # and a family of at least two peers.
    outliers: set[Path] = set()
    for group_rows in by_dir.values():
        current = [next(r for r in rows if r.path == old.path) for old in group_rows]
        candidates = [
            r for r in current
            if not r.aggregate_of and low <= r.duration <= high and r.bitrate_mbps > 0
        ]
        if len(candidates) < 3:
            continue
        best_cluster: list[TrackAnalysis] = []
        for center in candidates:
            cluster = [r for r in candidates if _relative_delta(r.bitrate_mbps, center.bitrate_mbps) <= 0.14]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
            elif len(cluster) == len(best_cluster) and cluster:
                # Prefer the higher-bitrate family when counts tie.  Long bonus
                # pieces are commonly encoded at a substantially lower rate.
                if _median([r.bitrate_mbps for r in cluster]) > _median([r.bitrate_mbps for r in best_cluster]):
                    best_cluster = cluster
        if len(best_cluster) < 2:
            continue
        center = _median([r.bitrate_mbps for r in best_cluster]) or 0.0
        if center <= 0:
            continue
        for row in candidates:
            if row in best_cluster:
                continue
            if _relative_delta(row.bitrate_mbps, center) <= 0.25:
                continue
            # Filename/title evidence can rescue a legitimate unusual encode.
            title_evidence = max((episode_title_similarity(row.path, ep.title) for ep in episodes), default=0.0)
            if title_evidence < 0.70 and epl_number(row.path) is None:
                outliers.add(row.path)

    return [replace(r, bitrate_outlier=(r.path in outliers)) for r in rows]


def _episode_candidate_rows(
    analyses: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> list[TrackAnalysis]:
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance = max(1.0, tolerance_minutes) * 60.0
    low = min(expected) - tolerance
    high = max(expected) + tolerance
    return [
        r for r in analyses
        if not r.aggregate_of
        and not r.bitrate_outlier
        and low <= r.duration <= high
    ]


def _disc_section_label(path: Path) -> Optional[tuple[str, int]]:
    """Return authored section labels such as A4/B1 from MakeMKV filenames.

    Some supplement discs expose menu/section identifiers rather than episode
    names.  A dense A/B/C/D... family across long and short tracks is strong
    evidence that the physical disc is a bonus-program index, not an episode
    sequence.  The trailing MakeMKV _tNN title number is deliberately ignored.
    """
    stem = re.sub(r"(?i)[ _-]*t\d+.*$", "", path.stem).strip()
    m = re.fullmatch(r"(?i)([A-Z])\s*0*(\d{1,2})", stem)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def classify_tv_disc_hypothesis(
    group: TvRipGroup,
    analyses: list[TrackAnalysis],
    candidates: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> DiscHypothesis:
    """Classify a physical TV rip directory before assigning episode slots.

    Runtime coincidence alone is intentionally insufficient.  The bonus score
    relies on whole-disc structure: many short programs, weak episode-title/EPL
    evidence, and especially dense authored section labels (A4, B1, B2, C1...).
    High scores are safe to archive as a bonus disc; medium scores require an
    explicit --disc-kind override rather than manufacturing episode mappings.
    """
    rows = [r for r in analyses if r.group.directory == group.directory]
    cands = [r for r in candidates if r.group.directory == group.directory]
    if not rows or not episodes:
        return DiscHypothesis("episodes", "low", 0, ())

    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance = max(1.0, tolerance_minutes) * 60.0
    low = max(10.0 * 60.0, min(expected) - tolerance)
    short_rows = [r for r in rows if r.duration < low]
    labels = [label for r in rows if (label := _disc_section_label(r.path)) is not None]
    letters = {letter for letter, _n in labels}
    label_fraction = len(labels) / len(rows)
    epl_hits = sum(1 for r in cands if epl_number(r.path) is not None)
    title_signal = max(
        (episode_title_similarity(r.path, ep.title) for r in cands for ep in episodes),
        default=0.0,
    )
    nums = [track_number(r.path)[0] for r in cands]
    nums = [n for n in nums if n < 10**9]

    score = 0
    reasons: list[str] = []
    candidate_fraction = len(cands) / len(rows) if rows else 0.0
    if len(rows) >= 6 and label_fraction >= 0.75 and len(letters) >= 3:
        # Section labels such as A4/B1/C1 are strong bonus-disc evidence only
        # when the disc is also structurally unlike an episode disc.  Anime
        # releases can use dense A/B/C authored labels for ordinary episode
        # material, so a disc where most titles are episode-length must not be
        # condemned as BONUS on labels alone.
        if candidate_fraction <= 0.55 or len(short_rows) >= max(3, len(rows) // 3):
            score += 5
            reasons.append(
                f"authored section labels on {len(labels)}/{len(rows)} tracks "
                f"across {len(letters)} sections"
            )
        else:
            score += 1
            reasons.append(
                f"authored section labels present, but {len(cands)}/{len(rows)} tracks are episode-length"
            )
    elif len(rows) >= 5 and label_fraction >= 0.60 and len(letters) >= 3:
        if candidate_fraction <= 0.55 or len(short_rows) >= max(3, len(rows) // 3):
            score += 3
            reasons.append("strong authored section-label pattern")
        else:
            score += 1
            reasons.append("section labels present on a mostly episode-length disc")

    if len(short_rows) >= 3 and len(short_rows) >= len(rows) / 2:
        score += 2
        reasons.append(f"{len(short_rows)}/{len(rows)} tracks are short-form programs")

    if cands and len(cands) <= 2 and len(rows) >= 6:
        score += 1
        reasons.append(f"only {len(cands)} episode-runtime candidate(s) among {len(rows)} tracks")

    if cands and epl_hits == 0 and title_signal < 0.45:
        score += 2
        reasons.append("episode-length tracks have weak title/EPL evidence")

    if len(nums) >= 2 and max(nums) - min(nums) >= max(4, len(rows) - 2):
        score += 1
        reasons.append("episode-runtime candidates are widely separated in MakeMKV title order")

    # Strong positive episode evidence defeats the bonus hypothesis.
    if epl_hits:
        score -= 6
        reasons.append(f"{epl_hits} candidate(s) carry explicit EPL ordinals")
    if title_signal >= 0.80:
        score -= 5
        reasons.append("strong episode-title filename evidence")

    if score >= 8:
        return DiscHypothesis("bonus", "high", score, tuple(reasons))
    if score >= 6:
        return DiscHypothesis("ambiguous", "medium", score, tuple(reasons))
    return DiscHypothesis("episodes", "normal", score, tuple(reasons))


def _suspected_missing_track_numbers(group: TvRipGroup, candidates: list[TrackAnalysis]) -> list[int]:
    """Return weak evidence for MakeMKV titles that failed to rip.

    This is intentionally only a hint for allocating already-known episode
    holes between discs.  It never creates an episode by itself.
    """
    if not candidates:
        return []
    all_nums = sorted({track_number(p)[0] for p in group.tracks if track_number(p)[0] < 10**9})
    cand_nums = sorted({track_number(r.path)[0] for r in candidates if track_number(r.path)[0] < 10**9})
    if not all_nums or not cand_nums:
        return []
    have = set(all_nums)
    missing: list[int] = []
    lo, hi = min(cand_nums), max(cand_nums)
    for n in range(lo, hi + 1):
        if n not in have:
            missing.append(n)
    # If a disc has t00 material but its first episode-like title begins later,
    # absent numbers between t00 and that first episode are useful evidence.
    if 0 in have and lo > 1:
        for n in range(1, lo):
            if n not in have:
                missing.append(n)
    # Likewise a hole immediately after the episode-like run is meaningful when
    # later bonus titles prove that MakeMKV continued scanning the disc.
    later = [n for n in all_nums if n > hi]
    if later:
        for n in range(hi + 1, min(later)):
            if n not in have:
                missing.append(n)
    return sorted(set(missing))


def infer_disc_slot_template(
    season_rows: dict[int, tuple[list[Episode], list[TvRipGroup], list[TrackAnalysis], list[TrackAnalysis]]]
) -> dict[tuple[int, int], int]:
    """Learn a disc episode-count template from structurally complete seasons.

    For a box set such as Picard, healthy seasons independently reveal the
    authored 3/4/3 episode distribution.  A damaged season can then retain
    holes on the correct physical disc instead of shifting later episodes.
    """
    observations: dict[tuple[int, int], list[int]] = {}
    for _season, (episodes, groups, _analysis, candidates) in season_rows.items():
        if any(g.final_season for g in groups):
            continue
        if len(candidates) != len(episodes):
            continue
        by_group: dict[Path, int] = {}
        for row in candidates:
            by_group[row.group.directory] = by_group.get(row.group.directory, 0) + 1
        for group in groups:
            # A disc deliberately excluded from episode candidacy (for example a
            # detected bonus disc) must not teach a reusable zero-slot template.
            if group.directory not in by_group:
                continue
            observations.setdefault(_disc_key(group), []).append(by_group[group.directory])

    result: dict[tuple[int, int], int] = {}
    for key, values in observations.items():
        if not values:
            continue
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        result[key] = max(sorted(counts), key=lambda value: (counts[value], value))
    return result


def infer_disc_slot_counts(
    groups: list[TvRipGroup],
    candidates: list[TrackAnalysis],
    episode_count: int,
    template: dict[tuple[int, int], int],
) -> tuple[list[int], str]:
    by_dir: dict[Path, list[TrackAnalysis]] = {g.directory: [] for g in groups}
    for row in candidates:
        by_dir.setdefault(row.group.directory, []).append(row)
    base = [len(by_dir.get(g.directory, [])) for g in groups]
    if sum(base) == episode_count:
        return base, "complete-disc-counts"

    templated = [template.get(_disc_key(g)) for g in groups]
    if all(v is not None for v in templated):
        vals = [int(v) for v in templated if v is not None]
        if sum(vals) == episode_count and all(vals[i] >= base[i] for i in range(len(vals))):
            return vals, "learned-disc-template"

    # If there are substantially more episode-length candidates than actual
    # episodes, candidate count is no longer a useful estimate of disc capacity.
    # This happens on authored anime sets with alternate/duplicate long titles.
    # Do not trim a huge first-disc count from the tail (which previously turned
    # 89 candidates / 25 episodes into nonsense such as 20/3/1/1).  Instead use
    # a conservative balanced allocation across every disc that has plausible
    # episode material.  Per-disc matching will choose the best tracks and send
    # the excess to Extras.
    if sum(base) > episode_count:
        active = [i for i, capacity in enumerate(base) if capacity > 0]
        if active:
            counts = [0] * len(base)
            remaining = episode_count
            while remaining > 0:
                eligible = [i for i in active if counts[i] < base[i]]
                if not eligible:
                    break
                idx = min(eligible, key=lambda i: (counts[i], i))
                counts[idx] += 1
                remaining -= 1
            return counts, "overcomplete-candidates/balanced"

    # We know how many episode slots are missing from the season.  Allocate
    # those holes using missing MakeMKV title numbers as weak disc-local evidence.
    counts = list(base)
    missing_total = max(0, episode_count - sum(counts))
    evidence = [_suspected_missing_track_numbers(g, by_dir.get(g.directory, [])) for g in groups]
    for idx, holes in enumerate(evidence):
        if missing_total <= 0:
            break
        add = min(len(holes), missing_total)
        counts[idx] += add
        missing_total -= add

    # Last-resort conservative distribution.  Put unresolved holes on discs
    # with the fewest currently assigned episode slots; this preserves all data
    # and prominently reports ambiguity rather than aborting the season.
    while missing_total > 0 and counts:
        idx = min(range(len(counts)), key=lambda i: (counts[i], i))
        counts[idx] += 1
        missing_total -= 1

    # If filtering left too many candidates, trim nominal slots from the end;
    # the per-disc assignment will leave excess tracks in Extras.
    overflow = sum(counts) - episode_count
    for idx in range(len(counts) - 1, -1, -1):
        if overflow <= 0:
            break
        removable = max(0, counts[idx] - 1)
        take = min(removable, overflow)
        counts[idx] -= take
        overflow -= take
    return counts, "gap-evidence/fallback"


def infer_complete_series_slot_counts(
    groups: list[TvRipGroup], candidates: list[TrackAnalysis], episode_count: int
) -> tuple[list[int], str]:
    """Fill a complete-series corpus in physical disc order.

    When a generic unlabeled tree contains almost exactly as many episode-like
    tracks as the entire regular series, physical source order is stronger
    evidence than balancing counts across discs.  Consume candidate capacity
    from Disc 1 onward and leave only the tail as bonus material.
    """
    by_dir: dict[Path, int] = {g.directory: 0 for g in groups}
    for row in candidates:
        by_dir[row.group.directory] = by_dir.get(row.group.directory, 0) + 1
    counts: list[int] = []
    remaining = episode_count
    for group in groups:
        capacity = by_dir.get(group.directory, 0)
        take = min(capacity, max(0, remaining))
        counts.append(take)
        remaining -= take
    return counts, "complete-series/ordered-fill"


def infer_track_number_direction(
    season_rows: dict[int, tuple[list[Episode], list[TvRipGroup], list[TrackAnalysis], list[TrackAnalysis]]],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> Optional[int]:
    """Learn whether MakeMKV tNN titles run with or against episode order.

    Only structurally complete seasons/discs participate.  We compare the two
    monotonic possibilities (ascending tNN and descending tNN) against TMDb
    runtimes.  A direction is returned only when several discs agree strongly;
    otherwise callers retain the general runtime/title matcher.

    Return +1 for ascending tNN episode order, -1 for descending, or None.
    """
    asc_cost = 0.0
    desc_cost = 0.0
    usable_discs = 0
    expected_by_season: dict[int, list[float]] = {}

    for season, (episodes, groups, _analysis, candidates) in season_rows.items():
        if len(candidates) != len(episodes) or not episodes:
            continue
        expected = expected_by_season.setdefault(
            season, episode_expected_seconds(episodes, show_runtime_minutes)
        )
        by_dir: dict[Path, list[TrackAnalysis]] = {g.directory: [] for g in groups}
        for row in candidates:
            by_dir.setdefault(row.group.directory, []).append(row)

        offset = 0
        for group in groups:
            rows = by_dir.get(group.directory, [])
            count = len(rows)
            disc_eps = episodes[offset: offset + count]
            disc_expected = expected[offset: offset + count]
            offset += count
            if count < 2 or len(disc_eps) != count:
                continue
            nums = [track_number(r.path)[0] for r in rows]
            if any(n >= 10**9 for n in nums) or len(set(nums)) != count:
                continue

            def direction_cost(reverse: bool) -> float:
                ordered = sorted(rows, key=lambda r: track_number(r.path)[0], reverse=reverse)
                total = 0.0
                for row, ep, exp in zip(ordered, disc_eps, disc_expected):
                    delta = abs(row.duration - exp)
                    title_sim = episode_title_similarity(row.path, ep.title)
                    if delta > max(1.0, tolerance_minutes) * 60.0 and title_sim < 0.85:
                        return float("inf")
                    total += (delta / 60.0) ** 2 - title_sim * 4.0
                return total

            a = direction_cost(False)
            d = direction_cost(True)
            if not (math.isfinite(a) and math.isfinite(d)):
                continue
            asc_cost += a
            desc_cost += d
            usable_discs += 1

    if usable_discs < 2:
        return None
    winner = min(asc_cost, desc_cost)
    loser = max(asc_cost, desc_cost)
    # Require a material advantage so random runtime coincidences do not turn
    # tNN into a false global rule for arbitrary box sets.
    if loser <= 0 or winner > loser * 0.70:
        return None
    return 1 if asc_cost < desc_cost else -1


def assign_disc_tracks(
    candidates: list[TrackAnalysis],
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
    track_number_direction: Optional[int] = None,
) -> tuple[list[EpisodeAssignment], list[Episode]]:
    """Minimum-cost one-to-one assignment within one physical disc.

    Track order is not required to equal episode order.  The primary objective
    is to maximize the number of plausible assignments; runtime/title cost is
    secondary.  Unmatched expected episodes are returned as explicit holes.
    """
    if not episodes:
        return [], []
    if not candidates:
        return [], list(episodes)
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance = max(1.0, tolerance_minutes) * 60.0
    n = len(candidates)
    m = len(episodes)

    # If complete discs elsewhere in this box set established a strong MakeMKV
    # tNN direction, use missing tNN values to place holes before consulting the
    # fuzzy runtime assignment.  This is especially valuable when TMDb rounds
    # several neighboring episodes to nearly identical minute runtimes.
    if track_number_direction in (-1, 1):
        numbered: dict[int, TrackAnalysis] = {}
        valid = True
        for row in candidates:
            num = track_number(row.path)[0]
            if num >= 10**9 or num in numbered:
                valid = False
                break
            numbered[num] = row
        if valid and candidates:
            missing_nums = _suspected_missing_track_numbers(candidates[0].group, candidates)
            slot_nums = sorted(
                set(numbered) | set(missing_nums),
                reverse=(track_number_direction < 0),
            )
            if len(slot_nums) == m:
                proposed: list[EpisodeAssignment] = []
                holes: list[Episode] = []
                plausible = True
                for num, ep, exp in zip(slot_nums, episodes, expected):
                    row = numbered.get(num)
                    if row is None:
                        holes.append(ep)
                        continue
                    delta = abs(row.duration - exp)
                    title_sim = episode_title_similarity(row.path, ep.title)
                    if delta > tolerance and title_sim < 0.85:
                        plausible = False
                        break
                    proposed.append(EpisodeAssignment(row.path, row.group, ep, row.duration))
                if plausible and len(proposed) == len(candidates):
                    return proposed, holes

    # Maximum-cardinality, minimum-cost bipartite matching.  Older versions
    # used an episode-subset bitmask DP here, which is exponential in the number
    # of episode slots (2^m) and could appear to hang on overcomplete anime discs.
    # Successive shortest augmenting paths are polynomial for these small graphs
    # and preserve the same objective: maximize plausible matches first, then
    # minimize runtime/title/order cost.
    source = 0
    cand0 = 1
    ep0 = cand0 + n
    sink = ep0 + m
    node_count = sink + 1

    # edge = [to, rev_index, residual_capacity, cost, candidate_index, episode_index]
    graph: list[list[list[object]]] = [[] for _ in range(node_count)]

    def add_edge(u: int, v: int, cost: float, ci: int = -1, ei: int = -1) -> list[object]:
        fwd: list[object] = [v, len(graph[v]), 1, cost, ci, ei]
        rev: list[object] = [u, len(graph[u]), 0, -cost, -1, -1]
        graph[u].append(fwd)
        graph[v].append(rev)
        return fwd

    for ci in range(n):
        add_edge(source, cand0 + ci, 0.0)
    for ei in range(m):
        add_edge(ep0 + ei, sink, 0.0)

    pair_edges: list[tuple[int, int, list[object]]] = []
    for ci, row in enumerate(candidates):
        for ei, ep in enumerate(episodes):
            delta = abs(row.duration - expected[ei])
            title_sim = episode_title_similarity(row.path, ep.title)
            if delta > tolerance and title_sim < 0.85:
                continue
            # +25 is a constant per matched edge, so it does not change ordering
            # among matchings of equal cardinality; it keeps initial edge costs
            # nonnegative while retaining the old runtime/title objective.
            local_cost = (delta / 60.0) ** 2 + (1.0 - title_sim) * 25.0
            if n > 1 and m > 1:
                local_cost += 0.02 * abs((ci / (n - 1)) - (ei / (m - 1)))
            edge = add_edge(cand0 + ci, ep0 + ei, local_cost, ci, ei)
            pair_edges.append((ci, ei, edge))

    # Bellman-Ford is intentionally used instead of a dependency-heavy Hungarian
    # implementation.  Graphs are tiny (dozens of nodes), and residual reverse
    # edges can have negative costs after an augmentation.
    while True:
        inf = float("inf")
        dist = [inf] * node_count
        prev: list[Optional[tuple[int, int]]] = [None] * node_count
        dist[source] = 0.0
        for _ in range(node_count - 1):
            changed = False
            for u in range(node_count):
                if not math.isfinite(dist[u]):
                    continue
                for edge_idx, edge in enumerate(graph[u]):
                    v = int(edge[0])
                    cap = int(edge[2])
                    cost = float(edge[3])
                    if cap <= 0:
                        continue
                    nd = dist[u] + cost
                    if nd + 1e-12 < dist[v]:
                        dist[v] = nd
                        prev[v] = (u, edge_idx)
                        changed = True
            if not changed:
                break
        if prev[sink] is None:
            break
        v = sink
        while v != source:
            u, edge_idx = prev[v]  # type: ignore[misc]
            edge = graph[u][edge_idx]
            rev_idx = int(edge[1])
            edge[2] = int(edge[2]) - 1
            graph[v][rev_idx][2] = int(graph[v][rev_idx][2]) + 1
            v = u

    matched_pairs = [(ci, ei) for ci, ei, edge in pair_edges if int(edge[2]) == 0]
    assignments = [
        EpisodeAssignment(candidates[ci].path, candidates[ci].group, episodes[ei], candidates[ci].duration)
        for ci, ei in matched_pairs
    ]
    assignments.sort(key=lambda a: (a.episode.season, a.episode.number))
    used = {(a.episode.season, a.episode.number) for a in assignments}
    missing = [ep for ep in episodes if (ep.season, ep.number) not in used]
    return assignments, missing


def select_episode_manifest_by_ordinal_ranges(
    groups: list[TvRipGroup],
    episodes: list[Episode],
    analyses: list[TrackAnalysis],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
) -> Optional[tuple[list[EpisodeAssignment], list[Episode], list[SkippedTrack]]]:
    """Map explicitly ranged packaging (1-10, 11-20, ...) to episode ordinals.

    Every participating group must carry a validated range and contain exactly
    one source track per ordinal.  The mapping is then positional within each
    range, with runtime verification as a sanity check.  This path is stronger
    than generic runtime matching because the package itself names the episode
    ordinals.
    """
    if not groups or any(group.episode_span is None for group in groups):
        return None

    range_rows = [
        (group.directory, group.episode_span, len(group.tracks))
        for group in groups if group.episode_span is not None
    ]
    if not _validate_episode_range_corpus(range_rows, require_multiple=len(groups) > 1):
        raise MKVPlexError(
            "Episode-range rip directories are inconsistent: ranges must be contiguous and "
            "each range must contain exactly one MKV per ordinal."
        )

    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    tolerance_seconds = max(1.0, float(tolerance_minutes)) * 60.0
    analysis_by_path = {row.path: row for row in analyses}
    assignments: list[EpisodeAssignment] = []
    used_keys: set[tuple[int, int]] = set()

    for group in sorted(
        groups, key=lambda g: ((g.episode_span or (10**6, 10**6))[0], str(g.directory).casefold())
    ):
        assert group.episode_span is not None
        start, end = group.episode_span
        if end > len(episodes):
            raise MKVPlexError(
                f"Episode-range directory {group.directory.name!r} claims ordinals {start}-{end}, "
                f"but provider metadata contains only {len(episodes)} regular episode(s)."
            )
        target = episodes[start - 1:end]
        tracks = sorted(group.tracks, key=track_number)
        if len(tracks) != len(target):
            raise MKVPlexError(
                f"Episode-range directory {group.directory.name!r} describes {len(target)} ordinal(s) "
                f"but contains {len(tracks)} MKV file(s)."
            )
        for ordinal, (track, ep) in enumerate(zip(tracks, target), start=start):
            row = analysis_by_path.get(track)
            if row is None:
                raise MKVPlexError(f"Missing media analysis for ordinal-range source: {track}")
            exp = expected[ordinal - 1]
            delta = abs(row.duration - exp)
            title_sim = episode_title_similarity(track, ep.title)
            if delta > tolerance_seconds and title_sim < 0.85:
                raise MKVPlexError(
                    f"Episode-range runtime contradiction at ordinal {ordinal}: {track.name!r} is "
                    f"{row.duration / 60.0:.1f}m but {ep.title!r} expects about {exp / 60.0:.1f}m. "
                    "Refusing to trust packaging ordinals against incompatible media."
                )
            assignments.append(EpisodeAssignment(track, group, ep, row.duration))
            used_keys.add((ep.season, ep.number))

    missing = [ep for ep in episodes if (ep.season, ep.number) not in used_keys]
    selected_paths = {assignment.source for assignment in assignments}
    skipped: list[SkippedTrack] = []
    for row in analyses:
        if row.path in selected_paths:
            continue
        reason = "aggregate/play-all" if row.aggregate_of else (
            "episode-length bitrate outlier" if row.bitrate_outlier else "non-ranged source/extra"
        )
        skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))

    assignments.sort(key=lambda a: (a.episode.season, a.episode.number))
    missing.sort(key=lambda ep: (ep.season, ep.number))
    return assignments, missing, skipped


def select_complete_series_manifest_discwise(
    groups: list[TvRipGroup],
    episodes: list[Episode],
    analyses: list[TrackAnalysis],
    candidates: list[TrackAnalysis],
    slot_counts: list[int],
) -> tuple[list[EpisodeAssignment], list[Episode], list[SkippedTrack]]:
    """Map a nearly-complete unlabeled series by physical disc + tNN order.

    This path is intentionally structural.  It is used only for an explicitly
    reviewed complete-series hypothesis, where filenames lack episode titles but
    the source corpus contains almost exactly the regular-series episode count.
    """
    by_dir: dict[Path, list[TrackAnalysis]] = {g.directory: [] for g in groups}
    for row in candidates:
        by_dir.setdefault(row.group.directory, []).append(row)

    assignments: list[EpisodeAssignment] = []
    missing: list[Episode] = []
    offset = 0
    for idx, group in enumerate(groups):
        count = slot_counts[idx] if idx < len(slot_counts) else 0
        disc_eps = episodes[offset:offset + count]
        offset += count
        rows = list(by_dir.get(group.directory, []))
        # Prefer MakeMKV title order when every candidate has a unique tNN.
        nums = [track_number(row.path)[0] for row in rows]
        if rows and all(n < 10**9 for n in nums) and len(set(nums)) == len(nums):
            rows.sort(key=lambda row: track_number(row.path)[0])
        else:
            rows.sort(key=lambda row: str(row.path))
        selected = rows[:count]
        for row, ep in zip(selected, disc_eps):
            assignments.append(EpisodeAssignment(row.path, row.group, ep, row.duration))
        if len(selected) < len(disc_eps):
            missing.extend(disc_eps[len(selected):])
    if offset < len(episodes):
        missing.extend(episodes[offset:])

    selected_paths = {a.source for a in assignments}
    candidate_paths = {r.path for r in candidates}
    skipped: list[SkippedTrack] = []
    for row in analyses:
        if row.path in selected_paths:
            continue
        if row.aggregate_of:
            reason = "aggregate/play-all"
        elif row.bitrate_outlier:
            reason = "episode-length bitrate outlier"
        elif row.path in candidate_paths:
            reason = "complete-series tail / bonus candidate"
        else:
            reason = "short/other"
        skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))
    assignments.sort(key=lambda a: (a.episode.season, a.episode.number))
    missing.sort(key=lambda ep: (ep.season, ep.number))
    return assignments, missing, skipped


def select_episode_manifest_discwise(
    groups: list[TvRipGroup],
    episodes: list[Episode],
    analyses: list[TrackAnalysis],
    candidates: list[TrackAnalysis],
    slot_counts: list[int],
    *,
    show_runtime_minutes: Optional[float] = None,
    tolerance_minutes: float = 12.0,
    track_number_direction: Optional[int] = None,
) -> tuple[list[EpisodeAssignment], list[Episode], list[SkippedTrack]]:
    assignments: list[EpisodeAssignment] = []
    missing: list[Episode] = []
    by_dir: dict[Path, list[TrackAnalysis]] = {g.directory: [] for g in groups}
    for row in candidates:
        by_dir.setdefault(row.group.directory, []).append(row)

    offset = 0
    for idx, group in enumerate(groups):
        count = slot_counts[idx] if idx < len(slot_counts) else 0
        disc_eps = episodes[offset: offset + count]
        offset += count
        disc_candidates = by_dir.get(group.directory, [])
        got, holes = assign_disc_tracks(
            disc_candidates, disc_eps,
            show_runtime_minutes=show_runtime_minutes,
            tolerance_minutes=tolerance_minutes,
            track_number_direction=track_number_direction,
        )
        assignments.extend(got)
        missing.extend(holes)
    if offset < len(episodes):
        missing.extend(episodes[offset:])

    selected_paths = {a.source for a in assignments}
    skipped: list[SkippedTrack] = []
    candidate_paths = {r.path for r in candidates}
    for row in analyses:
        if row.path in selected_paths:
            continue
        if row.aggregate_of:
            reason = "aggregate/play-all"
        elif row.bitrate_outlier:
            reason = "episode-length bitrate outlier"
        elif row.path in candidate_paths:
            reason = "unassigned episode-like"
        else:
            reason = "short/other"
        skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))
    assignments.sort(key=lambda a: (a.episode.season, a.episode.number))
    missing.sort(key=lambda ep: (ep.season, ep.number))
    return assignments, missing, skipped


_EPISODE_ORDINAL_RANGE_RE = re.compile(
    r"(?:^|[\s_])(\d{1,4})\s*[-–—]\s*(\d{1,4})\s*$"
)


def episode_ordinal_span_from_name(name: str) -> Optional[tuple[int, int]]:
    """Return a trailing authored episode-ordinal range such as ``1-10``.

    The range is deliberately conservative: descending/zero ranges, enormous
    spans, and year-like ranges are ignored.  A single range is only a hint;
    auto mode requires multiple contiguous, track-count-compatible ranges
    before treating the shape as strong TV evidence.
    """
    match = _EPISODE_ORDINAL_RANGE_RE.search(str(name).strip())
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end < start:
        return None
    if end - start + 1 > 500:
        return None
    if 1900 <= start <= 2199 and 1900 <= end <= 2199:
        return None
    return start, end


def _episode_span_from_relative_path(input_dir: Path, directory: Path) -> Optional[tuple[int, int]]:
    """Find the nearest trailing episode range in a rip directory path."""
    rel = directory.relative_to(input_dir) if directory != input_dir else Path(directory.name)
    parts = rel.parts if directory != input_dir else (directory.name,)
    for component in reversed(parts):
        span = episode_ordinal_span_from_name(component)
        if span is not None:
            return span
    return None


def _validate_episode_range_corpus(
    rows: list[tuple[Path, tuple[int, int], int]],
    *,
    require_multiple: bool = True,
) -> bool:
    """Validate contiguous ordinal-range groups against their physical track counts."""
    if require_multiple and len(rows) < 2:
        return False
    if not rows:
        return False
    ordered = sorted(rows, key=lambda row: (row[1][0], row[1][1], str(row[0]).casefold()))
    previous_end: Optional[int] = None
    for _directory, (start, end), track_count in ordered:
        if track_count != end - start + 1:
            return False
        if previous_end is not None and start != previous_end + 1:
            return False
        previous_end = end
    return True


def _hints_from_relative_path(input_dir: Path, directory: Path) -> tuple[Optional[int], Optional[int], bool]:
    """Collect season/disc hints from every path component below INPUT.

    This supports both:
        Show/Show Season 2 Disc 1/*.mkv
    and:
        Show/Season 2/Disc 1/*.mkv
    """
    season: Optional[int] = None
    disc: Optional[int] = None
    final_season = False

    rel = directory.relative_to(input_dir)
    for component in rel.parts:
        hints = parse_source_name(Path(component))
        if hints.season is not None:
            season = hints.season
        if hints.disc is not None:
            disc = hints.disc
        if hints.final_season:
            final_season = True
    return season, disc, final_season


def find_tv_rip_groups(input_dir: Path) -> list[TvRipGroup]:
    """Find directories containing MKVs under a TV-series input tree.

    A direct-disc input remains supported.  When INPUT is a series container,
    every descendant directory that directly contains MKVs becomes one disc
    group and is later sequenced by season/disc metadata.
    """
    if not input_dir.is_dir():
        raise MKVPlexError(f"Input is not a directory: {input_dir}")

    direct = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if direct:
        hints = parse_source_name(input_dir)
        return [TvRipGroup(
            directory=input_dir,
            season=hints.season,
            disc=hints.disc,
            final_season=hints.final_season,
            tracks=tuple(sorted(direct, key=track_number)),
            episode_span=episode_ordinal_span_from_name(input_dir.name),
        )]

    groups: list[TvRipGroup] = []
    for dirpath, _dirnames, filenames in os.walk(input_dir):
        directory = Path(dirpath)
        tracks = [
            directory / name for name in filenames
            if Path(name).suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not tracks:
            continue
        season, disc, final_season = _hints_from_relative_path(input_dir, directory)
        groups.append(TvRipGroup(
            directory=directory,
            season=season,
            disc=disc,
            final_season=final_season,
            tracks=tuple(sorted(tracks, key=track_number)),
            episode_span=_episode_span_from_relative_path(input_dir, directory),
        ))

    if not groups:
        raise MKVPlexError(f"No MKV files found in {input_dir} or its subdirectories")
    return groups


def looks_like_tv_tree(input_dir: Path) -> bool:
    """Return True when INPUT has strong TV-series physical structure.

    Explicit season/disc markers remain the strongest simple signal.  We also
    recognize authored episode-range packaging such as ``Show 1-10`` followed
    by ``Show 11-20`` when the ranges are contiguous and each directory
    contains exactly the number of MKVs described by its range.
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        return False

    # Direct season/disc hints still count.
    root_hints = parse_source_name(input_dir)
    if root_hints.season is not None or root_hints.final_season:
        return True

    range_rows: list[tuple[Path, tuple[int, int], int]] = []
    for dirpath, _dirnames, filenames in os.walk(input_dir):
        directory = Path(dirpath)
        if directory == input_dir:
            continue
        mkv_names = [name for name in filenames if Path(name).suffix.lower() in VIDEO_EXTENSIONS]
        if not mkv_names:
            continue
        season, disc, final_season = _hints_from_relative_path(input_dir, directory)
        if season is not None or disc is not None or final_season:
            return True
        span = _episode_span_from_relative_path(input_dir, directory)
        if span is not None:
            range_rows.append((directory, span, len(mkv_names)))

    return _validate_episode_range_corpus(range_rows, require_multiple=True)


__all__ = ['analyze_tv_tracks', '_episode_candidate_rows', '_disc_section_label', 'classify_tv_disc_hypothesis', '_suspected_missing_track_numbers', 'infer_disc_slot_template', 'infer_disc_slot_counts', 'infer_complete_series_slot_counts', 'infer_track_number_direction', 'assign_disc_tracks', 'select_episode_manifest_by_ordinal_ranges', 'select_complete_series_manifest_discwise', 'select_episode_manifest_discwise', '_EPISODE_ORDINAL_RANGE_RE', 'episode_ordinal_span_from_name', '_episode_span_from_relative_path', '_validate_episode_range_corpus', '_hints_from_relative_path', 'find_tv_rip_groups', 'looks_like_tv_tree']
