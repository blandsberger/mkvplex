"""TV-series plan construction, semantic signatures, and split execution."""
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
from .models import AggregateSplitPlan, COMPLETE_SERIES_SENTINEL, DiscHypothesis, Episode, EpisodeAssignment, MKVPlexError, Match, SkippedTrack, SplitBoundary, TrackAnalysis, Transfer, TvRipGroup, VERSION
from .common import _group_effective_season, _group_sort_key, _median, episode_expected_seconds, format_duration
from .discovery import _plan_key, _tv_plan_settings, discovery_db, remap_episode_seasons
from .naming import episode_match_confidence, epl_number, extras_group_directory, tv_directory_name
from .media import _aggregate_episode_sources, _split_segment_rows, allocate_episodes_to_aggregate_sources, build_aggregate_split_plans, contiguous_epl_count, episode_filename, execute_aggregate_split_plans, probe_durations, select_aggregate_source_subset, select_episode_tracks, select_episode_tracks_by_epl
from .discs import _episode_candidate_rows, analyze_tv_tracks, classify_tv_disc_hypothesis, infer_complete_series_slot_counts, infer_disc_slot_counts, infer_disc_slot_template, infer_track_number_direction, isolated_numbered_disc_scope_issue, select_complete_series_manifest_discwise, select_episode_manifest_by_ordinal_ranges, select_episode_manifest_discwise
from .tmdb import TMDbClient, _show_runtime_minutes, regular_series_episodes, season_episodes
from .fsops import _existing_destination_status, confirm, execute_plan, preflight_transfers, print_destination_conflicts, print_plan
from .volume import assign_numbered_volume_plan, infer_numbered_volume_plan

def _last_regular_season(client: TMDbClient, match: Match) -> int:
    data = client.tv_details(match.tmdb_id)
    numbers = [
        int(row["season_number"])
        for row in data.get("seasons", [])
        if row.get("season_number") is not None and int(row["season_number"]) > 0
    ]
    if not numbers:
        raise MKVPlexError(f"Could not determine regular seasons for {match.title}")
    return max(numbers)


def _validate_numbered_disc_sequences(resolved_groups: list[tuple[int, TvRipGroup]]) -> None:
    """Refuse to compact explicit numbered-disc gaps.

    When two or more discs in the same season/phase are explicitly numbered,
    a missing number is physical evidence of a missing source.  Later discs must
    never slide left to absorb that absence.  A single standalone Disc 2 remains
    valid because no surrounding sequence is being inferred.
    """
    buckets: dict[tuple[int, bool], list[TvRipGroup]] = {}
    for season, group in resolved_groups:
        if group.disc is None:
            continue
        buckets.setdefault((season, group.final_season), []).append(group)

    for (season, final_phase), groups in buckets.items():
        discs = sorted({int(group.disc) for group in groups if group.disc is not None})
        if len(discs) < 2:
            continue
        missing = [n for n in range(discs[0], discs[-1] + 1) if n not in discs]
        if not missing:
            continue
        label = "Final Season" if final_phase else (
            "Entire series" if season == COMPLETE_SERIES_SENTINEL else f"Season {season}"
        )
        missing_text = ", ".join(f"Disc {n}" for n in missing)
        raise MKVPlexError(
            f"Numbered physical-disc gap in {label}: missing {missing_text}. "
            "Refusing to rebalance later discs across a missing source."
        )


def _episode_json(ep: Episode) -> dict[str, Any]:
    return {
        "season": ep.season,
        "number": ep.number,
        "title": ep.title,
        "tmdb_id": ep.tmdb_id,
        "runtime_minutes": ep.runtime_minutes,
        "air_year": ep.air_year,
    }


def _boundary_json(boundary: SplitBoundary) -> dict[str, Any]:
    return {
        "predicted": boundary.predicted,
        "selected": boundary.selected,
        "black_start": boundary.black_start,
        "black_end": boundary.black_end,
        "black_duration": boundary.black_duration,
        "delta": boundary.delta,
        "confidence": boundary.confidence,
    }


def _semantic_time(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


def _tv_semantic_signature(
    *,
    mappings: list[tuple[Path, Episode, TvRipGroup, Optional[float]]],
    missing_manifest: list[Episode],
    all_split_plans: list[AggregateSplitPlan],
    split_destinations: list[Path],
    transfers: list[Transfer],
    extras_transfers: list[Transfer],
) -> dict[str, Any]:
    """Return the exact user-visible/executable TV plan identity.

    Discovery caches may be rebuilt or refreshed between dry-run and execution,
    but the approved mapping, destinations, missing holes, and split cut points
    must not change.
    """
    return {
        "mappings": sorted(
            (
                {
                    "source": str(source),
                    "episode": _episode_json(ep),
                    "group": str(group.directory),
                }
                for source, ep, group, _duration in mappings
            ),
            key=lambda row: (row["episode"]["season"], row["episode"]["number"], row["source"]),
        ),
        "missing": sorted(
            (_episode_json(ep) for ep in missing_manifest),
            key=lambda row: (row["season"], row["number"]),
        ),
        "splits": sorted(
            (
                {
                    "source": str(plan.source),
                    "group": str(plan.group.directory),
                    "episode": _episode_json(ep),
                    "selected_end": _semantic_time(boundary.selected),
                    "confidence": boundary.confidence,
                }
                for plan in all_split_plans
                for ep, boundary in zip(plan.episodes, plan.boundaries)
            ),
            key=lambda row: (row["episode"]["season"], row["episode"]["number"], row["source"]),
        ),
        "split_destinations": sorted(str(path) for path in split_destinations),
        "transfers": sorted(
            ({"source": str(t.source), "destination": str(t.destination)} for t in transfers),
            key=lambda row: (row["destination"], row["source"]),
        ),
        "extras_transfers": sorted(
            ({"source": str(t.source), "destination": str(t.destination)} for t in extras_transfers),
            key=lambda row: (row["destination"], row["source"]),
        ),
    }


def store_tv_plan_snapshot(
    args: argparse.Namespace,
    *,
    input_dir: Path,
    output_root: Path,
    extras_root: Path,
    query: str,
    year: Optional[int],
    match: Match,
    resolved_groups: list[tuple[int, TvRipGroup]],
    show_runtime_minutes: Optional[float],
    season_duration_maps: dict[int, dict[Path, float]],
    season_rows: dict[int, tuple[list[Episode], list[TvRipGroup], list[TrackAnalysis], list[TrackAnalysis]]],
    disc_template: dict[tuple[int, int], int],
    track_number_direction: Optional[int],
    mappings: list[tuple[Path, Episode, TvRipGroup, Optional[float]]],
    missing_manifest: list[Episode],
    all_split_plans: list[AggregateSplitPlan],
    split_destinations: list[Path],
    extras_mappings: list[tuple[Path, TvRipGroup, float, Path, str]],
    transfers: list[Transfer],
    extras_transfers: list[Transfer],
    deferred_giant_sources: dict[int, set[Path]],
    disc_hypotheses: dict[int, dict[Path, DiscHypothesis]],
) -> None:
    db = discovery_db()
    if db is None or not args.dry_run:
        return
    source_paths = [track for _season, group in resolved_groups for track in group.tracks]
    plan = {
        "command": "tv",
        "tool_version": VERSION,
        "query": query,
        "query_year": year,
        "match": {
            "media_type": match.media_type,
            "tmdb_id": match.tmdb_id,
            "imdb_id": match.imdb_id,
            "title": match.title,
            "year": match.year,
            "score": match.score,
        },
        "settings": _tv_plan_settings(args),
        "show_runtime_minutes": show_runtime_minutes,
        "track_number_direction": track_number_direction,
        "disc_template": [
            {"phase": phase, "disc": disc, "episode_slots": count}
            for (phase, disc), count in sorted(disc_template.items())
        ],
        "rip_groups": [
            {
                "season": season,
                "directory": str(group.directory),
                "disc": group.disc,
                "final_season": group.final_season,
                "episode_span": list(group.episode_span) if group.episode_span is not None else None,
                "tracks": [str(p) for p in group.tracks],
            }
            for season, group in resolved_groups
        ],
        "durations": {
            str(path): duration
            for season in sorted(season_duration_maps)
            for path, duration in season_duration_maps[season].items()
        },
        "episode_catalog": {
            str(season): [_episode_json(ep) for ep in season_rows[season][0]]
            for season in sorted(season_rows)
        },
        "disc_hypotheses": {
            str(season): [
                {
                    "directory": str(directory),
                    "kind": hypothesis.kind,
                    "confidence": hypothesis.confidence,
                    "score": hypothesis.score,
                    "reasons": list(hypothesis.reasons),
                }
                for directory, hypothesis in sorted(
                    disc_hypotheses.get(season, {}).items(), key=lambda item: str(item[0])
                )
            ]
            for season in sorted(season_rows)
        },
        "track_analysis": {
            str(season): [
                {
                    "path": str(row.path),
                    "group": str(row.group.directory),
                    "duration": row.duration,
                    "size_bytes": row.size_bytes,
                    "bitrate_mbps": row.bitrate_mbps,
                    "aggregate_of": [str(p) for p in row.aggregate_of],
                    "bitrate_outlier": row.bitrate_outlier,
                    "episode_candidate": any(c.path == row.path for c in season_rows[season][3]),
                }
                for row in season_rows[season][2]
            ]
            for season in sorted(season_rows)
        },
        "episode_mappings": [
            {
                "source": str(src),
                "episode": _episode_json(ep),
                "group": str(group.directory),
                "duration": duration,
            }
            for src, ep, group, duration in mappings
        ],
        "missing": [_episode_json(ep) for ep in missing_manifest],
        "aggregate_splits": [
            {
                "source": str(split.source),
                "group": str(split.group.directory),
                "source_duration": split.source_duration,
                "executable": split.executable,
                "episodes": [_episode_json(ep) for ep in split.episodes],
                "boundaries": [_boundary_json(b) for b in split.boundaries],
            }
            for split in all_split_plans
        ],
        "deferred_giant_sources": {
            str(season): [str(p) for p in sorted(paths, key=lambda p: str(p))]
            for season, paths in sorted(deferred_giant_sources.items())
        },
        "transfers": [
            {"source": str(t.source), "destination": str(t.destination)} for t in transfers
        ],
        "extras": [
            {
                "source": str(src), "group": str(group.directory),
                "duration": duration, "destination": str(destination), "reason": reason,
            }
            for src, group, duration, destination, reason in extras_mappings
        ],
        "extras_transfers": [
            {"source": str(t.source), "destination": str(t.destination)} for t in extras_transfers
        ],
        "tv_signature": _tv_semantic_signature(
            mappings=mappings, missing_manifest=missing_manifest,
            all_split_plans=all_split_plans, split_destinations=split_destinations, transfers=transfers,
            extras_transfers=extras_transfers,
        ),
    }
    key = _plan_key("tv", input_dir, output_root, extras_root, match, _tv_plan_settings(args))
    db.store_plan(
        plan_key=key, input_root=input_dir, output_root=output_root,
        extras_root=extras_root, tmdb_id=match.tmdb_id,
        source_paths=source_paths, plan=plan,
    )
    print(f"Dry-run plan saved to DB: {db.path}")
    print(f"  plan key: {key[:16]}…")
    print(f"  {db.stats_line()}")


def _do_tv_series(
    args: argparse.Namespace,
    client: TMDbClient,
    *,
    input_dir: Path,
    output_root: Path,
    extras_root: Path,
    query: str,
    year: Optional[int],
    groups: list[TvRipGroup],
    match: Match,
) -> int:
    """Process one logical TMDb series from a possibly mixed physical box set."""
    tree_mode = len(groups) > 1 or groups[0].directory != input_dir

    print()
    numbered_volume = getattr(args, "_numbered_volume", None)
    volume_mode = numbered_volume is not None
    complete_series_mode = bool(getattr(args, "_complete_series_mode", False)) and not volume_mode
    span_mode = complete_series_mode or volume_mode
    print(
        f"Series:   {match.title} ({match.year or '????'})  "
        f"TMDb {match.tmdb_id}  [{len(groups)} rip director{'y' if len(groups) == 1 else 'ies'}]"
    )
    if volume_mode:
        print(f"Scope:    RETAIL VOLUME {numbered_volume} (partial multi-season program slice)")
    elif complete_series_mode:
        print("Scope:    ENTIRE REGULAR SERIES (all TMDb seasons in order)")

    needs_final = any(g.final_season for g in groups)
    final_season_number = _last_regular_season(client, match) if needs_final else None
    if needs_final:
        print(f"Final Season -> season {final_season_number} (highest numbered regular season)")

    resolved_groups: list[tuple[int, TvRipGroup]] = []
    unresolved: list[Path] = []
    for group in groups:
        effective = _group_effective_season(group, final_season_number)

        if tree_mode and args.season is not None:
            if effective is not None and effective != args.season:
                continue
            if effective is None:
                effective = args.season
        elif not tree_mode and args.season is not None:
            effective = args.season

        if effective is None:
            unresolved.append(group.directory)
            continue
        resolved_groups.append((effective, group))

    if unresolved:
        details = "\n".join(f"  {p}" for p in unresolved)
        raise MKVPlexError(
            "Could not determine a season for these rip directories:\n" + details +
            "\nRename them to include Season N/Sxx, or process one season with --season N."
        )
    if not resolved_groups:
        if args.season is not None:
            raise MKVPlexError(f"No rip directories matched season {args.season}")
        raise MKVPlexError("No TV rip directories selected")

    seasons_present = sorted({season for season, _group in resolved_groups})
    if tree_mode and len(seasons_present) > 1 and args.episode_start != 1:
        raise MKVPlexError(
            "--episode-start other than 1 is ambiguous when processing multiple seasons. "
            "Use --season N to process one season."
        )
    if tree_mode and len(seasons_present) > 1 and args.episode_count is not None:
        raise MKVPlexError(
            "--episode-count is ambiguous when processing multiple seasons. "
            "Use --season N to process one season."
        )

    resolved_groups.sort(key=lambda item: _group_sort_key(item[1], item[0]))
    _validate_numbered_disc_sequences(resolved_groups)
    if any(group.episode_span is not None for _season, group in resolved_groups):
        if args.episode_start != 1 or args.episode_count is not None:
            raise MKVPlexError(
                "--episode-start/--episode-count cannot be combined with authored episode-range "
                "directories; their ordinals already define the physical episode positions."
            )

    if getattr(args, "disc_kind", "auto") != "auto" and len(resolved_groups) != 1:
        raise MKVPlexError(
            "--disc-kind episodes/bonus is only valid when exactly one TV rip directory is selected; "
            "use auto for a multi-disc tree."
        )

    db = discovery_db()
    prior_plan: Optional[dict[str, Any]] = None
    if db is not None and not args.dry_run:
        source_paths = [track for _season, group in resolved_groups for track in group.tracks]
        settings = _tv_plan_settings(args)
        prior_key = _plan_key("tv", input_dir, output_root, extras_root, match, settings)
        prior_plan, reason = db.load_valid_plan(
            prior_key, source_paths, match=match, expected_settings=settings
        )
        if prior_plan is None:
            raise MKVPlexError(
                "Refusing --db execution without an exact approved dry-run plan: "
                + reason + ". Re-run the same command with --dry-run --db first."
            )
        print(
            f"  Validated prior dry-run plan {prior_key[:16]}…; "
            "source sampled-MD5 inventory and media identity match. "
            "Reusing cached TMDb/ffprobe/fade discoveries."
        )

    print()
    print("Rip directories:")
    for season, group in resolved_groups:
        kind = (
            (f"Volume {numbered_volume}" if volume_mode and season == COMPLETE_SERIES_SENTINEL else "Entire series")
            if span_mode and season == COMPLETE_SERIES_SENTINEL
            else ("Final Season" if group.final_season else f"Season {season}")
        )
        disc = f", Disc {group.disc}" if group.disc is not None else ""
        span = (
            f", episodes {group.episode_span[0]}-{group.episode_span[1]}"
            if group.episode_span is not None else ""
        )
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
        print(f"  {rel} -> {kind}{disc}{span}: {len(group.tracks)} track(s)")

    destination_dir = output_root / tv_directory_name(match)
    extras_show_dir = extras_root / tv_directory_name(match)
    transfers: list[Transfer] = []
    extras_transfers: list[Transfer] = []
    mappings: list[tuple[Path, Episode, TvRipGroup, Optional[float]]] = []
    extras_mappings: list[tuple[Path, TvRipGroup, float, Path, str]] = []
    missing_manifest: list[Episode] = []
    season_episode_cache: dict[int, list[Episode]] = {}
    show_runtime_minutes = _show_runtime_minutes(client, match)

    # v0.7 first pass: probe and classify every selected season before assigning
    # episodes.  Complete sibling seasons can therefore teach a damaged season
    # the authored episode-per-disc layout.
    season_rows: dict[int, tuple[list[Episode], list[TvRipGroup], list[TrackAnalysis], list[TrackAnalysis]]] = {}
    season_duration_maps: dict[int, dict[Path, float]] = {}
    season_split_plans: dict[int, list[AggregateSplitPlan]] = {}
    season_deferred_giant_sources: dict[int, set[Path]] = {}
    season_disc_hypotheses: dict[int, dict[Path, DiscHypothesis]] = {}
    season_bonus_groups: dict[int, set[Path]] = {}
    season_ambiguous_groups: dict[int, set[Path]] = {}
    season_scope_issues: dict[int, str] = {}
    unresolved_sources: list[tuple[Path, TvRipGroup, float, str]] = []
    aggregate_episode_mappings: list[tuple[AggregateSplitPlan, Episode, float, float, SplitBoundary]] = []
    all_split_plans: list[AggregateSplitPlan] = []
    volume_program_plan: Optional[dict[str, Any]] = None

    for season in seasons_present:
        season_groups = [group for s, group in resolved_groups if s == season]
        season_groups.sort(key=lambda g: _group_sort_key(g, season))
        tracks_with_group: list[tuple[Path, TvRipGroup]] = []
        for group in season_groups:
            tracks_with_group.extend((track, group) for track in group.tracks)

        if span_mode and season == COMPLETE_SERIES_SENTINEL:
            if args.episode_start != 1 or args.episode_count is not None:
                raise MKVPlexError(
                    "--episode-start/--episode-count are not supported for the entire-series plan; "
                    "preview a specific season instead."
                )
            episodes = regular_series_episodes(client, match)
            season_counts = getattr(args, "season_counts", None)
            if season_counts:
                episodes = remap_episode_seasons(episodes, season_counts)
                print(
                    "  Plex presentation season map: "
                    + "/".join(str(count) for count in season_counts)
                    + f" ({len(episodes)} episodes across {len(season_counts)} seasons)"
                )
            for real_season in sorted({ep.season for ep in episodes}):
                season_episode_cache[real_season] = [ep for ep in episodes if ep.season == real_season]
            episode_list = list(episodes)
            start_ep = 1
        else:
            episodes = season_episodes(client, match, season)
            season_episode_cache[season] = episodes
            start_ep = args.episode_start
            episode_list = [e for e in episodes if e.number >= start_ep]
            if args.episode_count is not None:
                episode_list = episode_list[: args.episode_count]
        if not episode_list:
            label = "complete regular series" if complete_series_mode else f"season {season}"
            raise MKVPlexError(f"No episodes found for {label} from episode {start_ep}")

        if args.all_tracks:
            season_rows[season] = (episode_list, season_groups, [], [])
            season_duration_maps[season] = {}
            continue

        probe_scope = (
            f"retail Volume {numbered_volume}" if volume_mode and season == COMPLETE_SERIES_SENTINEL
            else ("complete series" if complete_series_mode and season == COMPLETE_SERIES_SENTINEL else f"Season {season}")
        )
        print(f"  Probing {probe_scope}: {len(tracks_with_group)} track(s) for {len(episode_list)} episode(s)...")
        duration_map = probe_durations(
            (src for src, _group in tracks_with_group),
            workers=args.probe_workers,
        )
        analyses = analyze_tv_tracks(
            tracks_with_group,
            episode_list,
            duration_map,
            show_runtime_minutes=show_runtime_minutes,
            tolerance_minutes=args.runtime_tolerance,
        )
        candidates = _episode_candidate_rows(
            analyses,
            episode_list,
            show_runtime_minutes=show_runtime_minutes,
            tolerance_minutes=args.runtime_tolerance,
        )
        aggregates = sum(1 for row in analyses if row.aggregate_of)
        outliers = sum(1 for row in analyses if row.bitrate_outlier)
        print(
            f"    structural pass: {len(candidates)} episode-like; "
            f"{aggregates} aggregate/play-all; {outliers} episode-length bitrate outlier(s)"
        )

        if volume_mode and season == COMPLETE_SERIES_SENTINEL:
            volume_program_plan = infer_numbered_volume_plan(
                client, match, volume_number=int(numbered_volume),
                groups=season_groups, analyses=analyses,
                show_runtime_minutes=show_runtime_minutes,
            )
            episode_list = list(volume_program_plan["episodes"])
            for real_season in sorted({ep.season for ep in episode_list}):
                season_episode_cache[real_season] = [ep for ep in episode_list if ep.season == real_season]
            candidates = list(volume_program_plan["programs"])
            per_disc: list[str] = []
            for group in season_groups:
                count = sum(1 for row in candidates if row.group.directory == group.directory)
                per_disc.append(str(count))
            first_ep, last_ep = episode_list[0], episode_list[-1]
            print(
                f"    numbered-volume structure: {len(candidates)} authored program title(s), "
                f"disc programs {'/'.join(per_disc)}"
            )
            print(
                f"    provider program alignment: {volume_program_plan['order_label']}; "
                f"Volume {numbered_volume} -> program ordinals "
                f"{volume_program_plan['start'] + 1}-{volume_program_plan['end']} "
                f"of {len(volume_program_plan['all_units'])}; "
                f"canonical span {first_ep.season}x{first_ep.number:02d}.."
                f"{last_ep.season}x{last_ep.number:02d}"
            )
            for refinement in volume_program_plan.get("topology_refinements", []):
                delta = refinement.get("chapter_delta")
                delta_text = f", nearest authored cut delta {delta:.1f}s" if delta is not None else ""
                print(
                    "    authored chapter topology corrected adjacent metadata identities: "
                    f"{Path(refinement['left_source']).name} -> {refinement['after_left']}; "
                    f"{Path(refinement['right_source']).name} -> {refinement['after_right']}"
                    f"{delta_text}"
                )
            print(
                f"    alignment runtime scale {volume_program_plan['runtime_scale']:.3f}x, "
                f"cost {volume_program_plan['runtime_cost']:.4f}"
            )
            season_rows[season] = (episode_list, season_groups, analyses, candidates)
            season_duration_maps[season] = duration_map
            season_disc_hypotheses[season] = {
                group.directory: DiscHypothesis(
                    "episodes", "numbered-volume", 1,
                    (f"verified play-all component set for retail Volume {numbered_volume}",),
                )
                for group in season_groups
            }
            season_bonus_groups[season] = set()
            season_ambiguous_groups[season] = set()
            continue

        disc_hypotheses: dict[Path, DiscHypothesis] = {}
        bonus_dirs: set[Path] = set()
        ambiguous_dirs: set[Path] = set()
        forced_kind = str(getattr(args, "disc_kind", "auto"))
        for group in season_groups:
            if forced_kind == "bonus":
                hypothesis = DiscHypothesis(
                    "bonus", "forced", 999, ("explicit --disc-kind bonus override",)
                )
            elif forced_kind == "episodes":
                hypothesis = DiscHypothesis(
                    "episodes", "forced", -999, ("explicit --disc-kind episodes override",)
                )
            else:
                group_candidate_count = sum(1 for row in candidates if row.group.directory == group.directory)
                group_giant_count = sum(
                    1 for row in analyses
                    if row.group.directory == group.directory
                    and row.duration >= max(
                        45.0 * 60.0,
                        (_median(episode_expected_seconds(
                            episode_list, show_runtime_minutes
                        )) or 0.0) * 2.25,
                    )
                )
                if (
                    group.episode_span is not None
                    and len(group.tracks) == group.episode_span[1] - group.episode_span[0] + 1
                ):
                    hypothesis = DiscHypothesis(
                        "episodes", "authored-range", 1,
                        (f"directory declares episode ordinals {group.episode_span[0]}-{group.episode_span[1]}",),
                    )
                elif complete_series_mode and (group_candidate_count > 0 or group_giant_count > 0):
                    reason = (
                        f"complete-series hypothesis retains {group_candidate_count} episode-like track(s)"
                        if group_candidate_count > 0
                        else f"complete-series aggregate hypothesis retains {group_giant_count} giant source(s)"
                    )
                    hypothesis = DiscHypothesis(
                        "episodes", "series-structure", -1, (reason,)
                    )
                else:
                    hypothesis = classify_tv_disc_hypothesis(
                        group, analyses, candidates, episode_list,
                        show_runtime_minutes=show_runtime_minutes,
                        tolerance_minutes=args.runtime_tolerance,
                    )
            disc_hypotheses[group.directory] = hypothesis
            rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
            if hypothesis.kind == "bonus":
                bonus_dirs.add(group.directory)
                why = "; ".join(hypothesis.reasons[:3])
                print(
                    f"    disc hypothesis [{rel}]: BONUS ({hypothesis.confidence}, score={hypothesis.score})"
                    + (f" -- {why}" if why else "")
                )
            elif hypothesis.kind == "ambiguous":
                ambiguous_dirs.add(group.directory)
                why = "; ".join(hypothesis.reasons[:3])
                print(
                    f"    disc hypothesis [{rel}]: AMBIGUOUS ({hypothesis.confidence}, score={hypothesis.score})"
                    + (f" -- {why}" if why else "")
                )
            elif hypothesis.score > 0 or forced_kind == "episodes":
                why = "; ".join(hypothesis.reasons[:2])
                print(
                    f"    disc hypothesis [{rel}]: EPISODES ({hypothesis.confidence}, score={hypothesis.score})"
                    + (f" -- {why}" if why else "")
                )

        season_disc_hypotheses[season] = disc_hypotheses
        season_bonus_groups[season] = bonus_dirs
        season_ambiguous_groups[season] = ambiguous_dirs

        # Bonus/ambiguous discs must not contribute runtime candidates or giant
        # aggregate sources to episode reconstruction.  Ambiguous discs remain
        # review-only; an explicit --disc-kind override is required.
        candidates = [r for r in candidates if r.group.directory not in bonus_dirs | ambiguous_dirs]
        if ambiguous_dirs:
            names = ", ".join(str(d.relative_to(input_dir)) if d != input_dir else "." for d in sorted(ambiguous_dirs, key=str))
            print(f"    review required before episode assignment: ambiguous disc(s): {names}")

        if not getattr(args, "no_aggregate_split", False) and len(candidates) < len(episode_list) and not ambiguous_dirs:
            episode_analyses = [
                r for r in analyses if r.group.directory not in bonus_dirs | ambiguous_dirs
            ]
            giant_candidates = _aggregate_episode_sources(
                episode_analyses, episode_list, show_runtime_minutes=show_runtime_minutes
            )
            giant_sources, deferred_giants = select_aggregate_source_subset(
                giant_candidates, episode_list, show_runtime_minutes=show_runtime_minutes
            ) if giant_candidates else ([], [])
            if deferred_giants:
                season_deferred_giant_sources[season] = {row.path for row in deferred_giants}
                names = ", ".join(row.path.name for row in deferred_giants)
                print(
                    f"    deferred {len(deferred_giants)} giant source(s) that do not fit "
                    f"the TV-season runtime envelope: {names}"
                )
            allocations = allocate_episodes_to_aggregate_sources(
                giant_sources, episode_list, show_runtime_minutes=show_runtime_minutes
            ) if giant_sources else None
            if allocations:
                counts = "/".join(str(len(eps)) for _row, eps in allocations)
                print(
                    f"    multi-episode source set: {len(allocations)} giant MKV(s), "
                    f"episode allocation {counts}; scanning black/runtime boundaries..."
                )
                plans = build_aggregate_split_plans(
                    allocations, show_runtime_minutes=show_runtime_minutes,
                    search_window=max(30.0, float(getattr(args, "split_search_window", 180.0))),
                    min_black_duration=max(0.05, float(getattr(args, "split_black_min", 0.30))),
                )
                season_split_plans[season] = plans
                executable = sum(1 for plan in plans if plan.executable)
                boundaries = sum(len(plan.boundaries) for plan in plans)
                unresolved_boundaries = sum(
                    1 for plan in plans for b in plan.boundaries
                    if b.selected is None or b.confidence == "unresolved" or b.confidence.startswith("low")
                )
                print(
                    f"    aggregate split scan: {boundaries - unresolved_boundaries}/{boundaries} "
                    f"usable boundaries; {executable}/{len(plans)} source plan(s) executable"
                )
        season_rows[season] = (episode_list, season_groups, analyses, candidates)
        season_duration_maps[season] = duration_map

    disc_template = infer_disc_slot_template(season_rows) if (not args.all_tracks and not volume_mode) else {}
    track_number_direction = (
        infer_track_number_direction(
            season_rows,
            show_runtime_minutes=show_runtime_minutes,
            tolerance_minutes=args.runtime_tolerance,
        )
        if (not args.all_tracks and not volume_mode) else None
    )
    if disc_template:
        pretty = ", ".join(
            f"Disc {disc if disc < 10**6 else '?'}={count}"
            for (_phase, disc), count in sorted(disc_template.items())
            if _phase == 0
        )
        if pretty:
            print(f"  Learned healthy-season disc layout: {pretty}")
    if track_number_direction is not None:
        direction_name = "ascending" if track_number_direction > 0 else "descending"
        print(f"  Learned MakeMKV episode title order: {direction_name} tNN")

    for season in seasons_present:
        episode_list, season_groups, analyses, candidates = season_rows[season]
        duration_map = season_duration_maps[season]
        tracks_with_group = [(track, group) for group in season_groups for track in group.tracks]

        season_assignments: list[EpisodeAssignment] = []
        season_missing: list[Episode] = []
        season_skipped: list[SkippedTrack] = []

        if volume_mode:
            if volume_program_plan is None:
                raise MKVPlexError("Internal error: numbered-volume plan missing during assignment")
            (
                season_assignments, season_skipped, volume_split_plans, volume_aggregate_rows
            ) = assign_numbered_volume_plan(
                volume_program_plan, analyses,
                show_runtime_minutes=show_runtime_minutes,
                split_search_window=float(getattr(args, "split_search_window", 180.0)),
                split_black_min=float(getattr(args, "split_black_min", 0.30)),
            )
            season_missing = []
            if volume_split_plans:
                season_split_plans[season] = volume_split_plans
                all_split_plans.extend(volume_split_plans)
                aggregate_episode_mappings.extend(volume_aggregate_rows)
            print(
                f"    mapped {len(volume_program_plan['episodes'])} canonical episode segment(s) "
                f"from {len(volume_program_plan['programs'])} physical program title(s); "
                f"archived {len(season_skipped)} non-program/master title(s)"
            )
        else:
            bonus_dirs = season_bonus_groups.get(season, set())
            ambiguous_dirs = season_ambiguous_groups.get(season, set())
            episode_groups = [g for g in season_groups if g.directory not in bonus_dirs | ambiguous_dirs]
            episode_analyses = [r for r in analyses if r.group.directory not in bonus_dirs | ambiguous_dirs]
            episode_candidates = [r for r in candidates if r.group.directory not in bonus_dirs | ambiguous_dirs]

            for row in analyses:
                if row.group.directory in bonus_dirs:
                    season_skipped.append(SkippedTrack(
                        row.path, row.group, row.duration, "bonus-disc hypothesis"
                    ))
                elif row.group.directory in ambiguous_dirs:
                    season_skipped.append(SkippedTrack(
                        row.path, row.group, row.duration, "ambiguous disc; review required"
                    ))

            aggregate_plans = season_split_plans.get(season, [])
            if ambiguous_dirs:
                # Do not manufacture episode identities from a disc that the
                # structural classifier itself considers ambiguous.
                print(
                    f"    episode assignment withheld for {len(ambiguous_dirs)} ambiguous disc(s); "
                    "use --disc-kind episodes or --disc-kind bonus on a single-disc input"
                )
            elif bonus_dirs and not episode_groups:
                print(
                    f"    bonus-disc plan: 0 episode mappings; archiving {len(season_skipped)} track(s) as Extras"
                )
            elif aggregate_plans:
                print("    using aggregate-source black/runtime episode reconstruction")
                used_sources = {plan.source for plan in aggregate_plans}
                mapped_episode_keys: set[tuple[int, int]] = set()
                for plan in aggregate_plans:
                    all_split_plans.append(plan)
                    start = 0.0
                    for ep, boundary in zip(plan.episodes, plan.boundaries):
                        if boundary.selected is not None:
                            aggregate_episode_mappings.append((plan, ep, start, boundary.selected, boundary))
                            mapped_episode_keys.add((ep.season, ep.number))
                            start = boundary.selected
                        else:
                            start = min(plan.source_duration, start + 60.0 * float(ep.runtime_minutes or show_runtime_minutes or 0.0))
                season_missing.extend(
                    ep for ep in episode_list
                    if (ep.season, ep.number) not in mapped_episode_keys
                )
                deferred_sources = season_deferred_giant_sources.get(season, set())
                for row in episode_analyses:
                    if row.path in deferred_sources:
                        # This giant source did not fit the TV season.  It may be a
                        # feature film or other long-form program; TV mode must not
                        # silently move it into Extras.
                        continue
                    if row.path in used_sources:
                        reason = "multi-episode source (split master)"
                    elif row.aggregate_of:
                        reason = "aggregate/play-all"
                    elif row.bitrate_outlier:
                        reason = "episode-length bitrate outlier"
                    else:
                        reason = "non-selected source/extra"
                    season_skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))
                print(
                    f"    reconstructed {len(mapped_episode_keys)} episode interval(s); "
                    f"missing/unresolved {len(season_missing)}; archived {len(season_skipped)} source track(s)"
                )
            elif args.all_tracks:
                episode_tracks_with_group = [(track, group) for group in episode_groups for track in group.tracks]
                if len(episode_tracks_with_group) > len(episode_list):
                    raise MKVPlexError(
                        f"--all-tracks selected, but season {season} has {len(episode_list)} "
                        f"episode(s) and {len(episode_tracks_with_group)} MKV track(s). Use "
                        "--episode-count to explicitly limit the mapping."
                    )
                for (src, group), ep in zip(episode_tracks_with_group, episode_list):
                    season_assignments.append(EpisodeAssignment(src, group, ep, 0.0))
                season_missing.extend(episode_list[len(season_assignments):])
            elif episode_groups and all(group.episode_span is not None for group in episode_groups):
                ranged = select_episode_manifest_by_ordinal_ranges(
                    episode_groups, episode_list, episode_analyses,
                    show_runtime_minutes=show_runtime_minutes,
                    tolerance_minutes=args.runtime_tolerance,
                )
                if ranged is None:
                    raise MKVPlexError("Internal error resolving explicit episode-range groups")
                ranged_assignments, ranged_missing, ranged_skipped = ranged
                season_assignments.extend(ranged_assignments)
                season_missing.extend(ranged_missing)
                season_skipped.extend(ranged_skipped)
                spans = ", ".join(
                    f"{group.episode_span[0]}-{group.episode_span[1]}"
                    for group in sorted(
                        episode_groups, key=lambda g: (g.episode_span or (10**6, 10**6))[0]
                    )
                )
                print(f"    using authored episode-range ordinals: {spans}")
            else:
                episode_tracks_with_group = [(track, group) for group in episode_groups for track in group.tracks]
                regular_tracks = [item for item in episode_tracks_with_group if not item[1].final_season]
                final_tracks = [item for item in episode_tracks_with_group if item[1].final_season]

                # Preserve the proven EPL path.  Complete EPL ordinals are stronger
                # than any inferred disc structure, especially split seasons such as
                # Breaking Bad Season 5 / Final Season.
                final_count = contiguous_epl_count(final_tracks) if final_tracks else None
                if (
                    final_tracks and regular_tracks and final_count is not None
                    and 0 < final_count < len(episode_list)
                ):
                    regular_eps = episode_list[:-final_count]
                    final_eps = episode_list[-final_count:]
                    print(
                        f"    Final Season EPL_01..{final_count:02d} -> "
                        f"{season}x{final_eps[0].number:02d}..{season}x{final_eps[-1].number:02d}"
                    )
                    regular_selected = select_episode_tracks_by_epl(
                        regular_tracks, regular_eps, duration_map,
                        show_runtime_minutes=show_runtime_minutes,
                        tolerance_minutes=args.runtime_tolerance,
                    )
                    final_selected = select_episode_tracks_by_epl(
                        final_tracks, final_eps, duration_map,
                        show_runtime_minutes=show_runtime_minutes,
                        tolerance_minutes=args.runtime_tolerance,
                    )
                    if regular_selected is None:
                        regular_selected = select_episode_tracks(
                            regular_tracks, regular_eps, duration_map,
                            show_runtime_minutes=show_runtime_minutes,
                            tolerance_minutes=args.runtime_tolerance,
                        )
                    if final_selected is None:
                        final_selected = select_episode_tracks(
                            final_tracks, final_eps, duration_map,
                            show_runtime_minutes=show_runtime_minutes,
                            tolerance_minutes=args.runtime_tolerance,
                        )
                    for (src, group), ep in zip(regular_selected[0], regular_eps):
                        season_assignments.append(EpisodeAssignment(src, group, ep, duration_map[src]))
                    for (src, group), ep in zip(final_selected[0], final_eps):
                        season_assignments.append(EpisodeAssignment(src, group, ep, duration_map[src]))
                    selected_paths = {a.source for a in season_assignments}
                    for row in episode_analyses:
                        if row.path not in selected_paths:
                            reason = "aggregate/play-all" if row.aggregate_of else (
                                "episode-length bitrate outlier" if row.bitrate_outlier else "extra/duplicate"
                            )
                            season_skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))
                else:
                    epl_selected = select_episode_tracks_by_epl(
                        episode_tracks_with_group, episode_list, duration_map,
                        show_runtime_minutes=show_runtime_minutes,
                        tolerance_minutes=args.runtime_tolerance,
                    )
                    if epl_selected is not None:
                        print("    using EPL_nn episode ordinals plus runtime verification")
                        for (src, group), ep in zip(epl_selected[0], episode_list):
                            season_assignments.append(EpisodeAssignment(src, group, ep, duration_map[src]))
                        selected_paths = {a.source for a in season_assignments}
                        for row in analyses:
                            if row.path not in selected_paths:
                                reason = "aggregate/play-all" if row.aggregate_of else (
                                    "episode-length bitrate outlier" if row.bitrate_outlier else "extra/duplicate"
                                )
                                season_skipped.append(SkippedTrack(row.path, row.group, row.duration, reason))
                    else:
                        scope_issue = isolated_numbered_disc_scope_issue(
                            episode_groups, episode_candidates, episode_list,
                            explicit_episode_window=(
                                args.episode_start != 1 or args.episode_count is not None
                            ),
                        )
                        if scope_issue is not None:
                            season_scope_issues[season] = scope_issue
                            print(f"    isolated-disc scope unresolved: {scope_issue}")
                            season_missing.extend(episode_list)
                            for row in episode_analyses:
                                unresolved_sources.append((
                                    row.path, row.group, row.duration,
                                    "isolated numbered disc; global episode offset unresolved",
                                ))
                            print(
                                f"    matched 0 episode track(s); missing {len(season_missing)}; "
                                f"archived 0; left {len(episode_analyses)} source track(s) untouched"
                            )
                            # Do not map or archive any title from an unresolved physical disc.
                            # The source remains exactly where it is until the user supplies a
                            # globally-positioned episode window.  Record the holes now because
                            # this branch intentionally skips the normal season-finalization path.
                            missing_manifest.extend(season_missing)
                            continue
                        if complete_series_mode:
                            slot_counts, slot_source = infer_complete_series_slot_counts(
                                episode_groups, episode_candidates, len(episode_list)
                            )
                        else:
                            slot_counts, slot_source = infer_disc_slot_counts(
                                episode_groups, episode_candidates, len(episode_list), disc_template
                            )
                        layout = "/".join(str(n) for n in slot_counts)
                        print(f"    disc episode slots: {layout} [{slot_source}]")
                        if complete_series_mode:
                            normal_assignments, normal_missing, normal_skipped = select_complete_series_manifest_discwise(
                                episode_groups, episode_list, episode_analyses, episode_candidates, slot_counts
                            )
                        else:
                            normal_assignments, normal_missing, normal_skipped = select_episode_manifest_discwise(
                                episode_groups,
                                episode_list,
                                episode_analyses,
                                episode_candidates,
                                slot_counts,
                                show_runtime_minutes=show_runtime_minutes,
                                tolerance_minutes=args.runtime_tolerance,
                                track_number_direction=track_number_direction,
                            )
                        season_assignments.extend(normal_assignments)
                        season_missing.extend(normal_missing)
                        season_skipped.extend(normal_skipped)

                print(
                    f"    matched {len(season_assignments)} episode track(s); "
                    f"missing {len(season_missing)}; archived {len(season_skipped)} non-selected track(s)"
                )

        season_assignments.sort(key=lambda a: (a.episode.season, a.episode.number))
        missing_manifest.extend(season_missing)
        for assignment in season_assignments:
            filename = episode_filename(match, assignment.episode, assignment.source)
            transfers.append(Transfer(assignment.source, destination_dir / filename))
            mappings.append((assignment.source, assignment.episode, assignment.group, duration_map.get(assignment.source)))

        for skipped in season_skipped:
            archive_dir = extras_group_directory(extras_show_dir, input_dir, skipped.group, season)
            destination = archive_dir / skipped.source.name
            extras_transfers.append(Transfer(skipped.source, destination))
            extras_mappings.append((skipped.source, skipped.group, skipped.duration, destination, skipped.reason))

    print()
    print("Episode manifest:")
    mappings_by_episode = {(ep.season, ep.number): (src, ep, group, duration) for src, ep, group, duration in mappings}
    missing_keys = {(ep.season, ep.number) for ep in missing_manifest}
    aggregate_by_episode = {
        (ep.season, ep.number): (plan, ep, start, end, boundary)
        for plan, ep, start, end, boundary in aggregate_episode_mappings
    }
    manifest_sections: list[tuple[int, list[Episode], set[Path], set[Path], list[TvRipGroup]]] = []
    if span_mode:
        synthetic = COMPLETE_SERIES_SENTINEL
        all_eps = season_rows[synthetic][0]
        for real_season in sorted({ep.season for ep in all_eps}):
            manifest_sections.append((
                real_season, [ep for ep in all_eps if ep.season == real_season],
                season_bonus_groups.get(synthetic, set()),
                season_ambiguous_groups.get(synthetic, set()),
                season_rows[synthetic][1],
            ))
    else:
        for season in seasons_present:
            manifest_sections.append((
                season, season_rows[season][0], season_bonus_groups.get(season, set()),
                season_ambiguous_groups.get(season, set()), season_rows[season][1],
            ))

    for season, episode_list, bonus_dirs, ambiguous_dirs, manifest_groups in manifest_sections:
        print(f"  Season {season}:")
        if bonus_dirs and not any(
            g.directory not in bonus_dirs | ambiguous_dirs for g in manifest_groups
        ):
            print("    [bonus disc: no episode mappings]")
        elif ambiguous_dirs:
            print("    [ambiguous disc: episode mappings withheld pending review]")
        previous_group: Optional[Path] = None
        for ep in episode_list:
            key = (ep.season, ep.number)
            aggregate_row = aggregate_by_episode.get(key)
            if aggregate_row is not None:
                plan, _ep, start, end, boundary = aggregate_row
                if plan.group.directory != previous_group:
                    rel = plan.group.directory.relative_to(input_dir) if plan.group.directory != input_dir else Path(".")
                    print(f"    [{rel}] [multi-episode source]")
                    previous_group = plan.group.directory
                expected = f" (~{ep.runtime_minutes}m)" if ep.runtime_minutes else ""
                air = f" ({ep.air_year})" if ep.air_year else ""
                black = (
                    f" black={boundary.black_duration:.2f}s"
                    if boundary.black_duration is not None else ""
                )
                delta = f" Δ={boundary.delta:.1f}s" if boundary.delta is not None else ""
                print(
                    f"      {plan.source.name:<28} [{format_duration(start)}-{format_duration(end)}] "
                    f"-> {ep.season}x{ep.number:02d}  {ep.title}{air}{expected} "
                    f"[{boundary.confidence}{black}{delta}]"
                )
                continue
            if key in missing_keys:
                expected = f" (~{ep.runtime_minutes}m)" if ep.runtime_minutes else ""
                air = f" ({ep.air_year})" if ep.air_year else ""
                print(f"    MISSING{'':<21} -> {ep.season}x{ep.number:02d}  {ep.title}{air}{expected}")
                continue
            row = mappings_by_episode.get(key)
            if row is None:
                continue
            src, _ep, group, duration = row
            if group.directory != previous_group:
                rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
                print(f"    [{rel}]")
                previous_group = group.directory
            runtime = f" [{format_duration(duration)}]" if duration is not None else ""
            expected = f" (~{ep.runtime_minutes}m)" if ep.runtime_minutes else ""
            epl = epl_number(src)
            hint = f" [EPL_{epl:02d}]" if epl is not None else ""
            air = f" ({ep.air_year})" if ep.air_year else ""
            confidence = ""
            if duration is not None:
                if volume_mode:
                    confidence = " [high:volume-program-order]"
                else:
                    season_eps = season_episode_cache.get(ep.season, [ep])
                    level, _candidates = episode_match_confidence(
                        src, duration, ep, season_eps,
                        show_runtime_minutes=show_runtime_minutes,
                    )
                    confidence = f" [{level}]"
            print(f"      {src.name:<28}{runtime:<12}{hint:<10} -> {ep.season}x{ep.number:02d}  {ep.title}{air}{expected}{confidence}")

    review_rows: list[tuple[Path, Episode, str, list[tuple[Episode, float, float]]]] = []
    for src, ep, _group, duration in mappings:
        if volume_mode:
            continue
        if duration is None:
            continue
        season_eps = season_episode_cache.get(ep.season, [ep])
        level, local_candidates = episode_match_confidence(
            src, duration, ep, season_eps,
            show_runtime_minutes=show_runtime_minutes,
        )
        if level.startswith("low"):
            review_rows.append((src, ep, level, local_candidates))
    if review_rows:
        print()
        print("Manual review suggested (runtime/title evidence is ambiguous):")
        for src, ep, level, local_candidates in review_rows:
            alts = []
            for cand, delta, title_sim in local_candidates:
                marker = "*" if cand.number == ep.number else " "
                alts.append(
                    f"{marker}{cand.season}x{cand.number:02d} {cand.title} "
                    f"(Δ{delta/60.0:.1f}m,title={title_sim:.2f})"
                )
            print(f"  {src.name} -> {ep.season}x{ep.number:02d} [{level}]")
            print("    candidates: " + "; ".join(alts))

    if missing_manifest:
        print()
        print("Missing episodes (left as holes; later episodes were NOT shifted):")
        for ep in sorted(missing_manifest, key=lambda e: (e.season, e.number)):
            runtime = f", ~{ep.runtime_minutes}m" if ep.runtime_minutes else ""
            print(f"  {ep.season}x{ep.number:02d} - {ep.title}{runtime}")

    if all_split_plans:
        print()
        print("Aggregate split sources:")
        for plan in all_split_plans:
            status = "EXECUTABLE" if plan.executable else "REVIEW REQUIRED"
            print(f"  {plan.source} [{format_duration(plan.source_duration)}] -> {len(plan.episodes)} episode(s) [{status}]")
            start = 0.0
            for ep, boundary in zip(plan.episodes, plan.boundaries):
                if boundary.selected is None:
                    print(
                        f"    {ep.season}x{ep.number:02d} {ep.title}: expected end "
                        f"{format_duration(boundary.predicted)} -> NO FADE FOUND [{boundary.confidence}]"
                    )
                    continue
                black = f", black {boundary.black_duration:.2f}s" if boundary.black_duration is not None else ""
                delta = f", delta {boundary.delta:.1f}s" if boundary.delta is not None else ""
                print(
                    f"    {ep.season}x{ep.number:02d} {format_duration(start)} -> "
                    f"{format_duration(boundary.selected)} [{boundary.confidence}{black}{delta}]"
                )
                start = boundary.selected
            tail = max(0.0, plan.source_duration - start)
            if tail > 2.0:
                print(f"    source tail after last proposed episode: {format_duration(tail)} (preserved in split master)")

    print()
    print("Extras archive:")
    if not extras_mappings:
        print("  (no skipped tracks)")
    else:
        previous_extra_group: Optional[Path] = None
        for src, group, duration, destination, reason in extras_mappings:
            if group.directory != previous_extra_group:
                rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
                print(f"  [{rel}]")
                previous_extra_group = group.directory
            print(f"    {src.name:<28} [{format_duration(duration)}] [{reason}] -> {destination}")
        print(f"  {len(extras_mappings)} skipped track(s) will be archived, not deleted.")

    if unresolved_sources:
        print()
        print("Unresolved source tracks (left in place):")
        for src, group, duration, reason in unresolved_sources:
            rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
            print(f"  [{rel}] {src.name} [{format_duration(duration)}] -- {reason}")

    print_plan(match, transfers, destination_dir, args.mode, "COPY" if args.copy else "MOVE")
    split_destinations: list[Path] = []
    if all_split_plans:
        print()
        print("Aggregate episode creation: STREAM COPY (ffmpeg; no re-encode)")
        for plan in all_split_plans:
            for ep, _start, _end, boundary in _split_segment_rows(plan):
                dst = destination_dir / episode_filename(match, ep, plan.source)
                split_destinations.append(dst)
                print(f"  {plan.source.name} [{boundary.confidence}] -> {dst}")
    print()
    print("Extras destination directory:")
    print(f"  {extras_show_dir}")
    print(f"Extras operation: {'COPY' if args.copy else 'MOVE'} ({len(extras_transfers)} file(s))")
    print(f"Extras permissions after completion: chmod -R {args.mode:o} {extras_show_dir}")

    destination_conflicts = preflight_transfers(
        [*transfers, *extras_transfers], allow_existing=args.dry_run
    )
    occupied = {t.destination for t in [*transfers, *extras_transfers]}
    for dst in split_destinations:
        if dst in occupied:
            raise MKVPlexError(f"Destination appears more than once in plan: {dst}")
        if dst.exists():
            if not args.dry_run:
                raise MKVPlexError(f"Refusing to overwrite existing file: {dst}")
            destination_conflicts.append((None, dst, _existing_destination_status(None, dst)))
        occupied.add(dst)
    non_executable = [plan for plan in all_split_plans if not plan.executable]
    ambiguous_disc_count = sum(len(v) for v in season_ambiguous_groups.values())
    if ambiguous_disc_count and not args.dry_run:
        raise MKVPlexError(
            "Disc classification review required before execution; rerun the single-disc input with "
            "--disc-kind episodes or --disc-kind bonus after inspection."
        )
    if non_executable and not args.dry_run:
        names = ", ".join(plan.source.name for plan in non_executable)
        raise MKVPlexError(
            "Aggregate split review required before execution; low-confidence/unresolved "
            f"split boundaries remain in: {names}. Use --dry-run to inspect the plan."
        )
    if season_scope_issues and not args.dry_run:
        details = "; ".join(season_scope_issues[season] for season in sorted(season_scope_issues))
        raise MKVPlexError(
            "Isolated numbered-disc episode scope is unresolved; refusing execution. " + details
        )

    tv_action = "COPY" if args.copy else "MOVE"
    tv_physical_sources = {t.source.resolve() for t in [*transfers, *extras_transfers]}
    tv_physical_sources.update(plan.source.resolve() for plan in all_split_plans)
    print()
    print("Episode vs extras by source:")
    mapped_sources_by_group: dict[Path, set[Path]] = {}
    for src, _ep, group, _duration in mappings:
        mapped_sources_by_group.setdefault(group.directory, set()).add(src)
    for plan, _ep, _start, _end, _boundary in aggregate_episode_mappings:
        mapped_sources_by_group.setdefault(plan.group.directory, set()).add(plan.source)
    extras_sources_by_group: dict[Path, set[Path]] = {}
    for src, group, _duration, _destination, _reason in extras_mappings:
        extras_sources_by_group.setdefault(group.directory, set()).add(src)
    for group in sorted(groups, key=lambda g: _group_sort_key(g, g.season or 0)):
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
        ep_sources = len(mapped_sources_by_group.get(group.directory, set()))
        extra_sources = len(extras_sources_by_group.get(group.directory, set()))
        print(f"  {str(rel):<28} episodes {ep_sources:>3}   extras {extra_sources:>3}   source files {len(group.tracks):>3}")
    if complete_series_mode:
        regular_total = len(regular_series_episodes(client, match))
        print(f"  complete-series target: {regular_total} regular episodes")
    elif volume_mode and volume_program_plan is not None:
        print(
            f"  retail-volume target: {len(volume_program_plan['programs'])} physical programs -> "
            f"{len(volume_program_plan['episodes'])} canonical episode segment(s)"
        )

    split_boundary_total = sum(len(plan.boundaries) for plan in all_split_plans)
    split_boundary_unresolved = sum(
        1 for plan in all_split_plans for b in plan.boundaries
        if b.selected is None or b.confidence == "unresolved"
    )
    split_boundary_low = sum(
        1 for plan in all_split_plans for b in plan.boundaries
        if b.confidence == "low" or b.confidence.startswith("low:")
    )
    split_boundary_runtime = sum(
        1 for plan in all_split_plans for b in plan.boundaries
        if "runtime" in b.confidence and not b.confidence.startswith("low")
    )
    split_boundary_chapter = sum(
        1 for plan in all_split_plans for b in plan.boundaries
        if ":chapter" in b.confidence and not b.confidence.startswith("low")
    )
    split_boundary_black = sum(
        1 for plan in all_split_plans for b in plan.boundaries
        if ":black" in b.confidence and not b.confidence.startswith("low")
    )

    blocker_messages: list[str] = []
    if destination_conflicts:
        blocker_messages.append(f"{len(destination_conflicts)} destination conflict(s)")
    if ambiguous_disc_count:
        blocker_messages.append(f"{ambiguous_disc_count} ambiguous disc classification(s)")
    if season_scope_issues:
        blocker_messages.append(f"{len(season_scope_issues)} isolated numbered-disc scope ambiguity/ambiguities")
    if split_boundary_unresolved:
        blocker_messages.append(f"{split_boundary_unresolved} unresolved split boundary/boundaries")
    if split_boundary_low:
        blocker_messages.append(f"{split_boundary_low} low-confidence split boundary/boundaries")

    print()
    print("Plan summary:")
    print(f"  Episodes identified:      {len(transfers) + len(split_destinations)}")
    print(f"  Whole-file episodes:      {len(transfers)}")
    print(f"  Split episodes to create: {len(split_destinations)}")
    print(f"  Missing episode holes:    {len(missing_manifest)}")
    print(f"  Extras to archive:        {len(extras_transfers)}")
    print(f"  Source files to {tv_action}:      {len(tv_physical_sources)}")
    print(f"  Destination conflicts:    {len(destination_conflicts)}")
    if split_boundary_total:
        print(f"  Split boundaries:         {split_boundary_total}")
        print(f"    authored chapter:       {split_boundary_chapter}")
        print(f"    observed black/fade:    {split_boundary_black}")
        print(f"    runtime reconstructed:  {split_boundary_runtime}")
        print(f"    low confidence:         {split_boundary_low}")
        print(f"    unresolved:             {split_boundary_unresolved}")
    print_destination_conflicts(destination_conflicts)
    if blocker_messages:
        print("  Execution blockers:")
        for blocker in blocker_messages:
            print(f"    - {blocker}")

    plan_executable = not blocker_messages

    if db is not None and not args.dry_run:
        if prior_plan is None:
            raise MKVPlexError("Internal error: approved TV plan was not loaded before execution")
        current_signature = _tv_semantic_signature(
            mappings=mappings, missing_manifest=missing_manifest, all_split_plans=all_split_plans,
            split_destinations=split_destinations, transfers=transfers, extras_transfers=extras_transfers,
        )
        if prior_plan.get("tv_signature") != current_signature:
            raise MKVPlexError(
                "Current TV episode/extras/split mapping differs from the approved dry run. "
                "Refusing execution; re-run the same command with --dry-run --db first."
            )
        print("  Semantic TV plan matches the approved dry run.")

    snapshot_kwargs = dict(
        input_dir=input_dir, output_root=output_root, extras_root=extras_root,
        query=query, year=year, match=match, resolved_groups=resolved_groups,
        show_runtime_minutes=show_runtime_minutes, season_duration_maps=season_duration_maps,
        season_rows=season_rows, disc_template=disc_template, track_number_direction=track_number_direction,
        mappings=mappings, missing_manifest=missing_manifest, all_split_plans=all_split_plans,
        split_destinations=split_destinations, extras_mappings=extras_mappings,
        transfers=transfers, extras_transfers=extras_transfers,
        deferred_giant_sources=season_deferred_giant_sources,
        disc_hypotheses=season_disc_hypotheses,
    )

    # When an unlabeled multi-season set is being explored, the dry run itself
    # becomes an interactive plan-review loop.  Do not hash/store every trial;
    # bind only the candidate the user explicitly accepts.
    if args.dry_run and getattr(args, "_season_review_active", False):
        rows = list(getattr(args, "_season_review_options", []))
        current = int(getattr(args, "_effective_season_choice"))
        series_available = bool(getattr(args, "_season_review_series_available", False))
        print()
        current_label = "Entire regular series" if current == COMPLETE_SERIES_SENTINEL else f"Season {current}"
        print(f"Season-plan review: currently previewing {current_label}.")
        if not plan_executable:
            print("  This candidate is not executable yet.")
            print("  Blocking reason(s): " + "; ".join(blocker_messages))
        choices = ", ".join(str(row[0]) for row in rows)
        extra_choice = ", [e] entire series" if series_available else ""
        while True:
            accept_choice = "[a] accept this plan, " if plan_executable else ""
            prompt = f"   {accept_choice}[{choices}] preview season{extra_choice}, [q] quit: "
            raw = input(prompt).strip().lower()
            if raw in {"a", "accept", "y", "yes"}:
                if not plan_executable:
                    print("   Accept is unavailable: " + "; ".join(blocker_messages))
                    continue
                store_tv_plan_snapshot(args, **snapshot_kwargs)
                setattr(args, "_season_review_action", "accept")
                print(
                    f"Dry run; accepted {current_label}. No media changed. Would "
                    f"{tv_action.lower()} {len(tv_physical_sources)} source file(s), create "
                    f"{len(split_destinations)} split episode file(s), and leave "
                    f"{len(missing_manifest)} episode hole(s)."
                )
                return 0
            if raw == "q":
                raise SystemExit(0)
            if raw in {"e", "entire", "series", "all"} and series_available:
                if current == COMPLETE_SERIES_SENTINEL:
                    print("   Entire regular series is already the displayed plan.")
                    continue
                setattr(args, "_season_review_action", COMPLETE_SERIES_SENTINEL)
                print("   Replanning as the entire regular series; cached probes will be reused.")
                return 0
            if raw.isdigit() and int(raw) in {row[0] for row in rows}:
                selected = int(raw)
                if selected == current:
                    print(f"   Season {selected} is already the displayed plan.")
                    continue
                setattr(args, "_season_review_action", selected)
                print(f"   Replanning as Season {selected}; cached probes will be reused.")
                return 0
            print("   Invalid selection.")

    if plan_executable:
        store_tv_plan_snapshot(args, **snapshot_kwargs)
    if not confirm("Apply this plan?", args.yes, args.dry_run):
        if args.dry_run:
            print(
                f"Dry run; no changes made. Would {tv_action.lower()} {len(tv_physical_sources)} "
                f"source file(s), create {len(split_destinations)} split episode file(s), "
                f"and leave {len(missing_manifest)} episode hole(s)."
            )
            if not plan_executable:
                print(
                    "Dry-run plan is NOT executable and was not saved as an approved --db plan; "
                    "resolve destination/disc/split review requirements and run --dry-run --db again."
                )
        else:
            print("Cancelled; no changes made.")
        return 2 if (args.dry_run and not plan_executable) else 0

    operations = execute_plan(
        destination_dir, transfers,
        copy_only=args.copy,
        verify_md5=args.verify_md5,
        mode=args.mode,
    )
    split_created = 0
    if all_split_plans:
        split_created = execute_aggregate_split_plans(
            all_split_plans, match, destination_dir, mode=args.mode
        )
        if split_created:
            operations.append("split")
    if extras_transfers:
        operations.extend(execute_plan(
            extras_show_dir, extras_transfers,
            copy_only=args.copy,
            verify_md5=args.verify_md5,
            mode=args.mode,
        ))
    print(
        "Done (" + ", ".join(sorted(set(operations))) + "). "
        f"Episodes: {len(transfers) + split_created}; missing: {len(missing_manifest)}; "
        f"extras archived: {len(extras_transfers)}."
    )
    if discovery_db() is not None:
        print(f"  {discovery_db().stats_line()}")
    return 0


__all__ = ['_last_regular_season', '_validate_numbered_disc_sequences', '_episode_json', '_boundary_json', '_semantic_time', '_tv_semantic_signature', 'store_tv_plan_snapshot', '_do_tv_series']
