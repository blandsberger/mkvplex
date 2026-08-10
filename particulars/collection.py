"""Mixed/boxed collection analysis and execution."""
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
from .models import Episode, MKVPlexError, Match, TrackAnalysis, Transfer, TvRipGroup, VERSION
from .common import _disc_key
from .discovery import _plan_key, _tv_plan_settings, discovery_db
from .naming import canonical_spaces, movie_basename, normalize_for_match, track_number, tv_directory_name
from .media import _presentation_stream_profile, episode_filename, movie_filename, probe_durations, probe_video_packet_fingerprints
from .discs import _episode_candidate_rows, analyze_tv_tracks
from .tmdb import TMDbClient, _alternate_episode_order_rows, _collection_order_sequences, _regular_tv_season_rows, _show_runtime_minutes, attach_imdb_id, regular_series_episodes, year_from_date
from .fsops import confirm, ensure_output_root, execute_plan, human_size, preflight_transfers, print_destination_conflicts
from .movie import _movie_extras_destination, _select_primary_movie_track, movie_expected_runtime_seconds

def _visual_clusters(
    rows: list[TrackAnalysis], fingerprints: dict[Path, str]
) -> list[list[TrackAnalysis]]:
    clusters: dict[str, list[TrackAnalysis]] = {}
    for row in rows:
        digest = fingerprints.get(row.path)
        if digest is None:
            continue
        clusters.setdefault(digest, []).append(row)
    ordered = list(clusters.values())
    for cluster in ordered:
        cluster.sort(key=lambda row: (_disc_key(row.group), track_number(row.path), str(row.path)))
    ordered.sort(key=lambda cluster: (_disc_key(cluster[0].group), track_number(cluster[0].path), str(cluster[0].path)))
    return ordered


_COLLECTION_GENERIC_WORDS = {
    "the", "a", "an", "and", "complete", "collection", "boxed", "boxset",
    "animated", "series", "show", "season", "seasons", "television", "tv",
}


def _distinctive_title_tokens(title: str) -> list[str]:
    return [t for t in normalize_for_match(title).split() if t not in _COLLECTION_GENERIC_WORDS and len(t) > 1]


def _regular_episode_count(client: TMDbClient, match: Match) -> int:
    return sum(row[1] for row in _regular_tv_season_rows(client, match))


def _collection_companion_candidates(
    client: TMDbClient, primary: Match, residual_episode_count: int,
) -> list[tuple[float, Match, int]]:
    """Find TV series whose identity/count plausibly explains a collection residual."""
    if residual_episode_count <= 0:
        return []
    core_tokens = _distinctive_title_tokens(primary.title)
    if not core_tokens:
        return []
    queries = []
    # Short franchise/core queries expose sequel series that a full-title query
    # can hide (e.g. Batman -> The New Batman Adventures).
    queries.append(" ".join(core_tokens[:2]))
    if len(core_tokens) > 2:
        queries.append(" ".join(core_tokens[:3]))
    seen: dict[int, dict[str, Any]] = {}
    for query in queries:
        for row in client.search_tv(query, None):
            try:
                tmdb_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            if tmdb_id == primary.tmdb_id:
                continue
            seen.setdefault(tmdb_id, row)

    # Candidate details are independent requests. Fetch/cache them together so
    # an all-flash media host cannot turn companion discovery into a long serial
    # metadata waterfall.
    client.prefetch(
        [(f"/tv/{tmdb_id}", None) for tmdb_id in seen],
        label="collection companion details",
    )

    results: list[tuple[float, Match, int]] = []
    primary_tokens = set(core_tokens)
    for tmdb_id, raw in seen.items():
        title = canonical_spaces(str(raw.get("name") or raw.get("original_name") or ""))
        if not title:
            continue
        candidate_tokens = set(_distinctive_title_tokens(title))
        overlap = len(primary_tokens & candidate_tokens) / max(1, len(primary_tokens))
        if overlap <= 0:
            continue
        details = client.tv_details(tmdb_id)
        count = 0
        for season in details.get("seasons", []) or []:
            try:
                number = int(season.get("season_number"))
                c = int(season.get("episode_count") or 0)
            except (TypeError, ValueError):
                continue
            if number > 0:
                count += max(0, c)
        if count <= 0:
            continue
        count_fit = max(0.0, 1.0 - abs(count - residual_episode_count) / max(1.0, float(residual_episode_count)))
        if count_fit < 0.55:
            continue
        year = year_from_date(raw.get("first_air_date"))
        year_fit = 0.5
        if primary.year is not None and year is not None:
            delta = abs(year - primary.year)
            year_fit = max(0.0, 1.0 - delta / 15.0)
        score = 0.58 * count_fit + 0.32 * overlap + 0.10 * year_fit
        match = Match(
            media_type="tv", tmdb_id=tmdb_id, title=title, year=year,
            imdb_id=None, score=score, raw=raw,
        )
        results.append((score, match, count))
    results.sort(key=lambda row: (-row[0], abs(row[2] - residual_episode_count), row[1].title))
    return results[:5]


def _choose_visual_cluster_primary(
    cluster: list[TrackAnalysis],
) -> tuple[TrackAnalysis, list[TrackAnalysis], dict[Path, dict[str, int]]]:
    """Choose a Plex-facing presentation while preserving every alternate.

    Equivalent-video titles frequently differ only in commentary/audio muxing.
    Prefer a presentation with ordinary (non-commentary) audio, then richer
    audio/subtitle coverage, then the larger container.  This is deliberately
    conservative because all non-selected presentations are archived, not lost.
    """
    profiles = {row.path: _presentation_stream_profile(row.path) for row in cluster}

    def score(row: TrackAnalysis) -> tuple[int, int, int, int, int]:
        profile = profiles[row.path]
        return (
            1 if profile["non_commentary_audio"] > 0 else 0,
            profile["non_commentary_audio"],
            profile["audio"],
            profile["subtitles"],
            row.size_bytes,
        )

    primary = max(cluster, key=score)
    alternates = [row for row in cluster if row.path != primary.path]
    alternates.sort(key=lambda row: (_disc_key(row.group), track_number(row.path), str(row.path)))
    return primary, alternates, profiles


def _movie_keyword_signal(text: str) -> bool:
    """Return True for a directory/title that explicitly advertises a film."""
    norm = normalize_for_match(text)
    tokens = set(norm.split())
    if tokens & {"movie", "movies", "film", "films", "feature", "featurefilm"}:
        return True
    return "motion picture" in norm or "the movie" in norm


def analyze_collection_overflow(
    args: argparse.Namespace,
    client: TMDbClient,
    *,
    input_dir: Path,
    match: Match,
    groups: list[TvRipGroup],
    total_regular_episodes: int,
) -> dict[str, Any]:
    """Explain a retail collection that cannot fit the selected series model.

    No execution plan is created here.  This stage discovers content-equivalent
    presentations and metadata hypotheses so a later collection planner can be
    built from evidence rather than from the raw MKV count.
    """
    all_tracks = [(track, group) for group in groups for track in group.tracks]
    regular_eps = regular_series_episodes(client, match)
    show_runtime = _show_runtime_minutes(client, match)
    print()
    print("Collection forensic preflight:")
    print(f"  probing {len(all_tracks)} source track(s) against {len(regular_eps)} regular episode runtime(s)...")
    duration_map = probe_durations((p for p, _g in all_tracks), workers=args.probe_workers)
    analyses = analyze_tv_tracks(
        all_tracks, regular_eps, duration_map,
        show_runtime_minutes=show_runtime,
        tolerance_minutes=args.runtime_tolerance,
    )
    episode_like = _episode_candidate_rows(
        analyses, regular_eps,
        show_runtime_minutes=show_runtime,
        tolerance_minutes=args.runtime_tolerance,
    )
    # Also fingerprint bitrate outliers in the episode runtime envelope so an
    # alternate/commentary presentation can still collapse into a normal-title
    # cluster. Standalone outlier clusters are not counted as regular episodes.
    fp_rows_by_path = {row.path: row for row in episode_like}
    for row in analyses:
        if row.bitrate_outlier and not row.aggregate_of:
            fp_rows_by_path.setdefault(row.path, row)
    fp_rows = list(fp_rows_by_path.values())
    fingerprints = probe_video_packet_fingerprints(fp_rows, workers=args.probe_workers)
    clusters = _visual_clusters(fp_rows, fingerprints)
    episode_like_paths = {row.path for row in episode_like}
    logical_clusters = [c for c in clusters if any(row.path in episode_like_paths for row in c)]
    duplicate_clusters = [c for c in logical_clusters if len(c) > 1]
    duplicate_presentations = sum(len(c) - 1 for c in duplicate_clusters)
    unique_episode_presentations = len(logical_clusters)

    print(f"  episode-like titles:             {len(episode_like)}")
    print(f"  unique visual presentations:    {unique_episode_presentations}")
    print(f"  alternate/duplicate titles:     {duplicate_presentations}")
    if duplicate_clusters:
        print("  visual-equivalent clusters:")
        for cluster in duplicate_clusters[:20]:
            names = []
            for row in cluster:
                rel = row.group.directory.relative_to(input_dir) if row.group.directory != input_dir else Path(".")
                names.append(f"{rel}/{row.path.name} ({human_size(row.size_bytes)})")
            print("    - " + " == ".join(names))
        if len(duplicate_clusters) > 20:
            print(f"    ... {len(duplicate_clusters) - 20} more duplicate cluster(s)")

    residual = unique_episode_presentations - total_regular_episodes
    print(f"  selected-series regular target: {total_regular_episodes}")
    print(f"  unexplained unique presentations: {residual:+d}")

    alternate_orders = _alternate_episode_order_rows(client, match)
    if alternate_orders:
        print("  TMDb alternate episode orderings available:")
        type_names = {3: "DVD", 6: "Production"}
        for row in alternate_orders:
            kind = int(row.get("type") or 0)
            print(
                f"    - {type_names.get(kind, str(kind))}: {row.get('name') or '(unnamed)'} "
                f"[{row.get('episode_count', '?')} episodes, group {row.get('id', '?')}]"
            )
    else:
        print("  TMDb alternate episode orderings: none advertised (DVD/Production)")

    companions = _collection_companion_candidates(client, match, residual)
    if residual > 0:
        if companions:
            print(f"  companion-series candidates for residual {residual}:")
            for score, candidate, count in companions:
                print(
                    f"    - {candidate.title} ({candidate.year or '????'}) TMDb {candidate.tmdb_id}: "
                    f"{count} regular episodes, collection-fit={score:.3f}"
                )
        else:
            print(f"  companion-series candidates for residual {residual}: none strong enough")

    exact_companion = next((row for row in companions if row[2] == residual and row[0] >= 0.80), None)
    coherent = unique_episode_presentations >= total_regular_episodes and (
        residual == 0 or exact_companion is not None
    )
    if coherent:
        print("  collection hypothesis: COHERENT (eligible for collection-plan review)")
    else:
        print("  collection hypothesis: UNRESOLVED (more structure/evidence required)")
    return {
        "episode_like": len(episode_like),
        "unique_visual": unique_episode_presentations,
        "duplicate_presentations": duplicate_presentations,
        "residual": residual,
        "alternate_orders": alternate_orders,
        "companions": companions,
        "exact_companion": exact_companion,
        "coherent": coherent,
        # Internal evidence reused by the collection planner. These objects are
        # intentionally not serialized here; the accepted execution snapshot
        # stores the resulting explicit mappings instead.
        "logical_clusters": logical_clusters,
        "duplicate_clusters": duplicate_clusters,
        "duration_map": duration_map,
        "analyses": analyses,
        "episode_like_paths": episode_like_paths,
    }


def _collection_match_identity(match: Match) -> dict[str, Any]:
    return {
        "media_type": match.media_type,
        "tmdb_id": match.tmdb_id,
        "imdb_id": match.imdb_id,
        "title": match.title,
        "year": match.year,
    }


def _collection_signature(
    *,
    primary_match: Match,
    companion_match: Match,
    order_choice: str,
    episode_rows: list[tuple[Path, Match, Episode, TvRipGroup]],
    alternate_rows: list[tuple[Path, Match, Path, TvRipGroup]],
    extras_transfers: list[Transfer],
    movie_rows: list[tuple[Path, Match, Transfer]],
) -> dict[str, Any]:
    return {
        "primary": _collection_match_identity(primary_match),
        "companion": _collection_match_identity(companion_match),
        "order": order_choice,
        "episodes": [
            {
                "source": str(source),
                "show_tmdb": owner.tmdb_id,
                "season": ep.season,
                "episode": ep.number,
                "episode_tmdb": ep.tmdb_id,
                "destination": str(
                    Path("TV") / tv_directory_name(owner) / episode_filename(owner, ep, source)
                ),
                "group": str(group.directory),
            }
            for source, owner, ep, group in episode_rows
        ],
        "alternates": [
            {
                "source": str(source), "show_tmdb": owner.tmdb_id,
                "primary_source": str(primary_source), "group": str(group.directory),
            }
            for source, owner, primary_source, group in alternate_rows
        ],
        "extras": [
            {"source": str(t.source), "destination": str(t.destination)}
            for t in extras_transfers
        ],
        "movies": [
            {
                "source": str(source),
                "match": _collection_match_identity(movie_match),
                "destination": str(transfer.destination),
            }
            for source, movie_match, transfer in movie_rows
        ],
    }


def _collection_plan_source_paths(
    tv_groups: list[TvRipGroup],
    movie_groups: list[tuple[str, Match, list[TvRipGroup]]],
    bonus_groups: list[TvRipGroup],
    unresolved_groups: list[tuple[str, list[TvRipGroup]]],
) -> list[Path]:
    paths: set[Path] = set()
    for group in tv_groups:
        paths.update(group.tracks)
    for _hint, _match, groups in movie_groups:
        for group in groups:
            paths.update(group.tracks)
    for group in bonus_groups:
        paths.update(group.tracks)
    for _hint, groups in unresolved_groups:
        for group in groups:
            paths.update(group.tracks)
    return sorted(paths, key=lambda p: str(p).casefold())


def _store_collection_plan_snapshot(
    args: argparse.Namespace,
    *,
    input_dir: Path,
    output_root: Path,
    extras_root: Path,
    primary_match: Match,
    companion_match: Match,
    source_paths: list[Path],
    signature: dict[str, Any],
    transfers: list[Transfer],
    extras_transfers: list[Transfer],
    movie_transfers: list[Transfer],
) -> str:
    db = discovery_db()
    if db is None or not args.dry_run:
        return ""
    settings = _tv_plan_settings(args)
    plan = {
        "command": "tv_collection",
        "tool_version": VERSION,
        "match": _collection_match_identity(primary_match),
        "companion_match": _collection_match_identity(companion_match),
        "settings": settings,
        "collection_signature": signature,
        "transfers": [
            {"source": str(t.source), "destination": str(t.destination)}
            for t in transfers
        ],
        "extras_transfers": [
            {"source": str(t.source), "destination": str(t.destination)}
            for t in extras_transfers
        ],
        "movie_transfers": [
            {"source": str(t.source), "destination": str(t.destination)}
            for t in movie_transfers
        ],
    }
    key = _plan_key("tv_collection", input_dir, output_root, extras_root, primary_match, settings)
    db.store_plan(
        plan_key=key, input_root=input_dir, output_root=output_root,
        extras_root=extras_root, tmdb_id=primary_match.tmdb_id,
        source_paths=source_paths, plan=plan,
    )
    print(f"Dry-run collection plan saved to DB: {db.path}")
    print(f"  plan key: {key[:16]}…")
    print(f"  {db.stats_line()}")
    return key


def _collection_movie_rows(
    args: argparse.Namespace,
    client: TMDbClient,
    *,
    input_dir: Path,
    extras_root: Path,
    movies_output: Optional[Path],
    movie_groups: list[tuple[str, Match, list[TvRipGroup]]],
) -> tuple[
    list[tuple[Path, Match, Transfer]],
    list[Transfer],
    list[tuple[str, Match, list[TvRipGroup]]],
]:
    """Resolve embedded movie groups into feature/extras transfers.

    If --movies-output is omitted, movie identities are still returned as
    deferred rows so the collection report can name them without moving them.
    """
    movie_rows: list[tuple[Path, Match, Transfer]] = []
    movie_extras: list[Transfer] = []
    deferred: list[tuple[str, Match, list[TvRipGroup]]] = []
    for hint, movie_match, groups in movie_groups:
        if movie_match.score < 0.78:
            deferred.append((hint, movie_match, groups))
            continue
        tracks = sorted({track for group in groups for track in group.tracks}, key=lambda p: str(p).casefold())
        if not tracks:
            continue
        durations = probe_durations(tracks, workers=max(1, int(getattr(args, "probe_workers", 4))))
        expected = movie_expected_runtime_seconds(client, movie_match)
        source = _select_primary_movie_track(input_dir, movie_match.title, tracks, durations, expected)
        if movies_output is None:
            deferred.append((hint, movie_match, groups))
            continue
        base = movie_basename(movie_match)
        transfer = Transfer(source, movies_output / base / movie_filename(movie_match, source))
        movie_rows.append((source, movie_match, transfer))
        for track in tracks:
            if track == source:
                continue
            movie_extras.append(Transfer(
                track, _movie_extras_destination(extras_root, base, input_dir, track)
            ))
    return movie_rows, movie_extras, deferred


def _execute_collection_transfers(
    args: argparse.Namespace,
    *,
    output_root: Path,
    extras_root: Path,
    movies_output: Optional[Path],
    episode_rows: list[tuple[Path, Match, Episode, TvRipGroup]],
    episode_transfers: list[Transfer],
    extras_transfers: list[Transfer],
    movie_rows: list[tuple[Path, Match, Transfer]],
) -> list[str]:
    operations: list[str] = []
    by_show: dict[int, tuple[Path, list[Transfer]]] = {}
    for (_source, owner, _ep, _group), transfer in zip(episode_rows, episode_transfers):
        root = output_root / tv_directory_name(owner)
        row = by_show.setdefault(owner.tmdb_id, (root, []))
        row[1].append(transfer)
    for root, rows in by_show.values():
        operations.extend(execute_plan(
            root, rows, copy_only=args.copy, verify_md5=args.verify_md5, mode=args.mode
        ))

    # Group extras by their canonical first-level owner so chmod stays local and
    # never recursively touches the user's entire Extras library.
    extras_by_root: dict[Path, list[Transfer]] = {}
    for transfer in extras_transfers:
        try:
            rel = transfer.destination.relative_to(extras_root)
            owner = extras_root / rel.parts[0]
        except (ValueError, IndexError):
            owner = transfer.destination.parent
        extras_by_root.setdefault(owner, []).append(transfer)
    for root, rows in extras_by_root.items():
        operations.extend(execute_plan(
            root, rows, copy_only=args.copy, verify_md5=args.verify_md5, mode=args.mode
        ))

    if movies_output is not None:
        for _source, _match, transfer in movie_rows:
            operations.extend(execute_plan(
                transfer.destination.parent, [transfer], copy_only=args.copy,
                verify_md5=args.verify_md5, mode=args.mode,
            ))
    return operations


def do_collection_plan(
    args: argparse.Namespace,
    client: TMDbClient,
    *,
    input_dir: Path,
    output_root: Path,
    extras_root: Path,
    primary_match: Match,
    tv_groups: list[TvRipGroup],
    report: dict[str, Any],
    total_regular_episodes: int,
) -> int:
    """Build/approve/execute a coherent multi-series physical collection plan."""
    exact = report.get("exact_companion")
    if exact is None:
        raise MKVPlexError("Collection planner requires an exact high-confidence companion-series residual.")
    _score, companion_match, companion_count = exact
    companion_match = attach_imdb_id(client, companion_match)
    primary_match = attach_imdb_id(client, primary_match)
    setattr(args, "_collection_mode", True)
    setattr(args, "_collection_companion_tmdb_id", companion_match.tmdb_id)

    logical_clusters: list[list[TrackAnalysis]] = list(report.get("logical_clusters") or [])
    if len(logical_clusters) != total_regular_episodes + companion_count:
        raise MKVPlexError(
            "Collection unique-presentation count changed before planning; refusing to guess."
        )

    # Every logical episode cluster must belong to one physical TV rip directory.
    cluster_rows: list[dict[str, Any]] = []
    all_cluster_paths: set[Path] = set()
    for cluster in logical_clusters:
        group_dirs = {row.group.directory for row in cluster}
        if len(group_dirs) != 1:
            raise MKVPlexError(
                "A visual-equivalent episode cluster spans multiple physical discs; "
                "collection boundary inference is ambiguous."
            )
        primary_row, alternate_rows, profiles = _choose_visual_cluster_primary(cluster)
        all_cluster_paths.update(row.path for row in cluster)
        cluster_rows.append({
            "group": primary_row.group,
            "primary": primary_row,
            "alternates": alternate_rows,
            "profiles": profiles,
            "track_order": min(track_number(row.path) for row in cluster),
        })
    cluster_rows.sort(key=lambda row: (
        _disc_key(row["group"]), row["track_order"], str(row["primary"].path)
    ))

    groups_sorted = sorted(tv_groups, key=lambda g: (_disc_key(g), str(g.directory).casefold()))
    clusters_by_group: dict[Path, list[dict[str, Any]]] = {}
    for row in cluster_rows:
        clusters_by_group.setdefault(row["group"].directory, []).append(row)

    print()
    print("Collection topology:")
    cumulative = 0
    boundary_group: Optional[Path] = None
    owner_by_group: dict[Path, Match] = {}
    for group in groups_sorted:
        count = len(clusters_by_group.get(group.directory, []))
        cumulative += count
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
        marker = ""
        if cumulative == total_regular_episodes:
            boundary_group = group.directory
            marker = "  <== primary-series boundary"
        print(f"  {str(rel):<48} unique programs {count:>3}   cumulative {cumulative:>3}{marker}")
    if boundary_group is None:
        raise MKVPlexError(
            f"The {total_regular_episodes}-episode primary-series boundary does not land exactly "
            "between physical discs. Collection execution remains disabled."
        )

    past_boundary = False
    for group in groups_sorted:
        owner_by_group[group.directory] = companion_match if past_boundary else primary_match
        if group.directory == boundary_group:
            past_boundary = True
    primary_clusters = [row for row in cluster_rows if owner_by_group[row["group"].directory].tmdb_id == primary_match.tmdb_id]
    companion_clusters = [row for row in cluster_rows if owner_by_group[row["group"].directory].tmdb_id == companion_match.tmdb_id]
    if len(primary_clusters) != total_regular_episodes or len(companion_clusters) != companion_count:
        raise MKVPlexError(
            "Physical-disc boundary counts do not exactly match the primary + companion episode inventories."
        )
    print(
        f"  boundary result: {len(primary_clusters)} -> {primary_match.title}; "
        f"{len(companion_clusters)} -> {companion_match.title}"
    )

    order_options = _collection_order_sequences(client, primary_match)
    requested = str(getattr(args, "_collection_order_choice", None) or getattr(args, "collection_order", "auto"))
    if requested == "auto":
        requested = "production" if "production" in order_options else (
            "dvd" if "dvd" in order_options else "regular"
        )
    if requested not in order_options:
        available = ", ".join(sorted(order_options))
        raise MKVPlexError(
            f"Collection order {requested!r} is not available for {primary_match.title}; available: {available}."
        )
    setattr(args, "_collection_order_choice", requested)

    companion_options = _collection_order_sequences(client, companion_match)
    companion_order = requested if requested in companion_options else "regular"
    primary_label, primary_sequence, primary_group_id = order_options[requested]
    companion_label, companion_sequence, companion_group_id = companion_options[companion_order]
    if len(primary_sequence) != len(primary_clusters) or len(companion_sequence) != len(companion_clusters):
        raise MKVPlexError("Provider episode ordering length does not match physical unique-program count.")

    print()
    print("Collection ordering:")
    print(
        f"  {primary_match.title}: {primary_label}"
        + (f" [group {primary_group_id}]" if primary_group_id else "")
    )
    print(
        f"  {companion_match.title}: {companion_label}"
        + (f" [group {companion_group_id}]" if companion_group_id else "")
    )
    if companion_order != requested:
        print(
            f"  note: {requested} order is not available for the companion series; "
            "using its regular TMDb order."
        )

    episode_rows: list[tuple[Path, Match, Episode, TvRipGroup]] = []
    alternate_rows: list[tuple[Path, Match, Path, TvRipGroup]] = []
    for cluster_row, ep in zip(primary_clusters, primary_sequence):
        owner = primary_match
        source = cluster_row["primary"].path
        group = cluster_row["group"]
        episode_rows.append((source, owner, ep, group))
        for alt in cluster_row["alternates"]:
            alternate_rows.append((alt.path, owner, source, group))
    for cluster_row, ep in zip(companion_clusters, companion_sequence):
        owner = companion_match
        source = cluster_row["primary"].path
        group = cluster_row["group"]
        episode_rows.append((source, owner, ep, group))
        for alt in cluster_row["alternates"]:
            alternate_rows.append((alt.path, owner, source, group))

    episode_transfers = [
        Transfer(
            source,
            output_root / tv_directory_name(owner) / episode_filename(owner, ep, source),
        )
        for source, owner, ep, _group in episode_rows
    ]

    extras_transfers: list[Transfer] = []
    extras_reason: dict[Path, str] = {}
    for source, owner, primary_source, group in alternate_rows:
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(group.directory.name)
        destination = extras_root / tv_directory_name(owner) / "Alternate Presentations" / rel / source.name
        extras_transfers.append(Transfer(source, destination))
        extras_reason[source] = f"visual duplicate/alternate of {primary_source.name}"

    # Non-cluster tracks on each TV disc remain attributable to whichever show
    # owns that physical disc after the exact collection boundary is established.
    for group in groups_sorted:
        owner = owner_by_group[group.directory]
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(group.directory.name)
        for track in group.tracks:
            if track in all_cluster_paths:
                continue
            destination = extras_root / tv_directory_name(owner) / rel / track.name
            extras_transfers.append(Transfer(track, destination))
            extras_reason[track] = "non-episode/bonus presentation"

    bonus_groups: list[TvRipGroup] = list(getattr(args, "_collection_bonus_groups", []))
    for group in bonus_groups:
        rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(group.directory.name)
        for track in group.tracks:
            extras_transfers.append(Transfer(
                track, extras_root / tv_directory_name(primary_match) / rel / track.name
            ))
            extras_reason[track] = "dedicated bonus/supplement disc"

    movies_output_arg = getattr(args, "movies_output", None)
    movies_output = ensure_output_root(Path(movies_output_arg)) if movies_output_arg else None
    movie_groups: list[tuple[str, Match, list[TvRipGroup]]] = list(
        getattr(args, "_collection_movie_groups", [])
    )
    movie_rows, movie_extras, deferred_movies = _collection_movie_rows(
        args, client, input_dir=input_dir, extras_root=extras_root,
        movies_output=movies_output, movie_groups=movie_groups,
    )
    extras_transfers.extend(movie_extras)
    for t in movie_extras:
        extras_reason[t.source] = "embedded movie extra/alternate"

    unresolved_groups: list[tuple[str, list[TvRipGroup]]] = list(
        getattr(args, "_collection_unresolved_groups", [])
    )

    print()
    print("Collection episode manifest:")
    previous_owner: Optional[int] = None
    previous_group: Optional[Path] = None
    for source, owner, ep, group in episode_rows:
        if owner.tmdb_id != previous_owner:
            print(f"  {owner.title} ({owner.year or '????'}) TMDb {owner.tmdb_id}:")
            previous_owner = owner.tmdb_id
            previous_group = None
        if group.directory != previous_group:
            rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
            print(f"    [{rel}]")
            previous_group = group.directory
        print(
            f"      {source.name:<58} -> {ep.season}x{ep.number:02d}  "
            f"{ep.title}{f' ({ep.air_year})' if ep.air_year else ''} [collection:{requested}]"
        )

    if alternate_rows:
        print()
        print("Alternate/commentary presentations:")
        for source, owner, primary_source, group in alternate_rows:
            profile = _presentation_stream_profile(source)
            primary_profile = _presentation_stream_profile(primary_source)
            rel = group.directory.relative_to(input_dir) if group.directory != input_dir else Path(".")
            print(
                f"  {rel}/{source.name} -> Extras [{owner.title}; visual duplicate of {primary_source.name}; "
                f"audio {profile['audio']} vs primary {primary_profile['audio']}]"
            )

    print()
    print("Embedded movies:")
    if movie_rows:
        for source, movie_match, transfer in movie_rows:
            print(
                f"  {source.relative_to(input_dir)} -> {movie_match.title} "
                f"({movie_match.year or '????'}) TMDb {movie_match.tmdb_id} -> {transfer.destination}"
            )
    if deferred_movies:
        for hint, movie_match, groups in deferred_movies:
            suffix = " (add --movies-output PATH to include it in the approved plan)" if movies_output is None else ""
            print(
                f"  detected but left untouched: {movie_match.title} ({movie_match.year or '????'}) "
                f"TMDb {movie_match.tmdb_id} from {len(groups)} rip dir(s){suffix}"
            )
    if not movie_rows and not deferred_movies:
        print("  (none detected)")

    print()
    print("Collection extras:")
    if extras_transfers:
        for transfer in extras_transfers:
            reason = extras_reason.get(transfer.source, "extra")
            print(f"  {transfer.source.relative_to(input_dir)} [{reason}] -> {transfer.destination}")
    else:
        print("  (none)")

    all_transfers = [*episode_transfers, *extras_transfers, *(row[2] for row in movie_rows)]
    destination_conflicts = preflight_transfers(all_transfers, allow_existing=args.dry_run)
    source_paths = _collection_plan_source_paths(
        tv_groups, movie_groups, bonus_groups, unresolved_groups
    )
    moved_sources = {t.source.resolve() for t in all_transfers}
    untouched_sources = [p for p in source_paths if p.resolve() not in moved_sources]

    print()
    print("Collection summary:")
    print(f"  {primary_match.title} episodes: {len(primary_clusters)}")
    print(f"  {companion_match.title} episodes: {len(companion_clusters)}")
    print(f"  TV episodes total:          {len(episode_transfers)}")
    print(f"  Alternate presentations:   {len(alternate_rows)}")
    print(f"  Other/bonus extras moved:   {len(extras_transfers) - len(alternate_rows)}")
    print(f"  Embedded movies moved:      {len(movie_rows)}")
    print(f"  Embedded movies deferred:   {len(deferred_movies)}")
    print(f"  Source files untouched:     {len(untouched_sources)}")
    print(f"  Destination conflicts:      {len(destination_conflicts)}")
    print_destination_conflicts(destination_conflicts)
    if unresolved_groups:
        print("  Unresolved collection groups:")
        for hint, groups in unresolved_groups:
            print(f"    - {hint!r}: {len(groups)} rip dir(s)")

    plan_executable = (
        not destination_conflicts
        and not unresolved_groups
        and not (movies_output is not None and deferred_movies)
    )
    signature = _collection_signature(
        primary_match=primary_match, companion_match=companion_match,
        order_choice=requested, episode_rows=episode_rows,
        alternate_rows=alternate_rows, extras_transfers=extras_transfers,
        movie_rows=movie_rows,
    )

    db = discovery_db()
    if db is not None and not args.dry_run:
        settings = _tv_plan_settings(args)
        key = _plan_key("tv_collection", input_dir, output_root, extras_root, primary_match, settings)
        prior_plan, reason = db.load_valid_plan(
            key, source_paths, match=primary_match, expected_settings=settings
        )
        if prior_plan is None:
            raise MKVPlexError(
                "Refusing collection execution without an exact approved dry-run plan: "
                + reason + ". Re-run the same command with --dry-run --db first."
            )
        if prior_plan.get("collection_signature") != signature:
            raise MKVPlexError(
                "Current collection episode/movie/alternate mapping differs from the approved dry run. "
                "Re-run with --dry-run --db first."
            )
        print(f"Validated approved collection plan {key[:16]}…")
    elif not args.dry_run:
        raise MKVPlexError(
            "Collection execution requires --db and an explicitly accepted dry-run plan."
        )

    if args.dry_run:
        if not plan_executable:
            print("Dry-run collection plan is NOT executable and will not be saved.")
            return 2
        available_keys = [key for key in ("regular", "dvd", "production") if key in order_options]
        if args.yes:
            if str(getattr(args, "collection_order", "auto")) == "auto" and len(available_keys) > 1:
                raise MKVPlexError(
                    "--yes will not choose among multiple collection episode orderings. "
                    "Add --collection-order regular|dvd|production."
                )
            _store_collection_plan_snapshot(
                args, input_dir=input_dir, output_root=output_root, extras_root=extras_root,
                primary_match=primary_match, companion_match=companion_match,
                source_paths=source_paths, signature=signature,
                transfers=episode_transfers, extras_transfers=extras_transfers,
                movie_transfers=[row[2] for row in movie_rows],
            )
            print(
                f"Dry run; accepted collection plan ({requested}). No media changed. "
                f"Would move {len(all_transfers)} source file(s)."
            )
            return 0

        key_labels = {"regular": "r", "dvd": "d", "production": "p"}
        choices = ", ".join(f"[{key_labels[k]}] {k}" for k in available_keys)
        print()
        print(f"Collection-plan review: currently previewing {requested} order.")
        while True:
            raw = input(f"   [a] accept this plan, {choices}, [q] quit: ").strip().lower()
            if raw in {"q", "quit"}:
                print("Dry run; collection plan not accepted. No media changed.")
                return 0
            if raw in {"a", "accept", "yes", "y"}:
                _store_collection_plan_snapshot(
                    args, input_dir=input_dir, output_root=output_root, extras_root=extras_root,
                    primary_match=primary_match, companion_match=companion_match,
                    source_paths=source_paths, signature=signature,
                    transfers=episode_transfers, extras_transfers=extras_transfers,
                    movie_transfers=[row[2] for row in movie_rows],
                )
                print(
                    f"Dry run; accepted collection plan ({requested}). No media changed. "
                    f"Would move {len(all_transfers)} source file(s)."
                )
                return 0
            selected = next((k for k, short in key_labels.items() if raw == short and k in available_keys), None)
            if selected is not None:
                if selected == requested:
                    print(f"   {selected} order is already displayed.")
                    continue
                setattr(args, "_collection_order_choice", selected)
                print(f"   Replanning collection with {selected} order; media fingerprints/TMDb responses remain cached.")
                return do_collection_plan(
                    args, client, input_dir=input_dir, output_root=output_root,
                    extras_root=extras_root, primary_match=primary_match,
                    tv_groups=tv_groups, report=report,
                    total_regular_episodes=total_regular_episodes,
                )
            print("   Invalid selection.")

    if not confirm("Apply this approved collection plan?", args.yes, False):
        print("Cancelled; no changes made.")
        return 0
    operations = _execute_collection_transfers(
        args, output_root=output_root, extras_root=extras_root,
        movies_output=movies_output, episode_rows=episode_rows,
        episode_transfers=episode_transfers, extras_transfers=extras_transfers,
        movie_rows=movie_rows,
    )
    print(
        "Done (" + ", ".join(sorted(set(operations))) + "). "
        f"TV episodes: {len(episode_transfers)}; alternates/extras: {len(extras_transfers)}; "
        f"movies: {len(movie_rows)}."
    )
    if discovery_db() is not None:
        print(f"  {discovery_db().stats_line()}")
    return 0


__all__ = ['_visual_clusters', '_COLLECTION_GENERIC_WORDS', '_distinctive_title_tokens', '_regular_episode_count', '_collection_companion_candidates', '_choose_visual_cluster_primary', '_movie_keyword_signal', 'analyze_collection_overflow', '_collection_match_identity', '_collection_signature', '_collection_plan_source_paths', '_store_collection_plan_snapshot', '_collection_movie_rows', '_execute_collection_transfers', 'do_collection_plan']
