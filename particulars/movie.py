"""Movie planning and execution flow."""
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
from .models import MKVPlexError, Match, Transfer, VERSION
from .common import format_duration
from .discovery import DiscoveryDB, _movie_plan_settings, _plan_key, discovery_db
from .naming import _match_tokens, canonical_spaces, find_movie_mkvs, movie_basename, normalize_for_match, parse_source_name, similarity
from .media import movie_filename, probe_durations
from .tmdb import TMDbClient, attach_imdb_id, build_matches, choose_match
from .fsops import cache_lookup, cache_store, confirm, ensure_output_root, execute_plan, human_size, md5sum, open_cache, preflight_transfers, print_destination_conflicts, resolve_title_and_year

def movie_expected_runtime_seconds(client: TMDbClient, match: Match) -> Optional[float]:
    """Return TMDb's movie runtime in seconds when available."""
    try:
        data = client.movie_details(match.tmdb_id)
    except MKVPlexError:
        return None
    runtime = data.get("runtime")
    try:
        minutes = float(runtime)
    except (TypeError, ValueError):
        return None
    return minutes * 60.0 if minutes > 0 else None


def _movie_track_title_hint(input_dir: Path, path: Path) -> Optional[str]:
    """Extract a possible distinct movie title from a track's local context.

    Parent-directory names are preferred for collection/multi-disc layouts.
    Direct MakeMKV filenames are useful only after stripping the tNN suffix.
    Generic provenance directories such as Extras/Bonus/Disc N are ignored.
    """
    generic = {
        "big", "extras", "extra", "bonus", "bonus disc", "special features",
        "special feature", "features", "featurettes", "disc", "disk",
    }
    candidates: list[str] = []
    if path.parent != input_dir:
        try:
            rel = path.parent.relative_to(input_dir)
            candidates.extend(reversed(rel.parts))
        except ValueError:
            candidates.append(path.parent.name)
    stem = re.sub(r"(?i)[ _.-]*t\d+\s*$", "", path.stem).strip()
    stem = re.sub(r"(?i)[ _.-]*(?:title|track)[ _.-]*\d+\s*$", "", stem).strip()
    if stem:
        candidates.append(stem)

    for raw in candidates:
        hints = parse_source_name(Path(raw))
        title = canonical_spaces(hints.title)
        norm = normalize_for_match(title)
        if not title or norm in generic:
            continue
        if re.fullmatch(r"(?:disc|disk|d)\s*\d+", norm):
            continue
        if re.fullmatch(r"t\s*\d+", norm):
            continue
        return title
    return None


def _same_movie_title_hint(hint: str, primary_title: str) -> bool:
    """Return True when HINT looks like another label for the primary movie."""
    if normalize_for_match(hint) == normalize_for_match(primary_title):
        return True
    generic = {"movie", "film", "disc", "disk", "bonus", "edition", "cut", "extended"}
    left = {t for t in _match_tokens(hint) if t not in generic}
    right = {t for t in _match_tokens(primary_title) if t not in generic}
    if not left or not right:
        return similarity(hint, primary_title) >= 0.86
    overlap = len(left & right) / max(1, min(len(left), len(right)))
    return overlap >= 0.80 or similarity(" ".join(sorted(left)), " ".join(sorted(right))) >= 0.86


def _select_primary_movie_track(
    input_dir: Path, primary_title: str, tracks: list[Path],
    durations: dict[Path, float], expected_runtime: Optional[float]
) -> Path:
    """Choose the main feature from title context + runtime + size."""
    def title_rank(path: Path) -> int:
        hint = _movie_track_title_hint(input_dir, path)
        if hint is None:
            return 1
        return 0 if _same_movie_title_hint(hint, primary_title) else 2

    if expected_runtime is None:
        return min(tracks, key=lambda p: (title_rank(p), -p.stat().st_size))
    # A real feature should usually be within 25%/25 minutes of provider runtime.
    tolerance = max(25.0 * 60.0, expected_runtime * 0.25)
    plausible = [p for p in tracks if abs(durations[p] - expected_runtime) <= tolerance]
    if not plausible:
        closest = min(
            tracks,
            key=lambda p: (abs(durations[p] - expected_runtime), title_rank(p), -p.stat().st_size),
        )
        raise MKVPlexError(
            "No physical track is compatible with the selected movie runtime: "
            f"{primary_title!r} expects about {expected_runtime / 60.0:.0f}m, but the closest "
            f"track is {closest.name!r} at {durations[closest] / 60.0:.1f}m. "
            "Refusing to manufacture a movie feature from an incompatible source corpus."
        )
    return min(
        plausible,
        key=lambda p: (title_rank(p), abs(durations[p] - expected_runtime), -p.stat().st_size),
    )


def _embedded_movie_matches(
    client: TMDbClient,
    input_dir: Path,
    primary_match: Match,
    primary_source: Path,
    tracks: list[Path],
    durations: dict[Path, float],
) -> tuple[list[tuple[Path, Match, float]], list[tuple[Path, float, str]]]:
    """Find confidently named companion movies among long-form secondary tracks.

    Returns (embedded movie rows, long-form review rows).  We intentionally do
    not guess from opaque tNN filenames: a second movie is promoted only when
    local naming supplies a distinct title and TMDb agrees strongly.
    """
    primary_runtime = movie_expected_runtime_seconds(client, primary_match)
    long_floor = 45.0 * 60.0
    if primary_runtime:
        long_floor = max(40.0 * 60.0, min(primary_runtime * 0.55, 75.0 * 60.0))

    by_hint: dict[str, list[Path]] = {}
    review: list[tuple[Path, float, str]] = []
    for path in tracks:
        if path == primary_source or durations[path] < long_floor:
            continue
        hint = _movie_track_title_hint(input_dir, path)
        if not hint or _same_movie_title_hint(hint, primary_match.title):
            review.append((path, durations[path], "long-form secondary / alternate feature"))
            continue
        by_hint.setdefault(hint, []).append(path)

    found: list[tuple[Path, Match, float]] = []
    claimed: set[Path] = set()
    for hint, hinted_tracks in sorted(by_hint.items(), key=lambda row: row[0].casefold()):
        matches = build_matches(client, "movie", hint, None)
        if not matches:
            for path in hinted_tracks:
                review.append((path, durations[path], f"long-form secondary; no TMDb movie match for {hint!r}"))
            continue
        candidate = matches[0]
        if candidate.tmdb_id == primary_match.tmdb_id or candidate.score < 0.72:
            for path in hinted_tracks:
                review.append((path, durations[path], f"long-form secondary; weak/distinct-title match {hint!r}"))
            continue
        expected = movie_expected_runtime_seconds(client, candidate)
        if expected is not None:
            tolerance = max(20.0 * 60.0, expected * 0.22)
            plausible = [p for p in hinted_tracks if abs(durations[p] - expected) <= tolerance]
            if not plausible:
                for path in hinted_tracks:
                    review.append((path, durations[path], f"long-form secondary; {hint!r} runtime disagrees with TMDb"))
                continue
            chosen = min(plausible, key=lambda p: (abs(durations[p] - expected), -p.stat().st_size))
        else:
            chosen = max(hinted_tracks, key=lambda p: p.stat().st_size)
        enriched = attach_imdb_id(client, candidate)
        found.append((chosen, enriched, durations[chosen]))
        claimed.add(chosen)
        for path in hinted_tracks:
            if path != chosen:
                review.append((path, durations[path], f"alternate/extra for embedded movie {enriched.title}"))

    # Avoid duplicate promotion if two title hints resolve to the same movie.
    dedup: dict[int, tuple[Path, Match, float]] = {}
    for row in found:
        path, match, duration = row
        old = dedup.get(match.tmdb_id)
        if old is None:
            dedup[match.tmdb_id] = row
            continue
        old_path, _old_match, old_duration = old
        expected = movie_expected_runtime_seconds(client, match)
        if expected is None:
            better = path.stat().st_size > old_path.stat().st_size
        else:
            better = abs(duration - expected) < abs(old_duration - expected)
        if better:
            review.append((old_path, old_duration, f"alternate/duplicate for embedded movie {match.title}"))
            dedup[match.tmdb_id] = row
        else:
            review.append((path, duration, f"alternate/duplicate for embedded movie {match.title}"))
    return list(dedup.values()), review


def _movie_extras_destination(
    extras_root: Path, owner_base: str, input_dir: Path, track: Path
) -> Path:
    """Preserve collection provenance under the owning movie's Extras directory."""
    owner_root = extras_root / owner_base
    try:
        rel_parent = track.parent.relative_to(input_dir)
    except ValueError:
        rel_parent = Path(input_dir.name)
    if str(rel_parent) in {"", "."}:
        rel_parent = Path(input_dir.name)
    return owner_root / rel_parent / track.name


def store_movie_plan_snapshot(
    args: argparse.Namespace, *, input_dir: Path, output_root: Path,
    extras_root: Optional[Path], match: Match, source_paths: Iterable[Path],
    transfers: list[Transfer], extras_transfers: list[Transfer],
    embedded_rows: list[tuple[Path, Match, float]],
    review_rows: list[tuple[Path, float, str]],
) -> None:
    db = discovery_db()
    if db is None or not args.dry_run:
        return
    settings = _movie_plan_settings(args)
    plan = {
        "command": "movie",
        "tool_version": VERSION,
        "match": DiscoveryDB._match_identity(match),
        "settings": settings,
        "embedded_movies": [
            {
                "source": str(source.resolve()),
                "duration": duration,
                "match": DiscoveryDB._match_identity(embedded),
            }
            for source, embedded, duration in embedded_rows
        ],
        "long_form_review": [
            {"source": str(source.resolve()), "duration": duration, "reason": reason}
            for source, duration, reason in review_rows
        ],
        "transfers": [
            {"source": str(t.source), "destination": str(t.destination)} for t in transfers
        ],
        "extras_transfers": [
            {"source": str(t.source), "destination": str(t.destination)} for t in extras_transfers
        ],
    }
    key = _plan_key("movie", input_dir, output_root, extras_root, match, settings)
    db.store_plan(
        plan_key=key, input_root=input_dir, output_root=output_root, extras_root=extras_root,
        tmdb_id=match.tmdb_id, source_paths=source_paths, plan=plan,
    )
    print(f"Dry-run plan saved to DB: {db.path}")
    print(f"  plan key: {key[:16]}…")
    print(f"  {db.stats_line()}")


def _embedded_identity_rows(rows: list[tuple[Path, Match, float]]) -> list[dict[str, Any]]:
    return [
        {
            "source": str(source.resolve()),
            "match": DiscoveryDB._match_identity(match),
        }
        for source, match, _duration in rows
    ]


def do_movie(args: argparse.Namespace, client: TMDbClient) -> int:
    input_dir = Path(args.input).expanduser().resolve()
    output_root = ensure_output_root(Path(args.output))
    extras_arg = getattr(args, "extras", None)
    extras_root = ensure_output_root(Path(extras_arg)) if extras_arg else None
    hints = parse_source_name(input_dir)
    query, year = resolve_title_and_year(args, hints)

    all_tracks = find_movie_mkvs(input_dir)
    tree_mode = any(track.parent != input_dir for track in all_tracks)
    print(f"Input:    {input_dir}")
    print(f"Parsed:   title={query!r}, year={year or 'unknown'}")
    print(
        f"Layout:   {'movie collection tree' if tree_mode else 'single movie rip directory'} "
        f"({len(all_tracks)} MKV file{'s' if len(all_tracks) != 1 else ''})"
    )

    match = choose_match(
        client, "movie", query, year,
        explicit_imdb=args.imdb,
        assume_yes=args.yes,
    )
    expected_runtime = movie_expected_runtime_seconds(client, match)
    print()
    print(f"Probing {len(all_tracks)} movie track(s) with ffprobe...")
    durations = probe_durations(all_tracks, workers=max(1, int(getattr(args, "probe_workers", 4))))
    source = _select_primary_movie_track(input_dir, match.title, all_tracks, durations, expected_runtime)
    runtime_note = f", TMDb ~{expected_runtime/60.0:.0f}m" if expected_runtime else ""
    print(
        f"Primary feature: {source.name} "
        f"[{format_duration(durations[source])}, {human_size(source.stat().st_size)}{runtime_note}]"
    )

    embedded_rows, review_rows = _embedded_movie_matches(
        client, input_dir, match, source, all_tracks, durations
    )
    embedded_rows.sort(key=lambda row: (row[1].title.casefold(), str(row[0]).casefold()))
    selected_sources = {source, *(row[0] for row in embedded_rows)}

    if embedded_rows:
        print()
        print("Embedded/companion movies detected:")
        for embedded_source, embedded_match, duration in embedded_rows:
            rel = embedded_source.relative_to(input_dir)
            print(
                f"  {rel} [{format_duration(duration)}] -> "
                f"{embedded_match.title} ({embedded_match.year or '????'}) "
                f"TMDb {embedded_match.tmdb_id} score={embedded_match.score:.3f}"
            )
    if review_rows:
        print()
        print("Long-form secondary tracks requiring no automatic movie identity:")
        for path, duration, reason in sorted(review_rows, key=lambda row: str(row[0]).casefold()):
            print(f"  {path.relative_to(input_dir)} [{format_duration(duration)}] [{reason}]")

    digest: Optional[str] = None
    cache: Optional[sqlite3.Connection] = None
    if args.hash:
        print(f"Hashing:  {source.name} (MD5)...")
        digest = md5sum(source)
        print(f"MD5:      {digest}")
        cache = open_cache(Path(args.cache))
        known = cache_lookup(cache, digest)
        if known:
            print(
                "Known hash: "
                f"{known['title']} ({known['year'] or '????'}) "
                f"{known['imdb_id'] or ''}".rstrip()
            )

    # Build one canonical movie transfer per confidently identified feature.
    movie_rows: list[tuple[Path, Match, Path, Transfer]] = []
    primary_base = movie_basename(match)
    primary_dir = output_root / primary_base
    movie_rows.append((source, match, primary_dir, Transfer(source, primary_dir / movie_filename(match, source))))
    for embedded_source, embedded_match, _duration in embedded_rows:
        base = movie_basename(embedded_match)
        dest_dir = output_root / base
        movie_rows.append((
            embedded_source,
            embedded_match,
            dest_dir,
            Transfer(embedded_source, dest_dir / movie_filename(embedded_match, embedded_source)),
        ))
    transfers = [row[3] for row in movie_rows]

    # Preserve all unselected tracks. Tracks in a named companion-movie subtree
    # are archived beneath that companion's canonical extras root.
    embedded_by_parent: dict[Path, Match] = {
        src.parent: ematch for src, ematch, _duration in embedded_rows if src.parent != input_dir
    }
    extras_transfers: list[Transfer] = []
    if extras_root is not None:
        for track in sorted(all_tracks, key=lambda p: str(p).casefold()):
            if track in selected_sources:
                continue
            owner = embedded_by_parent.get(track.parent, match)
            owner_base = movie_basename(owner)
            extras_transfers.append(Transfer(
                track, _movie_extras_destination(extras_root, owner_base, input_dir, track)
            ))

    # Validate an approved DB plan only after current structural/movie discovery
    # has been reconstructed from cached probes/TMDb data.
    db = discovery_db()
    if db is not None and not args.dry_run:
        settings = _movie_plan_settings(args)
        prior_key = _plan_key("movie", input_dir, output_root, extras_root, match, settings)
        prior_plan, reason = db.load_valid_plan(
            prior_key, all_tracks, match=match, expected_settings=settings
        )
        if prior_plan is None:
            raise MKVPlexError(
                "Refusing --db execution without an exact approved dry-run plan: "
                + reason + ". Re-run the same command with --dry-run --db first."
            )
        saved_embedded = [
            {"source": row.get("source"), "match": row.get("match")}
            for row in (prior_plan.get("embedded_movies") or [])
        ]
        current_embedded = _embedded_identity_rows(embedded_rows)
        if saved_embedded != current_embedded:
            raise MKVPlexError(
                "Embedded/companion movie identities differ from the approved dry run. "
                "Re-run with --dry-run --db first."
            )
        print(
            f"Validated prior dry-run plan {prior_key[:16]}…; source sampled-MD5 inventory, "
            "primary movie, and companion-movie identities match."
        )

    print()
    print("Movie plan:")
    for movie_source, movie_match, destination_dir, transfer in movie_rows:
        print(
            f"  {movie_match.title} ({movie_match.year or '????'}) "
            f"TMDb {movie_match.tmdb_id} IMDb {movie_match.imdb_id or '-'}"
        )
        print(
            f"    {movie_source.relative_to(input_dir)} "
            f"[{format_duration(durations[movie_source])}, {human_size(movie_source.stat().st_size)}]"
        )
        print(f"      -> {transfer.destination}")
        print(f"    permissions: chmod -R {args.mode:o} {destination_dir}")

    if extras_root is not None:
        print()
        print("Extras archive:")
        if extras_transfers:
            for item in extras_transfers:
                print(f"  {item.source.relative_to(input_dir)} -> {item.destination}")
        else:
            print("  (no non-feature MKV tracks)")
        print(f"Extras operation: {'COPY' if args.copy else 'MOVE'} ({len(extras_transfers)} file(s))")
    elif len(all_tracks) > len(selected_sources):
        print()
        print(
            f"Unselected tracks left untouched: {len(all_tracks) - len(selected_sources)} "
            "(no EXTRAS_OUTPUT supplied)"
        )

    destination_conflicts = preflight_transfers(
        [*transfers, *extras_transfers], allow_existing=args.dry_run
    )

    action = "COPY" if args.copy else "MOVE"
    physical_count = len(transfers) + len(extras_transfers)
    untouched = len(all_tracks) - len(selected_sources) - len(extras_transfers)
    print()
    print("Plan summary:")
    print(f"  Movies identified:        {len(transfers)}")
    print(f"  Primary movies:           1")
    print(f"  Companion movies:         {len(embedded_rows)}")
    print(f"  Extras to archive:        {len(extras_transfers)}")
    print(f"  Source files to {action}:      {physical_count}")
    if untouched:
        print(f"  Source files untouched:   {untouched}")
    print(f"  Destination conflicts:    {len(destination_conflicts)}")
    print_destination_conflicts(destination_conflicts)

    if not destination_conflicts:
        store_movie_plan_snapshot(
            args, input_dir=input_dir, output_root=output_root, extras_root=extras_root,
            match=match, source_paths=all_tracks, transfers=transfers,
            extras_transfers=extras_transfers, embedded_rows=embedded_rows,
            review_rows=review_rows,
        )
    if not confirm("Apply this plan?", args.yes, args.dry_run):
        if args.dry_run:
            print(
                f"Dry run; no changes made. Would {action.lower()} {physical_count} source file(s): "
                f"{len(transfers)} movie feature(s) and {len(extras_transfers)} extra(s)."
            )
            if destination_conflicts:
                print(
                    "Dry-run plan is NOT executable and was not saved as an approved --db plan; "
                    "resolve the destination conflict(s) and run --dry-run --db again."
                )
        else:
            print("Cancelled; no changes made.")
        return 2 if (args.dry_run and destination_conflicts) else 0

    operations: list[str] = []
    for _movie_source, _movie_match, destination_dir, transfer in movie_rows:
        operations.extend(execute_plan(
            destination_dir, [transfer], copy_only=args.copy,
            verify_md5=args.verify_md5, mode=args.mode,
        ))
    if extras_transfers:
        # Transfers can span multiple canonical extras directories; execute one
        # at a time so each owning movie tree receives the requested chmod.
        for transfer in extras_transfers:
            owner_dir = transfer.destination.parent
            operations.extend(execute_plan(
                owner_dir, [transfer], copy_only=args.copy,
                verify_md5=args.verify_md5, mode=args.mode,
            ))
    if digest and cache:
        cache_store(cache, digest, "movie", match.imdb_id, match.title, match.year)
    print(
        "Done (" + ", ".join(sorted(set(operations))) + "). "
        f"Movies: {len(transfers)}; extras archived: {len(extras_transfers)}; "
        f"source files processed: {physical_count}."
    )
    if discovery_db() is not None:
        print(f"  {discovery_db().stats_line()}")
    return 0


__all__ = ['movie_expected_runtime_seconds', '_movie_track_title_hint', '_same_movie_title_hint', '_select_primary_movie_track', '_embedded_movie_matches', '_movie_extras_destination', 'store_movie_plan_snapshot', '_embedded_identity_rows', 'do_movie']
