"""TV orchestration, season inference, and multi-series bucketing."""
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
from .models import COMPLETE_SERIES_SENTINEL, MKVPlexError, Match, TvRipGroup
from .discovery import discovery_db, remap_episode_seasons
from .naming import canonical_spaces, normalize_for_match, parse_source_name, similarity
from .media import _aggregate_complete_series_prefix_hint, probe_container_title
from .discs import find_tv_rip_groups
from .tmdb import TMDbClient, _regular_tv_season_rows, _show_runtime_minutes, attach_imdb_id, build_matches, choose_match, regular_series_episodes
from .fsops import ensure_output_root, resolve_title_and_year
from .collection import _movie_keyword_signal, analyze_collection_overflow, do_collection_plan
from .volume import _group_series_query, common_numbered_volume
from .tvplan import _do_tv_series

def _print_season_options(match: Match, rows: list[tuple[int, int, Optional[int], str]]) -> None:
    print()
    print(f"Unlabeled {match.title} rip directories: possible seasons:")
    for idx, (number, count, air_year, name) in enumerate(rows, start=1):
        year_text = f", {air_year}" if air_year else ""
        generic_name = normalize_for_match(name) == normalize_for_match(f"Season {number}")
        if generic_name:
            print(f"  {idx}. Season {number} ({count} episodes{year_text})")
        else:
            print(f"  {idx}. {name} (Season {number}, {count} episodes{year_text})")


def _apply_unlabeled_season(groups: list[TvRipGroup], season: int) -> list[TvRipGroup]:
    return [
        replace(g, season=season) if g.season is None and not g.final_season else g
        for g in groups
    ]


def _infer_unlabeled_series_season(
    client: TMDbClient,
    match: Match,
    groups: list[TvRipGroup],
    *,
    assume_yes: bool = False,
    forced_season: Optional[int] = None,
    selected_season: Optional[int] = None,
) -> tuple[list[TvRipGroup], Optional[int], str, list[tuple[int, int, Optional[int], str]]]:
    """Fill unlabeled disc seasons conservatively.

    The function itself no longer forces an interactive season decision before
    media analysis.  When several seasons are plausible it returns the choices
    to do_tv(), which can preview candidate plans repeatedly in one invocation.
    """
    unresolved = [g for g in groups if g.season is None and not g.final_season]
    if not unresolved:
        return groups, None, "", []

    if forced_season is not None:
        return _apply_unlabeled_season(groups, forced_season), forced_season, "--season override", []

    explicit = sorted({g.season for g in groups if g.season is not None})
    inferred: Optional[int] = explicit[0] if len(explicit) == 1 else None
    source = "sibling rip directories"

    if inferred is not None:
        return _apply_unlabeled_season(groups, inferred), inferred, source, []

    if explicit:
        return groups, None, "", []

    regular_rows = _regular_tv_season_rows(client, match)
    if len(regular_rows) == 1:
        inferred = regular_rows[0][0]
        return (
            _apply_unlabeled_season(groups, inferred), inferred,
            "TMDb single-season series", regular_rows,
        )

    if len(regular_rows) > 1:
        if selected_season is not None:
            if selected_season == COMPLETE_SERIES_SENTINEL:
                return (
                    _apply_unlabeled_season(groups, COMPLETE_SERIES_SENTINEL),
                    COMPLETE_SERIES_SENTINEL, "accepted complete-series plan", regular_rows,
                )
            valid = {row[0] for row in regular_rows}
            if selected_season not in valid:
                raise MKVPlexError(
                    f"Season {selected_season} is not a regular TMDb season for {match.title}."
                )
            return (
                _apply_unlabeled_season(groups, selected_season), selected_season,
                "accepted/previewed TMDb season", regular_rows,
            )
        if assume_yes:
            choices = ", ".join(
                f"{n} ({count} eps{', ' + str(y) if y else ''})"
                for n, count, y, _name in regular_rows
            )
            raise MKVPlexError(
                f"Season is ambiguous for unlabeled {match.title} rip directories. "
                f"Available regular seasons: {choices}. Re-run with --season N."
            )
        return groups, None, "season plan review required", regular_rows

    return groups, None, "", []


def _resolve_tv_series_buckets(
    args: argparse.Namespace,
    client: TMDbClient,
    input_dir: Path,
    root_query: str,
    root_year: Optional[int],
    groups: list[TvRipGroup],
    primary_match: Match,
) -> list[tuple[str, Optional[int], Match, list[TvRipGroup]]]:
    """Resolve each distinct show represented inside one physical input tree.

    A full-series package can legally contain a sequel/miniseries whose discs
    do not carry Season N in their names (e.g. Dexter + Dexter: New Blood).
    Distinct directory title hints are resolved once each and groups resolving
    to the same TMDb id are merged back together.
    """
    primary_groups: list[TvRipGroup] = []
    secondary_hints: dict[str, tuple[str, Optional[int], list[TvRipGroup]]] = {}
    collection_movies: list[tuple[str, Match, list[TvRipGroup]]] = []
    collection_bonus_groups: list[TvRipGroup] = []
    collection_unresolved_groups: list[tuple[str, list[TvRipGroup]]] = []

    for group in groups:
        hint = _group_series_query(input_dir, root_query, group)
        hint_key = normalize_for_match(hint)
        if similarity(hint, root_query) >= 0.88:
            primary_groups.append(group)
            continue
        parsed = parse_source_name(group.directory)
        row = secondary_hints.get(hint_key)
        if row is None:
            secondary_hints[hint_key] = (hint, parsed.year, [group])
        else:
            row[2].append(group)

    buckets_by_id: dict[int, tuple[str, Optional[int], Match, list[TvRipGroup]]] = {
        primary_match.tmdb_id: (root_query, root_year, primary_match, list(primary_groups))
    }

    for _key, (hint, hint_year, hint_groups) in sorted(secondary_hints.items()):
        print()
        print(f"Additional series-like rip group detected: {hint!r}")
        tv_matches = build_matches(client, "tv", hint, hint_year)
        movie_matches = build_matches(client, "movie", hint, hint_year)
        best_tv = tv_matches[0] if tv_matches else None
        best_movie = movie_matches[0] if movie_matches else None
        movie_keyword = _movie_keyword_signal(hint) or any(
            _movie_keyword_signal(group.directory.name) for group in hint_groups
        )
        if best_movie is not None and (
            best_tv is None
            or (best_movie.score >= 0.64 and best_movie.score >= best_tv.score + 0.08)
            or (best_movie.score >= 0.80 and best_tv.score < 0.72)
            or (movie_keyword and best_movie.score >= 0.75 and (best_tv is None or best_movie.score >= best_tv.score - 0.02))
        ):
            best_movie = attach_imdb_id(client, best_movie)
            signal = ", movie-keyword" if movie_keyword else ""
            print(
                f"  Looks like non-TV material: {best_movie.title} "
                f"({best_movie.year or '????'}) [movie, score={best_movie.score:.3f}{signal}]"
            )
            collection_movies.append((hint, best_movie, list(hint_groups)))
            print(
                f"  Collection planner will keep {len(hint_groups)} rip "
                f"director{'y' if len(hint_groups) == 1 else 'ies'} as embedded movie material."
            )
            continue
        if not tv_matches:
            hint_norm = normalize_for_match(hint)
            if set(hint_norm.split()) & {"bonus", "extras", "extra", "supplements", "supplement"}:
                collection_bonus_groups.extend(hint_groups)
                print(
                    f"  Bonus/supplement label detected; collection planner will archive "
                    f"{len(hint_groups)} rip director{'y' if len(hint_groups) == 1 else 'ies'} as Extras."
                )
            else:
                collection_unresolved_groups.append((hint, list(hint_groups)))
                print(
                    f"  No plausible TV/movie match; leaving {len(hint_groups)} rip "
                    f"director{'y' if len(hint_groups) == 1 else 'ies'} unresolved."
                )
            continue
        secondary_match = choose_match(
            client, "tv", hint, hint_year,
            explicit_imdb=None,
            assume_yes=args.yes,
        )
        existing = buckets_by_id.get(secondary_match.tmdb_id)
        if existing is not None:
            existing[3].extend(hint_groups)
            continue
        buckets_by_id[secondary_match.tmdb_id] = (
            hint, hint_year, secondary_match, list(hint_groups)
        )

    # Expose collection-level non-TV topology to the overflow planner without
    # polluting the ordinary logical-series buckets.
    setattr(args, "_collection_movie_groups", collection_movies)
    setattr(args, "_collection_bonus_groups", collection_bonus_groups)
    setattr(args, "_collection_unresolved_groups", collection_unresolved_groups)

    # A non-primary hint can legitimately resolve back to the primary show.
    # Ensure a root bucket with no groups is omitted rather than processed.
    return [row for row in buckets_by_id.values() if row[3]]



def _authored_title_fallback(groups: list[TvRipGroup], parsed_query: str) -> Optional[str]:
    """Return one consistent authored container title distinct from PARSED_QUERY.

    The fallback is deliberately conservative: probe at most a few tracks and
    use the title only when every non-empty authored title agrees.  This keeps
    disc labels from a mixed tree from becoming a new guessed series identity.
    """
    titles: list[str] = []
    for group in groups:
        for path in group.tracks:
            title = probe_container_title(path)
            if title:
                titles.append(title)
            if len(titles) >= 4:
                break
        if len(titles) >= 4:
            break
    if not titles:
        return None
    first = titles[0]
    first_norm = normalize_for_match(first)
    if any(normalize_for_match(value) != first_norm for value in titles[1:]):
        return None
    # Punctuation can itself carry discovery structure (notably a colon that
    # separates a localized subtitle), even when lexical normalization makes
    # the strings equal.  Only suppress a truly identical authored label.
    if canonical_spaces(first).casefold() == canonical_spaces(parsed_query).casefold():
        return None
    return first


def _choose_primary_tv_match(
    client: TMDbClient,
    query: str,
    year: Optional[int],
    groups: list[TvRipGroup],
    *,
    explicit_imdb: Optional[str],
    assume_yes: bool,
) -> Match:
    """Choose the primary TV identity, consulting authored title only on miss."""
    try:
        return choose_match(
            client, "tv", query, year,
            explicit_imdb=explicit_imdb, assume_yes=assume_yes,
        )
    except MKVPlexError as exc:
        if not str(exc).startswith("No tv matches found for"):
            raise
        authored = _authored_title_fallback(groups, query)
        if not authored:
            raise
        print(f"Authored media title fallback: {authored!r}")
        return choose_match(
            client, "tv", authored, year,
            explicit_imdb=explicit_imdb, assume_yes=assume_yes,
        )


def do_tv(args: argparse.Namespace, client: TMDbClient) -> int:
    input_dir = Path(args.input).expanduser().resolve()
    output_root = ensure_output_root(Path(args.output))
    extras_arg = getattr(args, "extras", None)
    if not extras_arg:
        raise MKVPlexError(
            "TV mode requires an extras archive root as the third directory: "
            "mkvplex tv INPUT TV_OUTPUT EXTRAS_OUTPUT"
        )
    extras_root = ensure_output_root(Path(extras_arg))
    movies_output_arg = getattr(args, "movies_output", None)
    if movies_output_arg:
        ensure_output_root(Path(movies_output_arg))
    root_hints = parse_source_name(input_dir)
    query, year = resolve_title_and_year(args, root_hints)
    groups = find_tv_rip_groups(input_dir)
    tree_mode = len(groups) > 1 or groups[0].directory != input_dir

    print(f"Input:    {input_dir}")
    print(f"Parsed:   title={query!r}, year={year or 'unknown'}")
    if tree_mode:
        print(f"Layout:   series tree ({len(groups)} rip director{'y' if len(groups) == 1 else 'ies'})")
    else:
        print("Layout:   single TV rip directory")

    primary_match = _choose_primary_tv_match(
        client, query, year, groups,
        explicit_imdb=args.imdb,
        assume_yes=args.yes,
    )

    # Mixed-series bucketing only makes sense when the input actually contains
    # multiple physical rip directories (or a nested rip directory below the
    # supplied root).  For a single rip directory the selected root identity is
    # authoritative.  Re-running the directory-name heuristic here can invent a
    # bogus secondary title from labels such as even when --title/--imdb already
    # resolved the show correctly.
    if tree_mode:
        buckets = _resolve_tv_series_buckets(
            args, client, input_dir, query, year, groups, primary_match
        )
    else:
        buckets = [(query, year, primary_match, list(groups))]
    if len(buckets) > 1:
        print()
        print("Mixed-series input detected:")
        for _q, _y, m, gs in buckets:
            print(f"  {m.title} ({m.year or '????'}) -> {len(gs)} rip director{'y' if len(gs) == 1 else 'ies'}")

    season_counts = getattr(args, "season_counts", None)
    if args.season is not None and season_counts:
        raise MKVPlexError("--season and --season-counts cannot be used together")
    if season_counts and (args.episode_start != 1 or args.episode_count is not None):
        raise MKVPlexError(
            "--season-counts remaps a complete provider episode sequence; "
            "do not combine it with --episode-start/--episode-count."
        )
    if len(buckets) > 1 and (args.season is not None or season_counts):
        raise MKVPlexError(
            "--season/--season-counts is ambiguous for a mixed-series input tree. Process the "
            "desired series subtree separately or omit the override."
        )

    for bucket_query, bucket_year, match, bucket_groups in buckets:
        if hasattr(args, "_effective_season_choice"):
            delattr(args, "_effective_season_choice")
        if hasattr(args, "_numbered_volume"):
            delattr(args, "_numbered_volume")
        original_groups = list(bucket_groups)
        approved_hint: Optional[int] = None
        db = discovery_db()
        if db is not None and not args.dry_run and args.season is None:
            approved_hint = db.approved_tv_season_hint(
                input_root=input_dir, output_root=output_root,
                extras_root=extras_root, match=match,
            )
            if approved_hint is not None:
                setattr(args, "_effective_season_choice", approved_hint)
                setattr(args, "_complete_series_mode", approved_hint == COMPLETE_SERIES_SENTINEL)
                approved_label = (
                    "entire regular series" if approved_hint == COMPLETE_SERIES_SENTINEL
                    else f"Season {approved_hint}"
                )
                print(f"  {match.title}: using approved dry-run {approved_label} choice from DB")

        numbered_volume = (
            common_numbered_volume(input_dir, original_groups)
            if args.season is None and not season_counts
            else None
        )
        if numbered_volume is not None:
            if any(g.season is not None or g.final_season for g in original_groups):
                raise MKVPlexError(
                    f"Retail Volume {numbered_volume} is mixed with explicit Season N labels; "
                    "refusing to guess which identity layer should win."
                )
            setattr(args, "_numbered_volume", numbered_volume)
            resolved = _apply_unlabeled_season(original_groups, COMPLETE_SERIES_SENTINEL)
            inferred_season = COMPLETE_SERIES_SENTINEL
            inference_source = f"numbered retail Volume {numbered_volume} partial-series analysis"
            season_options = _regular_tv_season_rows(client, match)
            setattr(args, "_effective_season_choice", COMPLETE_SERIES_SENTINEL)
            setattr(args, "_complete_series_mode", False)
        elif season_counts:
            if any(g.season is not None or g.final_season for g in original_groups):
                raise MKVPlexError(
                    "--season-counts is for an unlabeled/flattened complete-series corpus; "
                    "the input already contains explicit season markers."
                )
            ordinary = regular_series_episodes(client, match)
            # Validation happens here before expensive media probing and again
            # at the point of use to keep the identity layer fail-closed.
            remap_episode_seasons(ordinary, season_counts)
            resolved = _apply_unlabeled_season(original_groups, COMPLETE_SERIES_SENTINEL)
            inferred_season = COMPLETE_SERIES_SENTINEL
            inference_source = "explicit --season-counts presentation map"
            season_options = [
                (number, count, None, f"Season {number}")
                for number, count in enumerate(season_counts, start=1)
            ]
            setattr(args, "_effective_season_choice", COMPLETE_SERIES_SENTINEL)
            setattr(args, "_complete_series_mode", True)
        else:
            resolved, inferred_season, inference_source, season_options = _infer_unlabeled_series_season(
                client, match, original_groups, assume_yes=args.yes,
                forced_season=args.season, selected_season=approved_hint,
            )

        total_regular_episodes = sum(row[1] for row in season_options) if season_options else 0
        source_track_count = sum(len(g.tracks) for g in original_groups)
        # A corpus only modestly larger than the complete regular-series episode
        # count is a strong candidate for a complete-series archive with a small
        # number of specials/recaps/bonus programs.  This is only a preview
        # hypothesis; the user still sees every episode/extras assignment before
        # accepting it.
        complete_series_candidate = bool(
            season_counts
            or (
                len(season_options) > 1
                and total_regular_episodes > 0
                and source_track_count >= total_regular_episodes
                and source_track_count <= total_regular_episodes + max(12, int(total_regular_episodes * 0.20))
            )
        )

        # A second complete-series shape is the inverse of the ordinary one:
        # very few source files, each a many-hour physical-disc play-all title.
        # Probe that topology *before* defaulting an unlabeled tree to Season 1.
        aggregate_complete_hint: Optional[dict[str, Any]] = None
        if (
            not complete_series_candidate
            and args.season is None
            and numbered_volume is None
            and approved_hint is None
            and len(season_options) > 1
            and total_regular_episodes > 0
            and 1 < source_track_count < total_regular_episodes
        ):
            aggregate_complete_hint = _aggregate_complete_series_prefix_hint(
                original_groups,
                regular_series_episodes(client, match),
                show_runtime_minutes=_show_runtime_minutes(client, match),
                workers=args.probe_workers,
            )
            if aggregate_complete_hint is not None:
                complete_series_candidate = True
                ratio_pct = 100.0 * float(aggregate_complete_hint["ratio"])
                print()
                print("Aggregate complete-series hypothesis:")
                print(
                    f"  {aggregate_complete_hint['giant_count']} giant source title(s); "
                    f"the first {aggregate_complete_hint['prefix_count']} cover "
                    f"{ratio_pct:.1f}% of the {total_regular_episodes}-episode regular-series runtime"
                )
                if aggregate_complete_hint["deferred_count"]:
                    print(
                        f"  {aggregate_complete_hint['deferred_count']} later giant source(s) remain "
                        "outside that runtime envelope and will be deferred, not archived as Extras"
                    )

        # Safety guard for retail "complete/collection" packages whose physical
        # corpus is much larger than the selected TMDb show's entire regular
        # episode inventory.  Such sets can bundle a continuation/spin-off as
        # part of the same marketed series, or use an alternate production/DVD
        # order.  Falling through to a single-season preview in that situation
        # is exactly the kind of over-guess that can archive real episodes as
        # Extras.  An explicit --season remains an intentional escape hatch for
        # users processing a known subtree.
        packaging_tokens = set(normalize_for_match(query).split()) | set(
            normalize_for_match(input_dir.name).split()
        )
        collection_label = bool(packaging_tokens & {"complete", "collection", "boxed", "boxset"})
        collection_overflow_limit = (
            total_regular_episodes + max(20, int(total_regular_episodes * 0.25))
            if total_regular_episodes > 0 else 0
        )
        collection_overflow = bool(
            args.season is None
            and numbered_volume is None
            and approved_hint is None
            and len(original_groups) >= 2
            and len(season_options) > 1
            and collection_label
            and total_regular_episodes > 0
            and source_track_count > collection_overflow_limit
        )
        if collection_overflow:
            overflow = source_track_count - total_regular_episodes
            print()
            print("Collection-level ambiguity detected:")
            print(
                f"  retail/package label suggests a complete collection, but the selected "
                f"TMDb series has {total_regular_episodes} regular episode(s)"
            )
            print(
                f"  {len(original_groups)} TV rip directories contain {source_track_count} source track(s) "
                f"({overflow:+d} versus that complete-series target)"
            )
            print(
                "  Running collection-aware forensic discovery before building any execution plan: "
                "visual-equivalent title clustering, alternate TMDb orderings, companion-series search, "
                "and embedded movie/bonus topology."
            )
            if db is not None and not args.dry_run and str(getattr(args, "collection_order", "auto")) == "auto":
                approved_collection = db.approved_collection_hint(
                    input_root=input_dir, output_root=output_root, extras_root=extras_root, match=match
                )
                if approved_collection is not None:
                    approved_order, approved_companion = approved_collection
                    setattr(args, "_collection_order_choice", approved_order)
                    if approved_companion is not None:
                        setattr(args, "_collection_companion_tmdb_id", approved_companion)
                    print(
                        f"  using approved dry-run collection order {approved_order!r} "
                        "from DB before reconstructing the live plan"
                    )

            report = analyze_collection_overflow(
                args, client, input_dir=input_dir, match=match, groups=original_groups,
                total_regular_episodes=total_regular_episodes,
            )
            if report.get("coherent"):
                return do_collection_plan(
                    args, client, input_dir=input_dir, output_root=output_root,
                    extras_root=extras_root, primary_match=match, tv_groups=original_groups,
                    report=report, total_regular_episodes=total_regular_episodes,
                )
            raise MKVPlexError(
                "Refusing to preview or approve a single-series season plan for this complete collection; "
                "the collection forensic model is still unresolved and real episodes could be misclassified as Extras."
            )

        # Multiple unlabeled seasons are reviewed as candidate plans in one
        # invocation.  Prefer an entire-series hypothesis when the corpus size
        # plan is printed the user can accept it or preview another season.  The
        # SQLite probe cache survives each iteration, while source fingerprinting
        # happens only for the accepted plan.
        if season_options and len(season_options) > 1 and inferred_season is None:
            if not args.dry_run:
                if db is not None:
                    raise MKVPlexError(
                        f"The approved --db plan does not contain an unlabeled-season choice for "
                        f"{match.title}. Re-run with --dry-run --db and accept a season plan."
                    )
                _print_season_options(match, season_options)
                if complete_series_candidate:
                    print(f"  e. Entire regular series ({total_regular_episodes} episodes across {len(season_options)} seasons)")
                while True:
                    extra = ", e=entire series" if complete_series_candidate else ""
                    raw = input(
                        f"   Select season [1-{len(season_options)}{extra}, q=quit]: "
                    ).strip().lower()
                    if raw == "q":
                        raise SystemExit(0)
                    if raw in {"e", "entire", "series", "all"} and complete_series_candidate:
                        chosen = COMPLETE_SERIES_SENTINEL
                        resolved = _apply_unlabeled_season(original_groups, chosen)
                        inferred_season = chosen
                        inference_source = "interactive complete-series selection"
                        setattr(args, "_effective_season_choice", chosen)
                        setattr(args, "_complete_series_mode", True)
                        break
                    if raw.isdigit() and 1 <= int(raw) <= len(season_options):
                        chosen = season_options[int(raw) - 1][0]
                        resolved = _apply_unlabeled_season(original_groups, chosen)
                        inferred_season = chosen
                        inference_source = "interactive TMDb season selection"
                        setattr(args, "_effective_season_choice", chosen)
                        setattr(args, "_complete_series_mode", False)
                        break
                    print("   Invalid selection.")
            else:
                _print_season_options(match, season_options)
                if complete_series_candidate:
                    print(f"  e. Entire regular series ({total_regular_episodes} episodes across {len(season_options)} seasons)")
                    trial_season = COMPLETE_SERIES_SENTINEL
                    print(
                        f"   Previewing the entire {total_regular_episodes}-episode regular series first. "
                        "After the plan, accept it or preview a specific season; no restart is required."
                    )
                else:
                    trial_season = season_options[0][0]
                    print(
                        f"   Previewing Season {trial_season} first. After the plan, choose "
                        "'accept' or another season number; no restart is required."
                    )
                while True:
                    trial_groups = _apply_unlabeled_season(original_groups, trial_season)
                    setattr(args, "_effective_season_choice", trial_season)
                    setattr(args, "_complete_series_mode", trial_season == COMPLETE_SERIES_SENTINEL)
                    setattr(args, "_season_review_active", True)
                    setattr(args, "_season_review_options", season_options)
                    setattr(args, "_season_review_series_available", complete_series_candidate)
                    setattr(args, "_season_review_action", None)
                    if trial_season == COMPLETE_SERIES_SENTINEL:
                        print(
                            f"  {match.title}: unlabeled rip directories -> ENTIRE REGULAR SERIES "
                            "[candidate plan preview]"
                        )
                    else:
                        print(
                            f"  {match.title}: unlabeled rip directories -> Season {trial_season} "
                            "[candidate plan preview]"
                        )
                    _do_tv_series(
                        args, client,
                        input_dir=input_dir,
                        output_root=output_root,
                        extras_root=extras_root,
                        query=bucket_query,
                        year=bucket_year,
                        groups=trial_groups,
                        match=match,
                    )
                    action = getattr(args, "_season_review_action", None)
                    if action == "accept":
                        break
                    if isinstance(action, int):
                        trial_season = action
                        print()
                        continue
                    raise MKVPlexError("Season plan review ended without a selection")
                for attr in (
                    "_season_review_active", "_season_review_options",
                    "_season_review_series_available", "_season_review_action",
                ):
                    if hasattr(args, attr):
                        delattr(args, attr)
                continue

        if inferred_season is not None:
            setattr(args, "_effective_season_choice", inferred_season)
            setattr(args, "_complete_series_mode", inferred_season == COMPLETE_SERIES_SENTINEL)
            if getattr(args, "_numbered_volume", None) is not None:
                inferred_label = f"RETAIL VOLUME {getattr(args, '_numbered_volume')} (partial multi-season slice)"
            else:
                inferred_label = (
                    "ENTIRE REGULAR SERIES" if inferred_season == COMPLETE_SERIES_SENTINEL
                    else f"Season {inferred_season}"
                )
            print(
                f"  {match.title}: unlabeled rip directories -> {inferred_label} "
                f"[{inference_source}]"
            )
        _do_tv_series(
            args, client,
            input_dir=input_dir,
            output_root=output_root,
            extras_root=extras_root,
            query=bucket_query,
            year=bucket_year,
            groups=resolved,
            match=match,
        )
    return 0


__all__ = ['_authored_title_fallback', '_choose_primary_tv_match', '_print_season_options', '_apply_unlabeled_season', '_infer_unlabeled_series_season', '_resolve_tv_series_buckets', 'do_tv']
