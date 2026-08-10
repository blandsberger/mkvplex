"""Source parsing, matching text normalization, and Plex naming helpers."""
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
from .models import Episode, JUNK_TOKENS, MKVPlexError, Match, SourceHints, TvRipGroup, VIDEO_EXTENSIONS
from .common import episode_expected_seconds

def eprint(*args: object, **kwargs: object) -> None:
    kwargs.setdefault("flush", True)
    print(*args, file=sys.stderr, **kwargs)


def configure_output() -> None:
    """Keep progress visible even when stdout is piped through tee.

    Python normally block-buffers stdout when it is not a terminal.  mkvplex can
    spend minutes inside ffmpeg boundary scans, so line-buffer stdout to make
    ordinary status lines appear immediately in pipelines and logs.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


def canonical_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def filesystem_title(title: str) -> str:
    """Convert provider punctuation into the user's current library style.

    In particular, ':' -> '-' turns "Dune: Part Two" into
    "Dune- Part Two", matching the supplied example.
    """
    title = title.replace(":", "-")
    title = title.replace("/", "-")
    title = title.replace("\x00", "")
    return canonical_spaces(title)


def parse_source_name(path: Path) -> SourceHints:
    raw = path.resolve().name
    work = raw.translate(str.maketrans({"™": "", "®": "", "©": ""}))

    year: Optional[int] = None
    season: Optional[int] = None
    disc: Optional[int] = None
    final_season = False

    # Pull out an explicit year before replacing separators.
    year_patterns = [
        r"\((19\d{2}|20\d{2}|21\d{2})\)",
        r"\[(19\d{2}|20\d{2}|21\d{2})\]",
        r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)",
    ]
    for pat in year_patterns:
        m = re.search(pat, work)
        if m:
            year = int(m.group(1))
            work = work[:m.start()] + " " + work[m.end():]
            break

    # "Final Season" is common box-set marketing.  It is not a new season;
    # after resolving the series we map it to the highest numbered regular
    # season reported by the metadata provider.
    m = re.search(r"(?i)(?<![A-Za-z0-9])final[ ._-]*season(?![A-Za-z0-9])", work)
    if m:
        final_season = True
        work = work[:m.start()] + " " + work[m.end():]

    # S02 / Season 2 / season02
    m = re.search(r"(?i)(?<![A-Za-z0-9])(?:S|Season[ ._-]*)(0*\d{1,2})(?!\d)", work)
    if m:
        season = int(m.group(1))
        work = work[:m.start()] + " " + work[m.end():]

    # Physical-disc labels.  Besides Disc1 / D1, ripping directories often
    # preserve media shorthand such as BD01, BR02, DVD03, or Blu-ray Disc 1.
    # Require a numeric suffix so ordinary title words/acronyms are not eaten.
    m = re.search(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"(?:blu[ ._-]*ray|bluray)[ ._-]*(?:disc|disk)?|"
        r"bd|br|dvd|disc|disk|d"
        r")[ ._-]*0*(\d{1,2})(?!\d)",
        work,
    )
    if m:
        disc = int(m.group(1))
        work = work[:m.start()] + " " + work[m.end():]

    # Common release-name separators. Deliberately preserve apostrophes.
    work = re.sub(r"[._]+", " ", work)
    work = re.sub(r"\s*-\s*", " ", work)

    # Clean punctuation left behind after removing season/disc markers.
    work = re.sub(r"[,;]+", " ", work)

    # Remove obvious release tokens when they appear as whole words.
    words = work.split()
    kept: list[str] = []
    for word in words:
        cleaned = re.sub(r"[^A-Za-z0-9-]", "", word).lower()
        if cleaned in JUNK_TOKENS:
            continue
        if re.fullmatch(r"(?i)(?:x|h)26[45]", cleaned):
            continue
        kept.append(word)

    title = canonical_spaces(" ".join(kept))
    if not title:
        title = canonical_spaces(raw)

    return SourceHints(
        raw=raw, title=title, year=year, season=season, disc=disc,
        final_season=final_season,
    )


def track_number(path: Path) -> tuple[int, str]:
    # MakeMKV commonly produces title_t00.mkv, title_t01.mkv, ...
    m = re.search(r"(?i)(?:^|[_-])t(\d+)(?:$|[_-])", path.stem + "_")
    if not m:
        m = re.search(r"(?i)t(\d+)", path.stem)
    if m:
        return int(m.group(1)), path.name.lower()
    # Fall back to the last number in the stem, then filename.
    nums = re.findall(r"\d+", path.stem)
    return (int(nums[-1]) if nums else 10**9, path.name.lower())


def epl_number(path: Path) -> Optional[int]:
    """Return a MakeMKV/source episode-list hint such as EPL_08 -> 8.

    Some TV box sets preserve an episode-list ordinal in the MakeMKV output
    filename.  It is a stronger ordering signal than tNN when present.
    """
    m = re.search(r"(?i)(?:^|[^A-Za-z0-9])EPL[ ._-]*0*(\d{1,3})(?=$|[^0-9])", path.stem)
    return int(m.group(1)) if m else None


def _match_tokens(text: str) -> list[str]:
    """Normalize title-ish text for fuzzy source-name matching."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def episode_title_similarity(path: Path, episode_title: str) -> float:
    """Return 0..1 evidence that PATH's filename names EPISODE_TITLE.

    This is deliberately conservative: a full normalized episode title found
    inside the source stem is strong evidence; otherwise token containment and
    SequenceMatcher provide softer hints. Runtime still gates whether a track
    is episode-like.
    """
    src_tokens = _match_tokens(path.stem)
    ep_tokens = _match_tokens(episode_title)
    if not src_tokens or not ep_tokens:
        return 0.0
    src_norm = " ".join(src_tokens)
    ep_norm = " ".join(ep_tokens)
    if len(ep_norm) >= 4 and ep_norm in src_norm:
        return 1.0
    src_set = set(src_tokens)
    containment = sum(1 for token in ep_tokens if token in src_set) / len(ep_tokens)
    ratio = difflib.SequenceMatcher(None, src_norm, ep_norm).ratio()
    # Long source names dilute SequenceMatcher badly, so token containment is
    # the more useful fuzzy signal for MakeMKV filenames.
    return max(containment, ratio)


def episode_local_candidates(
    path: Path,
    duration: float,
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
    limit: int = 3,
) -> list[tuple[Episode, float, float]]:
    """Rank plausible episodes for one track by runtime + filename evidence.

    Returns (episode, runtime_delta_seconds, title_similarity). This ranking is
    diagnostic; the global selector still enforces disc/episode continuity.
    """
    expected = episode_expected_seconds(episodes, show_runtime_minutes)
    rows: list[tuple[float, Episode, float, float]] = []
    for ep, exp in zip(episodes, expected):
        delta = abs(duration - exp)
        title_sim = episode_title_similarity(path, ep.title)
        # Runtime is the baseline. A strong filename/title hit can overcome a
        # few minutes of provider rounding, but not make a short extra into an
        # episode because the caller's runtime tolerance still gates selection.
        score = (delta / 60.0) - (title_sim * 8.0)
        rows.append((score, ep, delta, title_sim))
    rows.sort(key=lambda row: (row[0], row[1].number))
    return [(ep, delta, sim) for _score, ep, delta, sim in rows[:max(1, limit)]]


def episode_match_confidence(
    path: Path,
    duration: float,
    assigned: Episode,
    episodes: list[Episode],
    *,
    show_runtime_minutes: Optional[float] = None,
) -> tuple[str, list[tuple[Episode, float, float]]]:
    """Describe how independently identifiable an assigned episode is."""
    if epl_number(path) is not None:
        return "high:EPL", episode_local_candidates(
            path, duration, episodes,
            show_runtime_minutes=show_runtime_minutes,
        )
    candidates = episode_local_candidates(
        path, duration, episodes,
        show_runtime_minutes=show_runtime_minutes,
    )
    title_sim = episode_title_similarity(path, assigned.title)
    if title_sim >= 0.85:
        return "high:title", candidates
    if title_sim >= 0.50:
        return "medium:title", candidates

    # TMDb commonly rounds runtimes to whole minutes. If several episodes are
    # effectively tied on runtime, order is doing the identification work and
    # the user should be shown that explicitly.
    assigned_row = next((row for row in candidates if row[0].number == assigned.number), None)
    if assigned_row is None:
        return "low:order", candidates
    assigned_delta = assigned_row[1]
    competing = [row for row in candidates if row[0].number != assigned.number]
    if competing and abs(competing[0][1] - assigned_delta) < 30.0:
        return "low:runtime-tie", candidates
    if assigned_delta <= 30.0:
        return "medium:runtime", candidates
    return "low:order", candidates


def find_mkvs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise MKVPlexError(f"Input is not a directory: {directory}")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not files:
        raise MKVPlexError(f"No MKV files found directly in {directory}")
    return files


def find_movie_mkvs(directory: Path) -> list[Path]:
    """Return MKVs for movie mode, including multi-disc/collection subdirectories.

    Direct single-disc inputs remain the common case.  If INPUT is a collection
    root, descendant MKVs are included so companion features on bonus/movie
    discs can be identified instead of silently ignored.
    """
    if not directory.is_dir():
        raise MKVPlexError(f"Input is not a directory: {directory}")
    files = sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: str(p).casefold(),
    )
    if not files:
        raise MKVPlexError(f"No MKV files found in {directory} or its subdirectories")
    return files


def largest_mkv(directory: Path) -> Path:
    return max(find_mkvs(directory), key=lambda p: p.stat().st_size)


def sorted_tv_tracks(directory: Path) -> list[Path]:
    return sorted(find_mkvs(directory), key=track_number)


def normalize_for_match(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return canonical_spaces(value)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_for_match(a), normalize_for_match(b)).ratio()


def movie_basename(match: Match) -> str:
    title = filesystem_title(match.title)
    year = match.year or "Unknown"
    if match.imdb_id:
        return f"{title} - ({year}) - {{imdb-{match.imdb_id}}}"
    return f"{title} - ({year})"


def tv_directory_name(match: Match) -> str:
    # The series premiere year is useful for lookup/disambiguation, but the
    # user's TV library keeps the series root yearless. Episode filenames carry
    # their own air year instead.
    return filesystem_title(match.title)


def extras_group_directory(
    extras_show_dir: Path,
    input_dir: Path,
    group: TvRipGroup,
    season: int,
) -> Path:
    """Return the archive directory for non-episode tracks from one rip group.

    Series-tree mode mirrors the original rip-directory structure below the
    canonical show directory so an unidentified extra remains attributable to
    its physical disc. For a single-disc input, preserve the source directory
    name as the first archive component.
    """
    if group.directory != input_dir:
        rel = group.directory.relative_to(input_dir)
        return extras_show_dir / rel

    # Direct-disc input has no relative child path to mirror. Keep the source
    # directory name; if it is somehow identical to the canonical show name,
    # add season/disc context to avoid a meaningless duplicate path.
    leaf = group.directory.name
    if leaf == extras_show_dir.name:
        parts = [f"Season {season}"]
        if group.disc is not None:
            parts.append(f"Disc {group.disc}")
        leaf = " - ".join(parts)
    return extras_show_dir / leaf


def episode_basename(match: Match, episode: Episode) -> str:
    show = filesystem_title(match.title)
    # Prefer the episode's actual air year. This matters for seasons spanning
    # calendar years (e.g. Breaking Bad season 5: 2012/2013).
    year = episode.air_year or match.year or "Unknown"
    ep_title = filesystem_title(episode.title)
    return f"{show} - ({year}) - {episode.season}x{episode.number:02d} - {ep_title}"


__all__ = ['eprint', 'configure_output', 'canonical_spaces', 'filesystem_title', 'parse_source_name', 'track_number', 'epl_number', '_match_tokens', 'episode_title_similarity', 'episode_local_candidates', 'episode_match_confidence', 'find_mkvs', 'find_movie_mkvs', 'largest_mkv', 'sorted_tv_tracks', 'normalize_for_match', 'similarity', 'movie_basename', 'tv_directory_name', 'extras_group_directory', 'episode_basename']
