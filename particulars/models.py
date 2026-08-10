"""Shared constants, exceptions, and immutable data models."""
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

VERSION = "0.13.3"


DISCOVERY_SCHEMA = 16


COMPLETE_SERIES_SENTINEL = -1


DEFAULT_DISCOVERY_DB = Path.home() / ".cache" / "mkvplex" / "discovery.sqlite3"


DEFAULT_TMDB_CACHE = Path.home() / ".cache" / "mkvplex" / "tmdb.sqlite3"


TMDB_API = "https://api.themoviedb.org/3"


DEFAULT_CACHE = Path.home() / ".cache" / "mkvplex" / "catalog.sqlite3"


VIDEO_EXTENSIONS = {".mkv"}


JUNK_TOKENS = {
    "2160p", "1080p", "720p", "480p", "uhd", "bluray", "blu-ray",
    "remux", "hevc", "x265", "x264", "h265", "h264", "hdr", "hdr10",
    "dv", "dolbyvision", "dolby", "vision", "disc", "disk",
}


class MKVPlexError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceHints:
    raw: str
    title: str
    year: Optional[int]
    season: Optional[int]
    disc: Optional[int]
    final_season: bool


@dataclass(frozen=True)
class Match:
    media_type: str  # movie | tv
    tmdb_id: int
    title: str
    year: Optional[int]
    imdb_id: Optional[str]
    score: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class Episode:
    season: int
    number: int
    title: str
    tmdb_id: Optional[int] = None
    runtime_minutes: Optional[int] = None
    air_year: Optional[int] = None
    # Preserve the provider air date as a structural clue.  Some DVD sets
    # package several short TMDb segment episodes into one broadcast program;
    # consecutive segments sharing an air date are therefore useful program
    # boundaries without changing the canonical Plex SxE identity.
    air_date: Optional[str] = None


@dataclass(frozen=True)
class Transfer:
    source: Path
    destination: Path


@dataclass(frozen=True)
class TvRipGroup:
    directory: Path
    season: Optional[int]
    disc: Optional[int]
    final_season: bool
    tracks: tuple[Path, ...]
    # Optional authored/packaging ordinal span inferred from a directory name
    # such as ``Show 1-10``.  This is physical-order evidence, not provider
    # season metadata.
    episode_span: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class TrackAnalysis:
    path: Path
    group: TvRipGroup
    duration: float
    size_bytes: int
    bitrate_mbps: float
    aggregate_of: tuple[Path, ...] = ()
    bitrate_outlier: bool = False


@dataclass(frozen=True)
class DiscHypothesis:
    kind: str  # episodes | bonus | ambiguous
    confidence: str
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeAssignment:
    source: Path
    group: TvRipGroup
    episode: Episode
    duration: float


@dataclass(frozen=True)
class SkippedTrack:
    source: Path
    group: TvRipGroup
    duration: float
    reason: str


@dataclass(frozen=True)
class BlackInterval:
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class SplitBoundary:
    predicted: float
    selected: Optional[float]
    black_start: Optional[float]
    black_end: Optional[float]
    black_duration: Optional[float]
    delta: Optional[float]
    confidence: str
    cached: bool = False


@dataclass(frozen=True)
class AggregateSplitPlan:
    source: Path
    group: TvRipGroup
    episodes: tuple[Episode, ...]
    source_duration: float
    boundaries: tuple[SplitBoundary, ...]

    @property
    def executable(self) -> bool:
        return bool(self.episodes) and len(self.boundaries) == len(self.episodes) and all(
            b.selected is not None
            and b.confidence != "unresolved"
            and not b.confidence.startswith("low")
            for b in self.boundaries
        )


__all__ = ['VERSION', 'DISCOVERY_SCHEMA', 'COMPLETE_SERIES_SENTINEL', 'DEFAULT_DISCOVERY_DB', 'DEFAULT_TMDB_CACHE', 'TMDB_API', 'DEFAULT_CACHE', 'VIDEO_EXTENSIONS', 'JUNK_TOKENS', 'MKVPlexError', 'SourceHints', 'Match', 'Episode', 'Transfer', 'TvRipGroup', 'TrackAnalysis', 'DiscHypothesis', 'EpisodeAssignment', 'SkippedTrack', 'BlackInterval', 'SplitBoundary', 'AggregateSplitPlan']
