"""Low-level planning helpers shared across media/disc/TV modules."""
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
from .models import Episode, MKVPlexError, TvRipGroup

def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def episode_expected_seconds(episodes: list[Episode], show_runtime_minutes: Optional[float] = None) -> list[float]:
    known = [float(ep.runtime_minutes) for ep in episodes if ep.runtime_minutes and ep.runtime_minutes > 0]
    fallback = _median(known)
    if fallback is None and show_runtime_minutes and show_runtime_minutes > 0:
        fallback = float(show_runtime_minutes)
    if fallback is None:
        raise MKVPlexError(
            "TMDb did not provide episode runtimes for this season/show. "
            "Use --all-tracks to restore sequential track mapping, or --episode-count "
            "after manually isolating the episode tracks."
        )
    return [60.0 * float(ep.runtime_minutes or fallback) for ep in episodes]


def _relative_delta(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _disc_key(group: TvRipGroup) -> tuple[int, int]:
    return (1 if group.final_season else 0, group.disc if group.disc is not None else 10**6)


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _group_effective_season(group: TvRipGroup, final_season_number: Optional[int]) -> Optional[int]:
    if group.final_season:
        return final_season_number
    return group.season


def _group_sort_key(group: TvRipGroup, season: int) -> tuple[int, int, int, int, str]:
    # Within the same numeric season, explicitly numbered season discs come
    # first and "Final Season" marketing discs come after them.  Authored
    # ordinal-range groups (e.g. 1-10, 11-20) provide a second structural order
    # when no disc number exists.
    phase = 1 if group.final_season else 0
    disc = group.disc if group.disc is not None else 10**6
    span_start = group.episode_span[0] if group.episode_span is not None else 10**6
    return season, phase, disc, span_start, str(group.directory).casefold()


__all__ = ['_median', 'episode_expected_seconds', '_relative_delta', '_disc_key', 'format_duration', '_group_effective_season', '_group_sort_key']
