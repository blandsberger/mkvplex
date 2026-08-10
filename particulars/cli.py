"""Command-line parser and top-level dispatch."""
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
from .models import DEFAULT_CACHE, DEFAULT_DISCOVERY_DB, DEFAULT_TMDB_CACHE, MKVPlexError, VERSION
from .discovery import DiscoveryDB, _DISCOVERY_DB, parse_season_counts, reset_discovery_db
from .naming import configure_output, eprint, parse_source_name
from .media import resolve_media_workers
from .discs import looks_like_tv_tree
from .tmdb import TMDbClient
from .movie import do_movie
from .tv import do_tv
from . import discovery as discovery_state

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="source directory containing MakeMKV MKVs or a supported collection/tree layout",
    )
    parser.add_argument(
        "output",
        help="Plex library root; mkvplex creates canonical title/show directories below it",
    )
    parser.add_argument("--title", help="override the title parsed from INPUT")
    parser.add_argument("--year", type=int, help="override the parsed release/premiere year")
    parser.add_argument("--imdb", help="require a specific IMDb tt ID for the primary title")
    parser.add_argument("--copy", action="store_true", help="copy sources instead of moving them")
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="analyze and print the complete plan without modifying media",
    )
    parser.add_argument(
        "--db", action="store_const", const=str(DEFAULT_DISCOVERY_DB), default=None,
        help=(
            "with --dry-run: erase/rebuild the default SQLite approval DB; "
            "without --dry-run: require and reuse that exact sampled-MD5-bound plan"
        ),
    )
    parser.add_argument(
        "--db-path", dest="db", metavar="PATH",
        help=(
            "same as --db, but use PATH; a fresh dry run replaces the DB and execution "
            "refuses changed sources/title/settings"
        ),
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="accept a sufficiently high-confidence primary metadata match without prompting",
    )
    parser.add_argument(
        "--mode",
        type=lambda s: int(s, 8),
        default=0o755,
        help="recursive permission mode applied to destination trees (octal; default: 755)",
    )
    parser.add_argument(
        "--verify-md5",
        action="store_true",
        help="when a transfer must copy bytes, verify source/destination MD5 before source deletion",
    )
    parser.add_argument(
        "--storage", choices=("auto", "hdd", "ssd"), default="auto",
        help=(
            "media scheduler storage hint: auto detects direct/ZFS backing devices; "
            "hdd favors low concurrency, ssd permits wider parallel probing (default: auto)"
        ),
    )
    parser.add_argument(
        "--tmdb-cache", default=str(DEFAULT_TMDB_CACHE), metavar="PATH",
        help=f"persistent TMDb response cache independent of --db (default: {DEFAULT_TMDB_CACHE})",
    )
    parser.add_argument(
        "--tmdb-cache-days", type=float, default=30.0, metavar="DAYS",
        help="reuse persistent TMDb metadata for this many days (default: 30)",
    )
    parser.add_argument(
        "--tmdb-rate", type=float, default=8.0, metavar="REQ_PER_SEC",
        help="TMDb request token-bucket rate; 429 is always respected (default: 8)",
    )
    parser.add_argument(
        "--tmdb-workers", type=int, default=8, metavar="N",
        help="maximum concurrent TMDb metadata-prefetch workers (default: 8)",
    )
    parser.add_argument(
        "--tmdb-refresh", action="store_true",
        help="bypass persistent/discovery TMDb cache reads for this run and refresh metadata",
    )


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        prog="mkvplex",
        formatter_class=formatter,
        description=(
            "Analyze MakeMKV rips, resolve metadata with TMDb, and build a safe Plex ingest plan.\n\n"
            "movie mode understands multi-disc trees, runtime-based main features, named companion\n"
            "movies, and extras. tv mode understands multi-season trees, extras, missing episodes,\n"
            "multiple series, aggregate/play-all titles, authored-chapter/runtime splitting of multi-episode MKVs, presentation-season remaps, and stream-derived technical filename suffixes."
        ),
        epilog=(
            "Examples:\n"
            "  mkvplex movie 'Incoming/Your Name' Movies Extras --dry-run\n"
            "  mkvplex movie 'Incoming/Box Set' Movies Extras --dry-run --db\n"
            "  mkvplex tv 'Incoming/Breaking Bad' 'Tv Shows Mature' Extras --dry-run\n"
            "  mkvplex tv 'Incoming/Neon Genesis Evangelion' 'Tv Shows Mature' Extras --dry-run --db\n"
            "  mkvplex auto INPUT OUTPUT EXTRAS --dry-run\n\n"
            "Approval DB workflow:\n"
            "  1. Run with --dry-run --db (the DB is replaced with a fresh analysis).\n"
            "  2. Inspect the printed plan.\n"
            "  3. Run the same command without --dry-run; source fingerprint/title/settings validation must match.\n\n"
            "This product uses the TMDB API but is not endorsed or certified by TMDB."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{movie,tv,auto}")

    movie = sub.add_parser(
        "movie",
        formatter_class=formatter,
        help="identify movie features/companion movies and archive remaining tracks",
        description=(
            "Movie ingest. INPUT may be a single MakeMKV rip directory or a multi-disc collection tree.\n"
            "The primary feature is chosen using TMDb runtime plus file size. Long-form tracks with a\n"
            "distinct local title can be resolved as companion movies; opaque long tracks are flagged\n"
            "for review rather than guessed. Unselected tracks can be preserved under EXTRAS_OUTPUT."
        ),
        epilog=(
            "Examples:\n"
            "  mkvplex movie 'Incoming/Dune Part Two' Movies Extras --dry-run\n"
            "  mkvplex movie 'Incoming/Collector Edition' Movies Extras --dry-run --db\n\n"
            "When EXTRAS_OUTPUT is omitted, unselected tracks are left untouched."
        ),
    )
    add_common_arguments(movie)
    movie.add_argument(
        "extras", nargs="?", metavar="EXTRAS_OUTPUT",
        help="optional extras archive root; original collection/rip-directory context is preserved",
    )
    movie.add_argument(
        "--probe-workers", type=int, default=0,
        help="parallel local media probes; 0 chooses from --storage (default: auto)",
    )
    movie.add_argument("--hash", action="store_true", help="compute primary-feature MD5 and record it in the legacy local catalog")
    movie.add_argument("--cache", default=str(DEFAULT_CACHE), help=f"legacy MD5 catalog path (default: {DEFAULT_CACHE})")

    tv = sub.add_parser(
        "tv",
        formatter_class=formatter,
        help="map TV rip trees to episodes while preserving extras and holes",
        description=(
            "TV ingest. INPUT may be one disc, one season, or a whole series/box-set tree.\n"
            "mkvplex uses filename hints, TMDb runtimes, ffprobe, bitrate/disc structure, and MakeMKV\n"
            "ordering clues. It classifies physical discs as episode/bonus/ambiguous before filling\n"
            "episode slots. Missing tracks remain explicit holes. Giant multi-episode titles can be\n"
            "split at detected black boundaries without re-encoding."
        ),
        epilog=(
            "Examples:\n"
            "  mkvplex tv 'Incoming/Breaking Bad' 'Tv Shows Mature' Extras --dry-run\n"
            "  mkvplex tv 'Incoming/Picard' 'Tv Shows' Extras --dry-run --db\n"
            "  mkvplex tv 'Incoming/Season 2 Disc 1' 'Tv Shows' Extras --season 2 --dry-run\n"
            "  mkvplex tv 'Incoming/Bonus Disc' 'Tv Shows' Extras --season 2 --disc-kind bonus --dry-run\n\n"
            "Unlabeled multi-season trees are previewed as complete candidate plans; after each preview\n"
            "you can accept it or try another season without restarting or re-probing the files.\n"
            "Ambiguous disc classification and low-confidence aggregate splits are never executed automatically."
        ),
    )
    add_common_arguments(tv)
    tv.add_argument(
        "extras", metavar="EXTRAS_OUTPUT",
        help="required archive root for non-episode MKV tracks; rip-directory context is preserved",
    )
    tv.add_argument(
        "--season", type=int,
        help="single rip: override season; series tree: process only this season",
    )
    tv.add_argument(
        "--season-counts", type=parse_season_counts, metavar="N,N,...",
        help=(
            "Plex presentation remap for providers that flatten a complete series into one season; "
            "counts are consecutive season sizes, e.g. 18,22,24,24,24,24,25"
        ),
    )
    tv.add_argument("--episode-start", type=int, default=1, help="first episode number on a manually scoped rip (default: 1)")
    tv.add_argument(
        "--episode-count", type=int,
        help="limit mapped episodes beginning at --episode-start",
    )
    tv.add_argument(
        "--runtime-tolerance", type=float, default=12.0,
        help="maximum automatic episode/runtime difference in minutes (default: 12)",
    )
    tv.add_argument(
        "--probe-workers", type=int, default=0,
        help="parallel local media probes; 0 chooses from --storage (default: auto)",
    )
    tv.add_argument(
        "--movies-output", metavar="PATH",
        help=(
            "optional movie-library root for confidently identified films embedded in a TV/collection tree; "
            "without it, embedded movies are reported and left untouched"
        ),
    )
    tv.add_argument(
        "--collection-order", choices=("auto", "regular", "dvd", "production"), default="auto",
        help=(
            "episode ordering for collection-aware plans; auto previews Production when available, then DVD, "
            "otherwise regular TMDb order (default: auto)"
        ),
    )
    tv.add_argument(
        "--all-tracks", action="store_true",
        help="legacy/manual mode: disable structural/runtime filtering and map tracks sequentially",
    )
    tv.add_argument(
        "--no-aggregate-split", action="store_true",
        help="disable detection/splitting of giant MKVs that contain multiple episodes",
    )
    tv.add_argument(
        "--disc-kind", choices=("auto", "episodes", "bonus"), default="auto",
        help=(
            "single-disc classification override: auto infers episode vs bonus structure; "
            "episodes forces episode assignment; bonus archives every track as Extras (default: auto)"
        ),
    )
    tv.add_argument(
        "--split-search-window", type=float, default=180.0,
        help="seconds around each predicted episode end searched for fade-to-black (default: 180)",
    )
    tv.add_argument(
        "--split-black-min", type=float, default=0.30,
        help="minimum sustained black duration in seconds for an aggregate boundary; TMDb runtime fitting is the fallback (default: 0.30)",
    )

    auto = sub.add_parser(
        "auto",
        formatter_class=formatter,
        help="infer TV vs movie from the input layout",
        description=(
            "Infer TV when season/disc structure is visible; otherwise use movie mode.\n"
            "For deterministic batch work, explicit 'movie' or 'tv' is preferable."
        ),
        epilog="Example:\n  mkvplex auto INPUT OUTPUT EXTRAS --dry-run",
    )
    add_common_arguments(auto)
    auto.add_argument(
        "extras", nargs="?", metavar="EXTRAS_OUTPUT",
        help="extras archive root (required when auto resolves to TV; optional for movies)",
    )
    auto.add_argument("--season", type=int, help="season number; forces TV behavior")
    auto.add_argument(
        "--season-counts", type=parse_season_counts, metavar="N,N,...",
        help="Plex presentation season sizes for a flattened complete-series provider order; forces TV behavior",
    )
    auto.add_argument("--episode-start", type=int, default=1)
    auto.add_argument("--episode-count", type=int)
    auto.add_argument("--runtime-tolerance", type=float, default=12.0)
    auto.add_argument("--probe-workers", type=int, default=0, help="parallel local media probes; 0 chooses from --storage")
    auto.add_argument(
        "--movies-output", metavar="PATH",
        help="optional movie-library root for films embedded in a TV/collection tree",
    )
    auto.add_argument(
        "--collection-order", choices=("auto", "regular", "dvd", "production"), default="auto",
        help="collection episode ordering (default: auto)",
    )
    auto.add_argument("--all-tracks", action="store_true")
    auto.add_argument("--no-aggregate-split", action="store_true")
    auto.add_argument("--split-search-window", type=float, default=180.0)
    auto.add_argument("--split-black-min", type=float, default=0.30)
    auto.add_argument("--hash", action="store_true")
    auto.add_argument("--cache", default=str(DEFAULT_CACHE))

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    configure_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "probe_workers"):
        workers, storage = resolve_media_workers(
            Path(args.input), int(getattr(args, "probe_workers", 0) or 0), str(getattr(args, "storage", "auto"))
        )
        args.probe_workers = workers
        setattr(args, "_storage_class", storage)
        print(f"Media scheduler: storage={storage}; local probe workers={workers}")

    if getattr(args, "db", None):
        db_path = Path(args.db)
        if args.dry_run:
            reset_discovery_db(db_path)
            print(f"Fresh dry-run DB: {db_path.expanduser().resolve()}")
        discovery_state._DISCOVERY_DB = DiscoveryDB(db_path)
        print(f"Discovery DB: {discovery_state._DISCOVERY_DB.path}")

    try:
        client = TMDbClient(
            bearer_token=os.environ.get("TMDB_BEARER_TOKEN"),
            api_key=os.environ.get("TMDB_API_KEY"),
            cache_path=Path(getattr(args, "tmdb_cache", DEFAULT_TMDB_CACHE)),
            cache_days=float(getattr(args, "tmdb_cache_days", 30.0)),
            rate_per_second=float(getattr(args, "tmdb_rate", 8.0)),
            workers=int(getattr(args, "tmdb_workers", 8)),
            refresh=bool(getattr(args, "tmdb_refresh", False)),
        )
        print(
            f"TMDb metadata cache: {client.cache.path} "
            f"(TTL {float(getattr(args, 'tmdb_cache_days', 30.0)):g}d; "
            f"rate {float(getattr(args, 'tmdb_rate', 8.0)):g}/s; workers {client.workers})"
        )

        if args.command == "movie":
            result = do_movie(args, client)
        elif args.command == "tv":
            result = do_tv(args, client)
        elif args.command == "auto":
            hints = parse_source_name(Path(args.input))
            if (
                args.season is not None
                or getattr(args, "season_counts", None)
                or hints.season is not None
                or looks_like_tv_tree(Path(args.input))
            ):
                result = do_tv(args, client)
            else:
                try:
                    result = do_movie(args, client)
                except MKVPlexError as exc:
                    if not str(exc).startswith("No movie matches found for"):
                        raise
                    if not getattr(args, "extras", None):
                        raise
                    print("Auto: movie metadata lookup found no match; trying TV discovery.")
                    result = do_tv(args, client)
        else:
            parser.error("unknown command")
            return 2
        print(f"  {client.stats_line()}")
        return result
    except MKVPlexError as exc:
        eprint(f"mkvplex: {exc}")
        return 1
    except KeyboardInterrupt:
        eprint("\nmkvplex: cancelled")
        return 130


__all__ = ['add_common_arguments', 'build_parser', 'main']
