"""Persistent discovery cache and dry-run plan state."""
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
from .models import DISCOVERY_SCHEMA, Episode, MKVPlexError, Match, SplitBoundary, VERSION
from .fsops import human_size, sampled_md5_fingerprint

class DiscoveryDB:
    """Persistent discovery cache used to turn a dry run into an execution plan.

    Per-file discovery cache entries are tied to file size + mtime_ns.  A saved
    execution plan adds a sampled MD5 fingerprint (fixed chunks from the start,
    middle, and end) to path + size + mtime_ns.  This catches accidental source
    replacement without rereading huge media files in full.  Expensive ffmpeg
    scans, ffprobe metadata, TMDb JSON, and the final dry-run plan snapshot all
    live in the SQLite database selected with --db.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self.probe_hits = 0
        self.probe_misses = 0
        self.black_hits = 0
        self.black_misses = 0
        self.chapter_hits = 0
        self.chapter_misses = 0
        self.tmdb_hits = 0
        self.tmdb_misses = 0
        self.visual_hits = 0
        self.visual_misses = 0
        self.profile_hits = 0
        self.profile_misses = 0
        self.order_hits = 0
        self.order_misses = 0
        with self.lock:
            self.con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS file_probe (
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (schema_version, path, size_bytes, mtime_ns)
                );
                CREATE TABLE IF NOT EXISTS video_stream_profile (
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (schema_version, path, size_bytes, mtime_ns)
                );
                CREATE TABLE IF NOT EXISTS black_scan (
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    current_start REAL NOT NULL,
                    expected_duration REAL NOT NULL,
                    source_duration REAL NOT NULL,
                    search_window REAL NOT NULL,
                    min_black REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        schema_version, path, size_bytes, mtime_ns,
                        current_start, expected_duration, source_duration,
                        search_window, min_black
                    )
                );
                CREATE TABLE IF NOT EXISTS chapter_scan (
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    chapter_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (schema_version, path, size_bytes, mtime_ns)
                );
                CREATE TABLE IF NOT EXISTS video_packet_fingerprint (
                    schema_version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    sample_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (schema_version, path, size_bytes, mtime_ns)
                );
                CREATE TABLE IF NOT EXISTS master_component_order (
                    schema_version INTEGER NOT NULL,
                    master_path TEXT NOT NULL,
                    master_size_bytes INTEGER NOT NULL,
                    master_mtime_ns INTEGER NOT NULL,
                    component_key TEXT NOT NULL,
                    order_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        schema_version, master_path, master_size_bytes, master_mtime_ns, component_key
                    )
                );
                CREATE TABLE IF NOT EXISTS tmdb_response (
                    schema_version INTEGER NOT NULL,
                    request_key TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (schema_version, request_key)
                );
                CREATE TABLE IF NOT EXISTS plan_snapshot (
                    schema_version INTEGER NOT NULL,
                    plan_key TEXT NOT NULL,
                    tool_version TEXT NOT NULL,
                    input_root TEXT NOT NULL,
                    output_root TEXT NOT NULL,
                    extras_root TEXT,
                    tmdb_id INTEGER,
                    source_inventory_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (schema_version, plan_key)
                );
                """
            )
            self.con.commit()

    @staticmethod
    def _stamp() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @staticmethod
    def fingerprint(path: Path) -> tuple[str, int, int]:
        p = path.expanduser().resolve()
        st = p.stat()
        return str(p), int(st.st_size), int(st.st_mtime_ns)

    @staticmethod
    def _round(value: float) -> float:
        # Stabilize keys across JSON/float round-trips while retaining sub-frame
        # precision relative to our boundary search tolerances.
        return round(float(value), 3)

    def get_duration(self, path: Path) -> Optional[float]:
        key = self.fingerprint(path)
        with self.lock:
            row = self.con.execute(
                "SELECT duration FROM file_probe WHERE schema_version=? AND path=? AND size_bytes=? AND mtime_ns=?",
                (DISCOVERY_SCHEMA, *key),
            ).fetchone()
            if row is not None:
                self.probe_hits += 1
                return float(row[0])
            self.probe_misses += 1
        return None

    def put_duration(self, path: Path, duration: float) -> None:
        key = self.fingerprint(path)
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO file_probe VALUES (?, ?, ?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, *key, float(duration), self._stamp()),
            )
            self.con.commit()

    def get_video_profile(self, path: Path) -> Optional[dict[str, Any]]:
        key = self.fingerprint(path)
        with self.lock:
            row = self.con.execute(
                """SELECT profile_json FROM video_stream_profile
                   WHERE schema_version=? AND path=? AND size_bytes=? AND mtime_ns=?""",
                (DISCOVERY_SCHEMA, *key),
            ).fetchone()
            if row is None:
                self.profile_misses += 1
                return None
            self.profile_hits += 1
        try:
            data = json.loads(row[0])
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def put_video_profile(self, path: Path, profile: dict[str, Any]) -> None:
        key = self.fingerprint(path)
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO video_stream_profile VALUES (?, ?, ?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, *key, json.dumps(profile, sort_keys=True), self._stamp()),
            )
            self.con.commit()

    def get_chapters(self, path: Path) -> Optional[list[float]]:
        key = self.fingerprint(path)
        with self.lock:
            row = self.con.execute(
                """SELECT chapter_json FROM chapter_scan
                   WHERE schema_version=? AND path=? AND size_bytes=? AND mtime_ns=?""",
                (DISCOVERY_SCHEMA, *key),
            ).fetchone()
            if row is None:
                self.chapter_misses += 1
                return None
            self.chapter_hits += 1
        try:
            data = json.loads(row[0])
            return [float(v) for v in data]
        except Exception:
            return None

    def put_chapters(self, path: Path, chapters: list[float]) -> None:
        key = self.fingerprint(path)
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO chapter_scan VALUES (?, ?, ?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, *key, json.dumps([round(float(v), 6) for v in chapters]), self._stamp()),
            )
            self.con.commit()

    def get_black_boundary(
        self, path: Path, *, current_start: float, expected_duration: float,
        source_duration: float, search_window: float, min_black: float,
    ) -> Optional[SplitBoundary]:
        fkey = self.fingerprint(path)
        params = tuple(self._round(v) for v in (
            current_start, expected_duration, source_duration, search_window, min_black
        ))
        with self.lock:
            row = self.con.execute(
                """SELECT result_json FROM black_scan
                   WHERE schema_version=? AND path=? AND size_bytes=? AND mtime_ns=?
                     AND current_start=? AND expected_duration=? AND source_duration=?
                     AND search_window=? AND min_black=?""",
                (DISCOVERY_SCHEMA, *fkey, *params),
            ).fetchone()
            if row is None:
                self.black_misses += 1
                return None
            self.black_hits += 1
        data = json.loads(row[0])
        return SplitBoundary(
            predicted=float(data["predicted"]),
            selected=None if data["selected"] is None else float(data["selected"]),
            black_start=None if data["black_start"] is None else float(data["black_start"]),
            black_end=None if data["black_end"] is None else float(data["black_end"]),
            black_duration=None if data["black_duration"] is None else float(data["black_duration"]),
            delta=None if data["delta"] is None else float(data["delta"]),
            confidence=str(data["confidence"]),
            cached=True,
        )

    def put_black_boundary(
        self, path: Path, boundary: SplitBoundary, *, current_start: float,
        expected_duration: float, source_duration: float, search_window: float,
        min_black: float,
    ) -> None:
        fkey = self.fingerprint(path)
        params = tuple(self._round(v) for v in (
            current_start, expected_duration, source_duration, search_window, min_black
        ))
        data = {
            "predicted": boundary.predicted,
            "selected": boundary.selected,
            "black_start": boundary.black_start,
            "black_end": boundary.black_end,
            "black_duration": boundary.black_duration,
            "delta": boundary.delta,
            "confidence": boundary.confidence,
        }
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO black_scan VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, *fkey, *params, json.dumps(data, sort_keys=True), self._stamp()),
            )
            self.con.commit()

    def get_video_fingerprint(self, path: Path) -> Optional[tuple[str, list[str]]]:
        key = self.fingerprint(path)
        with self.lock:
            row = self.con.execute(
                """SELECT fingerprint, sample_json FROM video_packet_fingerprint
                   WHERE schema_version=? AND path=? AND size_bytes=? AND mtime_ns=?""",
                (DISCOVERY_SCHEMA, *key),
            ).fetchone()
            if row is None:
                self.visual_misses += 1
                return None
            self.visual_hits += 1
        try:
            samples = list(json.loads(row[1]))
        except Exception:
            samples = []
        return str(row[0]), [str(v) for v in samples]

    def put_video_fingerprint(self, path: Path, fingerprint: str, samples: list[str]) -> None:
        key = self.fingerprint(path)
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO video_packet_fingerprint VALUES (?, ?, ?, ?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, *key, fingerprint, json.dumps(samples), self._stamp()),
            )
            self.con.commit()

    @staticmethod
    def _component_order_key(paths: Iterable[Path]) -> str:
        rows = []
        for path in sorted({Path(p).resolve() for p in paths}, key=lambda p: str(p)):
            st = path.stat()
            rows.append((str(path), int(st.st_size), int(st.st_mtime_ns)))
        payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_master_component_order(
        self, master: Path, components: Iterable[Path]
    ) -> Optional[list[Path]]:
        mkey = self.fingerprint(master)
        component_key = self._component_order_key(components)
        with self.lock:
            row = self.con.execute(
                """SELECT order_json FROM master_component_order
                   WHERE schema_version=? AND master_path=? AND master_size_bytes=?
                     AND master_mtime_ns=? AND component_key=?""",
                (DISCOVERY_SCHEMA, *mkey, component_key),
            ).fetchone()
            if row is None:
                self.order_misses += 1
                return None
            self.order_hits += 1
        try:
            values = [Path(v) for v in json.loads(row[0])]
        except Exception:
            return None
        expected = {Path(p).resolve() for p in components}
        resolved = [p.resolve() for p in values]
        if len(resolved) != len(expected) or set(resolved) != expected:
            return None
        return resolved

    def put_master_component_order(
        self, master: Path, components: Iterable[Path], ordered: Iterable[Path]
    ) -> None:
        mkey = self.fingerprint(master)
        component_key = self._component_order_key(components)
        order_json = json.dumps([str(Path(p).resolve()) for p in ordered])
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO master_component_order VALUES (?, ?, ?, ?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, *mkey, component_key, order_json, self._stamp()),
            )
            self.con.commit()

    def get_tmdb(self, request_key: str) -> Optional[dict[str, Any]]:
        with self.lock:
            row = self.con.execute(
                "SELECT response_json FROM tmdb_response WHERE schema_version=? AND request_key=?",
                (DISCOVERY_SCHEMA, request_key),
            ).fetchone()
            if row is None:
                self.tmdb_misses += 1
                return None
            self.tmdb_hits += 1
        return json.loads(row[0])

    def put_tmdb(self, request_key: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO tmdb_response VALUES (?, ?, ?, ?)",
                (DISCOVERY_SCHEMA, request_key, json.dumps(data, sort_keys=True), self._stamp()),
            )
            self.con.commit()

    def _build_inventory(self, source_paths: Iterable[Path], *, hashing_label: str) -> list[dict[str, Any]]:
        paths = sorted(set(Path(p).resolve() for p in source_paths), key=lambda p: str(p))
        inventory: list[dict[str, Any]] = []
        total = len(paths)
        for index, path in enumerate(paths, 1):
            if not path.exists():
                raise MKVPlexError(f"Plan source disappeared while {hashing_label}: {path}")
            spath, size, mtime_ns = self.fingerprint(path)
            print(
                f"  MD5 fingerprint {hashing_label} [{index}/{total}]: "
                f"{path.name} ({human_size(size)})"
            )
            digest = sampled_md5_fingerprint(path)
            inventory.append({
                "path": spath,
                "size": size,
                "mtime_ns": mtime_ns,
                "md5_sample": digest,
            })
        return inventory

    @staticmethod
    def _match_identity(match: Match) -> dict[str, Any]:
        return {
            "media_type": match.media_type,
            "tmdb_id": match.tmdb_id,
            "imdb_id": match.imdb_id,
            "title": match.title,
            "year": match.year,
        }

    def store_plan(
        self, *, plan_key: str, input_root: Path, output_root: Path,
        extras_root: Optional[Path], tmdb_id: Optional[int], source_paths: Iterable[Path],
        plan: dict[str, Any],
    ) -> None:
        print("Binding dry-run plan to source identity (path/size/mtime + sampled MD5):")
        inventory = self._build_inventory(source_paths, hashing_label="plan")
        with self.lock:
            self.con.execute(
                """INSERT OR REPLACE INTO plan_snapshot
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    DISCOVERY_SCHEMA, plan_key, VERSION, str(input_root), str(output_root),
                    None if extras_root is None else str(extras_root), tmdb_id,
                    json.dumps(inventory, sort_keys=True), json.dumps(plan, sort_keys=True),
                    self._stamp(),
                ),
            )
            self.con.commit()

    def load_valid_plan(
        self, plan_key: str, source_paths: Iterable[Path], *, match: Match,
        expected_settings: dict[str, Any],
    ) -> tuple[Optional[dict[str, Any]], str]:
        with self.lock:
            row = self.con.execute(
                """SELECT tool_version, source_inventory_json, plan_json
                   FROM plan_snapshot WHERE schema_version=? AND plan_key=?""",
                (DISCOVERY_SCHEMA, plan_key),
            ).fetchone()
        if row is None:
            return None, "no matching dry-run plan exists in this database"
        if str(row[0]) != VERSION:
            return None, f"plan was created by mkvplex {row[0]}, not {VERSION}"

        try:
            expected_inventory = json.loads(row[1])
            plan = json.loads(row[2])
        except Exception as exc:
            return None, f"saved plan is unreadable: {exc}"

        saved_match = plan.get("match") or {}
        expected_match = self._match_identity(match)
        saved_identity = {key: saved_match.get(key) for key in expected_match}
        if saved_identity != expected_match:
            return None, (
                "selected media identity differs from the approved dry run: "
                f"saved={saved_identity!r}, current={expected_match!r}"
            )

        saved_settings = plan.get("settings") or {}
        for key, value in expected_settings.items():
            if saved_settings.get(key) != value:
                return None, (
                    f"execution setting {key!r} differs from the approved dry run "
                    f"(saved={saved_settings.get(key)!r}, current={value!r})"
                )

        paths = sorted(set(Path(p).resolve() for p in source_paths), key=lambda p: str(p))
        expected_by_path = {str(row["path"]): row for row in expected_inventory}
        current_paths = [str(p) for p in paths]
        if current_paths != sorted(expected_by_path):
            missing = sorted(set(expected_by_path) - set(current_paths))
            added = sorted(set(current_paths) - set(expected_by_path))
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing[:5]))
            if added:
                details.append("added: " + ", ".join(added[:5]))
            return None, "source file set changed (" + "; ".join(details) + ")"

        print("Validating approved source identity (path/size/mtime + sampled MD5):")
        total = len(paths)
        for index, path in enumerate(paths, 1):
            expected = expected_by_path[str(path)]
            if not path.exists():
                return None, f"source file disappeared: {path}"
            spath, size, mtime_ns = self.fingerprint(path)
            if size != int(expected.get("size", -1)):
                return None, f"source size changed: {path}"
            if mtime_ns != int(expected.get("mtime_ns", -1)):
                return None, f"source mtime changed: {path}"
            print(
                f"  MD5 fingerprint validate [{index}/{total}]: "
                f"{path.name} ({human_size(size)})"
            )
            digest = sampled_md5_fingerprint(path)
            if digest != expected.get("md5_sample"):
                return None, f"source sampled-MD5 fingerprint changed: {path}"

        return plan, "ok"

    def approved_tv_season_hint(
        self, *, input_root: Path, output_root: Path, extras_root: Path, match: Match
    ) -> Optional[int]:
        """Recover an accepted unlabeled-season choice before full plan validation.

        A fresh --dry-run --db contains only plans from that planning session.
        This hint is used solely to reconstruct the same candidate plan; the normal
        load_valid_plan() path still validates version, settings, media identity,
        source inventory, mtimes, and sampled-MD5 before execution is allowed.
        """
        with self.lock:
            rows = self.con.execute(
                """SELECT tool_version, plan_json FROM plan_snapshot
                   WHERE schema_version=? AND input_root=? AND output_root=?
                     AND extras_root=? AND tmdb_id=?
                   ORDER BY updated_at DESC""",
                (
                    DISCOVERY_SCHEMA, str(input_root.resolve()), str(output_root.resolve()),
                    str(extras_root.resolve()), match.tmdb_id,
                ),
            ).fetchall()
        for tool_version, raw in rows:
            if str(tool_version) != VERSION:
                continue
            try:
                plan = json.loads(raw)
            except Exception:
                continue
            settings = plan.get("settings") or {}
            value = settings.get("resolved_unlabeled_season")
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def approved_collection_hint(
        self, *, input_root: Path, output_root: Path, extras_root: Path, match: Match
    ) -> Optional[tuple[str, Optional[int]]]:
        """Recover an accepted collection order/companion before full validation.

        This mirrors approved_tv_season_hint(): it is only a reconstruction hint.
        load_valid_plan() still validates the exact source inventory, version,
        settings, and primary media identity before execution.
        """
        with self.lock:
            rows = self.con.execute(
                """SELECT tool_version, plan_json FROM plan_snapshot
                   WHERE schema_version=? AND input_root=? AND output_root=?
                     AND extras_root=? AND tmdb_id=?
                   ORDER BY updated_at DESC""",
                (
                    DISCOVERY_SCHEMA, str(input_root.resolve()), str(output_root.resolve()),
                    str(extras_root.resolve()), match.tmdb_id,
                ),
            ).fetchall()
        for tool_version, raw in rows:
            if str(tool_version) != VERSION:
                continue
            try:
                plan = json.loads(raw)
            except Exception:
                continue
            if plan.get("command") != "tv_collection":
                continue
            settings = plan.get("settings") or {}
            order = settings.get("collection_order")
            companion = settings.get("collection_companion_tmdb_id")
            if not order or order == "auto":
                continue
            try:
                companion_id = None if companion is None else int(companion)
            except (TypeError, ValueError):
                companion_id = None
            return str(order), companion_id
        return None

    def stats_line(self) -> str:
        return (
            f"db cache: ffprobe {self.probe_hits} hit/{self.probe_misses} miss; "
            f"fade {self.black_hits} hit/{self.black_misses} miss; "
            f"chapters {self.chapter_hits} hit/{self.chapter_misses} miss; "
            f"visual {self.visual_hits} hit/{self.visual_misses} miss; "
            f"master-order {self.order_hits} hit/{self.order_misses} miss; "
            f"video-profile {self.profile_hits} hit/{self.profile_misses} miss; "
            f"TMDb {self.tmdb_hits} hit/{self.tmdb_misses} miss"
        )


_DISCOVERY_DB: Optional[DiscoveryDB] = None


def discovery_db() -> Optional[DiscoveryDB]:
    return _DISCOVERY_DB


def _plan_key(
    command: str, input_root: Path, output_root: Path, extras_root: Optional[Path],
    match: Match, settings: Optional[dict[str, Any]] = None,
) -> str:
    raw = json.dumps({
        "command": command,
        "input": str(input_root.resolve()),
        "output": str(output_root.resolve()),
        "extras": None if extras_root is None else str(extras_root.resolve()),
        "match": DiscoveryDB._match_identity(match),
        "settings": settings or {},
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _movie_plan_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "copy": bool(args.copy),
        "mode": int(args.mode),
        "verify_md5": bool(args.verify_md5),
        "probe_workers": int(getattr(args, "probe_workers", 4)),
    }


def parse_season_counts(value: str) -> tuple[int, ...]:
    """Parse a comma-separated Plex presentation season map.

    Example: ``18,22,24,24,24,24,25`` means absolute episodes 1-18 become
    S01E01-S01E18, the next 22 become S02E01-S02E22, and so on.  This is
    intentionally a presentation/identity layer only; it does not change the
    physical-source ordering or split-boundary reconstruction.
    """
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("season counts must be comma-separated positive integers")
    try:
        counts = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("season counts must be comma-separated positive integers") from exc
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("every season count must be greater than zero")
    return counts


def remap_episode_seasons(episodes: list[Episode], counts: tuple[int, ...]) -> list[Episode]:
    """Remap a flat provider episode sequence into Plex-facing seasons.

    Episode metadata (title, TMDb id, runtime, air year) stays attached to the
    same ordinal program.  Only ``season`` and ``number`` change.
    """
    expected = sum(counts)
    if expected != len(episodes):
        raise MKVPlexError(
            f"--season-counts describes {expected} episodes, but the selected metadata "
            f"sequence contains {len(episodes)}. Refusing to shift episode identities."
        )
    out: list[Episode] = []
    index = 0
    for season, count in enumerate(counts, start=1):
        for number in range(1, count + 1):
            ep = episodes[index]
            out.append(replace(ep, season=season, number=number))
            index += 1
    return out


def _tv_plan_settings(args: argparse.Namespace) -> dict[str, Any]:
    movies_output = getattr(args, "movies_output", None)
    if movies_output:
        movies_output = str(Path(movies_output).expanduser().resolve())
    collection_order = getattr(args, "_collection_order_choice", None)
    if collection_order is None:
        collection_order = str(getattr(args, "collection_order", "auto"))
    return {
        "copy": bool(args.copy),
        "mode": int(args.mode),
        "verify_md5": bool(args.verify_md5),
        "runtime_tolerance": float(args.runtime_tolerance),
        "split_search_window": float(getattr(args, "split_search_window", 180.0)),
        "split_black_min": float(getattr(args, "split_black_min", 0.30)),
        "episode_start": int(args.episode_start),
        "episode_count": args.episode_count,
        "season": args.season,
        "season_counts": list(getattr(args, "season_counts", None) or []),
        "numbered_volume": getattr(args, "_numbered_volume", None),
        "resolved_unlabeled_season": getattr(args, "_effective_season_choice", None),
        "complete_series_mode": bool(getattr(args, "_complete_series_mode", False)),
        "all_tracks": bool(getattr(args, "all_tracks", False)),
        "no_aggregate_split": bool(getattr(args, "no_aggregate_split", False)),
        "disc_kind": str(getattr(args, "disc_kind", "auto")),
        "collection_mode": bool(getattr(args, "_collection_mode", False)),
        "collection_order": collection_order,
        "collection_companion_tmdb_id": getattr(args, "_collection_companion_tmdb_id", None),
        "movies_output": movies_output,
    }


def reset_discovery_db(path: Path) -> None:
    """Delete a planning DB and SQLite sidecars before a fresh --dry-run --db."""
    db_path = path.expanduser().resolve()
    for candidate in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


__all__ = ['DiscoveryDB', '_DISCOVERY_DB', 'discovery_db', '_plan_key', '_movie_plan_settings', 'parse_season_counts', 'remap_episode_seasons', '_tv_plan_settings', 'reset_discovery_db']
