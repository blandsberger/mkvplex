"""Filesystem transfers, fingerprints, cache helpers, and execution preflight."""
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
from .models import MKVPlexError, Match, SourceHints, Transfer

def ensure_output_root(output: Path) -> Path:
    output = output.expanduser().resolve()
    if not output.exists():
        raise MKVPlexError(f"Output root does not exist: {output}")
    if not output.is_dir():
        raise MKVPlexError(f"Output root is not a directory: {output}")
    return output


def recursive_chmod(root: Path, mode: int) -> None:
    # Exact chmod -R semantics requested: all files and directories get MODE.
    root.chmod(mode)
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        for name in dirnames:
            (base / name).chmod(mode)
        for name in filenames:
            (base / name).chmod(mode)


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sampled_md5_fingerprint(path: Path, sample_size: int = 4 * 1024 * 1024) -> str:
    """Fast plan fingerprint using MD5 over fixed samples plus file size.

    This is intentionally not a full-file cryptographic identity check.  Plan
    validation separately requires the exact path, byte size, and nanosecond
    mtime recorded by the dry run.  Sampling the start, middle, and end keeps
    validation cheap for 30-40 GiB MakeMKV files while strongly detecting the
    accidental replacements/corruption this approval workflow is designed for.
    Files small enough to fit in three samples are hashed in full.
    """
    size = path.stat().st_size
    h = hashlib.md5()
    h.update(b"mkvplex-sampled-md5-v1\0")
    h.update(str(size).encode("ascii"))
    h.update(b"\0")

    with path.open("rb") as f:
        if size <= sample_size * 3:
            while True:
                chunk = f.read(sample_size)
                if not chunk:
                    break
                h.update(chunk)
        else:
            offsets = (0, max(0, (size - sample_size) // 2), size - sample_size)
            for offset in offsets:
                h.update(str(offset).encode("ascii"))
                h.update(b"\0")
                f.seek(offset)
                data = f.read(sample_size)
                if len(data) != sample_size:
                    raise MKVPlexError(
                        f"Short read while fingerprinting {path}: "
                        f"wanted {sample_size} bytes at {offset}, got {len(data)}"
                    )
                h.update(data)
    return h.hexdigest()


def copy_then_remove(src: Path, dst: Path, verify_md5: bool) -> None:
    if dst.exists():
        raise MKVPlexError(f"Refusing to overwrite existing file: {dst}")

    # Temp file lives in destination directory so the final os.replace is atomic.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
            shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        shutil.copystat(src, tmp, follow_symlinks=True)

        if tmp.stat().st_size != src.stat().st_size:
            raise MKVPlexError(f"Size verification failed copying {src} -> {dst}")

        if verify_md5:
            src_md5 = md5sum(src)
            dst_md5 = md5sum(tmp)
            if src_md5 != dst_md5:
                raise MKVPlexError(f"MD5 verification failed copying {src} -> {dst}")

        if dst.exists():
            raise MKVPlexError(f"Refusing to overwrite existing file: {dst}")
        os.replace(tmp, dst)
        src.unlink()
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def transfer_file(src: Path, dst: Path, copy_only: bool, verify_md5: bool) -> str:
    if dst.exists():
        raise MKVPlexError(f"Refusing to overwrite existing file: {dst}")

    if copy_only:
        # Copy-only preserves source; still use a temp file + verification.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
                shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            shutil.copystat(src, tmp, follow_symlinks=True)
            if tmp.stat().st_size != src.stat().st_size:
                raise MKVPlexError(f"Size verification failed copying {src} -> {dst}")
            if verify_md5 and md5sum(src) != md5sum(tmp):
                raise MKVPlexError(f"MD5 verification failed copying {src} -> {dst}")
            os.replace(tmp, dst)
            return "copy"
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    # Fast path: same filesystem, metadata-only move.
    try:
        os.rename(src, dst)
        return "rename"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    # Cross-filesystem move.
    copy_then_remove(src, dst, verify_md5=verify_md5)
    return "copy+remove"


def open_cache(path: Path) -> sqlite3.Connection:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS rips (
            md5 TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            imdb_id TEXT,
            title TEXT NOT NULL,
            year INTEGER,
            season INTEGER,
            episode INTEGER,
            added_at TEXT NOT NULL
        )
        """
    )
    con.commit()
    return con


def cache_lookup(con: sqlite3.Connection, digest: str) -> Optional[dict[str, Any]]:
    cur = con.execute(
        "SELECT md5, media_type, imdb_id, title, year, season, episode, added_at FROM rips WHERE md5 = ?",
        (digest,),
    )
    row = cur.fetchone()
    if not row:
        return None
    keys = ["md5", "media_type", "imdb_id", "title", "year", "season", "episode", "added_at"]
    return dict(zip(keys, row))


def cache_store(
    con: sqlite3.Connection,
    digest: str,
    media_type: str,
    imdb_id: Optional[str],
    title: str,
    year: Optional[int],
    season: Optional[int] = None,
    episode: Optional[int] = None,
) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO rips
            (md5, media_type, imdb_id, title, year, season, episode, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            digest, media_type, imdb_id, title, year, season, episode,
            dt.datetime.now(dt.timezone.utc).isoformat(),
        ),
    )
    con.commit()


def human_size(n: int) -> str:
    value = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def print_plan(match: Match, transfers: Iterable[Transfer], directory: Path, mode: int, operation: str) -> None:
    print()
    print(f"Resolved: {match.title} ({match.year or '????'})")
    print(f"TMDb:    {match.tmdb_id}")
    print(f"IMDb:    {match.imdb_id or '(not supplied by TMDb)'}")
    print(f"Score:   {match.score:.3f}")
    print()
    print(f"Destination directory:\n  {directory}")
    print()
    print(f"Operation: {operation}")
    for item in transfers:
        try:
            size = human_size(item.source.stat().st_size)
        except OSError:
            size = "?"
        print(f"  {item.source}  [{size}]")
        print(f"    -> {item.destination}")
    print()
    print(f"Permissions after completion: chmod -R {mode:o} {directory}")


def confirm(prompt: str, assume_yes: bool, dry_run: bool) -> bool:
    if dry_run:
        return False
    if assume_yes:
        return True
    response = input(prompt + " [Y/n]: ").strip().lower()
    return response in {"", "y", "yes"}


def _existing_destination_status(source: Optional[Path], destination: Path) -> str:
    """Describe an existing target without ever treating it as safe to overwrite.

    For whole-file transfers we can cheaply distinguish an apparent duplicate from
    a different file using the same sampled-MD5 fingerprint used by --db.  Split
    outputs have no single comparable source file, so they are reported simply as
    existing conflicts.
    """
    if source is None:
        return "EXISTS (split output; content comparison not applicable)"
    try:
        sst = source.stat()
        dst = destination.stat()
        if not destination.is_file():
            return "EXISTS (not a regular file)"
        if sst.st_size != dst.st_size:
            return f"DIFFERENT SIZE (source {human_size(sst.st_size)}, existing {human_size(dst.st_size)})"
        src_fp = sampled_md5_fingerprint(source)
        dst_fp = sampled_md5_fingerprint(destination)
        if src_fp == dst_fp:
            return "APPARENT DUPLICATE (same size + sampled MD5)"
        return "DIFFERENT CONTENT (same size, sampled MD5 differs)"
    except OSError as exc:
        return f"EXISTS (comparison failed: {exc})"


def preflight_transfers(
    transfers: Iterable[Transfer], *, allow_existing: bool = False
) -> list[tuple[Optional[Path], Path, str]]:
    """Validate plan-internal uniqueness and report/raise on existing targets.

    Plan-internal duplicate sources/destinations are always fatal.  During a dry
    run, however, existing filesystem targets are collected so the *entire* plan
    can be reviewed instead of aborting at the first collision.
    """
    seen_sources: set[Path] = set()
    seen_destinations: set[Path] = set()
    conflicts: list[tuple[Optional[Path], Path, str]] = []
    for item in transfers:
        if item.source in seen_sources:
            raise MKVPlexError(f"Source appears more than once in transfer plan: {item.source}")
        if item.destination in seen_destinations:
            raise MKVPlexError(f"Destination appears more than once in transfer plan: {item.destination}")
        if item.destination.exists():
            if not allow_existing:
                raise MKVPlexError(f"Refusing to overwrite existing file: {item.destination}")
            conflicts.append((item.source, item.destination, _existing_destination_status(item.source, item.destination)))
        seen_sources.add(item.source)
        seen_destinations.add(item.destination)
    return conflicts


def print_destination_conflicts(conflicts: list[tuple[Optional[Path], Path, str]]) -> None:
    if not conflicts:
        return
    print()
    print(f"Destination conflicts: {len(conflicts)}")
    for source, destination, status in conflicts:
        print(f"  {destination}")
        if source is not None:
            print(f"    planned source: {source}")
        print(f"    {status}")
    print("  No existing destination will be overwritten.")


def execute_plan(
    destination_dir: Path,
    transfers: list[Transfer],
    copy_only: bool,
    verify_md5: bool,
    mode: int,
) -> list[str]:
    if destination_dir.exists():
        # Existing show dirs are normal. Existing movie dirs may also be normal
        # during retries, but individual target files are never overwritten.
        if not destination_dir.is_dir():
            raise MKVPlexError(f"Destination exists and is not a directory: {destination_dir}")
    else:
        destination_dir.mkdir(parents=True, mode=mode)

    operations: list[str] = []
    for item in transfers:
        item.destination.parent.mkdir(parents=True, exist_ok=True, mode=mode)
        op = transfer_file(item.source, item.destination, copy_only=copy_only, verify_md5=verify_md5)
        operations.append(op)

    recursive_chmod(destination_dir, mode)
    return operations


def resolve_title_and_year(args: argparse.Namespace, hints: SourceHints) -> tuple[str, Optional[int]]:
    title = args.title if args.title else hints.title
    year = args.year if args.year is not None else hints.year
    return title, year


__all__ = ['ensure_output_root', 'recursive_chmod', 'md5sum', 'sampled_md5_fingerprint', 'copy_then_remove', 'transfer_file', 'open_cache', 'cache_lookup', 'cache_store', 'human_size', 'print_plan', 'confirm', '_existing_destination_status', 'preflight_transfers', 'print_destination_conflicts', 'execute_plan', 'resolve_title_and_year']
