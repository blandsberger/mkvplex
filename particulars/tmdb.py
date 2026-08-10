"""TMDb HTTP/cache client and provider metadata/order helpers."""
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
from .models import DEFAULT_TMDB_CACHE, Episode, MKVPlexError, Match, TMDB_API, VERSION
from .common import _median
from .discovery import discovery_db
from .naming import canonical_spaces, eprint, normalize_for_match, similarity

def year_from_date(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.match(r"(\d{4})", value)
    return int(m.group(1)) if m else None


def metadata_query_variants(query: str) -> list[str]:
    """Return conservative provider-search variants for packaging titles.

    Box-set directory names often preserve retail packaging rather than the
    canonical program title, for example ``Batman™- The Complete Animated
    Series``.  Provider lookup should not require the user to hand-normalize
    that label, but we also must not silently *replace* the source identity.

    The first item is always the literal query.  Later items only remove
    trademark glyphs and common packaging adjectives/phrases, so callers can
    search progressively and report which fallback actually produced a match.
    """
    variants: list[str] = []

    def add(value: str) -> None:
        value = canonical_spaces(value)
        if value and normalize_for_match(value) not in {normalize_for_match(v) for v in variants}:
            variants.append(value)

    add(query)

    # Retail names frequently include these glyphs even though metadata
    # providers index the underlying title without them.
    clean = query.translate(str.maketrans({"™": "", "®": "", "©": ""}))
    clean = canonical_spaces(re.sub(r"\s*[-–—]+\s*", " ", clean))
    add(clean)

    # Preserve semantic words such as "Animated Series" while stripping the
    # packaging adjective.  Thus "Batman The Complete Animated Series" ->
    # "Batman The Animated Series", not merely "Batman".
    no_complete = re.sub(
        r"(?i)\bthe\s+complete\s+(?=animated\s+series\b)",
        "The ",
        clean,
    )
    no_complete = re.sub(
        r"(?i)\bcomplete\s+(?=animated\s+series\b)",
        "",
        no_complete,
    )
    add(no_complete)

    # Common box-set suffixes can be absent from provider titles.  These are
    # fallbacks only and therefore cannot override a successful closer query.
    stripped_suffix = re.sub(
        r"(?i)\s+(?:the\s+)?complete\s+(?:television\s+)?series\s*$",
        "",
        clean,
    )
    add(stripped_suffix)
    stripped_collection = re.sub(
        r"(?i)\s+(?:the\s+)?complete\s+(?:collection|box\s*set)\s*$",
        "",
        clean,
    )
    add(stripped_collection)

    # Authored/localized titles often carry a subtitle that has no lexical
    # relationship to the provider's English release title.  When the source
    # itself preserves an explicit subtitle separator, search the parent title
    # as a low-priority discovery fallback rather than attempting translation.
    # Example shape: ``Franchise: Localized Subtitle`` -> ``Franchise``.
    # Do not progressively drop arbitrary trailing words.
    for base in list(variants):
        parent = re.split(r"\s*[:：]\s*", base, maxsplit=1)[0].strip()
        if parent and normalize_for_match(parent) != normalize_for_match(base):
            add(parent)

    # A surprisingly common release-label mismatch is singular/plural drift in
    # the leading noun of an ``X of the Y`` title.  For example, retail/rip
    # labels may say ``Legends Of The Galactic Heroes`` while the provider
    # indexes ``Legend of the Galactic Heroes``.  Keep this deliberately
    # narrow: only vary the first word, only for an ``of the`` construction,
    # and only as a lower-priority fallback.  A successful literal query (for
    # example ``Masters of the Universe``) therefore remains preferred.
    for base in list(variants):
        m = re.match(r"(?i)^([A-Za-z][A-Za-z'-]*)\s+(of\s+the\s+.+)$", base)
        if not m:
            continue
        head, tail = m.group(1), m.group(2)
        if len(head) > 3 and head.casefold().endswith("s") and not head.casefold().endswith("ss"):
            add(f"{head[:-1]} {tail}")
        elif len(head) > 2:
            add(f"{head}s {tail}")

    return variants


class PersistentTMDbCache:
    """Small persistent TMDb response cache independent of approval/discovery DBs.

    A fresh ``--dry-run --db`` intentionally destroys the approval/discovery
    database, but metadata does not need to be downloaded again.  This cache is
    therefore separate and survives planning runs.  Entries expire by age so
    metadata can still refresh naturally; ``--tmdb-refresh`` bypasses reads.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stale = 0
        with self.lock:
            self.con.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS response (
                    request_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self.con.commit()

    def get(self, request_key: str, *, max_age_seconds: float) -> Optional[dict[str, Any]]:
        with self.lock:
            row = self.con.execute(
                "SELECT response_json, updated_at FROM response WHERE request_key=?",
                (request_key,),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        age = max(0.0, time.time() - float(row[1]))
        if age > max_age_seconds:
            self.stale += 1
            return None
        try:
            data = json.loads(row[0])
        except Exception:
            self.misses += 1
            return None
        if not isinstance(data, dict):
            self.misses += 1
            return None
        self.hits += 1
        return data

    def get_stale(self, request_key: str) -> Optional[dict[str, Any]]:
        """Return an expired entry only as a network-failure fallback."""
        with self.lock:
            row = self.con.execute(
                "SELECT response_json FROM response WHERE request_key=?",
                (request_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def put(self, request_key: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        with self.lock:
            self.con.execute(
                "INSERT OR REPLACE INTO response(request_key,response_json,updated_at) VALUES(?,?,?)",
                (request_key, payload, time.time()),
            )
            self.con.commit()


class RequestRateLimiter:
    """Thread-safe token bucket for respectful bounded TMDb traffic."""

    def __init__(self, rate_per_second: float = 8.0, burst: Optional[float] = None) -> None:
        self.rate = max(0.1, float(rate_per_second))
        self.capacity = max(1.0, float(burst if burst is not None else min(8.0, self.rate)))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self.updated)
                self.updated = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(max(0.001, wait))


class TMDbClient:
    def __init__(
        self,
        bearer_token: Optional[str],
        api_key: Optional[str],
        *,
        cache_path: Path = DEFAULT_TMDB_CACHE,
        cache_days: float = 30.0,
        rate_per_second: float = 8.0,
        workers: int = 8,
        refresh: bool = False,
    ) -> None:
        self.bearer_token = bearer_token
        self.api_key = api_key
        if not bearer_token and not api_key:
            raise MKVPlexError(
                "TMDb credentials missing. Set TMDB_BEARER_TOKEN (preferred) "
                "or TMDB_API_KEY."
            )
        self.cache = PersistentTMDbCache(cache_path)
        self.cache_seconds = max(0.0, float(cache_days)) * 86400.0
        self.rate_limiter = RequestRateLimiter(rate_per_second, burst=min(8.0, max(2.0, rate_per_second)))
        self.workers = max(1, min(int(workers), 16))
        self.refresh = bool(refresh)
        self.network_requests = 0
        self.retries = 0
        self.prefetch_batches = 0
        self.lock = threading.Lock()
        self.session_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _request_key(endpoint: str, params: Optional[dict[str, Any]] = None) -> str:
        return json.dumps(
            {"endpoint": endpoint, "params": dict(params or {})},
            sort_keys=True, separators=(",", ":"),
        )

    def _remember(self, request_key: str, data: dict[str, Any]) -> None:
        with self.lock:
            self.session_cache[request_key] = data
        self.cache.put(request_key, data)
        db = discovery_db()
        if db is not None:
            db.put_tmdb(request_key, data)

    def get(self, endpoint: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        public_params = dict(params or {})
        request_key = self._request_key(endpoint, public_params)
        with self.lock:
            session = self.session_cache.get(request_key)
        if session is not None:
            return session
        db = discovery_db()
        if not self.refresh and db is not None:
            cached = db.get_tmdb(request_key)
            if cached is not None:
                return cached
        if not self.refresh and self.cache_seconds > 0:
            cached = self.cache.get(request_key, max_age_seconds=self.cache_seconds)
            if cached is not None:
                if db is not None:
                    db.put_tmdb(request_key, cached)
                return cached

        request_params = dict(public_params)
        if self.api_key and not self.bearer_token:
            request_params["api_key"] = self.api_key
        query = urllib.parse.urlencode({k: v for k, v in request_params.items() if v is not None})
        url = f"{TMDB_API}{endpoint}"
        if query:
            url += "?" + query

        headers = {
            "Accept": "application/json",
            "User-Agent": f"mkvplex/{VERSION}",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        last_error: Optional[BaseException] = None
        for attempt in range(5):
            self.rate_limiter.acquire()
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=20) as response:
                    data = json.load(response)
                if not isinstance(data, dict):
                    raise MKVPlexError(f"TMDb returned non-object JSON for {endpoint}")
                with self.lock:
                    self.network_requests += 1
                    self.retries += attempt
                self._remember(request_key, data)
                return data
            except urllib.error.HTTPError as exc:
                last_error = exc
                # TMDb explicitly asks clients to respect 429. Retry-After wins;
                # otherwise back off exponentially with a modest floor.
                if exc.code == 429 and attempt < 4:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except (TypeError, ValueError):
                        delay = 0.0
                    time.sleep(max(delay, min(16.0, 1.0 * (2 ** attempt))))
                    continue
                if 500 <= exc.code < 600 and attempt < 3:
                    time.sleep(min(8.0, 0.75 * (2 ** attempt)))
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                stale = self.cache.get_stale(request_key)
                if stale is not None and exc.code >= 500:
                    eprint(f"mkvplex: TMDb HTTP {exc.code}; using stale cached metadata for {endpoint}")
                    return stale
                raise MKVPlexError(f"TMDb HTTP {exc.code}: {body[:500]}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(min(8.0, 0.75 * (2 ** attempt)))
                    continue
                stale = self.cache.get_stale(request_key)
                if stale is not None:
                    eprint(f"mkvplex: TMDb unavailable; using stale cached metadata for {endpoint}")
                    return stale
                raise MKVPlexError(f"TMDb request failed: {exc.reason}") from exc
        raise MKVPlexError(f"TMDb request failed after retries: {last_error}")

    def prefetch(
        self,
        requests: Iterable[tuple[str, Optional[dict[str, Any]]]],
        *,
        label: Optional[str] = None,
    ) -> None:
        """Fetch a de-duplicated metadata work set with bounded concurrency.

        ``get`` still owns cache lookup, rate limiting, retry and 429 handling;
        this method only lets independent metadata objects overlap on the wire.
        """
        unique: list[tuple[str, Optional[dict[str, Any]]]] = []
        seen: set[str] = set()
        for endpoint, params in requests:
            key = self._request_key(endpoint, params)
            if key in seen:
                continue
            seen.add(key)
            unique.append((endpoint, params))
        if not unique:
            return
        self.prefetch_batches += 1
        if label:
            print(f"  TMDb prefetch: {label} ({len(unique)} object(s), up to {self.workers} worker(s))")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.workers, len(unique))) as pool:
            futures = [pool.submit(self.get, endpoint, params) for endpoint, params in unique]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def stats_line(self) -> str:
        return (
            f"TMDb cache {self.cache.hits} hit/{self.cache.misses} miss/{self.cache.stale} stale; "
            f"network {self.network_requests} request(s), {self.retries} retr{'y' if self.retries == 1 else 'ies'}, "
            f"{self.prefetch_batches} batch(es)"
        )

    def _prime_endpoint(self, endpoint: str, data: Any) -> None:
        if isinstance(data, dict):
            self._remember(self._request_key(endpoint, None), data)

    def movie_bundle(self, tmdb_id: int) -> dict[str, Any]:
        """Fetch movie details + external IDs in one TMDb request and prime both caches."""
        endpoint = f"/movie/{tmdb_id}"
        data = self.get(endpoint, {"append_to_response": "external_ids"})
        base = dict(data)
        external = base.pop("external_ids", None)
        self._prime_endpoint(endpoint, base)
        self._prime_endpoint(f"{endpoint}/external_ids", external)
        return base

    def tv_bundle(self, tmdb_id: int) -> dict[str, Any]:
        """Fetch TV details, external IDs and episode-group index in one request."""
        endpoint = f"/tv/{tmdb_id}"
        data = self.get(endpoint, {"append_to_response": "external_ids,episode_groups"})
        base = dict(data)
        external = base.pop("external_ids", None)
        groups = base.pop("episode_groups", None)
        self._prime_endpoint(endpoint, base)
        self._prime_endpoint(f"{endpoint}/external_ids", external)
        self._prime_endpoint(f"{endpoint}/episode_groups", groups)
        return base

    def search_movie(self, query: str, year: Optional[int]) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": query, "include_adult": "true"}
        if year:
            params["primary_release_year"] = year
        return self.get("/search/movie", params).get("results", [])

    def search_tv(self, query: str, year: Optional[int]) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": query, "include_adult": "true"}
        if year:
            params["first_air_date_year"] = year
        return self.get("/search/tv", params).get("results", [])

    def movie_external_ids(self, tmdb_id: int) -> dict[str, Any]:
        return self.get(f"/movie/{tmdb_id}/external_ids")

    def movie_details(self, tmdb_id: int) -> dict[str, Any]:
        return self.get(f"/movie/{tmdb_id}")

    def tv_external_ids(self, tmdb_id: int) -> dict[str, Any]:
        return self.get(f"/tv/{tmdb_id}/external_ids")

    def tv_details(self, tmdb_id: int) -> dict[str, Any]:
        return self.get(f"/tv/{tmdb_id}")

    def tv_season(self, tmdb_id: int, season: int) -> dict[str, Any]:
        return self.get(f"/tv/{tmdb_id}/season/{season}")

    def tv_episode_groups(self, tmdb_id: int) -> dict[str, Any]:
        return self.get(f"/tv/{tmdb_id}/episode_groups")

    def tv_episode_group_details(self, group_id: str) -> dict[str, Any]:
        return self.get(f"/tv/episode_group/{group_id}")


def score_result(query: str, hint_year: Optional[int], title: str, result_year: Optional[int], popularity: float) -> float:
    score = similarity(query, title)
    if hint_year is not None and result_year is not None:
        if hint_year == result_year:
            score += 0.18
        else:
            score -= min(abs(hint_year - result_year) * 0.04, 0.24)
    # Tiny tie-break only; title similarity and year should dominate.
    score += min(max(popularity, 0.0), 1000.0) / 100000.0
    return max(0.0, min(score, 1.0))


def build_matches(client: TMDbClient, media_type: str, query: str, year: Optional[int]) -> list[Match]:
    """Search TMDb, including conservative release/packaging fallbacks.

    Results from all useful query variants are merged by TMDb id.  A candidate
    is scored against the variant that found it, while a tiny penalty keeps an
    equally-good literal-title match ahead of a more speculative fallback.
    """
    merged: dict[int, Match] = {}
    variants = metadata_query_variants(query)
    used_fallback: Optional[str] = None

    for variant_index, search_query in enumerate(variants):
        if media_type == "movie":
            rows = client.search_movie(search_query, year)
        elif media_type == "tv":
            rows = client.search_tv(search_query, year)
        else:
            raise ValueError(media_type)

        if rows and variant_index > 0 and used_fallback is None:
            used_fallback = search_query

        for row in rows[:20]:
            if media_type == "movie":
                title = row.get("title") or row.get("original_title") or ""
                result_year = year_from_date(row.get("release_date"))
            else:
                title = row.get("name") or row.get("original_name") or ""
                result_year = year_from_date(row.get("first_air_date"))
            if not title:
                continue
            score = score_result(
                search_query, year, title, result_year,
                float(row.get("popularity") or 0.0),
            )
            # Prefer literal/less-normalized queries when scores are otherwise
            # identical, but keep the penalty small enough that an exact
            # canonical fallback wins over a bad literal fuzzy hit.
            score = max(0.0, score - variant_index * 0.008)
            candidate_raw = dict(row)
            candidate_raw["_mkvplex_search_query"] = search_query
            candidate = Match(
                media_type=media_type,
                tmdb_id=int(row["id"]),
                title=title,
                year=result_year,
                imdb_id=None,
                score=score,
                raw=candidate_raw,
            )
            previous = merged.get(candidate.tmdb_id)
            if previous is None or candidate.score > previous.score:
                merged[candidate.tmdb_id] = candidate

    matches = sorted(merged.values(), key=lambda m: m.score, reverse=True)
    if matches:
        winning_query = str(matches[0].raw.get("_mkvplex_search_query") or query)
        if normalize_for_match(winning_query) != normalize_for_match(query):
            print(f"Metadata query fallback: {winning_query!r}")
    return matches


def attach_imdb_id(client: TMDbClient, match: Match) -> Match:
    # Coalesce the selected title's immediately-needed metadata. TMDb supports
    # append_to_response on top-level movie/TV detail methods, so one request
    # primes details + external IDs (and TV episode-group index) for later use.
    if match.media_type == "movie":
        client.movie_bundle(match.tmdb_id)
        ids = client.movie_external_ids(match.tmdb_id)
    else:
        client.tv_bundle(match.tmdb_id)
        ids = client.tv_external_ids(match.tmdb_id)
    imdb = ids.get("imdb_id")
    return Match(
        media_type=match.media_type,
        tmdb_id=match.tmdb_id,
        title=match.title,
        year=match.year,
        imdb_id=imdb,
        score=match.score,
        raw=match.raw,
    )


def print_candidates(matches: list[Match], limit: int = 5) -> None:
    print("Candidates:")
    for i, m in enumerate(matches[:limit], start=1):
        year = str(m.year) if m.year else "????"
        print(f"  {i}. {m.title} ({year})  TMDb {m.tmdb_id}  score={m.score:.3f}")


def choose_match(
    client: TMDbClient,
    media_type: str,
    query: str,
    year: Optional[int],
    explicit_imdb: Optional[str],
    assume_yes: bool,
) -> Match:
    matches = build_matches(client, media_type, query, year)
    if not matches:
        raise MKVPlexError(f"No {media_type} matches found for {query!r}")

    # If an IMDb ID was given, use it as a hard filter by checking candidates.
    if explicit_imdb:
        explicit_imdb = explicit_imdb.strip()
        if not explicit_imdb.startswith("tt"):
            explicit_imdb = "tt" + explicit_imdb
        for candidate in matches[:20]:
            enriched = attach_imdb_id(client, candidate)
            if enriched.imdb_id == explicit_imdb:
                return enriched
        raise MKVPlexError(f"IMDb ID {explicit_imdb} did not match the search candidates for {query!r}")

    best = matches[0]
    print_candidates(matches)

    if assume_yes:
        if best.score < 0.72:
            raise MKVPlexError(
                f"Best automatic match is low confidence ({best.score:.3f}). "
                "Run interactively or specify --title/--year/--imdb."
            )
        return attach_imdb_id(client, best)

    default = "1"
    response = input(f"Select match [1-{min(5, len(matches))}, default 1, q=quit]: ").strip().lower()
    if response in {"q", "quit", "n", "no"}:
        raise MKVPlexError("Cancelled")
    if not response:
        response = default
    try:
        idx = int(response)
    except ValueError as exc:
        raise MKVPlexError(f"Invalid selection: {response!r}") from exc
    if idx < 1 or idx > min(5, len(matches)):
        raise MKVPlexError(f"Selection out of range: {idx}")
    return attach_imdb_id(client, matches[idx - 1])


def season_episodes(client: TMDbClient, match: Match, season: int) -> list[Episode]:
    data = client.tv_season(match.tmdb_id, season)
    episodes: list[Episode] = []
    for row in data.get("episodes", []):
        ep_num = row.get("episode_number")
        name = row.get("name")
        if ep_num is None or not name:
            continue
        runtime = row.get("runtime")
        try:
            runtime_minutes = int(runtime) if runtime is not None else None
        except (TypeError, ValueError):
            runtime_minutes = None
        episodes.append(Episode(
            season=season,
            number=int(ep_num),
            title=str(name),
            tmdb_id=int(row["id"]) if row.get("id") is not None else None,
            runtime_minutes=runtime_minutes,
            air_year=year_from_date(row.get("air_date")),
            air_date=str(row.get("air_date")) if row.get("air_date") else None,
        ))
    episodes.sort(key=lambda e: e.number)
    return episodes


def regular_series_episodes(client: TMDbClient, match: Match) -> list[Episode]:
    """Return every regular TMDb episode in season/episode order.

    Season objects are an independent metadata work set, so prefetch them as a
    bounded batch before media analysis. One season request yields all episode
    runtimes/names for that season; there is never one TMDb request per cut.
    """
    rows = _regular_tv_season_rows(client, match)
    client.prefetch(
        [(f"/tv/{match.tmdb_id}/season/{season}", None) for season, _count, _air_year, _name in rows],
        label=f"{match.title} regular seasons",
    )
    episodes: list[Episode] = []
    for season, _count, _air_year, _name in rows:
        episodes.extend(season_episodes(client, match, season))
    episodes.sort(key=lambda ep: (ep.season, ep.number))
    return episodes


def _alternate_episode_order_rows(client: TMDbClient, match: Match) -> list[dict[str, Any]]:
    """Return DVD/production episode groups exposed by TMDb for a show."""
    try:
        data = client.tv_episode_groups(match.tmdb_id)
    except MKVPlexError:
        return []
    rows = []
    for row in data.get("results", []) or []:
        try:
            kind = int(row.get("type") or 0)
        except (TypeError, ValueError):
            kind = 0
        if kind not in {3, 6}:  # TMDb: DVD=3, Production=6
            continue
        rows.append(row)
    rows.sort(key=lambda r: (0 if int(r.get("type") or 0) == 3 else 1, str(r.get("name") or "")))
    return rows


def _episode_group_sequence(
    client: TMDbClient, match: Match, group_row: dict[str, Any]
) -> list[Episode]:
    """Return ordinary TMDb Episode identities in one episode-group order.

    Episode-group rows are an ordering layer, not a new Plex numbering scheme.
    We therefore flatten the group in its advertised order, then map every
    episode back to the show's ordinary season/episode identity by TMDb episode
    id (or season/episode key as a fallback).  Destination filenames remain the
    canonical SxE names even when physical discs use DVD/production order.
    """
    group_id = str(group_row.get("id") or "")
    if not group_id:
        return []
    try:
        details = client.tv_episode_group_details(group_id)
    except MKVPlexError:
        return []

    ordinary = regular_series_episodes(client, match)
    by_id = {ep.tmdb_id: ep for ep in ordinary if ep.tmdb_id is not None}
    by_key = {(ep.season, ep.number): ep for ep in ordinary}

    def intish(value: Any, default: int = 10**9) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    sequence: list[Episode] = []
    seen: set[tuple[int, int]] = set()
    groups = list(details.get("groups") or [])
    groups.sort(key=lambda row: (intish(row.get("order")), str(row.get("name") or "")))
    for group in groups:
        episodes = list(group.get("episodes") or [])
        episodes.sort(key=lambda row: (
            intish(row.get("order")),
            intish(row.get("season_number")),
            intish(row.get("episode_number")),
            intish(row.get("id")),
        ))
        for row in episodes:
            ep: Optional[Episode] = None
            try:
                eid = int(row.get("id")) if row.get("id") is not None else None
            except (TypeError, ValueError):
                eid = None
            if eid is not None:
                ep = by_id.get(eid)
            if ep is None:
                try:
                    key = (int(row.get("season_number")), int(row.get("episode_number")))
                except (TypeError, ValueError):
                    key = None
                if key is not None:
                    ep = by_key.get(key)
            if ep is None:
                continue
            key = (ep.season, ep.number)
            if key in seen:
                continue
            seen.add(key)
            sequence.append(ep)
    return sequence


def _collection_order_sequences(
    client: TMDbClient, match: Match
) -> dict[str, tuple[str, list[Episode], Optional[str]]]:
    """Return usable ordinary/DVD/production sequences for a TV series."""
    regular = regular_series_episodes(client, match)
    expected = len(regular)
    options: dict[str, tuple[str, list[Episode], Optional[str]]] = {
        "regular": ("TMDb regular season/episode order", regular, None)
    }
    kind_names = {3: ("dvd", "DVD order"), 6: ("production", "Production order")}
    for row in _alternate_episode_order_rows(client, match):
        try:
            kind = int(row.get("type") or 0)
        except (TypeError, ValueError):
            continue
        meta = kind_names.get(kind)
        if meta is None:
            continue
        key, label = meta
        sequence = _episode_group_sequence(client, match, row)
        # An ordering with omitted/duplicated regular episodes is unsuitable for
        # an automatic complete-series assignment. Keep it visible in the
        # forensic report, but do not offer it as an executable ordering.
        if len(sequence) != expected:
            continue
        options[key] = (label, sequence, str(row.get("id") or "") or None)
    return options


def _volume_order_sequences(
    client: TMDbClient, match: Match
) -> dict[str, tuple[str, list[Episode], Optional[str]]]:
    """Return provider orderings usable for a numbered retail volume.

    Complete-series collection execution deliberately requires an alternate
    ordering to contain every regular episode.  A numbered retail volume is a
    partial slice, however, and TMDb DVD groups are sometimes nearly complete
    rather than identical to the regular inventory.  For physical DVD media, a
    high-coverage DVD group is stronger ordering evidence than aired order.
    Canonical Episode identities remain the ordinary SxE identities.
    """
    regular = regular_series_episodes(client, match)
    regular_keys = {(ep.season, ep.number) for ep in regular}
    expected = len(regular_keys)
    options: dict[str, tuple[str, list[Episode], Optional[str]]] = {
        "regular": ("TMDb regular season/episode order", regular, None)
    }
    if expected <= 0:
        return options

    # A numbered retail volume is physical chronology, not an assertion that
    # provider SxE numbers themselves are authored-disc order.  When TMDb has
    # complete ISO air dates, expose a chronological ordering layer while
    # retaining each Episode's canonical season/number identity.  This is a
    # conservative fallback for shows whose numeric SxE order differs locally
    # from the order in which complete programs aired/were packaged.
    iso_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if all(ep.air_date and iso_date.fullmatch(ep.air_date) for ep in regular):
        chronological = sorted(
            regular, key=lambda ep: (str(ep.air_date), ep.season, ep.number, ep.tmdb_id or 0)
        )
        if [(ep.season, ep.number) for ep in chronological] != [
            (ep.season, ep.number) for ep in regular
        ]:
            options["chronological"] = ("TMDb chronological air-date order", chronological, None)

    kind_names = {3: ("dvd", "TMDb DVD order"), 6: ("production", "TMDb Production order")}
    for row in _alternate_episode_order_rows(client, match):
        try:
            kind = int(row.get("type") or 0)
        except (TypeError, ValueError):
            continue
        meta = kind_names.get(kind)
        if meta is None:
            continue
        key, label = meta
        sequence = _episode_group_sequence(client, match, row)
        if not sequence:
            continue
        sequence_keys = {(ep.season, ep.number) for ep in sequence}
        if len(sequence_keys) != len(sequence):
            continue
        coverage = len(sequence_keys & regular_keys) / expected
        # This path is allowed to relax only the all-episodes requirement, not
        # series identity.  A sparse alternate group must never override aired
        # order merely because its runtimes happen to fit repetitive TV titles.
        if coverage < 0.90:
            continue
        if coverage < 0.999:
            label = f"{label} [{len(sequence_keys)}/{expected} canonical episodes]"
        options[key] = (label, sequence, str(row.get("id") or "") or None)
    return options


def _show_runtime_minutes(client: TMDbClient, match: Match) -> Optional[float]:
    details = client.tv_details(match.tmdb_id)
    raw = details.get("episode_run_time") or []
    vals: list[float] = []
    for item in raw:
        try:
            val = float(item)
        except (TypeError, ValueError):
            continue
        if val > 0:
            vals.append(val)
    return _median(vals)


def _regular_tv_season_rows(
    client: TMDbClient, match: Match
) -> list[tuple[int, int, Optional[int], str]]:
    """Return de-duplicated regular TMDb seasons in numeric order."""
    details = client.tv_details(match.tmdb_id)
    rows: list[tuple[int, int, Optional[int], str]] = []
    for row in details.get("seasons", []) or []:
        try:
            number = int(row.get("season_number"))
            episode_count = int(row.get("episode_count") or 0)
        except (TypeError, ValueError):
            continue
        if number <= 0 or episode_count <= 0:
            continue
        air_year = year_from_date(row.get("air_date"))
        name = canonical_spaces(str(row.get("name") or f"Season {number}"))
        rows.append((number, episode_count, air_year, name))
    dedup: dict[int, tuple[int, int, Optional[int], str]] = {}
    for item in rows:
        dedup[item[0]] = item
    return [dedup[n] for n in sorted(dedup)]


__all__ = ['year_from_date', 'metadata_query_variants', 'PersistentTMDbCache', 'RequestRateLimiter', 'TMDbClient', 'score_result', 'build_matches', 'attach_imdb_id', 'print_candidates', 'choose_match', 'season_episodes', 'regular_series_episodes', '_alternate_episode_order_rows', '_episode_group_sequence', '_collection_order_sequences', '_volume_order_sequences', '_show_runtime_minutes', '_regular_tv_season_rows']
