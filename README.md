# mkvplex

`mkvplex` turns MakeMKV `.mkv` rips into consistently named Plex media while preserving a dry-run-first, fail-closed workflow. It resolves metadata through TMDb, analyzes physical disc structure, separates extras/play-all masters, and can split multi-segment TV program files without re-encoding.

## Safety model

- Dry runs are first-class: `--dry-run --db` stores an approved plan.
- Source identity is bound with path, size, mtime, and sampled MD5 before execution.
- Ambiguous disc/episode mappings fail closed instead of shifting later episodes into holes.
- Authored chapter markers outrank runtime reconstruction when splitting multi-segment programs.
- Play-all masters can be used as a physical ordering oracle by matching encoded video packet content.
- Skipped masters and bonus material are archived to the extras tree rather than deleted.

## Layout

```text
mkvplex/
├── main.py
├── README.md
├── README.txt
├── CHANGELOG
├── particulars/
│   ├── __init__.py
│   ├── models.py
│   ├── common.py
│   ├── discovery.py
│   ├── naming.py
│   ├── media.py
│   ├── discs.py
│   ├── tmdb.py
│   ├── fsops.py
│   ├── movie.py
│   ├── collection.py
│   ├── volume.py
│   ├── tvplan.py
│   ├── tv.py
│   └── cli.py
└── tests/
    └── test_regressions.py
```

`particulars/__init__.py` re-exports the historical single-file API for compatibility, while new work should patch/import the logical module that owns the behavior.

## Requirements

- Python 3.10+
- `ffprobe` and `ffmpeg` in `PATH`
- TMDb credentials via `TMDB_BEARER_TOKEN` (preferred) or `TMDB_API_KEY`

The Python implementation otherwise uses the standard library.

## Usage

From the repository root:

```bash
python3 main.py movie INPUT MOVIES_OUTPUT EXTRAS_OUTPUT --dry-run --db
python3 main.py tv INPUT TV_OUTPUT EXTRAS_OUTPUT --dry-run --db
```

After reviewing an approved dry run, rerun without `--dry-run` to execute against the bound plan. Use `python3 main.py --help` for all options.

Example:

```bash
python3 main.py tv \
  'Incoming/x/Pinky And The Brain' \
  'Tv Shows' \
  'Extras' \
  --dry-run --db
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The regression suite includes prior failures around authored episode ranges, missing numbered discs, complete-series season projection, retail-volume labels, 6-program play-all detection, packet-content ordering, alternate episode orders, and chapter-topology correction.

## Development

The code was intentionally split at subsystem boundaries in 0.13.0. Keep low-level dependencies pointed downward: models/common → discovery/naming/fsops/TMDb/media/discs → movie/collection/volume → TV planning/orchestration → CLI. Prefer adding functionality to the owning module rather than rebuilding a monolith.
