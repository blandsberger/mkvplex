mkvplex 0.13.0
===============

MakeMKV -> Plex planning, naming, extras archival, and TV program splitting.

QUICK START
-----------
Set one of:
  TMDB_BEARER_TOKEN=...   (preferred)
  TMDB_API_KEY=...

Preview TV:
  python3 main.py tv INPUT TV_OUTPUT EXTRAS_OUTPUT --dry-run --db

Preview movie:
  python3 main.py movie INPUT MOVIES_OUTPUT EXTRAS_OUTPUT --dry-run --db

Run tests:
  python3 -m unittest discover -s tests -v

The implementation is now under particulars/.  particulars/__init__.py keeps the
historical API available, but changes should be made in the logical module that
owns the behavior.  See README.md for architecture and safety details.
