import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import particulars as m
from particulars import cli as cli_mod
from particulars import tv as tv_mod
from particulars import volume as volume_mod
from particulars import media as media_mod
from particulars import tmdb as tmdb_mod


class MKVPlexRegressionTests(unittest.TestCase):
    def make_versailles_tree(self, root: Path) -> Path:
        show = root / 'Rose of Versailles'
        show.mkdir()
        for start, end in ((1, 10), (11, 20), (21, 30), (31, 40)):
            directory = show / f'Rose of Versailles {start}-{end}'
            directory.mkdir()
            for i in range(10):
                (directory / f'A1_t{i:02d}.mkv').write_bytes(b'x')
        return show

    def test_versailles_shape_is_tv(self):
        with tempfile.TemporaryDirectory() as td:
            show = self.make_versailles_tree(Path(td))
            self.assertTrue(m.looks_like_tv_tree(show))
            groups = sorted(m.find_tv_rip_groups(show), key=lambda g: m._group_sort_key(g, 1))
            self.assertEqual([g.episode_span for g in groups], [(1, 10), (11, 20), (21, 30), (31, 40)])
            self.assertEqual([m._series_title_component(g.directory.name) for g in groups], ['Rose of Versailles'] * 4)

    def test_boxset_packaging_token_stays_with_root_series(self):
        with tempfile.TemporaryDirectory() as td:
            show = Path(td) / 'Mushi Shi'
            disc = show / 'MUSHI_SHI_BOXSET_D4'
            disc.mkdir(parents=True)
            track = disc / 'H1_t04.mkv'
            track.write_bytes(b'x')
            group = m.TvRipGroup(disc, None, 4, False, (track,))

            self.assertEqual(m._series_title_component(disc.name), 'MUSHI SHI')
            self.assertEqual(m._group_series_query(show, 'Mushi Shi', group), 'Mushi Shi')
            self.assertGreaterEqual(m.similarity('MUSHI SHI', 'Mushi Shi'), 0.88)

        for label in (
            'MUSHI_SHI_BOX_SET_D4',
            'MUSHI SHI BOXED SET DISC 4',
            'MUSHI_SHI_COMPLETE_SERIES_D4',
            'MUSHI_SHI_COMPLETE_COLLECTION_D4',
        ):
            self.assertEqual(m._series_title_component(label), 'MUSHI SHI')

    def test_bad_range_shape_is_not_auto_tv(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'Thing'; root.mkdir()
            for name, count in [('Thing 1-10', 10), ('Thing 12-20', 9)]:
                d = root / name; d.mkdir()
                for i in range(count):
                    (d / f'A1_t{i:02d}.mkv').write_bytes(b'x')
            self.assertFalse(m.looks_like_tv_tree(root))

    def test_ordinal_ranges_map_exactly_40(self):
        with tempfile.TemporaryDirectory() as td:
            show = self.make_versailles_tree(Path(td))
            groups = sorted(m.find_tv_rip_groups(show), key=lambda g: m._group_sort_key(g, 1))
            analyses = [
                m.TrackAnalysis(track, group, 23 * 60 + 44, 1, 1.0)
                for group in groups for track in group.tracks
            ]
            episodes = [m.Episode(1, i, f'Episode {i}', i, 24, 1979) for i in range(1, 41)]
            assignments, missing, skipped = m.select_episode_manifest_by_ordinal_ranges(
                groups, episodes, analyses, show_runtime_minutes=24, tolerance_minutes=12
            )
            self.assertEqual(len(assignments), 40)
            self.assertEqual([a.episode.number for a in assignments], list(range(1, 41)))
            self.assertEqual(missing, [])
            self.assertEqual(skipped, [])

    def test_movie_runtime_contradiction_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tracks = []
            durations = {}
            for i in range(3):
                p = root / f'A1_t{i:02d}.mkv'; p.write_bytes(b'x' * (i + 1)); tracks.append(p)
                durations[p] = 23 * 60 + 44
            with self.assertRaisesRegex(m.MKVPlexError, 'No physical track is compatible'):
                m._select_primary_movie_track(root, 'The Rose of Versailles', tracks, durations, 113 * 60)

    def test_numbered_disc_gap_fails_closed(self):
        root = Path('/synthetic')
        groups = [
            (1, m.TvRipGroup(root / f'Disc {n}', 1, n, False, tuple()))
            for n in (1, 2, 4)
        ]
        with self.assertRaisesRegex(m.MKVPlexError, 'missing Disc 3'):
            m._validate_numbered_disc_sequences(groups)

    def test_single_standalone_disc_does_not_require_disc_one(self):
        root = Path('/synthetic')
        m._validate_numbered_disc_sequences([(1, m.TvRipGroup(root / 'Disc 2', 1, 2, False, tuple()))])

    def test_season_projection_preserves_episode_identity(self):
        episodes = [m.Episode(1, i, f'Title {i}', 1000 + i, 23, 1990 + (i // 50)) for i in range(1, 162)]
        mapped = m.remap_episode_seasons(episodes, (18, 22, 24, 24, 24, 24, 25))
        self.assertEqual(len(mapped), 161)
        for before, after in zip(episodes, mapped):
            self.assertEqual((before.title, before.tmdb_id, before.runtime_minutes, before.air_year),
                             (after.title, after.tmdb_id, after.runtime_minutes, after.air_year))
        self.assertEqual((mapped[17].season, mapped[17].number), (1, 18))
        self.assertEqual((mapped[18].season, mapped[18].number), (2, 1))
        self.assertEqual((mapped[136].season, mapped[136].number), (7, 1))
        self.assertEqual((mapped[-1].season, mapped[-1].number), (7, 25))

    def test_auto_season_counts_forces_tv(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / 'PlainShow'; inp.mkdir()
            calls = []
            old_client, old_tv, old_movie, old_workers = cli_mod.TMDbClient, cli_mod.do_tv, cli_mod.do_movie, cli_mod.resolve_media_workers
            class FakeClient:
                def __init__(self, **kwargs):
                    self.cache = SimpleNamespace(path=Path(td) / 'tmdb.sqlite3')
                    self.workers = 1
                def stats_line(self): return 'fake stats'
            try:
                cli_mod.TMDbClient = FakeClient
                cli_mod.do_tv = lambda args, client: calls.append('tv') or 0
                cli_mod.do_movie = lambda args, client: calls.append('movie') or 0
                cli_mod.resolve_media_workers = lambda *args, **kwargs: (1, 'hdd')
                rc = m.main(['auto', str(inp), str(Path(td) / 'out'), str(Path(td) / 'extras'),
                             '--season-counts', '1', '--dry-run'])
                self.assertEqual(rc, 0)
                self.assertEqual(calls, ['tv'])
            finally:
                cli_mod.TMDbClient, cli_mod.do_tv, cli_mod.do_movie, cli_mod.resolve_media_workers = old_client, old_tv, old_movie, old_workers

    def test_numbered_volume_label_is_packaging_not_series_identity(self):
        root = Path('/synthetic/Pinky And The Brain')
        for label in (
            'PINKY AND THE BRAIN VOL 2 DISC 1',
            'PINKY AND THE BRAIN Vol. 2 Disc 2',
            'PINKY AND THE BRAIN Volume 2 DVD03',
        ):
            self.assertEqual(m._series_title_component(label), 'PINKY AND THE BRAIN')
            group = m.TvRipGroup(root / label, None, 1, False, tuple())
            self.assertEqual(
                m._group_series_query(root, 'Pinky And The Brain', group),
                'Pinky And The Brain',
            )


    def test_numbered_volume_common_across_discs(self):
        root = Path('/synthetic/Pinky And The Brain')
        groups = [
            m.TvRipGroup(root / f'PINKY AND THE BRAIN VOL 2 DISC {n}', None, n, False, tuple())
            for n in range(1, 5)
        ]
        self.assertEqual(m.common_numbered_volume(root, groups), 2)

    def test_airdate_segments_collapse_to_program_units(self):
        eps = [
            m.Episode(2, 14, 'Brain of the Future', 1, 21, 1997, '1997-02-08'),
            m.Episode(2, 15, 'Brinky', 2, 21, 1997, '1997-02-22'),
            m.Episode(2, 16, 'Hoop Schemes', 3, 21, 1997, '1997-05-17'),
            m.Episode(3, 1, 'Part A', 4, 12, 1997, '1997-09-08'),
            m.Episode(3, 2, 'Part B', 5, 9, 1997, '1997-09-08'),
        ]
        units = m.episodes_to_program_units(eps, target_seconds=21*60, show_runtime_minutes=21)
        self.assertEqual([[e.title for e in unit] for unit in units], [
            ['Brain of the Future'], ['Brinky'], ['Hoop Schemes'], ['Part A', 'Part B']
        ])

    def test_do_tv_numbered_volume_bypasses_single_season_preview(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inp = base / 'Pinky And The Brain'; inp.mkdir()
            out = base / 'Tv Shows'; out.mkdir()
            extras = base / 'Extras'; extras.mkdir()
            groups = []
            for disc in range(1, 5):
                d = inp / f'PINKY AND THE BRAIN VOL 2 DISC {disc}'; d.mkdir()
                p = d / 'x_t00.mkv'; p.write_bytes(b'x')
                groups.append(m.TvRipGroup(d, None, disc, False, (p,)))
            captured = {}
            old_find = tv_mod.find_tv_rip_groups
            old_choose = tv_mod.choose_match
            old_buckets = tv_mod._resolve_tv_series_buckets
            old_rows = tv_mod._regular_tv_season_rows
            old_series = tv_mod._do_tv_series
            try:
                tv_mod.find_tv_rip_groups = lambda _input: groups
                match = m.Match('tv', 2228, 'Pinky and the Brain', 1995, None, 1.0, {})
                tv_mod.choose_match = lambda *args, **kwargs: match
                tv_mod._resolve_tv_series_buckets = lambda args, client, input_dir, query, year, gs, primary: [(query, year, primary, gs)]
                tv_mod._regular_tv_season_rows = lambda client, match: [(1, 19, 1995, 'Season 1'), (2, 16, 1996, 'Season 2'), (3, 51, 1997, 'Season 3'), (4, 9, 1998, 'Season 4')]
                def fake_series(args, client, **kwargs):
                    captured['volume'] = getattr(args, '_numbered_volume', None)
                    captured['seasons'] = {g.season for g in kwargs['groups']}
                    return 0
                tv_mod._do_tv_series = fake_series
                args = SimpleNamespace(
                    input=str(inp), output=str(out), extras=str(extras), movies_output=None,
                    title=None, year=None, imdb=None, yes=False, season=None, season_counts=None,
                    episode_start=1, episode_count=None, dry_run=True, db=None, copy=False,
                    verify_md5=False, mode=0o755, probe_workers=1, all_tracks=False,
                    no_aggregate_split=False, disc_kind='auto', collection_order='auto',
                    split_search_window=180.0, split_black_min=0.30,
                )
                self.assertEqual(m.do_tv(args, object()), 0)
            finally:
                tv_mod.find_tv_rip_groups = old_find
                tv_mod.choose_match = old_choose
                tv_mod._resolve_tv_series_buckets = old_buckets
                tv_mod._regular_tv_season_rows = old_rows
                tv_mod._do_tv_series = old_series
            self.assertEqual(captured['volume'], 2)
            self.assertEqual(captured['seasons'], {m.COMPLETE_SERIES_SENTINEL})


    def test_six_program_playall_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'PINKY AND THE BRAIN VOL 2 DISC 1'; d.mkdir()
            names = [
                ('B1_t00.mkv', 5811348807, 2*3600 + 6*60 + 12),
                ('C1_t01.mkv', 955960826, 20*60 + 46),
                ('C2_t02.mkv', 975017880, 21*60 + 11),
                ('C3_t03.mkv', 974883149, 21*60 + 11),
                ('D1_t04.mkv', 975188701, 21*60 + 11),
                ('D3_t05.mkv', 960340480, 20*60 + 51),
                ('D4_t06.mkv', 970695984, 21*60 + 4),
            ]
            paths = []
            durations = {}
            for name, size, duration in names:
                path = d / name
                with path.open('wb') as f:
                    f.truncate(size)
                paths.append(path); durations[path] = duration
            group = m.TvRipGroup(d, None, 1, False, tuple(paths))
            eps = [m.Episode(1, i, f'Program {i}', i, 21, 1997, f'1997-01-{i:02d}') for i in range(1, 7)]
            rows = m.analyze_tv_tracks(
                [(path, group) for path in paths], eps, durations,
                show_runtime_minutes=21, tolerance_minutes=12,
            )
            master = next(row for row in rows if row.path.name == 'B1_t00.mkv')
            self.assertEqual(len(master.aggregate_of), 6)
            self.assertEqual([p.name for p in master.aggregate_of], [
                'C1_t01.mkv', 'C2_t02.mkv', 'C3_t03.mkv',
                'D1_t04.mkv', 'D3_t05.mkv', 'D4_t06.mkv',
            ])

    def test_playall_component_set_excludes_long_bonus(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'PINKY AND THE BRAIN VOL 2 DISC 4'; d.mkdir()
            names = [
                ('B1_t00.mkv', 5009685241, 1*3600 + 48*60 + 47),
                ('C1_t01.mkv', 987700286, 21*60 + 28),
                ('C3_t02.mkv', 1022036058, 22*60 + 12),
                ('C4_t03.mkv', 996430763, 21*60 + 38),
                ('C5_t04.mkv', 1027404519, 22*60 + 19),
                ('C6_t05.mkv', 976706613, 21*60 + 12),
                ('D1_t06.mkv', 1046645389, 29*60 + 49),
            ]
            paths = []; durations = {}
            for name, size, duration in names:
                path = d / name
                with path.open('wb') as f:
                    f.truncate(size)
                paths.append(path); durations[path] = duration
            group = m.TvRipGroup(d, None, 4, False, tuple(paths))
            eps = [m.Episode(3, i, f'Program {i}', i, 22, 1997, f'1997-02-{i:02d}') for i in range(1, 6)]
            rows = m.analyze_tv_tracks(
                [(path, group) for path in paths], eps, durations,
                show_runtime_minutes=22, tolerance_minutes=12,
            )
            master = next(row for row in rows if row.path.name == 'B1_t00.mkv')
            self.assertEqual(len(master.aggregate_of), 5)
            self.assertNotIn(d / 'D1_t06.mkv', master.aggregate_of)


    def test_numbered_volume_plan_can_span_seasons(self):
        root = Path('/synthetic/Pinky And The Brain')
        groups = []
        analyses = []
        program_no = 0
        # 21 authored programs, distributed 6/5/5/5; each disc also has a
        # structurally verified play-all master.
        for disc, count in enumerate((6, 5, 5, 5), 1):
            d = root / f'PINKY AND THE BRAIN VOL 2 DISC {disc}'
            tracks = []
            components = []
            for i in range(1, count + 1):
                p = d / f'C{i}_t{i:02d}.mkv'
                tracks.append(p); components.append(p); program_no += 1
            master = d / 'B1_t00.mkv'; tracks.insert(0, master)
            group = m.TvRipGroup(d, None, disc, False, tuple(tracks))
            groups.append(group)
            analyses.append(m.TrackAnalysis(master, group, count*21*60, count*1000, 1.0, tuple(components)))
            for p in components:
                analyses.append(m.TrackAnalysis(p, group, 21*60, 1000, 1.0))

        # 65 physical programs across the series.  Volume 2 should land on
        # zero-based program 22 when a 3-volume partition is inferred.  Make
        # program 23 the last three S2 programs and then cross into S3.
        episodes = []
        for i in range(65):
            season = 2 if i < 25 else 3
            number = i + 1 if season == 2 else i - 24
            episodes.append(m.Episode(season, number, f'P{i+1}', i+1, 21, 1997, f'1997-{(i//28)+1:02d}-{(i%28)+1:02d}'))

        class FakeClient:
            pass
        old_orders = volume_mod._volume_order_sequences
        old_playall = volume_mod.playall_component_order_by_video
        try:
            volume_mod.playall_component_order_by_video = lambda master, components: list(components)
            volume_mod._volume_order_sequences = lambda client, match: {
                'regular': ('regular', episodes, None)
            }
            plan = m.infer_numbered_volume_plan(
                FakeClient(), m.Match('tv', 2228, 'Pinky and the Brain', 1995, None, 1.0, {}),
                volume_number=2, groups=groups, analyses=analyses, show_runtime_minutes=21,
            )
        finally:
            volume_mod._volume_order_sequences = old_orders
            volume_mod.playall_component_order_by_video = old_playall
        self.assertEqual(plan['start'], 22)
        self.assertEqual(len(plan['programs']), 21)
        self.assertEqual(len(plan['episodes']), 21)
        self.assertEqual(plan['episodes'][0].season, 2)
        self.assertEqual(plan['episodes'][-1].season, 3)



    def test_numbered_volume_known_width_frontloads_remainder(self):
        # Regression from Pinky and the Brain Volume 2: TMDb regular metadata
        # collapses to 66 provider program units, while the physical Volume 2
        # contains 21 authored program titles.  With three retail volumes, the
        # other 45 units must balance 23 before / 22 after, so Volume 2 starts
        # at zero-based program 23.  The old equal-width formula chose 22.
        self.assertEqual(
            m._known_volume_balanced_start(
                total=66, count=21, volume_number=2, volume_total=3
            ),
            23,
        )
        # If provider grouping yields the true 65-program inventory instead,
        # the same rule naturally gives the 22/21/22 retail partition.
        self.assertEqual(
            m._known_volume_balanced_start(
                total=65, count=21, volume_number=2, volume_total=3
            ),
            22,
        )

    def test_numbered_volume_66_unit_window_starts_brain_future_ends_brainy_jack(self):
        root = Path('/synthetic/Pinky And The Brain')
        groups = []
        analyses = []
        for disc, count in enumerate((6, 5, 5, 5), 1):
            d = root / f'PINKY AND THE BRAIN VOL 2 DISC {disc}'
            tracks = []
            components = []
            for i in range(1, count + 1):
                p = d / f'C{i}_t{i:02d}.mkv'
                tracks.append(p); components.append(p)
            master = d / 'B1_t00.mkv'; tracks.insert(0, master)
            group = m.TvRipGroup(d, None, disc, False, tuple(tracks))
            groups.append(group)
            analyses.append(m.TrackAnalysis(master, group, count*21*60, count*1000, 1.0, tuple(components)))
            for p in components:
                analyses.append(m.TrackAnalysis(p, group, 21*60, 1000, 1.0))

        # 66 provider program units.  Keep every unit exactly 21 minutes so
        # runtime fit cannot manufacture the boundary; packaging position must
        # select zero-based 23.  Name the two boundary units explicitly.
        episodes = []
        for i in range(66):
            if i == 23:
                ep = m.Episode(2, 14, 'Brain of the Future', 2000+i, 21, 1997, f'1997-02-{(i%28)+1:02d}')
            elif i == 43:
                ep = m.Episode(3, 26, 'Brainy Jack', 2000+i, 21, 1997, f'1997-10-{(i%28)+1:02d}')
            else:
                ep = m.Episode(9, i+1, f'Provider Program {i+1}', 2000+i, 21, 1997, f'1997-{(i//28)+1:02d}-{(i%28)+1:02d}')
            episodes.append(ep)

        class FakeClient:
            pass
        old_orders = volume_mod._volume_order_sequences
        old_playall = volume_mod.playall_component_order_by_video
        try:
            volume_mod.playall_component_order_by_video = lambda master, components: list(components)
            volume_mod._volume_order_sequences = lambda client, match: {
                'regular': ('regular', episodes, None)
            }
            plan = m.infer_numbered_volume_plan(
                FakeClient(), m.Match('tv', 2228, 'Pinky and the Brain', 1995, None, 1.0, {}),
                volume_number=2, groups=groups, analyses=analyses, show_runtime_minutes=21,
            )
        finally:
            volume_mod._volume_order_sequences = old_orders
            volume_mod.playall_component_order_by_video = old_playall
        self.assertEqual(plan['start'], 23)
        self.assertEqual(plan['end'], 44)
        self.assertEqual(plan['episodes'][0].title, 'Brain of the Future')
        self.assertEqual(plan['episodes'][-1].title, 'Brainy Jack')

    def test_numbered_volume_boundary_preserves_29_canonical_segments(self):
        root = Path('/synthetic/Pinky And The Brain')
        groups = []
        analyses = []
        for disc, count in enumerate((6, 5, 5, 5), 1):
            d = root / f'PINKY AND THE BRAIN VOL 2 DISC {disc}'
            components = [d / f'C{i}_t{i:02d}.mkv' for i in range(1, count + 1)]
            master = d / 'B1_t00.mkv'
            group = m.TvRipGroup(d, None, disc, False, tuple([master, *components]))
            groups.append(group)
            analyses.append(m.TrackAnalysis(master, group, count*21*60, count*1000, 1.0, tuple(components)))
            analyses.extend(m.TrackAnalysis(p, group, 21*60, 1000, 1.0) for p in components)

        # 23 provider program units before Volume 2.
        episodes = [
            m.Episode(1, i+1, f'Pre {i+1}', 1000+i, 21, 1996, f'pre-{i:02d}')
            for i in range(23)
        ]
        # Volume 2 is 21 physical programs but 29 canonical TMDb segment rows.
        volume_units = [
            [(2,14,'Brain of the Future')],
            [(2,15,'Brinky')],
            [(2,16,'Hoop Schemes')],
            [(3,1,'Leave It to Beavers'),(3,2,'Cinebrania')],
            [(3,3,'Pinky & The Brain ...and Larry'),(3,4,'Where the Deer and The Mousealopes Play')],
            [(3,5,'Brain Noir')],
            [(3,6,"Brain's Bogie"),(3,7,'Say What, Earth')],
            [(3,8,'My Feldmans, My Friends')],
            [(3,9,'All You Need Is Narf'),(3,10,"Pinky's Plan")],
            [(3,11,'This Old Mouse')],
            [(3,12,'Brain Storm')],
            [(3,13,'A Meticulous Analysis Of History'),(3,14,"Funny, You Don't Look Rhennish")],
            [(3,15,'The Pinky Protocol')],
            [(3,16,"Mice Don't Dance"),(3,17,'Brain Drained')],
            [(3,18,'Brain Acres')],
            [(3,19,'Pinky And The Brainmaker'),(3,20,'Calvin Brain')],
            [(3,21,'Pinky Suavo'),(3,22,'T.H.E.Y.')],
            [(3,23,'The Real Life')],
            [(3,24,"Brain's Way")],
            [(3,25,'A Pinky And The Brain Halloween')],
            [(3,26,'Brainy Jack')],
        ]
        eid = 2000
        for unit_no, unit in enumerate(volume_units):
            date = f'vol2-{unit_no:02d}'
            for season, number, title in unit:
                runtime = 21 if len(unit) == 1 else (10 if unit.index((season, number, title)) == 0 else 11)
                episodes.append(m.Episode(season, number, title, eid, runtime, 1997, date))
                eid += 1
        # 22 provider program units after Volume 2 -> 66 total program units.
        episodes.extend(
            m.Episode(4, i+1, f'Post {i+1}', 3000+i, 21, 1998, f'post-{i:02d}')
            for i in range(22)
        )

        class FakeClient:
            pass
        old_orders = volume_mod._volume_order_sequences
        old_playall = volume_mod.playall_component_order_by_video
        try:
            volume_mod.playall_component_order_by_video = lambda master, components: list(components)
            volume_mod._volume_order_sequences = lambda client, match: {
                'regular': ('regular', episodes, None)
            }
            plan = m.infer_numbered_volume_plan(
                FakeClient(), m.Match('tv', 2228, 'Pinky and the Brain', 1995, None, 1.0, {}),
                volume_number=2, groups=groups, analyses=analyses, show_runtime_minutes=21,
            )
        finally:
            volume_mod._volume_order_sequences = old_orders
            volume_mod.playall_component_order_by_video = old_playall
        self.assertEqual(plan['start'], 23)
        self.assertEqual(plan['end'], 44)
        self.assertEqual(len(plan['programs']), 21)
        self.assertEqual(len(plan['episodes']), 29)
        self.assertEqual((plan['episodes'][0].season, plan['episodes'][0].number, plan['episodes'][0].title),
                         (2, 14, 'Brain of the Future'))
        self.assertEqual((plan['episodes'][-1].season, plan['episodes'][-1].number, plan['episodes'][-1].title),
                         (3, 26, 'Brainy Jack'))

    def test_playall_video_order_overrides_tnn_order(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'PINKY AND THE BRAIN VOL 2 DISC 1'; d.mkdir()
            master_path = d / 'B1_t00.mkv'; master_path.write_bytes(b'master')
            names = ['C1_t01.mkv', 'C2_t02.mkv', 'C3_t03.mkv', 'D1_t04.mkv', 'D3_t05.mkv', 'D4_t06.mkv']
            paths = []
            for name in names:
                path = d / name; path.write_bytes(name.encode()); paths.append(path)
            group = m.TvRipGroup(d, None, 1, False, tuple([master_path, *paths]))
            master = m.TrackAnalysis(master_path, group, 6*21*60, 6000, 1.0, tuple(paths))
            components = [m.TrackAnalysis(path, group, 21*60, 1000, 1.0) for path in paths]

            # Packet samples are unique per component and per seek region.  The
            # play-all master places D4 before D3 even though MakeMKV tNN says
            # t05 (D3) before t06 (D4).
            authored = ['C1_t01.mkv', 'C2_t02.mkv', 'C3_t03.mkv', 'D1_t04.mkv', 'D4_t06.mkv', 'D3_t05.mkv']
            packet_groups = {}
            hit_positions = {}
            counter = 1
            for slot, name in enumerate(authored, 1):
                groups = []
                for region in range(3):
                    hashes = []
                    base_time = slot * 1000.0 + region * 200.0
                    for packet in range(3):
                        digest = f'{counter:032x}'; counter += 1
                        hashes.append(digest)
                        hit_positions[digest] = [base_time + packet * 0.04]
                    groups.append(';'.join(f'100:{digest}' for digest in hashes))
                packet_groups[name] = groups

            old_fp = media_mod.probe_video_packet_fingerprint
            old_scan = media_mod._scan_master_packet_hash_positions
            try:
                media_mod.probe_video_packet_fingerprint = lambda path, duration: ('digest', packet_groups[path.name])
                media_mod._scan_master_packet_hash_positions = lambda master, targets, timeout=600: {
                    digest: hit_positions[digest] for digest in targets if digest in hit_positions
                }
                ordered = m.playall_component_order_by_video(master, components)
            finally:
                media_mod.probe_video_packet_fingerprint = old_fp
                media_mod._scan_master_packet_hash_positions = old_scan
            self.assertEqual([row.path.name for row in ordered], authored)

    def test_playall_video_order_fails_closed_when_regions_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); master_path = d / 'master_t00.mkv'; master_path.write_bytes(b'm')
            a = d / 'a_t01.mkv'; b = d / 'b_t02.mkv'; a.write_bytes(b'a'); b.write_bytes(b'b')
            group = m.TvRipGroup(d, None, 1, False, (master_path, a, b))
            master = m.TrackAnalysis(master_path, group, 42*60, 2000, 1.0, (a,b))
            components = [m.TrackAnalysis(a, group, 21*60, 1000, 1.0), m.TrackAnalysis(b, group, 21*60, 1000, 1.0)]
            samples = {
                'a_t01.mkv': ['1:' + '1'*32 + ';1:' + '2'*32, '1:' + '3'*32 + ';1:' + '4'*32],
                'b_t02.mkv': ['1:' + '5'*32 + ';1:' + '6'*32, '1:' + '7'*32 + ';1:' + '8'*32],
            }
            # Component A's first region appears in two distant locations, so
            # packet matching must refuse rather than silently fall back to tNN.
            hits = {
                '1'*32: [100.0, 1000.0], '2'*32: [100.1, 1000.1],
                '3'*32: [300.0], '4'*32: [300.1],
                '5'*32: [1500.0], '6'*32: [1500.1],
                '7'*32: [1700.0], '8'*32: [1700.1],
            }
            old_fp = media_mod.probe_video_packet_fingerprint
            old_scan = media_mod._scan_master_packet_hash_positions
            try:
                media_mod.probe_video_packet_fingerprint = lambda path, duration: ('digest', samples[path.name])
                media_mod._scan_master_packet_hash_positions = lambda master, targets, timeout=600: {
                    digest: hits[digest] for digest in targets if digest in hits
                }
                with self.assertRaisesRegex(m.MKVPlexError, 'at least two unique video regions'):
                    m.playall_component_order_by_video(master, components)
            finally:
                media_mod.probe_video_packet_fingerprint = old_fp
                media_mod._scan_master_packet_hash_positions = old_scan

    def test_master_component_order_cache_binds_component_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            master = root / 'master.mkv'; master.write_bytes(b'master')
            a = root / 'a.mkv'; a.write_bytes(b'a')
            b = root / 'b.mkv'; b.write_bytes(b'b')
            db = m.DiscoveryDB(root / 'discovery.sqlite3')
            db.put_master_component_order(master, [a, b], [b, a])
            self.assertEqual(db.get_master_component_order(master, [a, b]), [b.resolve(), a.resolve()])
            # Changing a component invalidates the cached authored order.
            b.write_bytes(b'changed')
            self.assertIsNone(db.get_master_component_order(master, [a, b]))


    def test_volume_order_sequences_accepts_nearly_complete_dvd_group(self):
        regular = [
            m.Episode(1, i + 1, f'E{i+1}', 1000 + i, 21, 1997, f'1997-01-{(i%28)+1:02d}')
            for i in range(20)
        ]
        dvd = list(regular[:-1])
        old_regular = tmdb_mod.regular_series_episodes
        old_rows = tmdb_mod._alternate_episode_order_rows
        old_seq = tmdb_mod._episode_group_sequence
        try:
            tmdb_mod.regular_series_episodes = lambda client, match: list(regular)
            tmdb_mod._alternate_episode_order_rows = lambda client, match: [
                {'id': 'dvd-group', 'type': 3, 'name': 'DVD Order'}
            ]
            tmdb_mod._episode_group_sequence = lambda client, match, row: list(dvd)
            match = m.Match('tv', 1, 'Show', 2000, None, 1.0, {})
            self.assertNotIn('dvd', m._collection_order_sequences(object(), match))
            volume_orders = m._volume_order_sequences(object(), match)
        finally:
            tmdb_mod.regular_series_episodes = old_regular
            tmdb_mod._alternate_episode_order_rows = old_rows
            tmdb_mod._episode_group_sequence = old_seq
        self.assertIn('dvd', volume_orders)
        self.assertIn('19/20 canonical episodes', volume_orders['dvd'][0])

    def test_numbered_volume_prefers_dvd_order_for_same_date_programs(self):
        root = Path('/synthetic/Pinky And The Brain')
        groups = []
        analyses = []
        for disc, count in enumerate((6, 5, 5, 5), 1):
            d = root / f'PINKY AND THE BRAIN VOL 2 DISC {disc}'
            components = [d / f'C{i}_t{i:02d}.mkv' for i in range(1, count + 1)]
            master = d / 'B1_t00.mkv'
            group = m.TvRipGroup(d, None, disc, False, tuple([master, *components]))
            groups.append(group)
            analyses.append(m.TrackAnalysis(master, group, count*21*60, count*1000, 1.0, tuple(components)))
            analyses.extend(m.TrackAnalysis(p, group, 21*60, 1000, 1.0) for p in components)

        pre = [m.Episode(1, i+1, f'Pre {i+1}', 100+i, 21, 1996, f'pre-{i:02d}') for i in range(23)]
        bf = m.Episode(2,14,'Brain of the Future',2014,20,1997,'1997-01-01')
        br = m.Episode(2,15,'Brinky',2015,21,1997,'1997-01-02')
        hs = m.Episode(2,16,'Hoop Schemes',2016,21,1997,'1997-01-03')
        lb = m.Episode(3,1,'Leave It to Beavers',3001,8,1997,'1997-09-08')
        ci = m.Episode(3,2,'Cinebrania',3002,12,1997,'1997-09-08')
        la = m.Episode(3,3,'Pinky & The Brain ...and Larry',3003,6,1997,'1997-09-13')
        mo = m.Episode(3,4,'Where the Deer and The Mousealopes Play',3004,15,1997,'1997-09-13')
        bn = m.Episode(3,5,'Brain Noir',3005,20,1997,'1997-09-13')
        tail_units = [[m.Episode(3,6+i,f'Tail {i+1}',3100+i,21,1997,f'1997-10-{(i%28)+1:02d}')] for i in range(15)]
        regular_units = [[bf],[br],[hs],[lb,ci],[la,mo],[bn],*tail_units]
        dvd_units = [[bf],[br],[hs],[lb,ci],[bn],[la,mo],*tail_units]
        post = [m.Episode(4, i+1, f'Post {i+1}', 4000+i, 21, 1998, f'post-{i:02d}') for i in range(22)]
        regular = pre + [ep for unit in regular_units for ep in unit] + post
        dvd = pre + [ep for unit in dvd_units for ep in unit] + post

        old_orders = volume_mod._volume_order_sequences
        old_playall = volume_mod.playall_component_order_by_video
        try:
            volume_mod.playall_component_order_by_video = lambda master, components: list(components)
            volume_mod._volume_order_sequences = lambda client, match: {
                'regular': ('TMDb regular season/episode order', regular, None),
                'dvd': ('TMDb DVD order', dvd, 'dvd-group'),
            }
            plan = m.infer_numbered_volume_plan(
                object(), m.Match('tv',2228,'Pinky and the Brain',1995,None,1.0,{}),
                volume_number=2, groups=groups, analyses=analyses, show_runtime_minutes=21,
            )
        finally:
            volume_mod._volume_order_sequences = old_orders
            volume_mod.playall_component_order_by_video = old_playall
        self.assertEqual(plan['order_key'], 'dvd')
        self.assertIn('DVD order', plan['order_label'])
        self.assertEqual([ep.title for ep in plan['units'][4]], ['Brain Noir'])
        self.assertEqual([ep.title for ep in plan['units'][5]], [
            'Pinky & The Brain ...and Larry', 'Where the Deer and The Mousealopes Play'
        ])


    def test_numbered_volume_uses_chronological_fallback_without_dvd_group(self):
        root = Path('/synthetic/Pinky And The Brain')
        groups = []
        analyses = []
        for disc, count in enumerate((6, 5, 5, 5), 1):
            d = root / f'PINKY AND THE BRAIN VOL 2 DISC {disc}'
            components = [d / f'C{i}_t{i:02d}.mkv' for i in range(1, count + 1)]
            master = d / 'B1_t00.mkv'
            group = m.TvRipGroup(d, None, disc, False, tuple([master, *components]))
            groups.append(group)
            analyses.append(m.TrackAnalysis(master, group, count*21*60, count*1000, 1.0, tuple(components)))
            analyses.extend(m.TrackAnalysis(p, group, 21*60, 1000, 1.0) for p in components)

        pre = [
            m.Episode(1, i+1, f'Pre {i+1}', 100+i, 21, 1996, f'1996-01-{i+1:02d}')
            for i in range(23)
        ]
        bf = m.Episode(2,14,'Brain of the Future',2014,20,1997,'1997-02-08')
        br = m.Episode(2,15,'Brinky',2015,21,1997,'1997-02-22')
        hs = m.Episode(2,16,'Hoop Schemes',2016,21,1997,'1997-05-17')
        lb = m.Episode(3,1,'Leave It to Beavers',3001,8,1997,'1997-09-08')
        ci = m.Episode(3,2,'Cinebrania',3002,12,1997,'1997-09-08')
        # Canonical numeric order puts Larry/Mousealopes before Brain Noir,
        # while provider air dates (and the physical retail disc) put Noir first.
        la = m.Episode(3,3,'Pinky & The Brain ...and Larry',3003,6,1997,'1997-09-13')
        mo = m.Episode(3,4,'Where the Deer and The Mousealopes Play',3004,15,1997,'1997-09-13')
        bn = m.Episode(3,5,'Brain Noir',3005,20,1997,'1997-09-12')
        tail_units = [
            [m.Episode(3,6+i,f'Tail {i+1}',3100+i,21,1997,f'1997-10-{i+1:02d}')]
            for i in range(15)
        ]
        regular_units = [[bf],[br],[hs],[lb,ci],[la,mo],[bn],*tail_units]
        post = [
            m.Episode(4, i+1, f'Post {i+1}', 4000+i, 21, 1998, f'1998-01-{i+1:02d}')
            for i in range(22)
        ]
        regular = pre + [ep for unit in regular_units for ep in unit] + post

        old_regular = tmdb_mod.regular_series_episodes
        old_rows = tmdb_mod._alternate_episode_order_rows
        old_playall = volume_mod.playall_component_order_by_video
        try:
            tmdb_mod.regular_series_episodes = lambda client, match: list(regular)
            tmdb_mod._alternate_episode_order_rows = lambda client, match: []
            volume_mod.playall_component_order_by_video = lambda master, components: list(components)
            plan = m.infer_numbered_volume_plan(
                object(), m.Match('tv',2228,'Pinky and the Brain',1995,None,1.0,{}),
                volume_number=2, groups=groups, analyses=analyses, show_runtime_minutes=21,
            )
        finally:
            tmdb_mod.regular_series_episodes = old_regular
            tmdb_mod._alternate_episode_order_rows = old_rows
            volume_mod.playall_component_order_by_video = old_playall

        self.assertEqual(plan['order_key'], 'chronological')
        self.assertIn('chronological air-date order', plan['order_label'])
        self.assertEqual([ep.title for ep in plan['units'][4]], ['Brain Noir'])
        self.assertEqual([ep.title for ep in plan['units'][5]], [
            'Pinky & The Brain ...and Larry', 'Where the Deer and The Mousealopes Play'
        ])

    def test_volume_chapter_topology_corrects_adjacent_single_multi_identity(self):
        root = Path('/synthetic/Pinky And The Brain/PINKY AND THE BRAIN VOL 2 DISC 1')
        group = m.TvRipGroup(root, None, 1, False, tuple())
        d3 = m.TrackAnalysis(root / 'D3_t05.mkv', group, 1251.392, 1, 1.0)
        d4 = m.TrackAnalysis(root / 'D4_t06.mkv', group, 1264.396, 1, 1.0)
        larry = m.Episode(3, 3, 'Pinky & The Brain ...and Larry', 3003, 6, 1997, '1997-09-13')
        mouse = m.Episode(3, 4, 'Where the Deer and The Mousealopes Play', 3004, 15, 1997, '1997-09-13')
        noir = m.Episode(3, 5, 'Brain Noir', 3005, 20, 1997, '1997-09-12')

        old_probe = volume_mod.probe_chapter_boundaries
        try:
            def fake_probe(path, *, source_duration=None):
                if Path(path).name == 'D3_t05.mkv':
                    # One real chapter spanning the program plus a sub-second
                    # tail; probe_chapter_boundaries would expose no meaningful
                    # internal marker after EOF filtering.
                    return []
                if Path(path).name == 'D4_t06.mkv':
                    # Actual MakeMKV-authored Disc 1 topology: the decisive
                    # Larry/Mousealopes cut is ~6:15.9 into the program.
                    return [65.5655, 375.908867, 1232.764867]
                return []
            volume_mod.probe_chapter_boundaries = fake_probe
            refined, changes = m._refine_volume_units_by_chapter_topology(
                [d3, d4], [[larry, mouse], [noir]], show_runtime_minutes=21,
            )
        finally:
            volume_mod.probe_chapter_boundaries = old_probe

        self.assertEqual([[ep.title for ep in unit] for unit in refined], [
            ['Brain Noir'],
            ['Pinky & The Brain ...and Larry', 'Where the Deer and The Mousealopes Play'],
        ])
        self.assertEqual(len(changes), 1)
        self.assertEqual(Path(changes[0]['left_source']).name, 'D3_t05.mkv')
        self.assertEqual(Path(changes[0]['right_source']).name, 'D4_t06.mkv')
        self.assertLess(changes[0]['chapter_delta'], 20.0)

    def test_numbered_volume_plan_integrates_authored_chapter_topology_swap(self):
        root = Path('/synthetic/Pinky And The Brain/PINKY AND THE BRAIN VOL 2 DISC 1')
        d3_path = root / 'D3_t05.mkv'
        d4_path = root / 'D4_t06.mkv'
        master_path = root / 'B1_t00.mkv'
        group = m.TvRipGroup(root, None, 1, False, (master_path, d3_path, d4_path))
        analyses = [
            m.TrackAnalysis(master_path, group, 2515.788, 3, 1.0, (d3_path, d4_path)),
            m.TrackAnalysis(d3_path, group, 1251.392, 1, 1.0),
            m.TrackAnalysis(d4_path, group, 1264.396, 1, 1.0),
        ]
        pre = [
            m.Episode(1, 1, 'Pre 1', 101, 21, 1996, '1996-01-01'),
            m.Episode(1, 2, 'Pre 2', 102, 21, 1996, '1996-01-02'),
        ]
        larry = m.Episode(3, 3, 'Pinky & The Brain ...and Larry', 303, 6, 1997, '1997-09-13')
        mouse = m.Episode(3, 4, 'Where the Deer and The Mousealopes Play', 304, 15, 1997, '1997-09-13')
        noir = m.Episode(3, 5, 'Brain Noir', 305, 20, 1997, '1997-09-12')
        post = [
            m.Episode(4, 1, 'Post 1', 401, 21, 1998, '1998-01-01'),
            m.Episode(4, 2, 'Post 2', 402, 21, 1998, '1998-01-02'),
        ]
        regular = [*pre, larry, mouse, noir, *post]

        old_orders = volume_mod._volume_order_sequences
        old_playall = volume_mod.playall_component_order_by_video
        old_probe = volume_mod.probe_chapter_boundaries
        try:
            volume_mod._volume_order_sequences = lambda client, match: {
                'regular': ('TMDb regular season/episode order', regular, None)
            }
            volume_mod.playall_component_order_by_video = lambda master, components: [
                next(row for row in components if row.path == d3_path),
                next(row for row in components if row.path == d4_path),
            ]
            def fake_probe(path, *, source_duration=None):
                if Path(path).name == 'D3_t05.mkv':
                    return []
                if Path(path).name == 'D4_t06.mkv':
                    return [65.5655, 375.908867, 1232.764867]
                return []
            volume_mod.probe_chapter_boundaries = fake_probe
            plan = m.infer_numbered_volume_plan(
                object(), m.Match('tv', 2228, 'Pinky and the Brain', 1995, None, 1.0, {}),
                volume_number=2, groups=[group], analyses=analyses, show_runtime_minutes=21,
            )
        finally:
            volume_mod._volume_order_sequences = old_orders
            volume_mod.playall_component_order_by_video = old_playall
            volume_mod.probe_chapter_boundaries = old_probe

        self.assertEqual(plan['start'], 2)
        self.assertEqual([[ep.title for ep in unit] for unit in plan['units']], [
            ['Brain Noir'],
            ['Pinky & The Brain ...and Larry', 'Where the Deer and The Mousealopes Play'],
        ])
        self.assertIn('authored MKV chapter-topology refinement', plan['order_label'])
        self.assertEqual(len(plan['topology_refinements']), 1)

    def test_volume_chapter_topology_does_not_swap_without_authored_cut(self):
        root = Path('/synthetic/Disc 1')
        group = m.TvRipGroup(root, None, 1, False, tuple())
        a = m.TrackAnalysis(root / 'A_t01.mkv', group, 1260.0, 1, 1.0)
        b = m.TrackAnalysis(root / 'B_t02.mkv', group, 1260.0, 1, 1.0)
        part1 = m.Episode(1, 1, 'Part A', 1, 8, 2000, '2000-01-01')
        part2 = m.Episode(1, 2, 'Part B', 2, 13, 2000, '2000-01-01')
        single = m.Episode(1, 3, 'Single', 3, 21, 2000, '2000-01-02')
        old_probe = volume_mod.probe_chapter_boundaries
        try:
            volume_mod.probe_chapter_boundaries = lambda path, *, source_duration=None: []
            refined, changes = m._refine_volume_units_by_chapter_topology(
                [a, b], [[part1, part2], [single]], show_runtime_minutes=21,
            )
        finally:
            volume_mod.probe_chapter_boundaries = old_probe
        self.assertEqual([[ep.number for ep in unit] for unit in refined], [[1, 2], [3]])
        self.assertEqual(changes, [])

    def test_tv_semantic_signature_binds_split_cut(self):
        match = m.Match('tv', 1, 'Show', 2000, 'tt1', 1.0, {})
        group = m.TvRipGroup(Path('/src/Disc 1'), 1, 1, False, (Path('/src/a.mkv'),))
        ep = m.Episode(1, 1, 'Pilot', 10, 23, 2000)
        b1 = m.SplitBoundary(100.0, 101.0, None, None, None, 1.0, 'high:chapter')
        b2 = m.SplitBoundary(100.0, 102.0, None, None, None, 2.0, 'high:chapter')
        p1 = m.AggregateSplitPlan(Path('/src/a.mkv'), group, (ep,), 101.0, (b1,))
        p2 = m.AggregateSplitPlan(Path('/src/a.mkv'), group, (ep,), 102.0, (b2,))
        kwargs = dict(mappings=[], missing_manifest=[], split_destinations=[], transfers=[], extras_transfers=[])
        s1 = m._tv_semantic_signature(all_split_plans=[p1], **kwargs)
        s2 = m._tv_semantic_signature(all_split_plans=[p2], **kwargs)
        self.assertNotEqual(s1, s2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
