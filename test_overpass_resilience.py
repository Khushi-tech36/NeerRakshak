"""Offline tests for Overpass 429 handling. Run:  python -m unittest tests.test_overpass_resilience -v"""
import sys, tempfile, threading, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import settlements as stl

BBOX = (87.0, 26.3, 87.6, 26.8)
TYPES = ("city", "town", "village")


class Resp:
    def __init__(self, status=200, headers=None, body=None):
        self.status_code, self.headers, self._body = status, headers or {}, body

    def json(self):
        return self._body


OK = Resp(200, body={"elements": [
    {"lat": 26.45, "lon": 87.28, "tags": {"name": "Biratnagar", "place": "city"}},
    {"lat": 26.60, "lon": 87.15, "tags": {"name:en": "Inaruwa", "place": "town"}}]})


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = [mock.patch.object(stl, "CACHE_DIR", Path(self.tmp.name)),
                  mock.patch.object(stl.random, "uniform", return_value=0.0)]
        for p in self.p: p.start()
        stl._reset_state()
        self.sleeps = []
        self.sl = mock.patch.object(stl.time, "sleep", side_effect=self.sleeps.append)
        self.sl.start()

    def tearDown(self):
        self.sl.stop()
        for p in self.p: p.stop()
        self.tmp.cleanup()

    def test_429_honours_retry_after_then_succeeds(self):
        with mock.patch.object(stl.requests, "post",
                               side_effect=[Resp(429, {"Retry-After": "7"}), OK]) as post:
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertEqual(r.source, "live"); self.assertEqual(len(r.df), 2)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(self.sleeps, [7.0])            # waited exactly Retry-After, no instant retry

    def test_exponential_backoff_without_header(self):
        with mock.patch.object(stl.requests, "post",
                               side_effect=[Resp(429), Resp(429), OK]) as post:
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertEqual(r.source, "live")
        self.assertEqual(self.sleeps, [5.0, 10.0])      # 5s then 10s
        self.assertEqual(post.call_count, 3)

    def test_gives_up_after_max_attempts_without_raising(self):
        with mock.patch.object(stl.requests, "post", return_value=Resp(429)) as post:
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertIsNone(r.df)
        self.assertEqual(r.warning, stl.UNAVAILABLE_MSG)
        self.assertEqual(post.call_count, stl.MAX_ATTEMPTS)
        self.assertGreater(stl.cooldown_remaining(), 0)

    def test_cooldown_blocks_further_requests(self):
        with mock.patch.object(stl.requests, "post", return_value=Resp(429)) as post:
            stl.get_osm_settlements(BBOX, TYPES)
            n = post.call_count
            r = stl.get_osm_settlements(BBOX, TYPES)    # immediate second click
        self.assertEqual(post.call_count, n)            # no new HTTP request
        self.assertIsNone(r.df)

    def test_huge_retry_after_does_not_block_ui(self):
        with mock.patch.object(stl.requests, "post",
                               return_value=Resp(429, {"Retry-After": "600"})) as post:
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertIsNone(r.df)
        self.assertEqual(post.call_count, 1)            # did not wait 10 minutes
        self.assertEqual(self.sleeps, [])
        self.assertGreater(stl.cooldown_remaining(), 500)

    def test_cache_prevents_repeat_requests(self):
        with mock.patch.object(stl.requests, "post", return_value=OK) as post:
            a = stl.get_osm_settlements(BBOX, TYPES)
            b = stl.get_osm_settlements(BBOX, TYPES)
            c = stl.get_osm_settlements(BBOX, ("village", "town", "city"))  # same set, other order
        self.assertEqual(post.call_count, 1)
        self.assertEqual((a.source, b.source, c.source), ("live", "cache", "cache"))

    def test_disk_cache_survives_memory_reset(self):
        with mock.patch.object(stl.requests, "post", return_value=OK) as post:
            stl.get_osm_settlements(BBOX, TYPES)
            stl._MEM.clear()
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertEqual(post.call_count, 1); self.assertEqual(r.source, "cache")

    def test_stale_cache_used_when_live_fails(self):
        with mock.patch.object(stl.requests, "post", return_value=OK):
            stl.get_osm_settlements(BBOX, TYPES)
        key = stl._cache_key(BBOX, TYPES)
        stl._MEM[key]["ts"] -= 3 * 24 * 3600            # make it 3 days old
        with mock.patch.object(stl.requests, "post", return_value=Resp(429)):
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertEqual(r.source, "stale-cache"); self.assertEqual(len(r.df), 2)
        self.assertIn("saved settlement data", r.note)

    def test_network_error_never_raises(self):
        with mock.patch.object(stl.requests, "post", side_effect=stl.requests.ConnectionError("down")):
            r = stl.get_osm_settlements(BBOX, TYPES)
        self.assertIsNone(r.df); self.assertEqual(r.warning, stl.UNAVAILABLE_MSG)

    def test_parallel_sessions_make_one_request(self):
        calls = []
        def slow_post(*a, **k):
            calls.append(1); stl.time.sleep(0); return OK
        out = []
        with mock.patch.object(stl.requests, "post", side_effect=slow_post):
            ts = [threading.Thread(target=lambda: out.append(stl.get_osm_settlements(BBOX, TYPES)))
                  for _ in range(5)]
            [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(o.df is not None and len(o.df) == 2 for o in out))

    def test_retry_after_http_date(self):
        import email.utils, time as _t
        d = email.utils.formatdate(_t.time() + 12, usegmt=True)
        v = stl._parse_retry_after(d)
        self.assertTrue(8 <= v <= 13)


class ReportNoSettlements(unittest.TestCase):
    def test_pdf_without_settlements(self):
        try:
            import report
        except ImportError:
            self.skipTest("reportlab missing")
        s = dict(region="X", after_dates=["2024-10-01"], before_dates=["2024-09-01"], orbit="ASCENDING",
                 area_km2=5.0, area_pct=1.0, n_total=0, n_inside=0, n_near=0, pop_text="n/a", settings="s")
        self.assertGreater(len(report.build_pdf(s, pd.DataFrame())), 1000)


if __name__ == "__main__":
    unittest.main()
