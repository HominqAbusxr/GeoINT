#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Offline checks for the skyline search (terrain.py find / check) and the terrain tiles. Run: uv run tests/test_skyline.py
No network: terrain comes from functions served as tiles, and Mapterhorn tiles are written into a temporary cache."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _skyfind as S  # noqa: E402
import terrain  # noqa: E402

C = (45.0, 7.0)
_rng = np.random.default_rng(11)
_HILLS = [(C[0] + _rng.uniform(-0.08, 0.08), C[1] + _rng.uniform(-0.11, 0.11), _rng.uniform(300, 1400), _rng.uniform(400, 1800))
          for _ in range(40)]


def hills(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    h = 500 + 0 * lat
    for la, lo, hh, w in _HILLS:
        dy, dx = (lat - la) * 110540, (lon - lo) * 111320 * math.cos(math.radians(C[0]))
        h = h + hh * np.exp(-(dx * dx + dy * dy) / (2 * w * w))
    return h


_TILES: dict = {}


def fake_fetch(z, x, y, cache, proxy):
    """The function above as a 256-px DEM tile."""
    if (z, x, y) not in _TILES:
        n = 256 * 2 ** z
        lon = (x * 256 + np.arange(256)) / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y * 256 + np.arange(256)) / n))))
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        _TILES[(z, x, y)] = hills(LA, LO).astype(np.float32)
    return _TILES[(z, x, y)]


def draw_trace(ground, cam, heading, hfov, tilt, roll, W=1200, H=800, noise=0.6, seed=0):
    """The skyline a camera there would photograph, traced every 12 px (ground nearer than 300 m left out)."""
    tn = S.turn(ground, *cam, 1.6, S.Layout("t", 0.02, 0.004), (10000.0,))
    az = heading + np.linspace(-hfov * 0.7, hfov * 0.7, 6000)
    sky, far = tn.at(az, 0)
    u, v = S.project(az, sky, heading, tilt, roll)
    f = S.hfov_to_f(hfov, W)
    x, y = W / 2 + u * f, H / 2 - v * f
    ok = np.isfinite(x) & (far > 300)
    o = np.argsort(x[ok])
    xs, ys = x[ok][o], y[ok][o]
    rng = np.random.default_rng(seed)
    pts = [[float(c), float(np.interp(c, xs, ys) + rng.normal(0, noise))] for c in range(12, W - 12, 12)
           if np.min(np.abs(xs - c)) < 3 and 0 < np.interp(c, xs, ys) < H]
    return {"image_size": [W, H], "ridge": pts, "f0": f, "f0_source": "exif"}


class CameraTests(unittest.TestCase):
    def test_project_inverts_directions(self):
        rng = np.random.default_rng(1)
        u, v = rng.uniform(-0.6, 0.6, 50), rng.uniform(-0.4, 0.4, 50)
        for tilt, roll in ((0, 0), (8, 0), (-12, 2.5), (20, -4)):
            az, el = S.directions(u, v, tilt, roll)
            u2, v2 = S.project(az + 123.0, el, 123.0, tilt, roll)
            self.assertLess(float(np.abs(u2 - u).max()), 1e-9)
            self.assertLess(float(np.abs(v2 - v).max()), 1e-9)

    def test_roll_turns_the_picture_clockwise(self):
        # a level horizon photographed with the picture turned clockwise runs lower on the right (terrain.py view --roll)
        u = np.array([-0.5, 0.5])
        uu, vv = S.project(np.degrees(np.arctan(u)), np.zeros(2), 0.0, 0.0, 3.0)
        self.assertGreater(vv[0], vv[1])


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.g = S.FuncGround(hills)
        self.cam = (C[0] - 0.01, C[1] + 0.015)
        self.tr = json.loads(json.dumps(draw_trace(self.g, self.cam, 30.0, 60.0, 3.0, 1.0)))
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "trace.json"
        self.path.write_text(json.dumps(self.tr))

    def tearDown(self):
        self.tmp.cleanup()

    def test_true_camera_explains_the_trace_and_decoys_do_not(self):
        trace = S.load_trace(self.path)
        ask = S.Ask(lenses_rough=[60.0], lenses_fine=[57.0, 60.0, 63.0], lens_bounds=(56.0, 64.0), cuts=(5000.0, 10000.0))
        sc = S.Scorer(self.g, [trace], ask, S.Layout("r", 0.3, 0.025), S.Layout("f", 0.1, 0.008), 0.3, 0.15)
        fits, tn = sc.full(*self.cam, most=2)
        self.assertGreater(fits[0].explained, 0.9)
        self.assertLess(abs((fits[0].heading - 30.0 + 180) % 360 - 180), 1.0)
        self.assertLess(abs(fits[0].tilt - 3.0), 0.8)
        self.assertLess(sc.decoy(*self.cam, tn=tn), fits[0].explained - 0.2)

    def test_a_straight_trace_has_no_shape(self):
        flat = {"image_size": [1200, 800], "ridge": [[x, 300 + 0.05 * x] for x in range(10, 1190, 10)], "f0": 1000}
        p = Path(self.tmp.name) / "flat.json"
        p.write_text(json.dumps(flat))
        t = S.load_trace(p)
        self.assertLess(t.shape_px(), S.FLAT_PX)
        sub = S.Sub.of(t)
        # however well a line lies on some hillside, it explains nothing of a shape it does not have
        self.assertLess(float(sub.explained(np.zeros(sub.u.size))), 0.05)
        self.assertEqual(t.turns(), 0)

    def test_weakest_part_caps_the_score(self):
        t = S.load_trace(self.path)
        sub = S.Sub.of(t)
        loss = np.zeros(sub.u.size)
        self.assertGreater(float(sub.explained(loss)), 0.99)
        # two of the four parts off the terrain: whatever the rest, the score falls to their level
        bad = sub.parts[0] | sub.parts[1]
        self.assertLess(float(sub.explained(np.where(bad, S.LOST_LOSS, 0.0))), 0.5)


class FindTests(unittest.TestCase):
    def test_find_puts_the_camera_first(self):
        cam = (C[0] + 0.004, C[1] - 0.006)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            g = S.Ground((cam[0] - 0.02, cam[1] - 0.03, cam[0] + 0.02, cam[1] + 0.03), 10000.0, fake_fetch, tmp, None, 13,
                         log=lambda *a: None)
            (tmp / "trace.json").write_text(json.dumps(draw_trace(g, cam, 200.0, 55.0, 2.0, -0.5, seed=3)))
            args = Namespace(ridge=tmp / "trace.json", f35=26.0, key=None, hfov=None, mode="ground", tilt=None, roll_max=5.0,
                             facing=None, heading=None, reach="5", clear=False, dist=None, floor=250.0, eye=1.6, floors=None,
                             selfie=False, zoom=13, photo=None, sheet_width=320, out=tmp / "find.json",
                             bbox=f"{cam[0] - 0.012},{cam[1] - 0.016},{cam[0] + 0.012},{cam[1] + 0.016}", at=None, radius=None,
                             spacing=150.0, stands="summit,slope,valley,plain", quick=True, top=5, sheet=None, sheet_n=3,
                             heat=str(tmp / "heat.png"), truth=f"{cam[0]},{cam[1]}", cache=tmp, proxy=None, workers=1)
            S.cmd_find(args, fake_fetch, ground=g)
            out = json.loads((tmp / "find.json").read_text())
            best = out["places"][0]
            self.assertLess(S.dist_m(cam, (best["lat"], best["lon"])), 300)
            self.assertGreater(best["explained"], 0.8)
            self.assertLess(best["decoy"], best["explained"])
            self.assertIn(out["trust"], ("solid", "check", "weak"))
            self.assertTrue((tmp / "heat.png").exists())


class TileTests(unittest.TestCase):
    def test_mapterhorn_tile_is_cut_into_four(self):
        """One 512-px Mapterhorn tile at zoom z is the four 256-px tiles at z + 1; a missing planet tile is the sea; a
        missing survey tile deeper than the planet falls back to the planet's tile enlarged."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            yy, xx = np.mgrid[0:512, 0:512]
            h = (100 + xx + 1000 * (yy // 256)).astype(np.float64)       # known heights
            v = h + 32768
            rgb = np.stack([np.floor(v / 256), np.floor(v % 256), np.round((v % 1) * 256)], -1).astype(np.uint8)
            Image.fromarray(rgb).save(cache / "mapterhorn_12_100_200.webp", lossless=True)
            q = terrain._fetch(13, 201, 401, cache, None)                 # right-bottom quarter
            self.assertEqual(q.shape, (256, 256))
            self.assertAlmostEqual(float(q[0, 0]), 100 + 256 + 1000, places=3)
            self.assertAlmostEqual(float(q[10, 20]), 100 + 256 + 20 + 1000, places=3)
            (cache / "mapterhorn_12_101_200.none").touch()                 # planet tile missing: the sea
            self.assertEqual(float(np.abs(terrain._fetch(13, 202, 400, cache, None)).max()), 0.0)
            (cache / "mapterhorn_13_201_401.none").touch()                 # no survey: the planet's tile, enlarged
            d = terrain._fetch(14, 402, 802, cache, None)
            self.assertEqual(d.shape, (256, 256))
            self.assertAlmostEqual(float(d[0, 0]), 100 + 256 + 1000, delta=1.0)

    def test_a_tile_of_one_height_is_a_tile(self):
        """The sea or a flat lake comes as a valid WebP of some 50 bytes: it must be read, not taken for a failure."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            v = -432 + 32768                                               # the Dead Sea's surface
            rgb = np.zeros((512, 512, 3), np.uint8)
            rgb[...] = (v // 256, v % 256, 0)
            p = cache / "mapterhorn_12_7_9.webp"
            Image.fromarray(rgb).save(p, lossless=True)
            self.assertLess(p.stat().st_size, 200)
            self.assertEqual(float(terrain._fetch(13, 14, 18, cache, None).max()), -432.0)


class RidgeTests(unittest.TestCase):
    def test_snow_under_a_blue_sky(self):
        """Snow brighter than the sky stops a brightness reading below it; read by blueness, the skyline is found."""
        W, H = 400, 300
        a = np.zeros((H, W, 3), np.uint8)
        a[:] = (70, 120, 200)                                            # blue sky
        top = (120 + 40 * np.sin(np.arange(W) / 30)).astype(int)
        for x in range(W):
            a[top[x]:, x] = (245, 245, 250)                              # snowfield, brighter than the sky
            a[top[x] + 60:, x] = (90, 80, 70)                            # rock below
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "snow.png"
            Image.fromarray(a).save(p)
            out = {}
            for by in ("luma", "blue"):
                r = Path(tmp) / f"{by}.json"
                subprocess.run([sys.executable, str(ROOT / "scripts" / "terrain.py"), "ridge", str(p), "--step", "10", "--by", by,
                                "--out", str(r)], check=True, capture_output=True)
                pts = np.array(json.loads(r.read_text())["ridge"], float)
                out[by] = float(np.median(np.abs(pts[:, 1] - top[pts[:, 0].astype(int)]))) if len(pts) else 999.0
            self.assertLess(out["blue"], 3.0)
            self.assertGreater(out["luma"], 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
