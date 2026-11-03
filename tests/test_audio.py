#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pillow", "h5py>=3.8"]
# ///
"""Offline checks for audio.py on simulated recordings. Run: uv run tests/test_audio.py

The recordings are simulated sample by sample: each sample carries the phase the aircraft emitted at its retarded time
(no Doppler formula involved), with 1/r loudness and white noise louder than the tones, so the fitter and the search
are checked against physics rather than against their own model."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audio  # noqa: E402

SR = 8000
C = 340.0
LAT0, LON0 = 45.0, 5.0          # simulated observer (flat ground)
T_START = 1_700_000_000.0       # first audio sample, unix seconds


def to_ll(e, n):
    return (LAT0 + np.degrees(n / audio.R_EARTH),
            LON0 + np.degrees(e / (audio.R_EARTH * math.cos(math.radians(LAT0)))))


class Path3:
    """A flight path in metres around the observer: straight (optionally accelerating), or turning at a constant rate."""

    def __init__(self, p0, heading, speed, alt, t_ref, turn=0.0, accel=0.0):
        self.p0, self.h0, self.v, self.alt, self.t_ref, self.turn = np.array(p0, float), heading, speed, alt, t_ref, turn
        self.accel = accel

    def pos(self, t):
        dt = np.asarray(t, float) - self.t_ref
        if self.turn == 0:
            h = math.radians(self.h0)
            s = self.v * dt + 0.5 * self.accel * dt * dt
            e = self.p0[0] + s * math.sin(h)
            n = self.p0[1] + s * math.cos(h)
        else:
            w = math.radians(self.turn)
            h = math.radians(self.h0) + w * dt
            e = self.p0[0] + self.v / w * (math.cos(math.radians(self.h0)) - np.cos(h))
            n = self.p0[1] + self.v / w * (np.sin(h) - math.sin(math.radians(self.h0)))
        return e, n, np.full_like(dt, self.alt)


def passing(lateral, heading, speed, alt, t_cpa, turn=0.0):
    """Path whose closest approach to the observer is `lateral` metres to the right of track at t_cpa."""
    h = math.radians(heading)
    return Path3((lateral * math.cos(h), -lateral * math.sin(h)), heading, speed, alt, t_cpa, turn)


def simulate(paths, dur, noise=1.0, seed=1):
    rng = np.random.default_rng(seed)
    t = T_START + np.arange(int(dur * SR)) / SR
    x = rng.normal(0, noise, len(t))
    for p, f0, k_amp, harmonics in paths:
        tau = t - 3000 / C
        for _ in range(8):
            e, n, u = p.pos(tau)
            r = np.sqrt(e * e + n * n + u * u)
            tau = t - r / C
        amp = k_amp / r
        for k, g in harmonics:
            x += g * amp * np.sin(2 * np.pi * k * f0 * (tau - T_START))
    return x


def write_wav(path, x):
    y = np.clip(x / (np.abs(x).max() * 1.05), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def write_tracks(path, flights, t0, t1, rng):
    with open(path, "w") as fh:
        for ident, p, desc in flights:
            ts = [t0]
            while ts[-1] < t1:
                ts.append(ts[-1] + rng.uniform(2, 8))
            ts = np.array(ts)
            e, n, u = p.pos(ts)
            lat, lon = to_ll(e + rng.normal(0, 5, len(ts)), n + rng.normal(0, 5, len(ts)))
            alt = np.round(u / 7.62) * 7.62
            fh.write(json.dumps({"id": ident, "seg": 0, "reg": ident.upper(), "type": "SIM", "desc": desc, "name": "",
                                 "t": ts.round(2).tolist(), "lat": lat.round(6).tolist(), "lon": lon.round(6).tolist(),
                                 "alt": alt.round(1).tolist(), "gs": [None] * len(ts)}) + "\n")


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        audio.main([str(a) for a in argv])
    return buf.getvalue()


class PhysicsTests(unittest.TestCase):
    def test_doppler_ratio_limits_and_retarded_time(self):
        v, d, th = 80.0, 2000.0, 100.0
        self.assertAlmostEqual(float(audio.doppler_ratio(np.array([th]), v, d, th, C)[0]), 1.0, places=9)
        far = audio.doppler_ratio(np.array([th - 5000, th + 5000]), v, d, th, C)
        self.assertAlmostEqual(far[0], C / (C - v), places=3)
        self.assertAlmostEqual(far[1], C / (C + v), places=3)
        # against a direct simulation of the emission times
        p = passing(-1500.0, 30.0, v, math.sqrt(d * d - 1500 ** 2), 0.0)
        t = np.linspace(-60, 60, 241) + d / C
        tau = t - 3000 / C
        for _ in range(30):
            e, n, u = p.pos(tau)
            tau = t - np.sqrt(e * e + n * n + u * u) / C
        e, n, u = p.pos(tau)
        r = np.sqrt(e * e + n * n + u * u)
        dt = 1e-3
        e2, n2, u2 = p.pos(tau + dt)
        vr = (np.sqrt(e2 * e2 + n2 * n2 + u2 * u2) - r) / dt
        self.assertLess(np.abs(C / (C + vr) - audio.doppler_ratio(t, v, d, d / C, C)).max(), 1e-5)

    def test_grid_ids_round_trip(self):
        g = audio.Grid(200)
        lat = np.array([45.0, -33.9, 64.1, 0.001])
        lon = np.array([5.0, 151.2, -21.9, -179.9])
        la, lo = g.center(g.cell(lat, lon))
        self.assertLess(np.abs(la - lat).max() * 111195, 110)
        self.assertLess((np.abs(lo - lon) * 111195 * np.cos(np.radians(lat))).max(), 110)

    def test_time_parsing(self):
        self.assertEqual(audio.parse_time("2025-09-08T20:37:00Z"), 1757363820.0)
        self.assertEqual(audio.parse_time("1757363820000"), 1757363820.0)
        a, b = audio.window("2025-09-08", "20:00", "24:00")
        self.assertEqual(b - a, 4 * 3600)


class AircraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(os.environ.get("AUDIO_TEST_KEEP") or cls.tmp.name)    # keep the files to look at them
        d.mkdir(parents=True, exist_ok=True)
        cls.dir = d
        rng = np.random.default_rng(7)
        cls.A = passing(2800.0, 60.0, 72.0, 1600.0, T_START + 300)            # event 1: slant ≈ 3230 m
        cls.B = passing(-1100.0, 200.0, 56.0, 800.0, T_START + 760, turn=0.25)  # event 2: turning, slant ≈ 1360 m
        dur = 1000
        x = simulate([(cls.A, 108.0, 320.0, [(1, 1.0), (2, 0.6), (3, 0.3)]),
                      (cls.B, 92.0, 150.0, [(1, 1.0), (2, 0.5)])], dur)
        write_wav(d / "rec.wav", x)
        flights = [("aaa111", cls.A, "L1P"), ("bbb222", cls.B, "L2T"),
                   ("jet333", passing(300.0, 10.0, 120.0, 900.0, T_START + 520), "L2J"),        # jet overhead: ignored
                   ("ccc444", passing(9000.0, 120.0, 65.0, 1200.0, T_START + 500), "L1P")]      # far away between events
        for k in range(14):                                                                      # background traffic
            p0 = rng.uniform(-25000, 25000, 2)
            flights.append((f"r{k:03d}", Path3(p0, rng.uniform(0, 360), rng.uniform(45, 110), rng.uniform(500, 2500),
                                               T_START + rng.uniform(-300, 1300)), "L1P"))
        write_tracks(d / "tracks.jsonl", flights, T_START - 1200, T_START + dur + 1200, rng)
        cls.dur = dur

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fit(self, name, start, end, band):
        out = self.dir / f"fit_{name}.json"
        run("fit", self.dir / "rec.wav", "--start", start, "--end", end, "--band", band, "--name", name, "--out", out,
            "--c", C)
        return json.loads(out.read_text())

    def test_1_events_finds_both_passes(self):
        run("events", self.dir / "rec.wav", "--out", self.dir / "events.json")
        ev = json.loads((self.dir / "events.json").read_text())["doppler_events"]
        mids = sorted(e["t_mid"] for e in ev)
        self.assertTrue(any(abs(m - (300 + 3230 / C)) < 15 for m in mids), mids)
        self.assertTrue(any(abs(m - (760 + 1360 / C)) < 15 for m in mids), mids)

    def test_2_fit_recovers_speed_and_distance(self):
        f1 = self.fit("e1", 200, 420, "80,140")
        d1 = math.hypot(2800, 1600)
        self.assertLess(abs(f1["best"]["v"] - 72) / 72, 0.06, f1["best"])
        self.assertLess(abs(f1["best"]["d"] - d1) / d1, 0.15, f1["best"])
        self.assertLess(abs(f1["best"]["t_cpa"] - (300 + d1 / C)), 1.5)
        self.assertLessEqual(f1["search"]["v"][0], 72)
        self.assertGreaterEqual(f1["search"]["v"][1], 72)
        self.assertTrue(f1["search"]["d"][0] <= d1 <= f1["search"]["d"][1])
        h2 = self.fit("e1h2", 200, 420, "175,275")      # second harmonic: same speed and distance
        self.assertLess(abs(h2["best"]["v"] - 72) / 72, 0.06, h2["best"])
        self.assertLess(abs(h2["best"]["d"] - d1) / d1, 0.15, h2["best"])
        f2 = self.fit("e2", 690, 840, "70,125")
        d2 = math.hypot(1100, 800)
        self.assertLess(abs(f2["best"]["v"] - 56) / 56, 0.12, f2["best"])   # turning: the straight-line fit is approximate
        self.assertTrue(f2["search"]["d"][0] <= d2 <= f2["search"]["d"][1], f2["search"])

    def test_3_locate_without_time(self):
        out = self.dir / "loc"
        run("locate", self.dir / "rec.wav", "--tracks", self.dir / "tracks.jsonl", "--fit", self.dir / "fit_e1.json",
            "--fit", self.dir / "fit_e2.json", "--flat", "--c", C, "--bbox", "44.75,4.65,45.25,5.35", "--out-dir", out)
        res = json.loads((out / "locate.json").read_text())
        b = res["best"]
        err = math.hypot((b["lat"] - LAT0) * 111195, (b["lon"] - LON0) * 111195 * math.cos(math.radians(LAT0)))
        self.assertLess(err, 150, (err, b))
        self.assertEqual([e["flight"]["id"] for e in b["events"]], ["aaa111", "bbb222"])
        self.assertLess(abs(audio.parse_time(b["recording_start"]) - T_START), 3)
        self.assertTrue((out / "map.png").exists() and (out / "curves.png").exists())

    def test_4_locate_one_event_with_known_time(self):
        out = self.dir / "loc1"
        run("locate", self.dir / "rec.wav", "--tracks", self.dir / "tracks.jsonl", "--fit", self.dir / "fit_e2.json",
            "--audio-start", audio.iso(T_START + 4), "--clock-tol", 10, "--flat", "--c", C,
            "--bbox", "44.75,4.65,45.25,5.35", "--out-dir", out)
        b = json.loads((out / "locate.json").read_text())["best"]
        err = math.hypot((b["lat"] - LAT0) * 111195, (b["lon"] - LON0) * 111195 * math.cos(math.radians(LAT0)))
        # one pass: position along the track trades against the clock error (≈ speed × error), so allow ~10 s × 56 m/s
        self.assertEqual(b["events"][0]["flight"]["id"], "bbb222")
        self.assertLess(err, 700, (err, b))


class ManoeuvreTests(unittest.TestCase):
    def test_accelerating_pass_widens_the_search(self):
        p = Path3((-600.0, 0.0), 0.0, 80.0, 1700.0, T_START + 200, accel=0.6)     # speeds up through the pass
        x = simulate([(p, 100.0, 300.0, [(1, 1.0), (2, 0.5)])], 420)
        with tempfile.TemporaryDirectory() as td:
            write_wav(Path(td) / "acc.wav", x)
            out = Path(td) / "fit.json"
            run("fit", Path(td) / "acc.wav", "--start", 80, "--end", 380, "--band", "70,140", "--out", out, "--c", C)
            r = json.loads(out.read_text())
            d = math.hypot(600, 1700)
            self.assertTrue(r["loose"], r["best"])
            self.assertTrue(r["search"]["v"][0] <= 80 <= r["search"]["v"][1], r["search"])
            self.assertTrue(r["search"]["d"][0] <= d <= r["search"]["d"][1], r["search"])


class ThunderAndHumTests(unittest.TestCase):
    def test_thunder_finds_the_area(self):
        rng = np.random.default_rng(3)
        dur, sr = 1500, 2000
        t0 = T_START + 3600
        n = 260
        st = T_START + rng.uniform(0, 3 * 3600, n)
        cx, cy = rng.normal(0, 15000, n), rng.normal(0, 15000, n)       # one storm over the observer…
        cx[:120] += 60000                                              # …and another far away
        x = rng.normal(0, 0.05, dur * sr)
        for k in range(n):
            r = math.sqrt(cx[k] ** 2 + cy[k] ** 2 + 2000 ** 2)
            ta = st[k] + r / 343 - t0
            if r > 25000 or not (0 <= ta < dur - 12):
                continue
            i = int(ta * sr)
            m = 10 * sr
            burst = rng.normal(0, 1, m) * np.exp(-np.arange(m) / (3.0 * sr)) * (2000 / r) ** 1
            x[i:i + m] += burst
        from scipy.signal import butter, sosfilt
        x = sosfilt(butter(4, [25, 200], "bandpass", fs=sr, output="sos"), x) + rng.normal(0, 0.01, len(x))
        with tempfile.TemporaryDirectory() as td:
            d = Path(os.environ.get("AUDIO_TEST_KEEP") or td)
            d.mkdir(parents=True, exist_ok=True)
            y = np.clip(x / np.abs(x).max(), -1, 1)
            with wave.open(str(d / "storm.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((y * 32767).astype("<i2").tobytes())
            jx, jy = cx + rng.normal(0, 1500, n), cy + rng.normal(0, 1500, n)     # located with km-scale errors
            lat, lon = to_ll(jx, jy)
            with open(d / "strikes.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["time", "lat", "lon"])
                for a, b, c in zip(st, lat, lon):
                    wr.writerow([audio.iso(a)[:-1] + f".{int((a % 1) * 1000):03d}Z", b, c])
            run("thunder", d / "storm.wav", "--strikes", d / "strikes.csv", "--bbox", "44.5,4.3,45.5,6.0",
                "--grid", 3000, "--rate", sr, "--from", audio.iso(T_START), "--to", audio.iso(T_START + 3 * 3600),
                "--out-dir", d / "th")
            res = json.loads((d / "th" / "thunder.json").read_text())
            b = res["best"]
            err = math.hypot((b["lat"] - LAT0) * 111195, (b["lon"] - LON0) * 111195 * math.cos(math.radians(LAT0)))
            self.assertLess(err, 8000, (err, b))
            self.assertLess(abs(audio.parse_time(b["recording_start"]) - t0), 12)

    def test_hum_grid_and_enf(self):
        sr, dur = 2000, 120
        t = np.arange(dur * sr) / sr
        f = 60 + 0.02 * np.sin(2 * np.pi * t / 40)
        ph = 2 * np.pi * np.cumsum(f) / sr
        x = 0.02 * np.sin(ph) + 0.03 * np.sin(2 * ph) + np.random.default_rng(0).normal(0, 0.2, len(t))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hum.wav"
            write_wav(p, x) if sr == SR else None
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((np.clip(x / np.abs(x).max(), -1, 1) * 32767).astype("<i2").tobytes())
            run("hum", p, "--rate", sr, "--out", Path(td) / "hum.json", "--csv", Path(td) / "enf.csv")
            res = json.loads((Path(td) / "hum.json").read_text())
            self.assertEqual(res["grid_hz"], 60)
            self.assertLess(abs(res["enf_mean"] - 60), 0.01)
            self.assertGreater(res["enf_std"], 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
