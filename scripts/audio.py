#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pillow", "h5py>=3.8"]
# ///
"""Sound as location evidence: the faint aircraft, thunder and mains hum in a recording, matched against open flight
and lightning records. A propeller aircraft's Doppler curve gives its speed and closest-approach distance; two passes a
known time apart, plus the silence between them, single out one spot among every flight of the day (often within
100–200 m). Thunder gives the city (a few km). Mains hum gives the grid (50 or 60 Hz) and a time fingerprint.
See references/audio.md for the method, its limits and the data sources.

  extract    a video's audio track → mono WAV
  look       whitened spectrogram sheets (rows of N minutes) that make inaudible tones visible; zoom a band or a time range
  events     candidate Doppler sweeps (aircraft), steady tones, mains hum and low-frequency bursts (thunder) → events.json
  fit        trace one Doppler curve, fit speed, closest-approach slant distance, source frequency and heard CPA time,
             with the ranges to search (fit_<name>.json + overlay); --trace takes hand-picked points for very faint curves
  hum        50 or 60 Hz grid, and the ENF trace (mains frequency over time); --reference finds the time in a grid record
  adsb       historical aircraft tracks for a day and area from the open adsb.lol archive (or a CSV) → tracks.jsonl
  locate     aircraft search: flights whose closest approach fits every fitted curve (speed, distance, time offsets), no
             other aircraft heard between them, then each candidate's exact modelled Doppler curve scored on the spectrogram
  lightning  lightning flashes for a time window and area from NOAA GOES GLM (the Americas, no key) → strikes.csv
  thunder    thunder search: the loudness envelope every grid point would have heard from the strikes, scored against the
             recording's 20–250 Hz envelope at every start time (cosine similarity)

Examples:
  audio.py extract clip.mp4 --out audio.wav
  audio.py look audio.wav --band 40,600 --out spec.png                       # 5-minute rows; look for falling curves
  audio.py look audio.wav --band 80,260 --start 340 --end 520 --out zoom.png # zoom on one pass
  audio.py events audio.wav --out events.json --png events.png
  audio.py fit audio.wav --start 350 --end 500 --band 90,240 --name e1         # → fit_e1.json, fit_e1.png
  audio.py fit audio.wav --trace e2_points.csv --name e2                      # hand-traced t,f points (seconds, Hz)
  audio.py adsb --date 2025-09-08 --bbox 40.4,-74.3,41.3,-72.6 --out tracks.jsonl       # streams one day (~3–4 GB)
  audio.py locate audio.wav --tracks tracks.jsonl --fit fit_e1.json --fit fit_e2.json \\
                  --date 2025-09-08 --bbox 40.4,-74.3,41.3,-72.6 --out-dir locate
  audio.py locate audio.wav --tracks tracks.jsonl --fit fit_e1.json --audio-start 2025-09-08T20:37:00Z --clock-tol 30
  audio.py lightning --date 2025-07-02 --from 17:00 --to 20:00 --bbox 27,-83,30,-80 --out strikes.csv
  audio.py thunder audio.wav --strikes strikes.csv --bbox 27,-83,30,-80 --date 2025-07-02 --from 17:00 --to 19:30
  audio.py hum audio.wav --csv enf.csv [--reference grid_frequency.csv]

Times are UTC unless they carry an offset. Distances in metres, speeds in m/s, frequencies in Hz.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from _net import PROXY_HELP, curl_args, fetch_bytes  # noqa: E402

R_EARTH = 6371008.8
FT = 0.3048
KT = 0.514444
CACHE = Path(os.environ.get("GEOINT_CACHE", Path.home() / ".cache" / "geoint")).expanduser()
TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types.js"


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- time helpers

def parse_time(s) -> float:
    """ISO 8601 (UTC unless an offset is given), or unix seconds / ms / µs / ns → unix seconds."""
    if isinstance(s, (int, float, np.floating, np.integer)):
        v = float(s)
    else:
        s = str(s).strip()
        try:
            v = float(s)
        except ValueError:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    for lim, div in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3)):
        if abs(v) > lim:
            return v / div
    return v


def iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def window(day: str | None, frm: str | None, to: str | None) -> tuple[float | None, float | None]:
    """--date 2025-09-08 --from 18:00 --to 23:30 (or full ISO times) → unix bounds; a date alone = that UTC day."""
    def one(v, default_hm):
        if v is None and day is None:
            return None
        v = v or default_hm
        if day and re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", v):
            if v.startswith("24"):
                return parse_time(day + "T00:00:00Z") + 86400
            return parse_time(f"{day}T{v}Z" if v.count(":") == 2 else f"{day}T{v}:00Z")
        return parse_time(v)
    return one(frm, "00:00"), one(to, "24:00")


def clock(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def bbox_arg(s: str | None):
    if not s:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 4 or v[0] >= v[2] or v[1] >= v[3]:
        raise SystemExit("--bbox is S,W,N,E (south, west, north, east)")
    return v


def pair(s: str) -> tuple[float, float]:
    a, b = (float(x) for x in s.split(","))
    return a, b


# ---------------------------------------------------------------- audio and spectrograms

def load_audio(path, rate: int = 8000) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        raise SystemExit(f"ffmpeg could not read {path}: {r.stderr.decode(errors='replace')[-300:]}")
    x = np.frombuffer(r.stdout, dtype=np.float32)
    if not len(x):
        raise SystemExit(f"No audio in {path}")
    return x


def stft_power(x: np.ndarray, sr: int, nfft: int, hop: int, fmin: float, fmax: float):
    """Power spectrogram restricted to [fmin, fmax] (frames × bins, float32), computed in chunks."""
    win = np.hanning(nfft).astype(np.float32)
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    n = 1 + (len(x) - nfft) // hop
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    b0 = int(np.searchsorted(freqs, fmin))
    b1 = int(np.searchsorted(freqs, fmax, side="right"))
    out = np.empty((n, b1 - b0), np.float32)
    view = np.lib.stride_tricks.sliding_window_view(x, nfft)
    step = max(1, 2 ** 24 // nfft)
    for i in range(0, n, step):
        idx = np.arange(i, min(n, i + step))
        sp = np.fft.rfft(view[idx * hop] * win, axis=1)[:, b0:b1]
        out[idx] = sp.real ** 2 + sp.imag ** 2
    times = (np.arange(n) * hop + nfft / 2) / sr
    return times, freqs[b0:b1], out


def whiten(P: np.ndarray) -> np.ndarray:
    """Robust z-score per frequency bin (removes steady noise and hum), minus each frame's median (broadband bursts)."""
    L = np.log10(P + 1e-20)
    med = np.median(L, axis=0)
    mad = np.median(np.abs(L - med), axis=0) * 1.4826 + 1e-6
    Z = (L - med) / mad
    Z -= np.median(Z, axis=1, keepdims=True)
    return Z.astype(np.float32)


class Spec:
    """Whitened spectrogram of a band: Z[frame, bin] with times t (s) and freqs f (Hz)."""

    def __init__(self, x: np.ndarray, sr: int, fmin: float, fmax: float, res: float = 1.0, hop_s: float = 0.25):
        self.sr = sr
        self.nfft = int(round(sr / res))
        self.hop = max(1, int(round(sr * hop_s)))
        self.t, self.f, P = stft_power(x, sr, self.nfft, self.hop, fmin, fmax)
        self.dt = self.hop / sr
        self.df = sr / self.nfft
        self.Z = whiten(P)

    def frames(self, t0: float, t1: float) -> np.ndarray:
        return np.nonzero((self.t >= t0) & (self.t <= t1))[0]

    def bin(self, f):
        return np.round((np.asarray(f) - self.f[0]) / self.df).astype(int)


# ---------------------------------------------------------------- drawing

_ANCH = [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)]
LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_ANCH)), np.array(_ANCH, float)[:, i])
                for i in range(3)], 1).astype(np.uint8)


def font(size: int = 13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _resize_max(A: np.ndarray, h: int, w: int) -> np.ndarray:
    """Shrink by max-pooling (thin faint lines survive), enlarge by repetition."""
    def ax(A, n, axis):
        m = A.shape[axis]
        if m == n or m == 0:
            return A
        if m > n:
            return np.maximum.reduceat(A, np.linspace(0, m, n + 1).astype(int)[:-1], axis=axis)
        return np.take(A, np.minimum((np.arange(n) * m / n).astype(int), m - 1), axis=axis)
    return ax(ax(A, h, 0), w, 1)


def _nice_step(span: float, target: int) -> float:
    raw = span / max(1, target)
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if s >= raw:
            return s
    return 5000


def render(spec: Spec, t0: float, t1: float, out: Path, row_s: float = 300, width: int = 1800, row_h: int = 360,
           vmax: float = 6.0, title: str = "", boxes=(), curves=(), rows_per_page: int = 6) -> list[Path]:
    """Spectrogram sheet(s): rows of row_s seconds, low frequencies at the bottom. boxes: (ta, tb, fa, fb, label);
    curves: (times, freqs, rgb) drawn as dots."""
    fnt = font(13)
    L, T, gap = 58, 26, 22
    fr = spec.frames(t0, t1)
    if not len(fr):
        raise SystemExit("Nothing to draw in that time range")
    t0, t1 = spec.t[fr[0]], spec.t[fr[-1]]
    nrows = max(1, math.ceil((t1 - t0) / row_s - 1e-3))
    fmin, fmax = spec.f[0], spec.f[-1]
    pages, paths = math.ceil(nrows / rows_per_page), []
    for p in range(pages):
        rows = range(p * rows_per_page, min(nrows, (p + 1) * rows_per_page))
        img = Image.new("RGB", (L + width + 12, T + len(rows) * (row_h + gap) + 6), (16, 16, 20))
        d = ImageDraw.Draw(img)
        d.text((L, 5), title + (f"  (page {p + 1}/{pages})" if pages > 1 else ""), fill=(230, 230, 230), font=fnt)
        for k, r in enumerate(rows):
            ra, rb = t0 + r * row_s, min(t1, t0 + (r + 1) * row_s)
            y0 = T + k * (row_h + gap)
            sel = spec.frames(ra, rb)
            wpx = max(1, int(round((rb - ra) / row_s * width)))
            A = _resize_max(spec.Z[sel].T[::-1], row_h, wpx)
            rgb = LUT[(np.clip(A / vmax, 0, 1) * 255).astype(np.uint8)]
            img.paste(Image.fromarray(rgb), (L, y0))
            fstep = _nice_step(fmax - fmin, 6)
            for fv in np.arange(math.ceil(fmin / fstep) * fstep, fmax + 1e-9, fstep):
                y = y0 + (1 - (fv - fmin) / (fmax - fmin)) * row_h
                d.line([(L - 5, y), (L - 1, y)], fill=(200, 200, 200))
                d.text((4, y - 7), f"{fv:g}", fill=(200, 200, 200), font=fnt)
            tstep = _nice_step(rb - ra, 10) if row_s <= 600 else 60
            for tv in np.arange(math.ceil(ra / tstep) * tstep, rb + 1e-9, tstep):
                x = L + (tv - ra) / row_s * width
                d.line([(x, y0 - 4), (x, y0)], fill=(200, 200, 200))
                d.text((x + 2, y0 - 17), clock(tv), fill=(200, 200, 200), font=fnt)

            def xy(tv, fv):
                return L + (tv - ra) / row_s * width, y0 + (1 - (fv - fmin) / (fmax - fmin)) * row_h
            for ta, tb, fa, fb, label in boxes:
                if tb < ra or ta > rb:
                    continue
                xa, ya = xy(max(ta, ra), min(fb, fmax))
                xb, yb = xy(min(tb, rb), max(fa, fmin))
                d.rectangle([xa, ya, xb, yb], outline=(80, 255, 120), width=2)
                d.text((xa + 3, ya + 2), label, fill=(80, 255, 120), font=fnt)
            for ts, fs, col in curves:
                ts, fs = np.asarray(ts), np.asarray(fs)
                m = (ts >= ra) & (ts <= rb) & (fs >= fmin) & (fs <= fmax)
                for tv, fv in zip(ts[m], fs[m]):
                    x, y = xy(tv, fv)
                    d.ellipse([x - 1.2, y - 1.2, x + 1.2, y + 1.2], fill=col)
        path = out if pages == 1 else out.with_name(f"{out.stem}_p{p + 1}{out.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------- Doppler physics

def doppler_ratio(t, v, d, th, c):
    """Received/emitted frequency for a source on a straight line at constant speed v (m/s), closest-approach slant
    distance d (m), heard at the observer's CPA time th (s, when the received frequency equals the emitted one).
    Uses the emission (retarded) time, solved exactly: the curve is not the radial-velocity curve shifted by a delay."""
    T = t - (th - d / c)                          # reception time relative to the emission time of the CPA
    a = c * c - v * v
    s = (c * c * T - np.sqrt(d * d * a + (v * c * T) ** 2)) / a   # emission time relative to CPA
    vr = v * v * s / np.sqrt(d * d + (v * s) ** 2)              # range rate at emission (+ = receding)
    return c / (c + vr)


def sound_speed(temp_c: float) -> float:
    return 331.3 * math.sqrt(1 + temp_c / 273.15)


def trace_ridge(Zc: np.ndarray, up: int = 1, down: int = 4, pen: float = 0.15) -> np.ndarray:
    """Best path through Zc (frames × bins) that mostly falls in frequency (Viterbi): up/down = max bins per frame."""
    n, m = Zc.shape
    score = Zc[0].astype(np.float64).copy()
    back = np.zeros((n, m), np.int8)
    for i in range(1, n):
        best = np.full(m, -np.inf)
        arg = np.zeros(m, np.int8)
        for k in range(-up, down + 1):
            prev = np.full(m, -np.inf)
            if k >= 0:
                prev[:m - k] = score[k:] - pen * k
            else:
                prev[-k:] = score[:m + k] + pen * k
            better = prev > best
            best[better] = prev[better]
            arg[better] = k
        score = best + Zc[i]
        back[i] = arg
    b = int(np.argmax(score))
    path = [b]
    for i in range(n - 1, 0, -1):
        b += int(back[i, b])
        path.append(b)
    return np.array(path[::-1])


def fit_curve(t: np.ndarray, f: np.ndarray, w: np.ndarray, c: float):
    """Robust least-squares fit of (f0, v, d, th) to traced points; returns params, relative residuals (‰), inlier mask."""
    from scipy.optimize import least_squares
    n = len(t)
    q = max(3, n // 7)
    f_hi, f_lo = float(np.median(f[:q])), float(np.median(f[-q:]))
    if f_hi <= f_lo * 1.003:
        raise SystemExit("The trace does not fall in frequency: not a Doppler pass, or only one side of it. "
                         "Widen --start/--end or check the band on a `look` sheet.")
    r = f_hi / f_lo
    v0 = c * (r - 1) / (r + 1)
    f00 = 2 / (1 / f_hi + 1 / f_lo)
    th0 = float(t[np.argmin(np.abs(f - f00))])
    sw = np.sqrt(w / w.mean())
    lo = [f.min() * 0.5, 3.0, 20.0, t[0] - 120]
    hi = [f.max() * 2.0, 0.9 * c, 60000.0, t[-1] + 120]

    def resid(p, m):
        return sw[m] * (p[0] * doppler_ratio(t[m], p[1], p[2], p[3], c) / f[m] - 1) * 1000

    def run(m, loss):
        best = None
        for d0 in (300.0, 1000.0, 2500.0, 6000.0, 15000.0):
            for vs in (1.0, 1.35, 1.8):
                p0 = np.clip([f00, v0 * vs, d0, th0], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
                res = least_squares(resid, p0, args=(m,), bounds=(lo, hi), loss=loss, f_scale=2.0,
                                    x_scale=[f00, 30.0, 1000.0, 5.0], max_nfev=400)
                if best is None or res.cost < best.cost:
                    best = res
        return best
    m = np.ones(n, bool)
    p = run(m, "soft_l1").x
    rel = (p[0] * doppler_ratio(t, p[1], p[2], p[3], c) / f - 1) * 1000
    sig = 1.4826 * np.median(np.abs(rel - np.median(rel))) + 1e-9
    m = np.abs(rel) <= max(4 * sig, 2.0)
    if m.sum() >= 12:
        p = run(m, "linear").x
        rel = (p[0] * doppler_ratio(t, p[1], p[2], p[3], c) / f - 1) * 1000
    return p, rel, m


def profile(t, f, w, p, c, eff, level=9.21):
    """Speed × distance profile (f0 solved analytically, th scanned); returns the region within Δχ² ≤ level (99%).
    A coarse wide pass, then a fine pass around the region; the best fit itself is always inside."""
    _, vb, db, thb = p
    W = w / w.mean()
    ths = thb + np.arange(-8, 8.001, 0.1)

    def cost(v, d):
        q = doppler_ratio(t[None, :], v, d, ths[:, None], c) / f[None, :]
        f0 = (W * q).sum(1) / (W * q * q).sum(1)
        rr = (f0[:, None] * q - 1) * 1000
        cs = (W * rr * rr).sum(1)
        k = int(np.argmin(cs))
        return cs[k], ths[k]

    def scan(vs, ds):
        chi = np.empty((len(vs), len(ds)))
        th = np.empty_like(chi)
        for i, v in enumerate(vs):
            for j, d in enumerate(ds):
                chi[i, j], th[i, j] = cost(v, d)
        return chi, th
    cbest = cost(vb, db)[0]
    s2 = None

    def region(vs, ds, chi, th):
        cmin = min(chi.min(), cbest)
        ok = eff * (chi - cmin) / s2 <= level
        if not ok.any():
            ok = chi <= chi.min()
        vi, di = np.nonzero(ok)
        return ok, vi, di
    vs = np.linspace(max(3.0, vb * 0.45), min(0.9 * c, vb * 1.8), 31)
    ds = np.geomspace(max(20.0, db / 5), min(80000.0, db * 5), 31)
    chi, th = scan(vs, ds)
    s2 = max(min(chi.min(), cbest) / max(1, len(t) - 4), 1e-12)
    _, vi, di = region(vs, ds, chi, th)
    edge = bool(vi.min() == 0 or vi.max() == len(vs) - 1 or di.min() == 0 or di.max() == len(ds) - 1)
    v_lo, v_hi = vs[max(0, vi.min() - 1)], vs[min(len(vs) - 1, vi.max() + 1)]
    d_lo, d_hi = ds[max(0, di.min() - 1)], ds[min(len(ds) - 1, di.max() + 1)]
    vs2 = np.linspace(min(v_lo, vb), max(v_hi, vb), 25)
    ds2 = np.geomspace(min(d_lo, db), max(d_hi, db), 25)
    chi2, th2 = scan(vs2, ds2)
    ok2, vi2, di2 = region(vs2, ds2, chi2, th2)
    vr = [min(float(vs2[vi2.min()]), vb), max(float(vs2[vi2.max()]), vb)]
    dr = [min(float(ds2[di2.min()]), db), max(float(ds2[di2.max()]), db)]
    tr = [min(float(th2[ok2].min()), thb), max(float(th2[ok2].max()), thb)]
    return dict(v=vr, d=dr, t_cpa=tr, at_grid_edge=edge)


# ---------------------------------------------------------------- subcommands: extract / look / events / fit

def cmd_extract(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", a.video, "-vn", "-ac", "1", "-ar", str(a.rate),
                        str(out)], capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode(errors="replace")[-400:])
    dur = len(load_audio(out, 1000)) / 1000
    print(json.dumps({"audio": str(out), "rate": a.rate, "duration_s": round(dur, 2)}))


def cmd_look(a):
    x = load_audio(a.audio, a.rate)
    lo, hi = pair(a.band)
    t0 = a.start or 0.0
    t1 = a.end if a.end is not None else len(x) / a.rate
    s0 = max(0, int((t0 - 2) * a.rate))
    seg = x[s0:int((t1 + 2) * a.rate)]
    spec = Spec(seg, a.rate, lo, hi, res=a.res, hop_s=a.hop)
    spec.t += s0 / a.rate
    row = a.row if a.row else max(30.0, min(300.0, t1 - t0))
    paths = render(spec, t0, t1, Path(a.out), row_s=row, width=a.width, row_h=a.height, vmax=a.contrast,
                   title=f"{Path(a.audio).name}  {lo:g}-{hi:g} Hz  {clock(t0)}-{clock(t1)}  (whitened; falling curves = passing aircraft)")
    print("\n".join(str(p) for p in paths))


def _tracks(spec: Spec, thr: float, min_dur: float):
    """Link per-frame spectral peaks of the 3-s smoothed spectrogram into tonal tracks, then join broken pieces."""
    from scipy.ndimage import uniform_filter1d
    k = max(1, int(round(3.0 / spec.dt)))
    Zs = uniform_filter1d(spec.Z, size=k, axis=0)
    noise = 1.4826 * np.median(np.abs(Zs - np.median(Zs))) + 1e-6
    Zs = Zs / noise
    pk = ((Zs >= np.roll(Zs, 1, 1)) & (Zs > np.roll(Zs, -1, 1)) & (Zs >= np.roll(Zs, 2, 1)) & (Zs > np.roll(Zs, -2, 1))
          & (Zs >= thr))
    pk[:, :2] = pk[:, -2:] = False
    fr, bi = np.nonzero(pk)
    order = np.argsort(fr, kind="stable")
    fr, bi = fr[order], bi[order]
    starts = np.searchsorted(fr, np.arange(len(spec.t) + 1))
    gapf = max(2, int(round(2.0 / spec.dt)))
    active, done = [], []
    for i in range(len(spec.t)):
        bins = bi[starts[i]:starts[i + 1]]
        still = []
        for tr in active:
            if i - tr["fr"][-1] > gapf:
                done.append(tr)
            else:
                still.append(tr)
        active = still
        used = set()
        for tr in sorted(active, key=lambda r: -len(r["fr"])):
            if not len(bins):
                break
            lb = tr["b"][-1]
            tol = max(2.0, 0.015 * spec.f[min(lb, len(spec.f) - 1)] / spec.df)
            dist = np.abs(bins - lb)
            j = int(np.argmin(dist))
            if dist[j] <= tol and j not in used:
                used.add(j)
                tr["fr"].append(i)
                tr["b"].append(int(bins[j]))
                tr["z"].append(float(Zs[i, bins[j]]))
        for j, b in enumerate(bins):
            if j not in used:
                active.append({"fr": [i], "b": [int(b)], "z": [float(Zs[i, b])]})
    done += active
    done = [tr for tr in done if len(tr["fr"]) >= 4]
    done.sort(key=lambda r: r["fr"][0])
    joined = []
    for tr in done:      # join a piece that starts where another ended: ≤ 12 s later, at most 3 % higher, and no
        for prev in joined:  # lower than the steepest Doppler fall (~2 %/s) allows across the gap
            gap = tr["fr"][0] - prev["fr"][-1]
            fa, fb = spec.f[min(prev["b"][-1], len(spec.f) - 1)], spec.f[min(tr["b"][0], len(spec.f) - 1)]
            gs_ = gap * spec.dt
            if 0 < gs_ <= 12 and fa * (0.97 - 0.02 * gs_) - 2 * spec.df <= fb <= fa * 1.03 + 2 * spec.df:
                for key in ("fr", "b", "z"):
                    prev[key] += tr[key]
                break
        else:
            joined.append(tr)
    out = []
    for tr in joined:
        tt = spec.t[tr["fr"]]
        if tt[-1] - tt[0] < min_dur:
            continue
        ff = spec.f[np.array(tr["b"])]
        out.append({"t": tt, "f": ff, "z": np.array(tr["z"])})
    return out


def cmd_events(a):
    from scipy.ndimage import median_filter, percentile_filter, gaussian_filter1d
    from scipy.signal import find_peaks
    x = load_audio(a.audio, a.rate)
    dur = len(x) / a.rate
    lo, hi = pair(a.band)
    spec = Spec(x, a.rate, lo, hi, res=a.res, hop_s=0.5)
    tracks = _tracks(spec, a.threshold, a.min_dur)
    doppler, steady = [], []
    for tr in tracks:
        f = median_filter(tr["f"], size=5, mode="nearest")
        n = len(f)
        q = max(2, n // 7)
        f1, f2 = float(np.median(f[:q])), float(np.median(f[-q:]))
        spread = (np.percentile(f, 90) - np.percentile(f, 10)) / np.median(f)
        steps = np.diff(f)
        mono = float(np.mean(steps <= spec.df * 0.51)) if len(steps) else 0
        rec = dict(t_start=float(tr["t"][0]), t_end=float(tr["t"][-1]), f_start=round(f1, 1), f_end=round(f2, 1),
                   f_min=float(f.min()), f_max=float(f.max()), z=round(float(np.mean(tr["z"])), 2))
        if spread < 0.006:
            fm = float(np.median(f))
            fam = [g for g in (50, 60) if abs(fm / g - round(fm / g)) * g < 0.6 and round(fm / g) >= 1]
            rec.update(f=round(fm, 2), mains=f"{fam[0]} Hz harmonic" if fam else None)
            steady.append(rec)
        elif f1 / f2 >= 1.03 and mono >= 0.75:
            rr = f1 / f2
            c = sound_speed(a.temp)
            rec.update(speed_min=round(c * (rr - 1) / (rr + 1), 1),
                       t_mid=float(tr["t"][np.argmin(np.abs(f - 2 / (1 / f1 + 1 / f2)))]))
            doppler.append(rec)
    # group harmonics of one pass
    doppler.sort(key=lambda r: r["t_start"])
    groups = []
    for r in doppler:
        for g in groups:
            h = g[0]
            ov = min(h["t_end"], r["t_end"]) - max(h["t_start"], r["t_start"])
            if ov < 0.5 * min(h["t_end"] - h["t_start"], r["t_end"] - r["t_start"]):
                continue
            ratio = (r["f_start"] / h["f_start"] + r["f_end"] / h["f_end"]) / 2
            if any(abs(ratio - p / q) / (p / q) < 0.025 for p in range(1, 7) for q in range(1, 7)):
                g.append(r)
                break
        else:
            groups.append([r])
    events = []
    for k, g in enumerate(groups, 1):
        best = max(g, key=lambda r: r["z"] * (r["t_end"] - r["t_start"]))
        events.append(dict(id=f"E{k}", t_start=round(min(r["t_start"] for r in g), 1),
                           t_end=round(max(r["t_end"] for r in g), 1), t_mid=round(best["t_mid"], 1),
                           tracks=len(g), lowest_f=[min(r["f_end"] for r in g), min(r["f_start"] for r in g)],
                           fit_band=[round(best["f_min"] * 0.9, 1), round(best["f_max"] * 1.1, 1)],
                           speed_min=best["speed_min"], z=best["z"]))
    # low-frequency bursts (thunder candidates; also wind, doors, traffic)
    P = stft_power(x, a.rate, a.rate // 4, a.rate // 4, 20, 250)[2].sum(1)
    db = 10 * np.log10(P + 1e-20)
    base = percentile_filter(db, 30, size=max(3, int(60 / 0.25)), mode="nearest")
    ex = gaussian_filter1d(db - base, 2)
    pk, props = find_peaks(ex, height=a.burst_db, distance=int(4 / 0.25))
    bursts = [dict(t=round(float(i * 0.25 + 0.125), 1), db_above=round(float(h), 1)) for i, h in zip(pk, props["peak_heights"])]
    mains = [s for s in steady if s.get("mains")]
    res = dict(audio=str(a.audio), duration_s=round(dur, 1), band=[lo, hi], doppler_events=events,
               steady_tones=steady, mains_hint=sorted({s["mains"] for s in mains}), low_freq_bursts=bursts)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(events)} Doppler event(s), {len(steady)} steady tone(s), {len(bursts)} low-frequency burst(s) → {a.out}")
    for ev in events:
        print(f"  {ev['id']}  {clock(ev['t_start'])}–{clock(ev['t_end'])}  CPA≈{clock(ev['t_mid'])}  "
              f"{ev['lowest_f'][1]:.0f}→{ev['lowest_f'][0]:.0f} Hz  {ev['tracks']} harmonic(s)  speed≥{ev['speed_min']} m/s  z={ev['z']}")
        print(f"      fit: audio.py fit {a.audio} --start {max(0, ev['t_start'] - 10):.0f} --end {ev['t_end'] + 10:.0f} "
              f"--band {ev['fit_band'][0]:.0f},{ev['fit_band'][1]:.0f} --name {ev['id'].lower()}")
    if mains:
        print("  mains hum:", ", ".join(res["mains_hint"]), "→ run `audio.py hum`")
    if a.png:
        boxes = [(ev["t_start"], ev["t_end"], ev["fit_band"][0], ev["fit_band"][1], ev["id"]) for ev in events]
        paths = render(spec, 0, dur, Path(a.png), row_s=min(300.0, max(30.0, dur)), vmax=a.contrast, boxes=boxes,
                       title=f"{Path(a.audio).name}: candidate Doppler events (green)")
        print("  sheet:", ", ".join(str(p) for p in paths))


def cmd_fit(a):
    c = a.c or sound_speed(a.temp)
    x = load_audio(a.audio, a.rate)
    if a.trace:
        rows = [r for r in csv.reader(open(a.trace)) if r and not r[0].startswith("#")]
        pts = [(float(r[0]), float(r[1]), float(r[2]) if len(r) > 2 else 1.0) for r in rows if re.match(r"^[\d.+-]", r[0])]
        t, f, w = (np.array(v, float) for v in zip(*sorted(pts)))
        t0, t1 = float(t.min()) - 10, float(t.max()) + 10
        lo, hi = float(f.min()) * 0.85, float(f.max()) * 1.15
        s0 = max(0, int((t0 - 2) * a.rate))
        spec = Spec(x[s0:int((t1 + 2) * a.rate)], a.rate, lo, hi, res=a.res, hop_s=a.hop)
        spec.t += s0 / a.rate
        eff = 1.0
    else:
        if a.start is None or a.end is None or not a.band:
            raise SystemExit("fit needs --start, --end and --band (or --trace)")
        t0, t1 = a.start, a.end
        lo, hi = pair(a.band)
        s0 = max(0, int((t0 - 2) * a.rate))
        spec = Spec(x[s0:int((t1 + 2) * a.rate)], a.rate, lo, hi, res=a.res, hop_s=a.hop)
        spec.t += s0 / a.rate
        fr = spec.frames(t0, t1)
        from scipy.ndimage import uniform_filter1d
        Zc = np.clip(uniform_filter1d(spec.Z[fr], size=max(1, int(round(1.0 / spec.dt))), axis=0), -2, 8)
        slope = a.max_slope or max(12.0, 0.06 * hi)            # Hz/s; scales with the tone (a kHz jet tone falls fast)
        down = max(2, int(math.ceil(slope * spec.dt / spec.df)))
        path = trace_ridge(Zc, up=1, down=down, pen=0.4)
        z = Zc[np.arange(len(fr)), path]
        # keep the longest bright stretch (gaps up to 3 s), so noise before/after the pass is not fitted
        good = uniform_filter1d(z, size=max(3, int(4 / spec.dt))) >= a.min_z
        lim = int(3 / spec.dt)
        on = np.nonzero(good)[0]
        i0 = i1 = 0
        if len(on):
            runs = np.split(on, np.nonzero(np.diff(on) > lim)[0] + 1)
            longest = max(runs, key=lambda r: r[-1] - r[0])
            i0, i1 = int(longest[0]), int(longest[-1])
        if i1 - i0 < 20:
            raise SystemExit("No continuous curve found: lower --min-z, adjust the band, or hand-trace it (--trace)")
        sel = np.arange(i0, i1 + 1)
        b = path[sel]
        Zr = spec.Z[fr]
        zl = Zr[sel, np.clip(b - 1, 0, Zr.shape[1] - 1)]
        z0 = Zr[sel, b]
        zr = Zr[sel, np.clip(b + 1, 0, Zr.shape[1] - 1)]
        den = zl - 2 * z0 + zr
        delta = np.where(np.abs(den) > 1e-6, 0.5 * (zl - zr) / den, 0.0).clip(-0.5, 0.5)
        t = spec.t[fr[sel]]
        f = spec.f[0] + (b + delta) * spec.df
        w = np.clip(z0, 0.3, 8.0)
        eff = min(1.0, a.hop * a.res)
    p, rel, inl = fit_curve(t, f, w, c)
    reg = profile(t[inl], f[inl], w[inl], p, c, eff)
    vpad = max(a.speed_pad, 0.1 * p[1])
    search = dict(v=[round(max(3.0, reg["v"][0] - vpad), 1), round(reg["v"][1] + vpad, 1)],
                  d=[round(reg["d"][0] * 0.75), round(reg["d"][1] * 1.33)],
                  t_cpa=[round(reg["t_cpa"][0] - 2, 2), round(reg["t_cpa"][1] + 2, 2)])
    rms = float(np.sqrt(np.mean(rel[inl] ** 2)))
    loose = a.loose or rms > a.loose_rms
    if loose:   # a turn, climb or speed change bends the curve: the straight-line numbers are biased, not just noisy
        search = dict(v=[round(min(search["v"][0], 0.5 * p[1]), 1), round(max(search["v"][1], 1.6 * p[1]), 1)],
                      d=[round(min(search["d"][0], 0.33 * reg["d"][0])), round(max(search["d"][1], 3.0 * reg["d"][1]))],
                      t_cpa=[round(min(search["t_cpa"][0], p[3] - 8), 2), round(max(search["t_cpa"][1], p[3] + 8), 2)])
        log(f"note: residual {rms:.1f}‰ — the pass is not a straight constant-speed line (turn, climb, acceleration); "
            "search ranges widened, the exact-curve scoring in `locate` decides")
    name = a.name or "e1"
    res = dict(audio=str(a.audio), name=name, c=round(c, 1), band=[round(float(spec.f[0]), 1), round(float(spec.f[-1]), 1)],
               span=[round(float(t[0]), 2), round(float(t[-1]), 2)], n_points=int(inl.sum()),
               best=dict(f0=round(float(p[0]), 3), v=round(float(p[1]), 1), d=round(float(p[2])), t_cpa=round(float(p[3]), 2),
                         rms_permille=round(rms, 2)),
               region_99=reg, search=search, loose=loose,
               note="speed is relative to the air (wind shifts it vs ADS-B ground speed); the search ranges are padded for that",
               trace=[[round(float(a_), 2), round(float(b_), 2), round(float(c_), 2)] for a_, b_, c_ in zip(t, f, w)])
    out = Path(a.out or f"fit_{name}.json")
    out.write_text(json.dumps(res, indent=1))
    tt = np.linspace(t[0] - 5, t[-1] + 5, 400)
    model = p[0] * doppler_ratio(tt, p[1], p[2], p[3], c)
    png = out.with_suffix(".png")
    t_lo, t_hi = max(spec.t[0], t[0] - 15), min(spec.t[-1], t[-1] + 15)
    render(spec, t_lo, t_hi, png, row_s=t_hi - t_lo, width=1200, row_h=420, vmax=a.contrast,
           title=f"{name}: v={p[1]:.0f} m/s  d={p[2]:.0f} m  f0={p[0]:.2f} Hz  CPA heard {clock(p[3])}  (cyan trace, white fit)",
           curves=[(t[inl], f[inl], (0, 230, 255)), (tt, model, (255, 255, 255))])
    print(json.dumps({k: res[k] for k in ("name", "best", "region_99", "search", "n_points")}))
    print(f"→ {out}  {png}")
    edges = []
    if not a.trace:
        q = max(3, len(t) // 10)
        if np.median(f[:q]) >= hi - max(3 * spec.df, 0.03 * hi) or t[0] <= t0 + 3:
            edges.append("starts at the top of the band or the start of the window (the approach is cut off: lower --start, raise the band top)")
        if np.median(f[-q:]) <= lo + max(3 * spec.df, 0.03 * lo) or t[-1] >= t1 - 3:
            edges.append("ends at the bottom of the band or the end of the window (the departure is cut off: raise --end, lower the band bottom)")
    for e_ in edges:
        log("warning: the traced curve " + e_ + "; a one-sided pass gives wrong speed and distance")
    if edges:
        res["warnings"] = edges
        out.write_text(json.dumps(res, indent=1))
    if reg["at_grid_edge"]:
        log("warning: the 99% region reaches the edge of the profile grid — the curve constrains speed/distance weakly "
            "(only one side traced, or a manoeuvre); keep the ranges generous")


# ---------------------------------------------------------------- mains hum / ENF

def cmd_hum(a):
    x = load_audio(a.audio, a.rate)
    sr = a.rate
    nfft = int(sr * 8)
    if len(x) < nfft:
        raise SystemExit("hum needs at least 8 s of audio")
    _, f, P = stft_power(x, sr, nfft, sr * 4, 40, min(sr / 2 - 1, 400))
    S = np.median(P, axis=0)
    fam = {}
    for g in (50, 60):
        prom = []
        for h in range(1, 7):
            fc = g * h
            if fc > f[-1] - 6:
                break
            near = (f > fc - 6) & (f < fc + 6)
            peak = (f > fc - 0.6 * h) & (f < fc + 0.6 * h)
            base = np.median(S[near & ~peak]) + 1e-20
            prom.append((h, float(10 * np.log10(S[peak].max() / base))))
        fam[g] = prom
    score = {g: sum(max(0, p) for _, p in v) for g, v in fam.items()}
    g = max(score, key=score.get)
    hbest, pbest = max(fam[g], key=lambda r: r[1])
    verdict = g if pbest >= a.min_db else None
    res = dict(audio=str(a.audio), grid_hz=verdict, prominence_db={str(k): v for k, v in fam.items()},
               note="50 Hz: Europe, Africa, most of Asia, Australia, much of South America; 60 Hz: North America, "
                    "Brazil, Colombia, Venezuela, Peru (mostly), Saudi Arabia, South Korea, Taiwan, the Philippines, "
                    "western Japan (eastern Japan is 50 Hz). A recorder on battery away from wiring may show none.")
    if verdict:
        # ENF trace from the strongest harmonic: 8 s windows every second, parabolic peak interpolation
        tt, ff, PP = stft_power(x, sr, nfft, sr, hbest * g - hbest * 1.0, hbest * g + hbest * 1.0)
        k = np.argmax(PP, axis=1)
        k = np.clip(k, 1, PP.shape[1] - 2)
        L = np.log(PP + 1e-30)
        r = np.arange(len(k))
        den = L[r, k - 1] - 2 * L[r, k] + L[r, k + 1]
        dl = np.where(np.abs(den) > 1e-12, 0.5 * (L[r, k - 1] - L[r, k + 1]) / den, 0).clip(-0.5, 0.5)
        enf = (ff[0] + (k + dl) * (ff[1] - ff[0])) / hbest
        res.update(harmonic=hbest, enf_mean=round(float(enf.mean()), 4), enf_std=round(float(enf.std()), 4),
                   enf_range=[round(float(enf.min()), 4), round(float(enf.max()), 4)])
        if a.csv:
            with open(a.csv, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["t_s", "freq_hz"])
                for tv, fv in zip(tt, enf):
                    wr.writerow([round(float(tv), 2), round(float(fv), 5)])
            res["enf_csv"] = a.csv
        if a.reference:
            rt, rf = [], []
            for row in csv.DictReader(open(a.reference)):
                keys = {k.lower().strip(): v for k, v in row.items() if k}
                tk = next((k for k in keys if k in ("time", "timestamp", "datetime", "dtm", "t", "date")), None)
                fk = next((k for k in keys if k in ("freq", "frequency", "f", "value", "hz", "freq_hz")), None)
                if tk and fk and keys[fk]:
                    rt.append(parse_time(keys[tk]))
                    rf.append(float(keys[fk]))
            rt, rf = np.array(rt), np.array(rf)
            o = np.argsort(rt)
            grid_t = np.arange(rt[o][0], rt[o][-1], 1.0)
            ref = np.interp(grid_t, rt[o], rf[o])
            obs = np.interp(np.arange(0, tt[-1] - tt[0], 1.0), tt - tt[0], enf)
            obs = obs - obs.mean()
            n = len(obs)
            if len(ref) <= n:
                raise SystemExit("The reference record is shorter than the recording")
            from numpy.lib.stride_tricks import sliding_window_view
            W = sliding_window_view(ref, n)
            Wc = W - W.mean(1, keepdims=True)
            sc = (Wc @ obs) / (np.linalg.norm(Wc, axis=1) * np.linalg.norm(obs) + 1e-12)
            top = np.argsort(sc)[::-1][:5]
            res["reference_match"] = [dict(start=iso(grid_t[i] - tt[0]), corr=round(float(sc[i]), 3)) for i in top]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- ADS-B tracks

def type_table(proxy=None) -> dict:
    p = CACHE / "adsb" / "icao_aircraft_types.json"
    if not p.exists():
        raw = fetch_bytes(TYPES_URL, proxy, timeout=60)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    return json.loads(p.read_text())


def adsblol_assets(day: date, proxy=None, tag: str | None = None):
    repo = f"adsblol/globe_history_{day.year}"
    tags = [tag] if tag else [f"v{day:%Y.%m.%d}-planes-readsb-prod-{n}" for n in range(3)]
    for tg in tags:
        try:
            rel = json.loads(fetch_bytes(f"https://api.github.com/repos/{repo}/releases/tags/{tg}", proxy, 30,
                                         {"Accept": "application/vnd.github+json"}))
        except RuntimeError:
            continue
        assets = sorted((x for x in rel.get("assets", []) if ".tar" in x["name"]), key=lambda x: x["name"])
        if assets:
            return tg, [(x["name"], x["browser_download_url"], x["size"]) for x in assets]
    raise SystemExit(f"No adsb.lol archive found for {day} (tried {', '.join(tags)} in {repo}); "
                     f"list releases at https://github.com/{repo}/releases and pass --tag, or bring a CSV (--csv)")


def _segments(t, lat, lon, alt, gs, max_gap=90.0):
    cut = np.nonzero(np.diff(t) > max_gap)[0] + 1
    for s in np.split(np.arange(len(t)), cut):
        if len(s) >= 3:
            yield s


def cmd_adsb(a):
    bbox = bbox_arg(a.bbox)
    if not bbox:
        raise SystemExit("adsb needs --bbox S,W,N,E")
    t_from, t_to = window(a.date, a.from_, a.to)
    pad = a.pad
    mlat = a.margin / 111195.0
    S, W, N, E = bbox
    mlon = a.margin / (111195.0 * max(0.1, math.cos(math.radians((S + N) / 2))))
    S, W, N, E = S - mlat, W - mlon, N + mlat, E + mlon
    types = type_table(a.proxy)
    out = open(a.out, "w")
    nkeep = 0
    counts = {}

    def emit(ident, reg, typ, name, t, lat, lon, alt, gs):
        nonlocal nkeep
        sel = (t >= t_from - pad) & (t <= t_to + pad) & np.isfinite(alt)
        inb = (lat >= S) & (lat <= N) & (lon >= W) & (lon <= E)
        if not (sel & inb).any():
            return
        # keep the time-window part, trimmed to the stretch near the box (plus a few points around it)
        idx = np.nonzero(sel)[0]
        ib = np.nonzero(sel & inb)[0]
        lo_i = max(idx[0], ib[0] - 3)
        hi_i = min(idx[-1], ib[-1] + 3)
        r = np.arange(lo_i, hi_i + 1)
        r = r[np.isfinite(alt[r])]
        desc = (types.get(typ) or {}).get("desc", "") if typ else ""
        for k, s in enumerate(_segments(t[r], lat[r], lon[r], alt[r], gs[r])):
            s = r[s]
            out.write(json.dumps({"id": ident, "seg": k, "reg": reg, "type": typ, "desc": desc, "name": name,
                                  "t": [round(float(v), 2) for v in t[s]], "lat": [round(float(v), 6) for v in lat[s]],
                                  "lon": [round(float(v), 6) for v in lon[s]], "alt": [round(float(v), 1) for v in alt[s]],
                                  "gs": [None if not np.isfinite(v) else round(float(v), 1) for v in gs[s]]}) + "\n")
            nkeep += 1
            cls = desc[2:3] or "?"
            counts[cls] = counts.get(cls, 0) + 1

    if a.csv:
        rows = {}
        for row in csv.DictReader(open(a.csv)):
            k = {x.lower().strip(): v for x, v in row.items() if x}
            fid = k.get("id") or k.get("icao24") or k.get("icao") or k.get("hex") or k.get("callsign") or "x"
            alt = k.get("alt_m") or k.get("geoaltitude") or k.get("altitude_m")
            if alt in (None, ""):
                af = k.get("alt_ft") or k.get("altitude") or k.get("baroaltitude_ft")
                alt = float(af) * FT if af not in (None, "") else float(k.get("baroaltitude") or "nan")
            gs = k.get("gs_ms") or k.get("velocity") or ""
            rows.setdefault(fid, []).append((parse_time(k.get("time") or k.get("timestamp") or k["t"]), float(k["lat"]),
                                             float(k.get("lon") or k.get("lng") or k["longitude"]), float(alt),
                                             float(gs) if gs else float("nan"), k.get("type") or k.get("typecode") or ""))
        for fid, pts in rows.items():
            pts.sort()
            arr = np.array([p[:5] for p in pts], float)
            emit(fid, "", pts[0][5], "", arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
    else:
        if not a.date:
            raise SystemExit("adsb needs --date (or --csv)")
        day = date.fromisoformat(a.date)
        tag, assets = adsblol_assets(day, a.proxy, a.tag)
        size = sum(s for _, _, s in assets) / 1e9
        log(f"adsb.lol {tag}: streaming {len(assets)} part(s), {size:.1f} GB (not stored); filtering to the box and window…")
        cmd = ["curl", "-q", "-sSL", "--fail", *curl_args(a.proxy)] + [u for _, u, _ in assets]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        n = 0
        t_start = _time.time()
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or "trace_full_" not in m.name:
                    continue
                raw = tf.extractfile(m).read()
                n += 1
                if n % 20000 == 0:
                    log(f"  {n} aircraft read, {nkeep} kept, {_time.time() - t_start:.0f} s")
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                d = json.loads(raw)
                ts, tr = float(d.get("timestamp", 0)), d.get("trace") or []
                if not tr or ts + tr[-1][0] < t_from - pad or ts + tr[0][0] > t_to + pad:
                    continue
                t = np.array([p[0] for p in tr], float) + ts
                lat = np.array([p[1] for p in tr], float)
                lon = np.array([p[2] for p in tr], float)
                if not ((lat >= S) & (lat <= N) & (lon >= W) & (lon <= E)).any():
                    continue

                def altm(p):
                    if p[6] is not None and int(p[6]) & 1:      # stale position
                        return float("nan")
                    g = p[10] if len(p) > 10 and isinstance(p[10], (int, float)) else None
                    b = p[3] if isinstance(p[3], (int, float)) else None
                    v = g if g is not None else b
                    return float(v) * FT if v is not None else float("nan")
                alt = np.array([altm(p) for p in tr], float)
                gs = np.array([float(p[4]) * KT if isinstance(p[4], (int, float)) else float("nan") for p in tr])
                emit(d.get("icao", m.name[-11:-5]), d.get("r", ""), d.get("t", ""), d.get("desc", ""), t, lat, lon, alt, gs)
        proc.wait()
        if proc.returncode:
            log(f"warning: download ended with curl exit {proc.returncode}; the track file may be incomplete")
    out.close()
    print(json.dumps({"tracks": a.out, "segments": nkeep, "by_engine": counts,
                      "engine_key": "P piston, T turboprop/turboshaft, J jet, E electric, ? unknown type"}))


class Flight:
    def __init__(self, r: dict):
        from scipy.interpolate import PchipInterpolator
        t = np.asarray(r["t"], float)
        o = np.argsort(t)
        t = t[o]
        keep = np.concatenate([[True], np.diff(t) > 0.05])
        t = t[keep]
        self.id = f'{r.get("id")}' + (f'#{r["seg"]}' if r.get("seg") else "")
        self.reg, self.type, self.desc, self.name = r.get("reg", ""), r.get("type", ""), r.get("desc", ""), r.get("name", "")
        self.t = t
        lat = np.asarray(r["lat"], float)[o][keep]
        lon = np.asarray(r["lon"], float)[o][keep]
        alt = np.asarray(r["alt"], float)[o][keep]
        self.lat, self.lon, self.alt = lat, lon, alt
        self.ilat = PchipInterpolator(t, lat, extrapolate=False)
        self.ilon = PchipInterpolator(t, lon, extrapolate=False)
        self.ialt = PchipInterpolator(t, alt, extrapolate=False)
        self.dlat, self.dlon, self.dalt = self.ilat.derivative(), self.ilon.derivative(), self.ialt.derivative()

    @property
    def jet(self) -> bool:
        return self.desc[2:3] == "J"

    def info(self) -> dict:
        return {"id": self.id, "reg": self.reg, "type": self.type, "desc": self.desc, "name": self.name}


def load_flights(path, kinds: str, t0=None, t1=None) -> list[Flight]:
    out = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if len(r["t"]) < 3:
            continue
        if t0 is not None and (r["t"][-1] < t0 or r["t"][0] > t1):
            continue
        desc = r.get("desc", "") or ""
        e = desc[2:3]
        if kinds == "nonjet" and e == "J":
            continue
        if kinds == "prop" and e not in ("P", "T"):
            continue
        if kinds == "heli" and desc[:1] not in ("H", "G"):
            continue
        out.append(Flight(r))
    return out


# ---------------------------------------------------------------- spatial helpers

class Grid:
    """Square-ish cells of g metres; integer ids that encode row and column."""

    def __init__(self, g: float):
        self.g = g
        self.dlat = math.degrees(g / R_EARTH)

    def cell(self, lat, lon):
        i = np.round(np.asarray(lat) / self.dlat).astype(np.int64)
        dlon = self.dlat / np.maximum(np.cos(np.radians(i * self.dlat)), 0.02)
        j = np.round(np.asarray(lon) / dlon).astype(np.int64)
        return ((i + (1 << 20)) << 26) | (j + (1 << 25))

    def center(self, ids):
        ids = np.asarray(ids, np.int64)
        i = (ids >> 26) - (1 << 20)
        j = (ids & ((1 << 26) - 1)) - (1 << 25)
        lat = i * self.dlat
        return lat, j * self.dlat / np.maximum(np.cos(np.radians(lat)), 0.02)


class Elev:
    """Ground elevation (m) from AWS Terrain Tiles (Terrarium), cached under $GEOINT_CACHE/dem; --flat gives 0."""

    def __init__(self, zoom: int = 11, flat: bool = False, proxy=None):
        self.z, self.flat, self.proxy = zoom, flat, proxy
        self.tiles: dict = {}
        self.cache = CACHE / "dem"

    def sample(self, lat, lon) -> np.ndarray:
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        if self.flat or not len(lat):
            return np.zeros_like(lat)
        from terrain import _fetch
        z = self.z
        while True:
            n = 256 * 2 ** z
            x = (lon + 180) / 360 * n
            y = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n
            tx, ty = (x // 256).astype(np.int64), (y // 256).astype(np.int64)
            keys, inv = np.unique(tx * (1 << 22) + ty, return_inverse=True)
            if len(keys) <= 1500 or z <= 7:
                break
            z -= 1
        need = [(z, int(k >> 22), int(k & ((1 << 22) - 1))) for k in keys if (z, int(k >> 22), int(k & ((1 << 22) - 1))) not in self.tiles]
        if need:
            self.cache.mkdir(parents=True, exist_ok=True)
            if len(need) > 50:
                log(f"  elevation: fetching {len(need)} tiles (zoom {z})…")
            with ThreadPoolExecutor(16) as ex:
                for key, arr in zip(need, ex.map(lambda q: _fetch(q[0], q[1], q[2], self.cache, self.proxy), need)):
                    self.tiles[key] = arr
        out = np.empty_like(lat)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
        for k, key in enumerate(keys):
            idx = order[bounds[k]:bounds[k + 1]]
            a, b = int(key >> 22), int(key & ((1 << 22) - 1))
            arr = self.tiles[(z, a, b)]
            px = np.clip((x[idx] - a * 256).astype(int), 0, 255)
            py = np.clip((y[idx] - b * 256).astype(int), 0, 255)
            out[idx] = arr[py, px]
        return np.where(out < -120, 0.0, out)      # deep negative = sea (bathymetry): the surface is at 0


def enu(lat, lon, lat0, lon0):
    return (R_EARTH * np.cos(np.radians(lat0)) * np.radians(lon - lon0), R_EARTH * np.radians(lat - lat0))


# ---------------------------------------------------------------- aircraft search

def swath(fl: Flight, t0: float, t1: float, grid: Grid, elev: Elev, c: float, mic: float,
          speed=None, slant=None, mask_cells=None, step_t: float = 1.0):
    """Every grid cell this flight passed abeam of within the time window: (cell, heard CPA time, slant distance,
    3-D speed, |along-track offset|). speed/slant (lo, hi) restrict to an event's measured ranges."""
    a0, a1 = max(t0, fl.t[0]), min(t1, fl.t[-1])
    if a1 - a0 < 2:
        return None
    tt = np.arange(a0, a1, step_t)
    lat, lon, alt = fl.ilat(tt), fl.ilon(tt), fl.ialt(tt)
    vn = R_EARTH * np.radians(fl.dlat(tt))
    ve = R_EARTH * np.cos(np.radians(lat)) * np.radians(fl.dlon(tt))
    vu = fl.dalt(tt)
    vh = np.hypot(ve, vn)
    sp = np.sqrt(vh ** 2 + vu ** 2)
    ok = np.isfinite(lat) & np.isfinite(alt) & (vh > 12)
    if speed is not None:
        ok &= (sp >= speed[0]) & (sp <= speed[1])
    if not ok.any():
        return None
    tt, lat, lon, alt, ve, vn, vh, sp = (v[ok] for v in (tt, lat, lon, alt, ve, vn, vh, sp))
    hg = alt - elev.sample(lat, lon) - mic
    g = grid.g
    smax_all = (slant[1] if slant else 12000.0) + g
    offs = np.arange(-smax_all, smax_all + 1e-6, g / 2)
    if slant:
        smin = np.sqrt(np.maximum(0, slant[0] ** 2 - hg ** 2)) - g
        smax = np.sqrt(np.maximum(0, slant[1] ** 2 - hg ** 2)) + g
    else:
        smin = np.full(len(tt), -1.0)
        smax = np.full(len(tt), smax_all)
    ux, uy = ve / vh, vn / vh
    res = []
    for i in range(0, len(tt), 1500):
        sl = slice(i, i + 1500)
        m = (np.abs(offs)[None, :] >= smin[sl, None]) & (np.abs(offs)[None, :] <= smax[sl, None])
        si, oi = np.nonzero(m)
        if not len(si):
            continue
        si = si + i
        s = offs[oi]
        pe, pn = s * uy[si], -s * ux[si]          # perpendicular to the track (right-hand side for s > 0)
        clat = lat[si] + np.degrees(pn / R_EARTH)
        clon = lon[si] + np.degrees(pe / (R_EARTH * np.cos(np.radians(lat[si]))))
        cid = grid.cell(clat, clon)
        if mask_cells is not None:
            pos = np.searchsorted(mask_cells, cid)
            keep = (pos < len(mask_cells)) & (mask_cells[np.minimum(pos, len(mask_cells) - 1)] == cid)
            if not keep.any():
                continue
            si, cid = si[keep], cid[keep]
        res.append((si, cid))
    if not res:
        return None
    si = np.concatenate([r[0] for r in res])
    cid = np.concatenate([r[1] for r in res])
    clat, clon = grid.center(cid)
    ae, an = enu(lat[si], lon[si], clat, clon)            # aircraft relative to the cell centre
    along = ae * ux[si] + an * uy[si]
    lateral = np.abs(ae * uy[si] - an * ux[si])
    zc = elev.sample(clat, clon) + mic
    h = alt[si] - zc
    d = np.sqrt(lateral ** 2 + h ** 2)
    tcpa = tt[si] - along / vh[si]
    th = tcpa + d / c
    keep = np.ones(len(si), bool) if slant is None else (d >= slant[0]) & (d <= slant[1])
    if not keep.any():
        return None
    cid, th, d, spd, aa = cid[keep], th[keep], d[keep], sp[si][keep], np.abs(along[keep])
    # one row per (cell, pass): the sample nearest abeam
    o = np.lexsort((th, cid))
    cid, th, d, spd, aa = cid[o], th[o], d[o], spd[o], aa[o]
    newg = np.concatenate([[True], (np.diff(cid) != 0) | (np.diff(th) > 60)])
    gid = np.cumsum(newg)
    o = np.lexsort((aa, gid))
    first = np.concatenate([[True], np.diff(gid[o]) != 0])
    o = o[first]
    return dict(cell=cid[o], th=th[o], d=d[o], v=spd[o])


def _cat(parts: list[dict], fi: list[int]) -> dict:
    if not parts:
        return dict(cell=np.zeros(0, np.int64), th=np.zeros(0), d=np.zeros(0), v=np.zeros(0), fl=np.zeros(0, int))
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("cell", "th", "d", "v")}
    out["fl"] = np.concatenate([np.full(len(p["cell"]), f) for p, f in zip(parts, fi)])
    return out


def _ranges(i0, i1):
    cnt = i1 - i0
    rep = np.repeat(np.arange(len(i0)), cnt)
    idx = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(i0, cnt)
    return rep, idx


def model_ratio(fl: Flight, lat0: float, lon0: float, z0: float, t_abs: np.ndarray, c: float):
    """Received/emitted frequency ratio heard at (lat0, lon0, z0) at absolute times t_abs (emission-time solution)."""
    tau = t_abs - 3000 / c
    for _ in range(7):
        lat, lon, alt = fl.ilat(tau), fl.ilon(tau), fl.ialt(tau)
        e, n = enu(lat, lon, lat0, lon0)
        r = np.sqrt(e * e + n * n + (alt - z0) ** 2)
        tau = t_abs - r / c
    lat, lon, alt = fl.ilat(tau), fl.ilon(tau), fl.ialt(tau)
    e, n = enu(lat, lon, lat0, lon0)
    u = alt - z0
    r = np.sqrt(e * e + n * n + u * u)
    ve = R_EARTH * np.cos(np.radians(lat)) * np.radians(fl.dlon(tau))
    vn = R_EARTH * np.radians(fl.dlat(tau))
    vr = (e * ve + n * vn + u * fl.dalt(tau)) / r
    return c / (c + vr), r


class Scorer:
    """Scores modelled curves on the spectrogram (brightness under the curve) or against traced points."""

    def __init__(self, spec: Spec, fits: list[dict], mode: str, c: float, t_sweep: float, f_sweep: float, frame_step: int = 2):
        from scipy.ndimage import uniform_filter1d
        self.spec, self.fits, self.mode, self.c = spec, fits, mode, c
        # 1-s smoothing along time, read between bins: a curve 1 % off loses most of its brightness
        self.Zm = np.clip(uniform_filter1d(spec.Z, size=max(1, int(round(1.0 / spec.dt))), axis=0), -2, 8)
        self.shifts = np.arange(-t_sweep, t_sweep + 1e-9, spec.dt)
        self.eps = np.linspace(-f_sweep, f_sweep, 21)
        self.frames, self.trace = [], []
        for ft in fits:
            a, b = ft["span"]
            fr = spec.frames(a, b)[::frame_step]
            self.frames.append(fr)
            tr = np.array(ft["trace"], float)
            self.trace.append(tr)

    def _event(self, k, fl, lat, lon, z0, T0, shifts):
        """Score of event k for each shift (rows) and f0 tweak (cols)."""
        f0 = self.fits[k]["best"]["f0"]
        if self.mode == "trace":
            ta = self.trace[k][:, 0]
            fobs = self.trace[k][:, 1]
        else:
            ta = self.spec.t[self.frames[k]]
        t_abs = ta[None, :] + T0 + shifts[:, None]
        ratio, _ = model_ratio(fl, lat, lon, z0, t_abs.ravel(), self.c)
        ratio = ratio.reshape(t_abs.shape)
        fm = f0 * ratio[:, None, :] * (1 + self.eps[None, :, None])         # shifts × eps × frames
        if self.mode == "trace":
            err = np.abs(fm / fobs[None, None, :] - 1) * 1000
            err = np.where(np.isfinite(err), err, 50.0)
            return -np.median(err, axis=2)
        b = (fm - self.spec.f[0]) / self.spec.df
        inb = np.isfinite(b) & (b >= 0) & (b < self.Zm.shape[1] - 1)
        b = np.where(inb, b, 0.0)
        bi = b.astype(int)
        w = b - bi
        fr = np.broadcast_to(self.frames[k][None, None, :], bi.shape)
        val = np.where(inb, (1 - w) * self.Zm[fr, bi] + w * self.Zm[fr, bi + 1], 0.0)
        return val.mean(axis=2)

    def score(self, fls: list[Flight], lat: float, lon: float, z0: float, T0: float, locked: bool = True):
        """locked: shift chosen on event 1, later events placed by flight geometry (absolute time unknown)."""
        if locked:
            s1 = self._event(0, fls[0], lat, lon, z0, T0, self.shifts)
            i, j = np.unravel_index(np.argmax(s1), s1.shape)
            sh = self.shifts[i:i + 1]
            per = [float(s1[i, j])] + [float(self._event(k, fls[k], lat, lon, z0, T0, sh)[0].max()) for k in range(1, len(fls))]
            return sum(per), per, float(self.shifts[i])
        mats = [self._event(k, fls[k], lat, lon, z0, T0, self.shifts).max(axis=1) for k in range(len(fls))]
        tot = np.sum(mats, axis=0)
        i = int(np.argmax(tot))
        return float(tot[i]), [float(m[i]) for m in mats], float(self.shifts[i])


def cmd_locate(a):
    c = a.c or sound_speed(a.temp)
    fits = sorted((json.load(open(p)) for p in a.fit), key=lambda r: r["best"]["t_cpa"])
    K = len(fits)
    t_from, t_to = window(a.date, a.from_, a.to)
    T0_known = parse_time(a.audio_start) if a.audio_start else None
    bbox = bbox_arg(a.bbox)
    x = load_audio(a.audio, a.rate)
    dur = len(x) / a.rate
    if T0_known is not None:
        t_from = T0_known - a.clock_tol - 600
        t_to = T0_known + dur + a.clock_tol + 600
    flights = load_flights(a.tracks, a.types, t_from, t_to)
    if not flights:
        raise SystemExit("No flights of the selected types in the track file for that window")
    if t_from is None:
        t_from = min(f.t[0] for f in flights)
        t_to = max(f.t[-1] for f in flights)
    log(f"{len(flights)} flight segments ({a.types}); window {iso(t_from)} → {iso(t_to)}; {K} event(s); c={c:.1f} m/s")
    grid, elev = Grid(a.grid), Elev(a.dem_zoom, a.flat, a.proxy)

    def in_box(cells):
        if not bbox:
            return np.ones(len(cells), bool)
        la, lo = grid.center(cells)
        return (la >= bbox[0]) & (la <= bbox[2]) & (lo >= bbox[1]) & (lo <= bbox[3])

    funnel = []
    E = []
    for k, ft in enumerate(fits):
        sp, sl = ft["search"]["v"], ft["search"]["d"]
        parts, fi = [], []
        for i, fl in enumerate(flights):
            r = swath(fl, t_from, t_to, grid, elev, c, a.mic_height, speed=sp, slant=sl)
            if r is not None:
                m = in_box(r["cell"])
                if m.any():
                    parts.append({kk: vv[m] for kk, vv in r.items()})
                    fi.append(i)
        ek = _cat(parts, fi)
        E.append(ek)
        funnel.append(dict(stage=f"event {k + 1} ({ft['name']}): speed {sp[0]:.0f}-{sp[1]:.0f} m/s, slant {sl[0]:.0f}-{sl[1]:.0f} m",
                           cells=int(len(np.unique(ek["cell"]))), flights=int(len(set(fi)))))
        log("  " + funnel[-1]["stage"] + f": {funnel[-1]['cells']} cells from {funnel[-1]['flights']} flights")
    # time offsets between events (locked to the audio)
    ta = [ft["best"]["t_cpa"] for ft in fits]
    half = [max(0.0, (ft["search"]["t_cpa"][1] - ft["search"]["t_cpa"][0]) / 2 - 2) for ft in fits]
    e1 = E[0]
    keep = np.ones(len(e1["cell"]), bool)
    if T0_known is not None:
        tol = a.clock_tol + half[0] + a.offset_tol
        keep &= np.abs(e1["th"] - (T0_known + ta[0])) <= tol
    allc = np.unique(np.concatenate([e["cell"] for e in E]))
    tbase = t_from - 7200
    combo = dict(cell=e1["cell"][keep], th=[e1["th"][keep]], fl=[e1["fl"][keep]], d=[e1["d"][keep]], v=[e1["v"][keep]])
    for k in range(1, K):
        ek = E[k]
        dk = np.searchsorted(allc, ek["cell"]).astype(np.int64)
        tds = np.clip(np.round((ek["th"] - tbase) * 10), 0, (1 << 24) - 1).astype(np.int64)
        key = (dk << 24) | tds
        o = np.argsort(key)
        ks = key[o]
        d1 = np.searchsorted(allc, combo["cell"]).astype(np.int64)
        off = ta[k] - ta[0]
        tol = a.offset_tol + half[k] + half[0]
        lo = (d1 << 24) | np.clip(np.round((combo["th"][0] + off - tol - tbase) * 10), 0, (1 << 24) - 1).astype(np.int64)
        hi = (d1 << 24) | np.clip(np.round((combo["th"][0] + off + tol - tbase) * 10), 0, (1 << 24) - 1).astype(np.int64)
        rep, idx = _ranges(np.searchsorted(ks, lo, "left"), np.searchsorted(ks, hi, "right"))
        j = o[idx]
        ok = ~((ek["fl"][j] == combo["fl"][-1][rep]) & (np.abs(ek["th"][j] - combo["th"][-1][rep]) < 120))
        rep, j = rep[ok], j[ok]
        combo = dict(cell=combo["cell"][rep], th=[v[rep] for v in combo["th"]] + [ek["th"][j]],
                     fl=[v[rep] for v in combo["fl"]] + [ek["fl"][j]], d=[v[rep] for v in combo["d"]] + [ek["d"][j]],
                     v=[v[rep] for v in combo["v"]] + [ek["v"][j]])
    n = len(combo["cell"])
    funnel.append(dict(stage=("absolute time ±%gs and " % a.clock_tol if T0_known is not None else "") +
                       f"time offsets between events ±{a.offset_tol:g}s", candidates=n, cells=int(len(np.unique(combo["cell"])))))
    log(f"  after time offsets: {n} candidates in {funnel[-1]['cells']} cells")
    # silence: no other aircraft heard close by between the events
    if n and a.silence != "off":
        A = a.audible or max(ft["best"]["d"] for ft in fits)
        cu = np.unique(combo["cell"])
        sflights = flights if a.silence_types == a.types else load_flights(a.tracks, a.silence_types, t_from, t_to)
        parts, fi = [], []
        ids = {id(f): i for i, f in enumerate(flights)}
        for fl in sflights:
            r = swath(fl, t_from, t_to, grid, elev, c, a.mic_height, slant=(0.0, A), mask_cells=cu)
            if r is not None:
                parts.append(r)
                fi.append(ids.get(id(fl), -1 - len(fi)))
        S = _cat(parts, fi)
        dk = np.searchsorted(cu, S["cell"]).astype(np.int64)
        key = (dk << 24) | np.clip(np.round((S["th"] - tbase) * 10), 0, (1 << 24) - 1).astype(np.int64)
        ks = np.sort(key)
        dc = np.searchsorted(cu, combo["cell"]).astype(np.int64)
        m = a.margin
        bad = np.zeros(n, bool)
        ivals = [(combo["th"][k] + m, combo["th"][k + 1] - m) for k in range(K - 1)]
        if a.silence == "whole":
            t_start = combo["th"][0] - ta[0]
            ivals += [(t_start + 5, combo["th"][0] - m), (combo["th"][-1] + m, t_start + dur - 5)]
        for lo_t, hi_t in ivals:
            okv = hi_t > lo_t
            lo = (dc << 24) | np.clip(np.round((lo_t - tbase) * 10), 0, (1 << 24) - 1).astype(np.int64)
            hi = (dc << 24) | np.clip(np.round((hi_t - tbase) * 10), 0, (1 << 24) - 1).astype(np.int64)
            cnt = np.searchsorted(ks, hi, "right") - np.searchsorted(ks, lo, "left")
            bad |= okv & (cnt > 0)
        combo = dict(cell=combo["cell"][~bad], th=[v[~bad] for v in combo["th"]], fl=[v[~bad] for v in combo["fl"]],
                     d=[v[~bad] for v in combo["d"]], v=[v[~bad] for v in combo["v"]])
        n = len(combo["cell"])
        funnel.append(dict(stage=f"silence ({a.silence}): no {a.silence_types} aircraft within {A:.0f} m slant between events",
                           candidates=n, cells=int(len(np.unique(combo["cell"])))))
        log(f"  after silence: {n} candidates in {funnel[-1]['cells']} cells")
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not n:
        (outdir / "locate.json").write_text(json.dumps(dict(funnel=funnel, groups=[]), indent=1))
        raise SystemExit("No candidate survived. Widen the ranges (--offset-tol, the fits' search ranges), the window, "
                         "the box, or the aircraft types (--types all); check the tracks cover the area.")
    # exact scoring
    lo_f = min(ft["band"][0] for ft in fits) * 0.95
    hi_f = max(ft["band"][1] for ft in fits) * 1.05
    spec = Spec(x, a.rate, lo_f, hi_f, res=a.res, hop_s=0.25)
    scorer = Scorer(spec, fits, a.score, c, a.t_sweep if T0_known is None else a.clock_tol, a.f_sweep)
    flist = np.stack(combo["fl"], 1)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(tuple(int(v) for v in flist[i]), []).append(i)
    rng = np.random.default_rng(0)
    sel = []
    per_group = max(5, a.max_score // max(1, len(groups)))
    for g, idx in groups.items():
        sel += idx if len(idx) <= per_group else list(rng.choice(idx, per_group, replace=False))
    log(f"  scoring {len(sel)} candidates ({len(groups)} flight combinations)…")
    clat, clon = grid.center(combo["cell"][sel])
    zc = elev.sample(clat, clon) + a.mic_height
    rows = []
    t_start = _time.time()
    for q, i in enumerate(sel):
        fls = [flights[int(v)] for v in flist[i]]
        T0 = T0_known if T0_known is not None else combo["th"][0][i] - ta[0]
        tot, per, sh = scorer.score(fls, float(clat[q]), float(clon[q]), float(zc[q]), T0, locked=T0_known is None)
        rows.append(dict(i=i, lat=float(clat[q]), lon=float(clon[q]), z=float(zc[q]), score=tot, per=per, T0=T0 + sh,
                         g=tuple(int(v) for v in flist[i])))
        if q and q % 2000 == 0:
            log(f"    {q}/{len(sel)}  {_time.time() - t_start:.0f} s")
    best_by = {}
    for r in rows:
        if r["g"] not in best_by or r["score"] > best_by[r["g"]]["score"]:
            best_by[r["g"]] = r
    top = sorted(best_by.values(), key=lambda r: -r["score"])[:a.refine]
    # refine around each top group's best cell
    final = []
    for r in top:
        fls = [flights[k] for k in r["g"]]
        best = r
        for step in (a.grid / 4, a.grid / 16):
            offs = np.arange(-4, 5) * step
            la0, lo0 = best["lat"], best["lon"]
            for dy in offs:
                for dx in offs:
                    la = la0 + math.degrees(dy / R_EARTH)
                    lo = lo0 + math.degrees(dx / (R_EARTH * math.cos(math.radians(la0))))
                    z = float(elev.sample(la, lo)[0]) + a.mic_height
                    T0 = T0_known if T0_known is not None else r["T0"]
                    tot, per, sh = scorer.score(fls, la, lo, z, T0, locked=T0_known is None)
                    if tot > best["score"]:
                        best = dict(r, lat=la, lon=lo, z=z, score=tot, per=per, T0=T0 + sh)
        details = []
        for k, fl in enumerate(fls):
            tt = np.arange(fl.t[0], fl.t[-1], 0.5)
            ratio, rr = model_ratio(fl, best["lat"], best["lon"], best["z"], tt, c)
            ok = np.isfinite(rr)
            j = int(np.nanargmin(np.where(ok, rr, np.inf)))
            th = float(tt[j])
            details.append(dict(event=fits[k]["name"], flight=fl.info(), cpa_slant_m=round(float(rr[j])),
                                heard_cpa=iso(th), audio_cpa=clock(fits[k]["best"]["t_cpa"])))
        final.append(dict(lat=round(best["lat"], 6), lon=round(best["lon"], 6), score=round(best["score"], 3),
                          per_event=[round(v, 3) for v in best["per"]], recording_start=iso(best["T0"]),
                          events=details, candidates_in_group=len(groups[r["g"]])))
    final.sort(key=lambda r: -r["score"])
    margin = round(final[0]["score"] - final[1]["score"], 3) if len(final) > 1 else None
    res = dict(audio=str(a.audio), c=round(c, 1), grid_m=a.grid, score_mode=a.score, funnel=funnel,
               best=final[0], margin_to_second=margin, groups=final,
               read_me="score = mean whitened brightness under each modelled curve (trace mode: −median error ‰), summed "
                       "over events; trust a winner that leads clearly, matches every event, and survives a rerun with "
                       "wider ranges. The point is the microphone, ± the grid step; verify on street level.")
    (outdir / "locate.json").write_text(json.dumps(res, indent=1))
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
              "properties": {"score": round(r["score"], 3), "flights": "+".join(flights[k].id for k in r["g"])}} for r in rows]
    feats += [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [g["lon"], g["lat"]]},
               "properties": {"rank": k + 1, "score": g["score"], "best": True}} for k, g in enumerate(final)]
    (outdir / "candidates.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    _plot_map(rows, final, flights, outdir / "map.png")
    _plot_curves(final[:3], fits, flights, spec, c, elev, a.mic_height, outdir / "curves.png")
    b = final[0]
    print(f"best: {b['lat']:.6f}, {b['lon']:.6f}  score {b['score']}  (margin to 2nd {margin})  recording start ≈ {b['recording_start']}")
    for d in b["events"]:
        print(f"  {d['event']}: {d['flight']['id']} {d['flight']['reg']} {d['flight']['type']} — CPA {d['cpa_slant_m']} m at {d['heard_cpa']}")
    print(f"→ {outdir}/locate.json, candidates.geojson, map.png, curves.png")


def _plot_map(rows, final, flights, path: Path):
    if not rows:
        return
    la = np.array([r["lat"] for r in rows])
    lo = np.array([r["lon"] for r in rows])
    sc = np.array([r["score"] for r in rows])
    pts = [(g["lat"], g["lon"]) for g in final]
    lat_all = np.concatenate([la, [p[0] for p in pts]])
    lon_all = np.concatenate([lo, [p[1] for p in pts]])
    S, N = lat_all.min(), lat_all.max()
    W, E = lon_all.min(), lon_all.max()
    k = math.cos(math.radians((S + N) / 2))
    span = max((N - S), (E - W) * k, 1e-3) * 1.1
    cx, cy = (W + E) / 2, (S + N) / 2
    size = 900
    img = Image.new("RGB", (size, size + 40), (250, 250, 250))
    d = ImageDraw.Draw(img)

    def xy(lat, lon):
        return size / 2 + (lon - cx) * k / span * size, size / 2 - (lat - cy) / span * size
    for fl in flights:
        if fl.lat.min() > N + span or fl.lat.max() < S - span:
            continue
        line = [xy(a_, b_) for a_, b_ in zip(fl.lat, fl.lon)]
        d.line(line, fill=(200, 200, 210), width=1)
    lo_s, hi_s = np.percentile(sc, 5), sc.max()
    for o in np.argsort(sc):
        v = (sc[o] - lo_s) / max(1e-9, hi_s - lo_s)
        col = tuple(int(x) for x in LUT[int(np.clip(v, 0, 1) * 230)])
        x, y = xy(la[o], lo[o])
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    for i, (a_, b_) in enumerate(pts[:5], 1):
        x, y = xy(a_, b_)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(0, 150, 0), width=3)
        d.text((x + 10, y - 8), str(i), fill=(0, 120, 0), font=font(16))
    d.text((10, size + 10), f"scored candidates (hotter = better), grey = flight tracks; width {span * 111.2:.1f} km",
           fill=(30, 30, 30), font=font(14))
    img.save(path)


def _plot_curves(groups, fits, flights_all, spec, c, elev, mic, path: Path):
    panels = []
    idx = {f.id: f for f in flights_all}
    for gi, g in enumerate(groups, 1):
        for k, ev in enumerate(g["events"]):
            fl = idx.get(ev["flight"]["id"])
            ft = fits[k]
            a, b = ft["span"]
            t0, t1 = max(spec.t[0], a - 15), min(spec.t[-1], b + 15)
            fr = spec.frames(t0, t1)
            lo, hi = ft["band"]
            bsel = (spec.f >= lo) & (spec.f <= hi)
            Z = spec.Z[np.ix_(fr, np.nonzero(bsel)[0])]
            A = _resize_max(Z.T[::-1], 220, 520)
            im = Image.fromarray(LUT[(np.clip(A / 6, 0, 1) * 255).astype(np.uint8)])
            d = ImageDraw.Draw(im)
            if fl is not None:
                tt = spec.t[fr]
                z = float(elev.sample(g["lat"], g["lon"])[0]) + mic
                ratio, _ = model_ratio(fl, g["lat"], g["lon"], z, tt + parse_time(g["recording_start"]), c)
                fm = ft["best"]["f0"] * ratio
                for tv, fv in zip(tt, fm):
                    if np.isfinite(fv) and lo <= fv <= hi:
                        x = (tv - t0) / (t1 - t0) * 520
                        y = (1 - (fv - lo) / (hi - lo)) * 220
                        d.point((x, y), fill=(255, 255, 255))
            d.text((4, 4), f"#{gi} {ft['name']} score {g['per_event'][k]:.2f}", fill=(255, 255, 255), font=font(13))
            panels.append(im)
    if not panels:
        return
    cols = len(fits)
    rows_n = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * 524, rows_n * 224), (0, 0, 0))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % cols) * 524, (i // cols) * 224))
    sheet.save(path)


# ---------------------------------------------------------------- lightning and thunder

def read_strikes(path):
    t, la, lo = [], [], []
    for row in csv.DictReader(open(path)):
        k = {x.lower().strip(): v for x, v in row.items() if x}
        tk = next((x for x in ("time", "timestamp", "datetime", "t", "utc") if x in k), None)
        lk = next((x for x in ("lat", "latitude") if x in k), None)
        ok_ = next((x for x in ("lon", "lng", "longitude", "long") if x in k), None)
        if not (tk and lk and ok_) or not k[tk]:
            continue
        t.append(parse_time(k[tk]))
        la.append(float(k[lk]))
        lo.append(float(k[ok_]))
    if not t:
        raise SystemExit(f"No strikes read from {path} (need time, lat, lon columns)")
    o = np.argsort(t)
    return np.array(t)[o], np.array(la)[o], np.array(lo)[o]


def _glm_sat(day: date, lon_c: float) -> list[str]:
    if lon_c < -106:
        return ["noaa-goes18" if day >= date(2023, 1, 4) else "noaa-goes17", "noaa-goes19", "noaa-goes16"]
    return ["noaa-goes19" if day >= date(2025, 4, 7) else "noaa-goes16", "noaa-goes18", "noaa-goes17"]


def cmd_lightning(a):
    import h5py
    bbox = bbox_arg(a.bbox)
    t0, t1 = window(a.date, a.from_, a.to)
    if bbox is None or t0 is None:
        raise SystemExit("lightning needs --bbox and --date (with --from/--to)")
    keys = []
    sats = [a.sat] if a.sat else _glm_sat(datetime.fromtimestamp(t0, timezone.utc).date(), (bbox[1] + bbox[3]) / 2)
    for bucket in sats:
        h = datetime.fromtimestamp(t0, timezone.utc).replace(minute=0, second=0, microsecond=0)
        while h.timestamp() <= t1:
            prefix = f"GLM-L2-LCFA/{h:%Y}/{h.timetuple().tm_yday:03d}/{h:%H}/"
            token = None
            while True:
                url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}" + (f"&continuation-token={token}" if token else "")
                xml = fetch_bytes(url.replace("+", "%2B").replace("=", "%3D", 0), a.proxy, 60).decode()
                for kk in re.findall(r"<Key>([^<]+)</Key>", xml):
                    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", kk)
                    if m:
                        ts = datetime(int(m[1]), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(m[2]) - 1, hours=int(m[3]), minutes=int(m[4]), seconds=int(m[5]))
                        if t0 - 20 <= ts.timestamp() <= t1:
                            keys.append(f"https://{bucket}.s3.amazonaws.com/{kk}")
                nt = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
                if not nt:
                    break
                from urllib.parse import quote
                token = quote(nt[1], safe="")
            h += timedelta(hours=1)
        if keys:
            break
    if not keys:
        raise SystemExit("No GLM files for that window (GLM covers the Americas and nearby oceans, from 2017)")
    log(f"GLM: {len(keys)} files from {keys[0].split('//')[1].split('.')[0]}…")

    def get(u):
        try:
            return fetch_bytes(u, a.proxy, 60)
        except RuntimeError:
            return None
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for raw in ex.map(get, keys):
            if raw is None:
                continue
            f = h5py.File(io.BytesIO(raw), "r")
            if "flash_lat" not in f or f["flash_lat"].shape[0] == 0:
                continue
            lat = f["flash_lat"][:].astype(float)
            lon = f["flash_lon"][:].astype(float)
            v = f["flash_time_offset_of_first_event"]
            r = v[:].view(np.uint16).astype(float) if v.attrs.get("_Unsigned") in (b"true", "true") else v[:].astype(float)
            r = r * float(v.attrs.get("scale_factor", [1])[0]) + float(v.attrs.get("add_offset", [0])[0])
            units = v.attrs["units"]
            units = units.decode() if isinstance(units, bytes) else str(units)
            base = parse_time(units.split("since", 1)[1].strip().replace(" ", "T") + "Z")
            en = f["flash_energy"]
            e = en[:].view(np.uint16).astype(float) * float(en.attrs.get("scale_factor", [1])[0]) + float(en.attrs.get("add_offset", [0])[0])
            m = (lat >= bbox[0]) & (lat <= bbox[2]) & (lon >= bbox[1]) & (lon <= bbox[3])
            for tt, la, lo, ee in zip(base + r[m], lat[m], lon[m], e[m]):
                if t0 <= tt <= t1:
                    rows.append((tt, la, lo, ee))
    rows.sort()
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "lat", "lon", "energy_j", "source"])
        for tt, la, lo, ee in rows:
            w.writerow([f"{iso(tt)[:-1]}.{int(round((tt % 1) * 1000)):03d}Z", round(la, 5), round(lo, 5), f"{ee:.3g}", "GLM"])
    print(json.dumps({"strikes": a.out, "flashes": len(rows), "files": len(keys),
                      "note": "GLM flash centroids: total lightning (cloud and ground), position error several km"}))


def cmd_thunder(a):
    from scipy.ndimage import gaussian_filter1d, percentile_filter
    bbox = bbox_arg(a.bbox)
    if not bbox:
        raise SystemExit("thunder needs --bbox")
    x = load_audio(a.audio, a.rate)
    sr = a.rate
    P = stft_power(x, sr, sr // 4, sr // 4, 20, 250)[2].sum(1)
    n1 = len(P) // 4
    e = P[:n1 * 4].reshape(-1, 4).mean(1)                      # 1-s energy, 20–250 Hz
    floor = percentile_filter(e, 20, size=61, mode="nearest")
    obs = gaussian_filter1d(np.maximum(e - floor * 1.5, 0) ** a.power, a.smooth)
    N = len(obs)
    if obs.max() <= 0:
        raise SystemExit("No low-frequency bursts above the background")
    obs = obs / np.linalg.norm(obs)
    from scipy.signal import find_peaks
    nb = len(find_peaks(obs, prominence=obs.max() * 0.08, distance=5)[0])
    st, sla, slo = read_strikes(a.strikes)
    t_from, t_to = window(a.date, a.from_, a.to)
    if t_from is None:
        t_from, t_to = st[0] - 60, st[-1]
    maxd = a.max_range / a.c + 30
    T_s = math.floor(t_from - 10)
    M = int(math.ceil(t_to + N + maxd - T_s)) + 1
    m = (st >= T_s - maxd) & (st <= T_s + M)
    st, sla, slo = st[m], sla[m], slo[m]
    w = np.ones(len(st))
    if a.balance:
        ex, ny = enu(sla, slo, (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        from scipy.spatial import cKDTree
        cnt = np.array([len(v) for v in cKDTree(np.c_[ex, ny]).query_ball_point(np.c_[ex, ny], 3000)])
        w = 1 / np.sqrt(cnt)
    min_strikes = a.min_strikes or max(3, int(round(0.5 * nb)))
    log(f"{len(st)} strikes in window; {nb} bursts heard; require ≥{min_strikes} strikes within {a.max_range / 1000:.0f} km")
    S, W_, Nn, E = bbox
    dlat = math.degrees(a.grid / R_EARTH)
    lats = np.arange(S, Nn + 1e-9, dlat)
    cells = []
    for la in lats:
        dlon = dlat / math.cos(math.radians(la))
        for lo in np.arange(W_, E + 1e-9, dlon):
            cells.append((la, lo))
    cells = np.array(cells)
    from scipy.spatial import cKDTree
    lat0, lon0 = (S + Nn) / 2, (W_ + E) / 2
    sx, sy = enu(sla, slo, lat0, lon0)
    cx, cy = enu(cells[:, 0], cells[:, 1], lat0, lon0)
    tree = cKDTree(np.c_[sx, sy])
    near = tree.query_ball_point(np.c_[cx, cy], a.max_range)
    L = 1 << int(math.ceil(math.log2(M + N + 1)))
    kern = np.zeros(L)
    kt = np.arange(0, 40)
    kern[:40] = np.exp(-kt / a.decay)
    Kf = np.fft.rfft(kern)
    Of = np.conj(np.fft.rfft(obs, L))
    Wf = np.fft.rfft(np.ones(N), L)
    k_lo = max(0, int(math.floor(t_from - T_s)))
    k_hi = min(M - N, int(math.ceil(t_to - T_s)))
    if k_hi < k_lo:
        raise SystemExit("Window shorter than needed; widen --from/--to")
    out = []
    t_start = _time.time()
    for ci, idx in enumerate(near):
        if len(idx) < min_strikes:
            continue
        idx = np.array(idx)
        hd = np.hypot(sx[idx] - cx[ci], sy[idx] - cy[ci])
        sl = np.sqrt(hd ** 2 + a.source_height ** 2)
        arr = st[idx] + sl / a.c - T_s
        o = np.sort(arr)
        if (np.searchsorted(o, o + N) - np.arange(len(o))).max() < min_strikes:
            continue
        amp = w[idx] * (a.source_height / np.maximum(sl, a.source_height)) ** 2
        ok = (arr >= 0) & (arr < M)
        imp = np.bincount(arr[ok].astype(int), weights=amp[ok], minlength=M)[:M]
        me = np.fft.irfft(np.fft.rfft(imp, L) * Kf, L)[:M]
        mm = gaussian_filter1d(np.maximum(me, 0) ** a.power, a.smooth)
        corr = np.fft.irfft(np.fft.rfft(mm, L) * Of, L)[:M - N + 1]
        energy = np.fft.irfft(np.fft.rfft(mm * mm, L) * Wf, L)[N - 1:M]          # Σ m² over each window
        nrm = np.sqrt(np.maximum(energy, 0))
        # windows with almost no modelled sound are noise / round-off, not evidence
        sc = np.where(nrm > 1e-3 * max(1e-30, nrm.max()), np.clip(corr / np.maximum(nrm, 1e-30), -1, 1), 0)[k_lo:k_hi + 1]
        if not len(sc):
            continue
        j = int(np.argmax(sc))
        out.append((float(sc[j]), float(cells[ci, 0]), float(cells[ci, 1]), T_s + k_lo + j, len(idx)))
        if len(out) % 2000 == 0:
            log(f"  {len(out)} cells scored, {_time.time() - t_start:.0f} s")
    if not out:
        raise SystemExit("No grid cell had enough strikes nearby; widen the box/window or lower --min-strikes")
    out.sort(key=lambda r: -r[0])
    scores = np.array([r[0] for r in out])
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    top = [dict(lat=round(r[1], 5), lon=round(r[2], 5), score=round(r[0], 4), recording_start=iso(r[3]), strikes_near=r[4])
           for r in out[:a.top]]
    res = dict(audio=str(a.audio), bursts_heard=nb, cells_scored=len(out), grid_m=a.grid,
               score_percentiles={p: round(float(np.percentile(scores, p)), 4) for p in (50, 90, 99)},
               best=top[0], top=top,
               read_me="cosine similarity 0–1 between the heard and modelled envelopes at the best start time; trust a "
                       "cluster that stands clear of the 99th percentile, then rerun on a smaller box with a finer grid")
    (outdir / "thunder.json").write_text(json.dumps(res, indent=1))
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r[2], r[1]]},
              "properties": {"score": round(r[0], 4), "start": iso(r[3])}} for r in out]
    (outdir / "thunder.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    rows = [dict(lat=r[1], lon=r[2], score=r[0]) for r in out]
    _plot_map(rows, [dict(lat=t["lat"], lon=t["lon"]) for t in top[:5]], [], outdir / "thunder_map.png")
    # observed vs modelled envelope at the best cell
    b = out[0]
    hd = np.hypot(*(np.array(enu(sla, slo, b[1], b[2]))))
    sl = np.sqrt(hd ** 2 + a.source_height ** 2)
    sel = hd <= a.max_range
    arr = st[sel] + sl[sel] / a.c - b[3]
    imp = np.bincount(np.clip(arr[(arr >= -40) & (arr < N)] + 40, 0, None).astype(int),
                      weights=(w[sel] * (a.source_height / np.maximum(sl[sel], a.source_height)) ** 2)[(arr >= -40) & (arr < N)],
                      minlength=N + 40)[:N + 40]
    me = np.convolve(imp, np.exp(-np.arange(40) / a.decay))[:N + 40][40:]
    mm = gaussian_filter1d(np.maximum(me, 0) ** a.power, a.smooth)
    img = Image.new("RGB", (1200, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for series, col in ((obs, (40, 90, 220)), (mm, (220, 60, 40))):
        if series.max() > 0:
            s = series / series.max()
            d.line([(10 + i / max(1, N - 1) * 1180, 280 - v * 250) for i, v in enumerate(s)], fill=col, width=2)
    d.text((12, 6), f"blue: heard 20-250 Hz envelope; red: modelled at {b[1]:.4f},{b[2]:.4f} starting {iso(b[3])}  (score {b[0]:.3f})",
           fill=(0, 0, 0), font=font(14))
    img.save(outdir / "thunder_best.png")
    print(json.dumps({k: res[k] for k in ("best", "score_percentiles", "cells_scored", "bursts_heard")}, indent=1))
    print(f"→ {outdir}/thunder.json, thunder.geojson, thunder_map.png, thunder_best.png")


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, rate=8000):
        p.add_argument("--rate", type=int, default=rate, help="decode sample rate (Hz)")
        return p

    p = sub.add_parser("extract", help="audio track → mono WAV")
    p.add_argument("video")
    p.add_argument("--out", default="audio.wav")
    p.add_argument("--rate", type=int, default=16000)
    p.set_defaults(func=cmd_extract)

    p = common(sub.add_parser("look", help="whitened spectrogram sheet(s)"))
    p.add_argument("audio")
    p.add_argument("--band", default="40,600", help="lo,hi Hz (aircraft fundamentals are usually 60–300 Hz)")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--row", type=float, help="seconds per row (default 300, or the range when shorter)")
    p.add_argument("--res", type=float, default=1.0, help="frequency resolution, Hz (window = 1/res s)")
    p.add_argument("--hop", type=float, default=0.25, help="seconds between frames")
    p.add_argument("--width", type=int, default=1800)
    p.add_argument("--height", type=int, default=360, help="pixels per row")
    p.add_argument("--contrast", type=float, default=6.0, help="z-score shown as white (lower = brighter)")
    p.add_argument("--out", default="spec.png")
    p.set_defaults(func=cmd_look)

    p = common(sub.add_parser("events", help="find Doppler sweeps, steady tones, hum and low-frequency bursts"))
    p.add_argument("audio")
    p.add_argument("--band", default="50,800")
    p.add_argument("--res", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=3.5, help="peak height on the 3-s smoothed spectrogram, in noise σ")
    p.add_argument("--min-dur", type=float, default=10.0, help="shortest track kept, s")
    p.add_argument("--burst-db", type=float, default=6.0, help="low-frequency burst threshold above the local floor, dB")
    p.add_argument("--temp", type=float, default=15.0, help="air temperature °C (speed of sound)")
    p.add_argument("--contrast", type=float, default=6.0)
    p.add_argument("--out", default="events.json")
    p.add_argument("--png", help="annotated sheet")
    p.set_defaults(func=cmd_events)

    p = common(sub.add_parser("fit", help="fit one Doppler curve"))
    p.add_argument("audio")
    p.add_argument("--start", type=float, help="seconds into the audio (include the whole pass)")
    p.add_argument("--end", type=float)
    p.add_argument("--band", help="lo,hi Hz around ONE harmonic of the pass")
    p.add_argument("--trace", help="CSV of hand-traced points: t_seconds,freq_hz[,weight]")
    p.add_argument("--name", help="event name (default e1)")
    p.add_argument("--res", type=float, default=1.0)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--min-z", type=float, default=1.0, help="weakest smoothed brightness that still counts as the curve")
    p.add_argument("--max-slope", type=float, help="fastest frequency fall to follow, Hz/s (default: 6 %% of the band top, ≥12)")
    p.add_argument("--temp", type=float, default=15.0, help="air temperature °C (speed of sound)")
    p.add_argument("--c", type=float, help="speed of sound m/s (overrides --temp)")
    p.add_argument("--speed-pad", type=float, default=10.0, help="m/s added to the speed range (wind, manoeuvres)")
    p.add_argument("--loose", action="store_true", help="wide search ranges (a manoeuvring pass); automatic above --loose-rms")
    p.add_argument("--loose-rms", type=float, default=5.0, help="fit residual (‰) above which the ranges are widened")
    p.add_argument("--contrast", type=float, default=6.0)
    p.add_argument("--out", help="default fit_<name>.json")
    p.set_defaults(func=cmd_fit)

    p = common(sub.add_parser("hum", help="mains hum: 50/60 Hz grid and ENF trace"))
    p.add_argument("audio")
    p.add_argument("--min-db", type=float, default=8.0, help="prominence needed to call a grid")
    p.add_argument("--csv", help="write the ENF trace (t_s, freq_hz)")
    p.add_argument("--reference", help="grid-frequency record CSV (time, frequency) to find when it was recorded")
    p.add_argument("--out", default="hum.json")
    p.set_defaults(func=cmd_hum)

    p = sub.add_parser("adsb", help="historical aircraft tracks for a day and box")
    p.add_argument("--date", help="UTC day YYYY-MM-DD (the archive is per UTC day)")
    p.add_argument("--from", dest="from_", help="HH:MM or ISO (default 00:00)")
    p.add_argument("--to", help="HH:MM or ISO (default 24:00)")
    p.add_argument("--bbox", required=True, help="S,W,N,E")
    p.add_argument("--margin", type=float, default=15000, help="metres added around the box")
    p.add_argument("--pad", type=float, default=900, help="seconds added around the window")
    p.add_argument("--tag", help="adsb.lol release tag, when the default name is not found")
    p.add_argument("--csv", help="use this CSV instead (id,time,lat,lon,alt_m|alt_ft[,gs_ms,type])")
    p.add_argument("--out", default="tracks.jsonl")
    p.add_argument("--proxy", help=PROXY_HELP)
    p.set_defaults(func=cmd_adsb)

    p = common(sub.add_parser("locate", help="aircraft search: funnel + modelled-curve scoring"))
    p.add_argument("audio")
    p.add_argument("--tracks", required=True)
    p.add_argument("--fit", action="append", required=True, help="fit JSON, one per event (repeat)")
    p.add_argument("--date")
    p.add_argument("--from", dest="from_")
    p.add_argument("--to")
    p.add_argument("--audio-start", help="UTC time of the recording's first sample, when known")
    p.add_argument("--clock-tol", type=float, default=30.0, help="± seconds on --audio-start")
    p.add_argument("--bbox", help="S,W,N,E search box")
    p.add_argument("--grid", type=float, default=200.0, help="cell size, m")
    p.add_argument("--types", default="nonjet", choices=["nonjet", "prop", "heli", "all"], help="aircraft that can be an event")
    p.add_argument("--silence", default="between", choices=["between", "whole", "off"])
    p.add_argument("--silence-types", default="nonjet", choices=["nonjet", "prop", "heli", "all"])
    p.add_argument("--audible", type=float, help="slant distance heard for sure, m (default: the farthest fitted event)")
    p.add_argument("--margin", type=float, default=30.0, help="s around each event where other passes are ignored")
    p.add_argument("--offset-tol", type=float, default=5.0, help="± s on the time between events")
    p.add_argument("--score", default="spectrogram", choices=["spectrogram", "trace"])
    p.add_argument("--t-sweep", type=float, default=3.0, help="± s shift tried on event 1")
    p.add_argument("--f-sweep", type=float, default=0.01, help="± fraction on each source frequency")
    p.add_argument("--max-score", type=int, default=20000, help="cap on candidates scored (subsampled per flight pair)")
    p.add_argument("--refine", type=int, default=6, help="top flight combinations refined on a finer grid")
    p.add_argument("--res", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=15.0)
    p.add_argument("--c", type=float)
    p.add_argument("--mic-height", type=float, default=2.0)
    p.add_argument("--dem-zoom", type=int, default=11)
    p.add_argument("--flat", action="store_true", help="ignore terrain (ground at 0 m)")
    p.add_argument("--out-dir", default="locate")
    p.add_argument("--proxy", help=PROXY_HELP)
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser("lightning", help="GOES GLM lightning flashes → CSV")
    p.add_argument("--date", required=True)
    p.add_argument("--from", dest="from_")
    p.add_argument("--to")
    p.add_argument("--bbox", required=True)
    p.add_argument("--sat", help="S3 bucket, e.g. noaa-goes19 (East) or noaa-goes18 (West)")
    p.add_argument("--out", default="strikes.csv")
    p.add_argument("--proxy", help=PROXY_HELP)
    p.set_defaults(func=cmd_lightning)

    p = common(sub.add_parser("thunder", help="thunder search against a strikes CSV"))
    p.add_argument("audio")
    p.add_argument("--strikes", required=True, help="CSV with time, lat, lon")
    p.add_argument("--bbox", required=True)
    p.add_argument("--date")
    p.add_argument("--from", dest="from_", help="earliest recording start (HH:MM or ISO)")
    p.add_argument("--to", help="latest recording start")
    p.add_argument("--grid", type=float, default=2000.0)
    p.add_argument("--max-range", type=float, default=25000.0, help="farthest thunder heard, m")
    p.add_argument("--source-height", type=float, default=2000.0, help="height of the sound source, m")
    p.add_argument("--c", type=float, default=343.0)
    p.add_argument("--decay", type=float, default=4.0, help="rumble decay time constant, s")
    p.add_argument("--smooth", type=float, default=4.0, help="Gaussian smoothing, s (≈ strike position error / 343 m/s)")
    p.add_argument("--power", type=float, default=0.25, help="exponent on energy: low values let every burst count, not just the loudest (0.5 = amplitude)")
    p.add_argument("--balance", action="store_true", help="down-weight strikes in dense clusters")
    p.add_argument("--min-strikes", type=int, help="strikes needed near a cell (default: half the bursts heard, ≥3)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out-dir", default="thunder")
    p.set_defaults(func=cmd_thunder)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
