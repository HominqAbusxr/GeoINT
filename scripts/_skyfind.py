"""Find where a photo was taken from its skyline: every place of an area scored by how much of the traced
skyline's shape the terrain there explains, with checks that say how far the best place can be trusted.

Used by `terrain.py find` (an area) and `terrain.py check` (one place). Method and how to read the results:
references/skyline.md.

The score. The trace (ridge.json: pixel points along the skyline) is projected through a candidate camera
(heading, tilt, roll, lens) onto the 360° skyline the terrain draws from the place. Each point's miss is
measured in pixels of a frame PLANE_W wide whatever the photo's size, and costs a Huber loss capped at
LOST_PX: a stretch traced over the wrong edge costs what a point off the terrain does, however far off it
lies, so it cannot outweigh the rest. The place's cost is set against what the trace costs against its own
best straight line: `explained` is the share of the trace's shape the terrain accounts for, 1 when every
bump lies on the terrain, 0 when the terrain does no better than a straight line (which any hillside does,
given a little roll). The score is the lesser of the whole trace's and its weakest parts' (the trace cut in
PARTS across the frame, the very weakest let go): a wrong place lays the easy stretches on some ridge and
misses the peaks. The trace's sharpest stretches (peaks and notches) are held to their own fit too.

What makes a result trustworthy is not a high score but a score far above what the area reaches by chance,
no rival elsewhere, neighbours that fit almost as well, and decoys of the trace (mirrored, upside down) that
fit much worse at the same place. `trust()` reads all of that into solid / check / weak with reasons.

Terrain: DEM tiles through terrain.py's `_fetch` (Mapterhorn: Copernicus GLO-30 worldwide, national lidar
where published), read at three levels of detail so a near ridge comes from 19 m cells and a range 40 km
out from coarser ones. Earth curvature and refraction as in terrain.py (k = 0.13).
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

R_EARTH = 6371008.8
K_REFRACTION = 0.13

# The plane residuals are measured on, whatever the photo's resolution.
PLANE_W = 1200.0
# The loss: square near the line, straight past HUBER_PX, level past LOST_PX.
HUBER_PX = 3.0
LOST_PX = 4 * HUBER_PX
LOST_LOSS = HUBER_PX * (LOST_PX - 0.5 * HUBER_PX)
# The score's parts: the trace cut in PARTS spans across the frame, the weakest LET_GO let go, the next
# WEAKEST averaged. A part a straight line already follows within PART_FLOOR_PX has no shape of its own.
PARTS = 4
LET_GO = 1
WEAKEST = 2
PART_FLOOR_PX = 5.0
LINE_FLOOR_PX = 1.5
MIN_POINTS = 6
# Peaks and notches held to their own fit: points whose stretch (DISTINCT_REACH of the trace's width either
# way) bends most off its straight line, the top DISTINCT_SHARE, past DISTINCT_FLOOR_PX.
DISTINCT_SHARE = 0.25
DISTINCT_REACH = 0.05
DISTINCT_FLOOR_PX = 6.0
KEY_COST = 0.5
# Verdict on a trace and its best place.
FLAT_PX = 5.0                 # 98% of the trace within this of a straight line: no shape to place
FLAT_SHARE = 0.98
NONE_BELOW = 0.25
LOOSE_BELOW = 0.5
AMBIGUOUS_MARGIN = 0.04       # a separate place this close to the best: two answers
AMBIGUOUS_SHARE = 0.75        # one place: a second heading explaining this share of the best fits as well
# Trust.
SEPARATE_M = 3000.0
RIVAL_MARGIN = 0.10
CHANCE_CLEAR = 0.15
DECOY_MARGIN = 0.15
SPOT_NEAR_M = 150.0
SPOT_SHARE = 0.75
SPOT_POINTS = 6
SPOT_RISE_M = 40.0            # … standing within this height of it
SPOT_SHARE_OF_RIDGE = 0.05    # … and within this share of the distance to the ridges its fit lies on
SPOT_NEAR_MIN = 50.0
SPOT_BLOCKED = 0.3            # … unless near ground hides this share of the view the best place looks out on
TURN_PX = 6.0
TURNS_MIN = 5
TILT_STEEP = 12.0
# Camera freedoms: tilt bounds by who held the camera, roll, lens.
TILTS = {"ground": (-20.0, 25.0), "drone": (-80.0, 25.0), "aircraft": (-80.0, 15.0)}
ROLL_MAX = 5.0
SLIDE_ROLL = 4.0
SLIDE_ROLL_PX = 3 * LOST_PX
LENS_REACH = 1.5
FOV_MIN, FOV_MAX = 1.0, 150.0
# Distances: a band said for the skyline is widened by BAND_SLACK; unless said otherwise, a point whose skyline
# is set by ground nearer than FLOOR_M (the eye's own slope, as the terrain's grain draws it) is lost.
BAND_SLACK = 1.33
FLOOR_M = 250.0
# Haze hides far ridges the terrain still holds: unless told the reach, each place is read cut at these
# distances too, and says which cut explained the trace.
AUTO_CUTS = (5000.0, 10000.0, 20000.0, 50000.0)
NEAR_M = 30.0
# Levels of detail: zoom z (256-px tiles; z13 ≈ 19 m at the equator) is read out to LOD_NEAR × 2^(13 - z).
LOD_NEAR = 11000.0
LEVEL_PX_MAX = 64_000_000
# Search stages.
KEEP_FIRST = 60
KEEP_NEXT = 30
LISTED = 10
POLISH_PLACES = 5
FILL_SHARE = 0.1              # a cell is filled in at this share of the distance to the ridges its fit lies on
FILL_BUDGET = 1.0             # … adding at most this many times the grid's points
GRID_MAX = 25000              # points a grid laid for near ridges (--dist) or a long lens may hold
LENS_WIDE = 60.0              # lenses narrower than this get a grid as many times denser as their width squares


# ------------------------------------------------------------------------------------------------ the trace

def _loss(px: np.ndarray) -> np.ndarray:
    a = np.abs(px)
    return np.minimum(np.where(a <= HUBER_PX, 0.5 * a * a, HUBER_PX * (a - 0.5 * HUBER_PX)), LOST_LOSS)


PART_FLOOR_LOSS = float(_loss(np.array([PART_FLOOR_PX]))[0])
LINE_FLOOR_LOSS = float(_loss(np.array([LINE_FLOOR_PX]))[0])


def off_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each point's distance from the trace's best straight line (any slope), fitted with the loss's own give."""
    A = np.column_stack([x, np.ones_like(x)])
    w = np.ones_like(x)
    off = y - y.mean()
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        off = y - A @ coef
        a = np.abs(off)
        w = np.sqrt(np.where(a <= HUBER_PX, 1.0, HUBER_PX / np.maximum(a, 1e-9)))
    return off


def distinct(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """The trace's sharpest stretches as a mask, or None when nothing bends DISTINCT_FLOOR_PX."""
    if x.size < 3 * MIN_POINTS:
        return None
    reach = DISTINCT_REACH * float(x.max() - x.min())
    lo = np.searchsorted(x, x - reach, side="left")
    hi = np.searchsorted(x, x + reach, side="right")

    def run(v):
        c = np.concatenate([[0.0], np.cumsum(v)])
        return c[hi] - c[lo]

    n = (hi - lo).astype(float)
    sx, sy, sxx, sxy = run(x), run(y), run(x * x), run(x * y)
    den = n * sxx - sx * sx
    ok = np.abs(den) > 1e-9
    slope = np.where(ok, (n * sxy - sx * sy) / np.where(ok, den, 1.0), 0.0)
    cut = (sy - slope * sx) / np.maximum(n, 1.0)
    bend = np.where(n >= 5, np.abs(y - (cut + slope * x)), 0.0)
    if bend.max() < DISTINCT_FLOOR_PX:
        return None
    a = np.searchsorted(x, x - reach / 2, side="left")
    b = np.searchsorted(x, x + reach / 2, side="right")
    near = np.array([bend[i:j].max() for i, j in zip(a, b)])
    top = near >= max(float(np.quantile(near, 1 - DISTINCT_SHARE)), DISTINCT_FLOOR_PX)
    return top if top.sum() >= MIN_POINTS else None


def turns(y: np.ndarray) -> int:
    """How many times the trace turns up or down by TURN_PX or more, read left to right (plane px)."""
    n, way, held = 0, 0, None
    for v in y:
        if held is None:
            held = v
        elif way >= 0 and v < held - TURN_PX:
            n += way > 0
            way, held = -1, v
        elif way <= 0 and v > held + TURN_PX:
            n += way < 0
            way, held = 1, v
        elif (way > 0 and v > held) or (way < 0 and v < held):
            held = v
    return int(n)


@dataclass
class Trace:
    """Skyline points traced on a photo, in pixels (x right, y down), with the photo's size and lens."""
    x: np.ndarray
    y: np.ndarray
    W: int
    H: int
    f0: float
    f_source: str = "assumed"
    photo: str | None = None
    key: np.ndarray | None = None          # points marked key (--key), else the sharpest stretches
    label: str = "as shot"

    @property
    def scale(self) -> float:
        return PLANE_W / self.W

    def hfov0(self) -> float:
        return 2 * math.degrees(math.atan(self.W / 2 / self.f0))

    def mirrored(self) -> "Trace":
        o = np.argsort(self.W - 1 - self.x, kind="stable")
        key = self.key[o] if self.key is not None else None
        return replace(self, x=(self.W - 1 - self.x)[o], y=self.y[o], key=key, label="mirrored")

    def flipped(self) -> "Trace":
        m = float(np.median(self.y))
        return replace(self, y=2 * m - self.y, key=None, label="upside down")

    def shape_px(self) -> float:
        s = self.scale
        return float(np.quantile(np.abs(off_line(self.x * s, -self.y * s)), FLAT_SHARE))

    def turns(self) -> int:
        return turns(-self.y * self.scale)


def load_trace(path: Path, f35: float = 26.0, key: str | None = None) -> Trace:
    """Read terrain.py ridge output (or any {"image_size": [W, H], "ridge": [[x, y], …]})."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = np.array(j.get("ridge") or j.get("points") or [], dtype=float).reshape(-1, 2)
    if len(pts) < 3 * MIN_POINTS:
        sys.exit(f"{path}: {len(pts)} skyline points; the search needs at least {3 * MIN_POINTS} (terrain.py ridge --step smaller)")
    W, H = (int(v) for v in j["image_size"])
    o = np.argsort(pts[:, 0], kind="stable")
    x, y = pts[o, 0], pts[o, 1]
    f0 = j.get("f0")
    src = j.get("f0_source", "assumed")
    if not f0:
        f0, src = f35 / 43.2666 * math.hypot(W, H), "assumed"
    k = None
    if key:
        k = np.zeros(x.size, bool)
        for part in key.split(","):
            a, b = (float(v) for v in part.split(":"))
            k |= (x >= min(a, b)) & (x <= max(a, b))
        if k.sum() < MIN_POINTS:
            sys.exit(f"--key covers {int(k.sum())} trace points; it needs at least {MIN_POINTS}")
    return Trace(x=x, y=y, W=W, H=H, f0=float(f0), f_source=src, photo=j.get("photo"), key=k)


@dataclass
class Sub:
    """The trace as one evaluation reads it: a subset of points with the straight-line baseline of each part."""
    trace: Trace
    u: np.ndarray            # right of the centre, px
    v: np.ndarray            # up from the centre, px
    xp: np.ndarray           # plane px across, for the parts
    line: np.ndarray         # each point's loss against the trace's straight line
    parts: np.ndarray        # (n_parts, n) masks
    key: np.ndarray | None
    memo: dict = field(default_factory=dict)

    def lens(self, hfov: float, tilts: tuple) -> tuple:
        """Through one lens, once per search: each point's direction on a 1° grid of tilts (a tilt is not a shift:
        off-centre points rise less and turn sideways as the camera tilts), how far it rises per degree of tilt
        and of roll, and the plane px a degree of elevation spans there."""
        k = (round(hfov, 6), tilts)
        if k not in self.memo:
            f = hfov_to_f(hfov, self.trace.W)
            u, v = self.u / f, self.v / f
            grid = np.arange(math.floor(tilts[0]), math.ceil(tilts[1]) + 1.0, 1.0)
            daz, el = directions(u[None, :], v[None, :], grid[:, None], 0.0)
            kt = directions(u, v, 1.0, 0.0)[1] - directions(u, v, 0.0, 0.0)[1]
            kr = directions(u, v, 0.0, 1.0)[1] - directions(u, v, 0.0, 0.0)[1]
            self.memo[k] = (grid, daz.astype(np.float32), el.astype(np.float32), kt.astype(np.float32),
                            kr.astype(np.float32), _per_degree(self, f).astype(np.float32))
        return self.memo[k]

    @classmethod
    def of(cls, tr: Trace, most: int | None = None) -> "Sub":
        idx = np.arange(tr.x.size)
        if most and tr.x.size > most:
            idx = np.unique(np.round(np.linspace(0, tr.x.size - 1, most)).astype(int))
        s = tr.scale
        x, y = tr.x[idx], tr.y[idx]
        line = _loss(off_line(tr.x * s, -tr.y * s))[idx]
        xp = x * s
        edges = np.linspace(xp.min(), xp.max(), PARTS + 1)
        parts = []
        for i in range(PARTS):
            m = (xp >= edges[i]) & ((xp < edges[i + 1]) if i < PARTS - 1 else (xp <= edges[i + 1]))
            if m.sum() >= MIN_POINTS:
                parts.append(m)
        key = tr.key[idx] if tr.key is not None else None
        if key is None:
            d = distinct(tr.x * s, -tr.y * s)
            key = d[idx] if d is not None else None
        if key is not None and key.sum() < 3:
            key = None
        return cls(tr, u=x - tr.W / 2, v=tr.H / 2 - y, xp=xp, line=line,
                   parts=np.array(parts) if parts else np.zeros((0, x.size), bool), key=key)

    def explained(self, loss: np.ndarray) -> np.ndarray:
        """Share of the trace's shape a fit explains, for loss (..., n_points) → (...)."""
        line = float(self.line.mean())
        # a trace with next to no shape can be explained no more than it has: a straight one, nothing
        whole = (line - loss.mean(axis=-1)) / max(line, LINE_FLOOR_LOSS)
        score = whole
        if len(self.parts):
            cnt = self.parts.sum(axis=1)
            pl = loss @ self.parts.T.astype(loss.dtype) / cnt
            base = np.maximum((self.line @ self.parts.T.astype(float)) / cnt, PART_FLOOR_LOSS)
            each = np.sort(1 - pl / base, axis=-1)
            k = each[..., LET_GO:LET_GO + WEAKEST] if each.shape[-1] > LET_GO else each
            score = np.minimum(score, k.mean(axis=-1))
        if self.key is not None:
            kl = max(float(self.line[self.key].mean()), PART_FLOOR_LOSS)
            ks = 1 - loss[..., self.key].mean(axis=-1) / kl
            score = score - KEY_COST * np.maximum(0, whole - ks)
        return np.clip(score, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- the camera

def directions(u, v, tilt, roll):
    """Azimuth from the heading and elevation (degrees) of image points u, v (right and up from the centre, in
    focal lengths), the camera pitched up by `tilt` and its picture turned clockwise by `roll` (as
    terrain.py view --roll). Heading is the outermost turn, so turning the camera only slides the azimuths."""
    r = np.radians(-np.asarray(roll, float))
    cr, sr = np.cos(r), np.sin(r)
    up = u * cr + v * sr
    vp = -u * sr + v * cr
    t = np.radians(np.asarray(tilt, float))
    ct, st = np.cos(t), np.sin(t)
    fwd = ct - vp * st
    return np.degrees(np.arctan2(up, fwd)), np.degrees(np.arctan2(vp * ct + st, np.hypot(up, fwd)))


def project(az, el, heading, tilt, roll):
    """Image u, v (focal lengths) of directions az/el (degrees); NaN behind the camera. Inverse of `directions`."""
    a = np.radians(np.asarray(az, float) - heading)
    e = np.radians(np.asarray(el, float))
    E, N, U = np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)
    t = math.radians(tilt)
    b = -N * math.sin(t) + U * math.cos(t)
    c = N * math.cos(t) + U * math.sin(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        up, vp = np.where(c > 1e-6, E / c, np.nan), np.where(c > 1e-6, b / c, np.nan)
    r = math.radians(-roll)
    return up * math.cos(r) - vp * math.sin(r), up * math.sin(r) + vp * math.cos(r)


def hfov_to_f(hfov: float, W: int) -> float:
    return W / 2 / math.tan(math.radians(hfov) / 2)


def _per_degree(sub: Sub, f: float) -> np.ndarray:
    """Plane px a degree of elevation spans at each point."""
    u, v = sub.u / f, sub.v / f
    return f * sub.trace.scale * math.pi / 180 * (1 + u * u + v * v) / np.sqrt(1 + u * u)


# ---------------------------------------------------------------------------------------------- the terrain

def _dest(lat0, lon0, az, dist):
    la1, lo1 = math.radians(lat0), math.radians(lon0)
    b = np.radians(az)[:, None]
    d = (np.asarray(dist, float) / R_EARTH)[None, :]
    la2 = np.arcsin(math.sin(la1) * np.cos(d) + math.cos(la1) * np.sin(d) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * np.sin(d) * math.cos(la1), np.cos(d) - math.sin(la1) * np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def _gpx(z: int, lat, lon):
    n = 256 * 2 ** z
    x = (np.asarray(lon, float) + 180) / 360 * n
    y = (1 - np.arcsinh(np.tan(np.radians(np.asarray(lat, float)))) / math.pi) / 2 * n
    return x, y


def _bilinear(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    np.clip(x0, 0, h.shape[1] - 2, out=x0)
    np.clip(y0, 0, h.shape[0] - 2, out=y0)
    a = h[y0, x0]
    b = h[y0, x0 + 1]
    c = h[y0 + 1, x0]
    d = h[y0 + 1, x0 + 1]
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


class Level:
    def __init__(self, z: int, bbox, fetch, cache: Path, proxy, threads: int = 8):
        s, w, n, e = bbox
        x0, y0 = _gpx(z, n, w)
        x1, y1 = _gpx(z, s, e)
        self.z = z
        self.tx0, self.ty0 = int(x0 // 256), int(y0 // 256)
        tx1, ty1 = int(x1 // 256), int(y1 // 256)
        nx, ny = tx1 - self.tx0 + 1, ty1 - self.ty0 + 1
        self.h = np.zeros((ny * 256, nx * 256), np.float32)
        jobs = [(x, y) for y in range(self.ty0, ty1 + 1) for x in range(self.tx0, tx1 + 1)]
        with ThreadPoolExecutor(threads) as ex:
            for (x, y), a in zip(jobs, ex.map(lambda t: fetch(z, t[0], t[1], cache, proxy), jobs)):
                self.h[(y - self.ty0) * 256:(y - self.ty0 + 1) * 256, (x - self.tx0) * 256:(x - self.tx0 + 1) * 256] = a
        self.ox, self.oy = self.tx0 * 256, self.ty0 * 256

    @staticmethod
    def pixels(z: int, bbox) -> int:
        s, w, n, e = bbox
        x0, y0 = _gpx(z, n, w)
        x1, y1 = _gpx(z, s, e)
        return int((x1 // 256 - x0 // 256 + 1) * (y1 // 256 - y0 // 256 + 1) * 65536)

    def sample(self, lat, lon):
        x, y = _gpx(self.z, lat, lon)
        return _bilinear(self.h, x - self.ox, y - self.oy)


def _grow(bbox, m: float):
    s, w, n, e = bbox
    dl = m / 110540
    dk = m / (111320 * math.cos(math.radians((s + n) / 2)))
    return (max(-85.0, s - dl), w - dk, min(85.0, n + dl), e + dk)


class Ground:
    """Heights for an area and all it can see, from DEM tiles at a few levels of detail: zoom z is read out to
    LOD_NEAR × 2^(13 − z), so a near ridge comes from the finest cells and a far range from coarser ones."""

    def __init__(self, bbox, reach: float, fetch, cache: Path, proxy, zoom: int = 13, log=print):
        self.levels: list[tuple[float, float, Level]] = []        # (from m, to m, level)
        lo, z = 0.0, zoom
        while lo < reach and z >= 8:
            hi = reach if z == 8 else min(reach, LOD_NEAR * 2 ** (13 - z) if z <= 13 else LOD_NEAR / 2 ** (z - 13))
            box = _grow(bbox, hi + 1500)
            while Level.pixels(z, box) > LEVEL_PX_MAX and z > 8:
                z -= 1
                hi = reach if z == 8 else min(reach, LOD_NEAR * 2 ** (13 - z))
                box = _grow(bbox, hi + 1500)
            t = time.time()
            lv = Level(z, box, fetch, cache, proxy)
            log(f"  terrain z{z} out to {hi / 1000:.0f} km: {lv.h.shape[1]}×{lv.h.shape[0]} px ({time.time() - t:.0f} s)")
            self.levels.append((lo, hi, lv))
            lo, z = hi, z - 1
        self.detail = 156543.03392 * math.cos(math.radians((bbox[0] + bbox[2]) / 2)) / 2 ** self.levels[0][2].z
        self._off: dict = {}

    def point(self, lat, lon) -> np.ndarray:
        return self.levels[0][2].sample(np.atleast_1d(lat), np.atleast_1d(lon))

    def rays(self, lat0: float, lon0: float, az: np.ndarray, dist: np.ndarray, layout: str) -> np.ndarray:
        out = np.empty((az.size, dist.size), np.float32)
        band = round(lat0, 2)
        last = len(self.levels) - 1
        for n_, (lo, hi, lv) in enumerate(self.levels):
            i0 = np.searchsorted(dist, lo, "left") if n_ else 0
            i1 = np.searchsorted(dist, hi, "right") if n_ < last else dist.size
            if i1 <= i0:
                continue
            key = (layout, lv.z, band, i0, i1)
            if key not in self._off:
                if len(self._off) > 64:
                    self._off.clear()
                p0 = math.radians(band)
                dd = dist[i0:i1][None, :]
                dy, dx = dd * np.cos(np.radians(az))[:, None], dd * np.sin(np.radians(az))[:, None]
                p = p0 + dy / R_EARTH
                lam = dx / (R_EARTH * np.cos((p0 + p) / 2))
                n = 256 * 2 ** lv.z
                self._off[key] = ((lam / (2 * math.pi) * n).astype(np.float32),
                                  (-(np.arcsinh(np.tan(p)) - math.asinh(math.tan(p0))) / (2 * math.pi) * n).astype(np.float32))
            ox, oy = self._off[key]
            x0, y0 = _gpx(lv.z, lat0, lon0)
            out[:, i0:i1] = _bilinear(lv.h, ox + np.float32(x0 - lv.ox), oy + np.float32(y0 - lv.oy))
        return out


class FuncGround:
    """Heights from a function of (lat, lon): for tests and synthetic checks."""

    def __init__(self, f, detail: float = 20.0):
        self.f = f
        self.detail = detail

    def point(self, lat, lon):
        return np.asarray(self.f(np.atleast_1d(lat), np.atleast_1d(lon)), np.float32)

    def rays(self, lat0, lon0, az, dist, layout):
        la, lo = _dest(lat0, lon0, az, dist)
        return np.asarray(self.f(la, lo), np.float32)


# ---------------------------------------------------------------------------------------------- the skyline

@dataclass
class Turn:
    """The skyline round a place, cut at a few reaches: elevation (degrees) and the distance of the ground
    that sets it, per azimuth column."""
    step: float
    sky: np.ndarray           # (n_cuts, n)
    far: np.ndarray           # (n_cuts, n)
    cuts: tuple
    ground: float
    eye: float

    def __post_init__(self):
        # the turn laid out three times over, so azimuths from -360° to 720° read it without wrapping
        n = self.sky.shape[1]
        self._s = np.concatenate([self.sky] * 3 + [self.sky[:, :1]], axis=1).astype(np.float32)
        self._d = np.diff(self._s, axis=1, append=self._s[:, -1:]).astype(np.float32)
        self._f = np.concatenate([self.far] * 3 + [self.far[:, :1]], axis=1).astype(np.float32)
        self._inv = np.float32(1.0 / self.step)
        self._top = 3 * n - 1
        # cuts that draw the same skyline as the nearer one add nothing to search: from a valley the far cuts
        # often see nothing new
        self.uniq = [0]
        for c in range(1, len(self.cuts)):
            a, b = self.sky[c], self.sky[self.uniq[-1]]
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any() or float(np.abs(a[m] - b[m]).max()) > 1e-3:
                self.uniq.append(c)
        self._su, self._du, self._fu = self._s[self.uniq], self._d[self.uniq], self._f[self.uniq]

    def at(self, az: np.ndarray, cut: int | None = None, uniq: bool = False):
        """Skyline elevation (interpolated) and the distance that sets it, at azimuths `az` (-360° to 720°): for
        every cut (leading axis; only the cuts in `uniq` with uniq=True), or for one."""
        i = (np.asarray(az, np.float32) + np.float32(360.0)) * self._inv
        i0 = np.clip(i.astype(np.int64), 0, self._top)
        fr = i - i0
        if cut is not None:
            return self._s[cut][i0] + self._d[cut][i0] * fr, self._f[cut][i0]
        if uniq:
            return self._su[:, i0] + self._du[:, i0] * fr, self._fu[:, i0]
        return self._s[:, i0] + self._d[:, i0] * fr, self._f[:, i0]


@dataclass
class Layout:
    name: str
    az_step: float
    ratio: float

    def __post_init__(self):
        self.az_step = 360.0 / round(360.0 / self.az_step)          # columns that close the turn exactly

    def dist(self, reach: float) -> np.ndarray:
        n = int(math.ceil(math.log(reach / NEAR_M) / math.log(1 + self.ratio))) + 1
        return NEAR_M * (1 + self.ratio) ** np.arange(n)


def turn(ground, lat: float, lon: float, eye: float, layout: Layout, cuts: tuple, sectors=None) -> Turn:
    n = int(round(360 / layout.az_step))
    az = np.arange(n) * layout.az_step
    keep = np.ones(n, bool) if not sectors else _in_sectors(az, sectors)
    dist = layout.dist(max(cuts))
    g0 = float(ground.point(lat, lon)[0])
    sky = np.full((len(cuts), n), np.nan, np.float32)
    far = np.full((len(cuts), n), np.nan, np.float32)
    if keep.any():
        h = ground.rays(lat, lon, az[keep], dist, layout.name + ("" if not sectors else str(sectors)))
        drop = dist ** 2 / (2 * R_EARTH) * (1 - K_REFRACTION)
        q = (h - (drop + g0 + eye)[None, :].astype(np.float32)) / dist[None, :].astype(np.float32)
        run = np.maximum.accumulate(q, axis=1)
        for c, reach in enumerate(cuts):
            i = int(np.searchsorted(dist, reach, "right")) - 1
            sky[c, keep] = np.degrees(np.arctan(run[:, i]))
            far[c, keep] = dist[np.argmax(q[:, :i + 1], axis=1)]
    return Turn(layout.az_step, sky, far, tuple(cuts), g0, g0 + eye)


def _in_sectors(az, sectors) -> np.ndarray:
    m = np.zeros(np.shape(az), bool)
    for mid, half in sectors:
        m |= np.abs((np.asarray(az) - mid + 180) % 360 - 180) <= half
    return m


# ---------------------------------------------------------------------------------------------- the search

@dataclass
class Ask:
    """What the search is told."""
    lenses_rough: list
    lenses_fine: list
    lens_bounds: tuple
    tilts: tuple = TILTS["ground"]
    roll_max: float = ROLL_MAX
    sectors: list | None = None
    cuts: tuple = AUTO_CUTS
    band: tuple | None = None          # metres the skyline may stand at, widened by BAND_SLACK
    floor: float = FLOOR_M
    eye: float = 1.6


@dataclass
class Fit:
    explained: float
    heading: float
    hfov: float
    tilt: float
    roll: float
    cut: float
    ridge_m: float = float("nan")      # median distance of the ground the matched points lie on
    lost: float = 0.0                  # share of points off the terrain
    trace: str = "as shot"

    def row(self) -> dict:
        return {"explained": round(self.explained, 4), "heading": round(self.heading % 360, 2),
                "hfov": round(self.hfov, 2), "tilt": round(self.tilt, 2), "roll": round(self.roll, 2),
                "reach_km": round(self.cut / 1000, 1), "ridge_km": round(self.ridge_m / 1000, 2) if np.isfinite(self.ridge_m) else None,
                "lost": round(self.lost, 3), "trace": self.trace}


def _lost_mask(far: np.ndarray, sky: np.ndarray, ask: Ask) -> np.ndarray:
    lost = ~np.isfinite(sky)
    if ask.band:
        lost |= (far < ask.band[0] / BAND_SLACK) | (far > ask.band[1] * BAND_SLACK)
    elif ask.floor > 0:
        lost |= far < ask.floor
    return lost


def _headings(step: float, sectors) -> np.ndarray:
    hs = np.arange(0, 360, step)
    if sectors:
        hs = hs[_in_sectors(hs, sectors)]
    return hs


def _nanmedian(a: np.ndarray) -> np.ndarray:
    """Median along the last axis ignoring NaN, to within a rank: the NaN filled alternately above and below
    everything, then one partition (numpy's nanmedian falls back to masked arrays on short rows: slow)."""
    n = a.shape[-1]
    fill = np.where(np.arange(n) % 2 == 0, np.float32(np.inf), np.float32(-np.inf))
    bad = np.isnan(a)
    b = np.where(bad, fill, a)
    m = np.partition(b, n // 2, axis=-1)[..., n // 2]
    return np.where((~bad).any(axis=-1) & np.isfinite(m), m, np.nan)


def _apart(fits: list, apart: float, most: int) -> list:
    kept = []
    for r in fits:
        if all(abs((r.heading - k.heading + 180) % 360 - 180) >= apart for k in kept):
            kept.append(r)
            if len(kept) >= most:
                break
    return kept


def slide(sub: Sub, tn: Turn, ask: Ask, lenses: list, step: float, most: int = 6) -> list[Fit]:
    """Every heading × lens × cut: the tilt laid as the median of what each point asks, the roll as the slope
    the rest asks once the tilt is laid, then the trace's residuals in plane px scored. Returns the best
    `most` fits whose headings stand apart."""
    hs = _headings(step, ask.sectors)
    if hs.size == 0:
        return []
    t_lo, t_hi = ask.tilts
    width = float(np.ptp(np.degrees(np.arctan(sub.u / hfov_to_f(lenses[len(lenses) // 2], sub.trace.W)))))
    apart = max(5.0, 0.5 * width)
    picks: list[Fit] = []
    for hfov in lenses:
        grid, daz_g, el_g, kt, kr, ppd = sub.lens(hfov, ask.tilts)
        i0 = int(np.clip(round(0.0 - grid[0]), 0, grid.size - 1))
        daz0, el0 = daz_g[i0], el_g[i0] + kt * (0.0 - grid[i0])
        # pass 1: the tilt each heading asks, read through a level camera
        sky, far = tn.at(hs[:, None] + daz0[None, :], uniq=True)            # (C, H, P), C the distinct cuts
        lost = _lost_mask(far, sky, ask)
        tilt = _nanmedian(np.where(lost, np.nan, (sky - el0) / kt))         # (C, H)
        tilt = np.clip(np.nan_to_num(tilt, nan=0.0), t_lo, t_hi).astype(np.float32)
        # pass 2: each heading's points read again through its own tilt, from the lens's grid of tilts
        ti = np.clip(np.rint(tilt - grid[0]).astype(np.int64), 0, grid.size - 1)
        daz = daz_g[ti]                                                      # (C, H, P)
        el = el_g[ti] + kt * (tilt - grid[ti].astype(np.float32))[..., None]
        A = hs[None, :, None].astype(np.float32) + daz
        sky2 = np.empty(el.shape, np.float32)
        far2 = np.empty(el.shape, np.float32)
        for c, cut in enumerate(tn.uniq):
            sky2[c], far2[c] = tn.at(A[c], cut)
        lost = _lost_mask(far2, sky2, ask)
        res = np.where(lost, 0.0, sky2 - el)
        # tilt touched up and roll laid together on the points already near the line, so a stretch traced
        # wrong cannot steer them
        w = ((~lost) & (np.abs(res * ppd) < SLIDE_ROLL_PX)).astype(np.float32)
        stt, srr, s_tr = (w * kt * kt).sum(-1), (w * kr * kr).sum(-1), (w * kt * kr).sum(-1)
        st, sr = (w * kt * res).sum(-1), (w * kr * res).sum(-1)
        det = stt * srr - s_tr * s_tr
        ok = det > 1e-9
        dt = np.where(ok, (st * srr - sr * s_tr) / np.where(ok, det, 1), 0.0)
        dr = np.where(ok, (sr * stt - st * s_tr) / np.where(ok, det, 1), 0.0)
        lim = min(SLIDE_ROLL, ask.roll_max)
        dr = np.clip(dr, -lim, lim)
        dt = np.clip(dt, t_lo - tilt, t_hi - tilt)
        px = (res - dt[..., None] * kt - dr[..., None] * kr) * ppd
        score = sub.explained(np.where(lost, LOST_LOSS, _loss(px)))         # (C, H)
        flat = score.ravel()
        k = min(flat.size, 200)
        order = np.argpartition(-flat, k - 1)[:k]
        order = order[np.argsort(-flat[order])]
        here: list[Fit] = []
        for j in order:
            c, i = divmod(int(j), hs.size)
            if any(abs((hs[i] - r.heading + 180) % 360 - 180) < apart for r in here):
                continue
            ok_pts = ~lost[c, i]
            here.append(Fit(float(flat[j]), float(hs[i]), hfov, float(tilt[c, i] + dt[c, i]), float(dr[c, i]), tn.cuts[tn.uniq[c]],
                            float(np.median(far2[c, i][ok_pts])) if ok_pts.any() else float("nan"),
                            float(lost[c, i].mean()), sub.trace.label))
            if len(here) >= most:
                break
        picks += here
    picks.sort(key=lambda r: -r.explained)
    return _apart(picks, apart, most)


def evaluate(sub: Sub, tn: Turn, ask: Ask, fit: Fit) -> tuple[float, np.ndarray, np.ndarray]:
    """One camera exactly: (explained, per-point plane px, lost mask) at the fit's cut."""
    c = tn.cuts.index(fit.cut)
    f = hfov_to_f(fit.hfov, sub.trace.W)
    daz, el = directions(sub.u / f, sub.v / f, fit.tilt, fit.roll)
    sky, far = tn.at(fit.heading + daz)
    lost = _lost_mask(far[c], sky[c], ask)
    px = (sky[c] - el) * _per_degree(sub, f)
    loss = np.where(lost, LOST_LOSS, _loss(px))
    return float(sub.explained(loss)), px, lost


def refine(sub: Sub, tn: Turn, ask: Ask, fit: Fit, step: float, rounds: int = 5) -> Fit:
    """Polish one fit exactly: heading, tilt, roll and lens eased together, every cut tried."""
    lo, hi = ask.lens_bounds
    sH, sT, sR, sF = step, 0.4, 1.0, 1.04
    cur = fit
    f_best = None
    for _ in range(rounds * 2):
        cands = []
        for dh in (-sH, 0, sH):
            for dt in (-sT, 0, sT):
                for dr in (-sR, 0, sR):
                    for k in (1 / sF, 1, sF):
                        hf = min(hi, max(lo, cur.hfov * k))
                        t = min(ask.tilts[1], max(ask.tilts[0], cur.tilt + dt))
                        r = max(-ask.roll_max, min(ask.roll_max, cur.roll + dr))
                        cands.append((cur.heading + dh, t, r, hf))
        H = np.array([c[0] for c in cands])
        Tt = np.array([c[1] for c in cands])
        Rr = np.array([c[2] for c in cands])
        Hf = np.array([c[3] for c in cands])
        F = sub.trace.W / 2 / np.tan(np.radians(Hf) / 2)
        u, v = sub.u[None, :] / F[:, None], sub.v[None, :] / F[:, None]
        daz, el = directions(u, v, Tt[:, None], Rr[:, None])
        sky, far = tn.at(H[:, None] + daz)                                   # (C, K, P)
        lost = _lost_mask(far, sky, ask)
        ppd = F[:, None] * sub.trace.scale * math.pi / 180 * (1 + u * u + v * v) / np.sqrt(1 + u * u)
        loss = np.where(lost, LOST_LOSS, _loss((sky - el[None]) * ppd[None]))
        score = sub.explained(loss)                                          # (C, K)
        c, k = np.unravel_index(int(np.argmax(score)), score.shape)
        nf = Fit(float(score[c, k]), float(H[k]), float(Hf[k]), float(Tt[k]), float(Rr[k]), tn.cuts[c],
                 float(np.median(far[c, k][~lost[c, k]])) if (~lost[c, k]).any() else float("nan"),
                 float(lost[c, k].mean()), sub.trace.label)
        if f_best is not None and nf.explained <= f_best.explained + 1e-6:
            sH, sT, sR, sF = sH / 2, sT / 2, sR / 2, 1 + (sF - 1) / 2
        if f_best is None or nf.explained > f_best.explained:
            f_best = nf
        cur = f_best
        if sH < 0.01:
            break
    return f_best


@dataclass
class Scorer:
    """Scores places for one search: the trace sets, the layouts and what the search was told."""
    ground: object
    traces: list
    ask: Ask
    rough_layout: Layout
    fine_layout: Layout
    rough_step: float
    fine_step: float
    rough_subs: list = field(default_factory=list)
    fine_subs: list = field(default_factory=list)

    def __post_init__(self):
        self.rough_subs = [Sub.of(t, 80) for t in self.traces]
        self.fine_subs = [(Sub.of(t, 120), Sub.of(t, 240)) for t in self.traces]

    def sectors_for_turn(self):
        if not self.ask.sectors:
            return None
        pad = max(self.ask.lens_bounds[1], 10.0) / 2 + 6
        return tuple((m, min(180.0, h + pad)) for m, h in self.ask.sectors)

    def turn(self, lat: float, lon: float, fine: bool) -> Turn:
        return turn(self.ground, lat, lon, self.ask.eye, self.fine_layout if fine else self.rough_layout,
                    self.ask.cuts, self.sectors_for_turn())

    def rough(self, lat: float, lon: float) -> Fit | None:
        tn = self.turn(lat, lon, False)
        fits = [f for s in self.rough_subs for f in slide(s, tn, self.ask, self.ask.lenses_rough, self.rough_step, most=1)]
        return max(fits, key=lambda r: r.explained) if fits else None

    def full(self, lat: float, lon: float, subs=None, most: int = 3, tn: Turn | None = None) -> tuple[list[Fit], Turn]:
        """Every heading searched finely, the best few polished exactly; returns fits at headings apart (or of
        another trace: the mirrored one of a selfie)."""
        tn = tn or self.turn(lat, lon, True)
        out = []
        for slid, whole in (subs or self.fine_subs):
            for f in slide(slid, tn, self.ask, self.ask.lenses_fine, self.fine_step, most=4):
                out.append(refine(whole, tn, self.ask, f, self.fine_step))
        out.sort(key=lambda r: -r.explained)
        kept = []
        for f in out:
            if all(abs((f.heading - k.heading + 180) % 360 - 180) >= 5 or f.trace != k.trace for k in kept):
                kept.append(f)
        return kept[:most], tn

    def decoy(self, lat: float, lon: float, tn: Turn | None = None) -> float:
        """What the best decoy of the trace (upside down, and mirrored unless a selfie's mirror is already
        searched) explains at this place. A real place explains its own trace far better than its decoys."""
        subs = []
        selfie = any(t.label == "mirrored" for t in self.traces)
        for t in self.traces:
            if t.label == "mirrored":
                continue
            for d in ([t.flipped()] + ([] if selfie else [t.mirrored()])):
                subs.append((Sub.of(d, 120), Sub.of(d, 240)))
        fits, _ = self.full(lat, lon, subs=subs, most=1, tn=tn)
        return fits[0].explained if fits else 0.0


# ------------------------------------------------------------------------------------------------- the area

def hex_grid(center, half_x: float, half_y: float, spacing: float, inside) -> list[tuple[float, float]]:
    lat0, lon0 = center
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540.0
    pts = []
    dy = spacing * math.sqrt(3) / 2
    for j, y in enumerate(np.arange(-half_y, half_y + 1e-6, dy)):
        for x in np.arange(-half_x, half_x + 1e-6, spacing) + (spacing / 2 if j % 2 else 0.0):
            la, lo = lat0 + y / ky, lon0 + x / kx
            if inside(la, lo):
                pts.append((la, lo))
    return pts


def patch(p, spacing: float, radius: float) -> list[tuple[float, float]]:
    """A small hex patch of this spacing about p, out to radius."""
    lat0, lon0 = p
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540.0
    out = []
    n = int(math.ceil(radius / spacing)) + 1
    dy = spacing * math.sqrt(3) / 2
    for j in range(-n, n + 1):
        off = spacing / 2 if j % 2 else 0.0
        for i in range(-n, n + 1):
            x, y = i * spacing + off, j * dy
            if 0 < x * x + y * y <= radius * radius * 1.0001:
                out.append((lat0 + y / ky, lon0 + x / kx))
    return out


def dist_m(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(d))


LANDFORMS = ("summit", "slope", "valley", "plain", "water")


def landform(ground, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Where each point stands: summit (or ridge), slope, valley, plain, or water (no relief at all)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    kx = 111320 * np.cos(np.radians(lat))[:, None]
    ang = np.radians(np.arange(16) * 22.5)

    def ring(r, k=16):
        a = ang[:: 16 // k]
        return ground.point((lat[:, None] + r * np.cos(a)[None, :] / 110540).ravel(),
                            (lon[:, None] + r * np.sin(a)[None, :] / kx).ravel()).reshape(lat.size, -1)

    h0 = ground.point(lat, lon)
    w = ring(150, 8)
    local = np.concatenate([ring(200), ring(400)], axis=1)
    around = np.concatenate([ring(600), ring(1500)], axis=1)
    share = (local < h0[:, None]).mean(axis=1)
    kind = np.where(share >= 0.75, "summit", np.where(share <= 0.25, "valley", "slope")).astype(object)
    flat = (np.ptp(local, axis=1) < 8) | (np.ptp(around, axis=1) < 30)
    kind[flat] = "plain"
    water = np.maximum(np.ptp(w, axis=1), np.abs(w - h0[:, None]).max(axis=1)) < 0.05
    kind[water] = "water"
    return kind


def _parse_sectors(facing: str | None, heading: str | None):
    if heading:
        a, b = (float(v) for v in heading.split(":"))
        width = (b - a) % 360 or 360
        return [((a + width / 2) % 360, width / 2)]
    if not facing:
        return None
    names = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
    out = []
    for w in facing.upper().replace(" ", "").split(","):
        if w not in names:
            sys.exit(f"--facing takes compass directions N,NE,E,SE,S,SW,W,NW (got {w!r})")
        out.append((float(names[w]), 22.5))
    return out


def _lenses(trace: Trace, hfov: str | None):
    """Lens widths to try (rough, fine) and the bounds the polish may ease within."""
    if hfov:
        b = [float(v) for v in hfov.split(":")]
        lo, hi = (b[0], b[0]) if len(b) == 1 else (min(b), max(b))
        if lo == hi:
            lo, hi = lo / 1.04, hi * 1.04
    elif trace.f_source in ("exif", "arg"):
        h0 = trace.hfov0()
        lo, hi = h0 / 1.08, h0 * 1.08
    else:
        h0 = trace.hfov0()
        lo, hi = h0 / LENS_REACH, h0 * LENS_REACH
    lo, hi = max(FOV_MIN, lo), min(FOV_MAX, hi)

    def ladder(ratio):
        if hi / lo <= ratio:
            return [math.sqrt(lo * hi)]
        n = int(math.ceil(math.log(hi / lo) / math.log(ratio))) + 1
        return list(np.geomspace(lo, hi, n))

    return ladder(1.25), ladder(1.08), (lo, hi)


# ------------------------------------------------------------------------------------------------- the verdict

def trust(best: dict, places: list, chance: float | None, around: float | None, trace: Trace, ask: Ask,
          lens_guessed: bool, mode: str, spot_m: float = SPOT_NEAR_M) -> tuple[str, str, list]:
    """(verdict, level, reasons). verdict: match / loose / ambiguous / none / flat; level: solid / check / weak.
    `around`: what the ground SPOT_NEAR_M round the best place explains, as a share of what the best explains
    on the same (rough) reading. A verdict naming no single place is weak; one that does is solid with
    nothing against it, worth a check with one thing, weak with two or more."""
    reasons = []
    e = best["explained"]
    rivals = [p for p in places[1:] if p["from_best_m"] >= SEPARATE_M and p["explained"] >= e - RIVAL_MARGIN]
    if trace.shape_px() < FLAT_PX:
        verdict = "flat"
    elif e < NONE_BELOW:
        verdict = "none"
    elif any(p["explained"] >= e - AMBIGUOUS_MARGIN for p in rivals):
        verdict = "ambiguous"
    elif e < LOOSE_BELOW:
        verdict = "loose"
    else:
        verdict = "match"
    if rivals:
        near = min(p["from_best_m"] for p in rivals)
        reasons.append(f"{len(rivals)} other place{'s' if len(rivals) > 1 else ''}, the nearest {near / 1000:.0f} km away, explain{'s' if len(rivals) == 1 else ''} nearly as much")
    if chance is not None and e - chance < CHANCE_CLEAR:
        reasons.append(f"barely above what the area fits by chance ({chance:.0%})")
    if around is not None and around < SPOT_SHARE:
        reasons.append(f"the ground {spot_m:.0f} m around explains much less ({around:.0%} of the best): a lone spike")
    n = trace.turns()
    if n < TURNS_MIN:
        reasons.append(f"the trace turns only {n} time{'s' if n != 1 else ''}: a plain shape fits many places")
    lo, hi = ask.lens_bounds
    if lens_guessed and best["hfov"] <= lo * 1.05:
        reasons.append(f"the lens found ({best['hfov']:.0f}°) is the narrowest tried")
    elif lens_guessed and best["hfov"] >= hi / 1.05:
        reasons.append(f"the lens found ({best['hfov']:.0f}°) is the widest tried")
    if abs(best["roll"]) >= ask.roll_max - 0.2:
        reasons.append(f"the camera had to roll {abs(best['roll']):.0f}°, the most allowed")
    if mode == "ground" and abs(best["tilt"]) >= TILT_STEEP:
        reasons.append(f"the camera tilts {abs(best['tilt']):.0f}° {'down' if best['tilt'] < 0 else 'up'}")
    if best.get("decoy") is not None and best["decoy"] >= e - DECOY_MARGIN:
        reasons.append(f"a decoy of the trace fits there nearly as well ({best['decoy']:.0%})")
    level = "weak" if verdict != "match" or len(reasons) >= 2 else ("check" if reasons else "solid")
    return verdict, level, reasons


# ------------------------------------------------------------------------------------------------- pictures

def _font(size):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def overlay(photo: Path | None, trace: Trace, fit: Fit, tn: Turn, ask: Ask, width: int, title: str) -> Image.Image:
    """The photo (or grey) with the place's skyline drawn through the fitted camera (red) and the trace (green
    where it lies within a few pixels, orange where it misses)."""
    W, H = trace.W, trace.H
    s = width / W
    if photo and Path(photo).exists():
        im = Image.open(photo).convert("RGB").resize((width, int(H * s)))
    else:
        im = Image.new("RGB", (width, int(H * s)), (110, 110, 110))
    d = ImageDraw.Draw(im)
    f = hfov_to_f(fit.hfov, W)
    span = fit.hfov * 0.65 + 2
    az = fit.heading + np.linspace(-span, span, 1400)
    c = tn.cuts.index(fit.cut)
    sky, _ = tn.at(az)
    u, v = project(az, sky[c], fit.heading, fit.tilt, fit.roll)
    x, y = W / 2 + u * f, H / 2 - v * f
    seg = []
    for xi, yi in zip(x, y):
        if np.isfinite(xi) and -W < xi < 2 * W:
            seg.append((xi * s, yi * s))
        elif len(seg) > 1:
            d.line(seg, fill=(255, 40, 40), width=2)
            seg = []
        else:
            seg = []
    if len(seg) > 1:
        d.line(seg, fill=(255, 40, 40), width=2)
    sub = Sub.of(trace)
    _, px, lost = evaluate(sub, tn, ask, fit)
    r = max(2, int(width / 260))
    for xi, yi, p, l in zip(trace.x, trace.y, px, lost):
        col = (60, 230, 90) if (not l and abs(p) <= 2 * HUBER_PX) else (255, 170, 0)
        d.ellipse([xi * s - r, yi * s - r, xi * s + r, yi * s + r], fill=col)
    out = Image.new("RGB", (width, im.height + 36), (0, 0, 0))        # the title above the picture, not over its skyline
    out.paste(im, (0, 36))
    ImageDraw.Draw(out).text((5, 2), title, fill=(255, 235, 90), font=_font(13))
    return out


def heatmap(ground, scored: list, bbox, places: list, truth, out: Path, chance: float | None, width: int = 900) -> None:
    s, w, n, e = bbox
    kx = 111320 * math.cos(math.radians((s + n) / 2))
    ar = (n - s) * 110540 / ((e - w) * kx)
    W, H = width, max(200, int(width * ar))
    la = np.linspace(n, s, H)
    lo = np.linspace(w, e, W)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    try:
        h = ground.point(LA.ravel(), LO.ravel()).reshape(H, W).astype(float)
        cell = (e - w) * kx / W
        gy, gx = np.gradient(h, cell)
        shade = np.clip(0.55 + 0.45 * (-gx * 0.7 + gy * 0.7) / np.sqrt(1 + gx * gx + gy * gy), 0, 1)
    except Exception:  # noqa: BLE001
        shade = np.full((H, W), 0.6)
    img = (np.stack([shade] * 3, -1) * 200 + 30).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")

    def xy(p):
        return (p[1] - w) / (e - w) * (W - 1), (n - p[0]) / (n - s) * (H - 1)

    def col(v):
        v = min(1.0, max(0.0, v))
        stops = [(0.0, (40, 30, 90)), (0.35, (40, 120, 180)), (0.6, (60, 190, 120)), (0.8, (240, 220, 60)), (1.0, (255, 80, 40))]
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if v <= b:
                t = (v - a) / (b - a)
                return tuple(int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
        return stops[-1][1]

    for p in sorted(scored, key=lambda r: (-r["spacing"], r["explained"])):
        x, y = xy((p["lat"], p["lon"]))
        r = max(1.5, p["spacing"] / ((e - w) * kx) * W * 0.55)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col(p["explained"]) + (220,))
    f = _font(15)
    for i, p in enumerate(places[:9]):
        x, y = xy((p["lat"], p["lon"]))
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 255, 255, 255), width=2)
        d.text((x + 8, y - 8), str(i + 1), fill=(255, 255, 255, 255), font=f)
    if truth:
        x, y = xy(truth)
        d.regular_polygon((x, y, 9), 5, fill=(255, 0, 200, 255))
    d.rectangle([0, 0, W, 22], fill=(0, 0, 0, 200))
    d.text((5, 3), "explained: dark 0% → blue → green → yellow → red 100%" + (f"   chance fit {chance:.0%}" if chance is not None else "")
           + ("   ★ truth" if truth else ""), fill=(255, 255, 255, 255), font=_font(13))
    im.save(out)


# ------------------------------------------------------------------------------------------------- setup

def _ask(args, trace: Trace) -> tuple[Ask, bool]:
    rough, fine, bounds = _lenses(trace, args.hfov)
    tilts = TILTS[args.mode]
    if args.tilt:
        a, b = (float(v) for v in args.tilt.split(":"))
        tilts = (min(a, b), max(a, b))
    if args.clear and args.reach == "auto":
        cuts = (max(AUTO_CUTS),)
    elif args.reach == "auto":
        cuts = AUTO_CUTS
    else:
        r = float(args.reach) * 1000
        cuts = (r,) if args.clear else (r, 2 * r)
    band = None
    if args.dist:
        a, b = (float(v) * 1000 for v in args.dist.split(":"))
        band = (min(a, b), max(a, b))
        top = min(band[1] * BAND_SLACK, 100000.0)
        # the haze cuts that fall inside the band still apply: a far band can be hidden past any of them
        cuts = tuple(c for c in AUTO_CUTS if band[0] <= c < top) + (top,)
    eye = args.eye + 3.0 * (args.floors or 0)
    ask = Ask(lenses_rough=rough, lenses_fine=fine, lens_bounds=bounds, tilts=tilts, roll_max=args.roll_max,
              sectors=_parse_sectors(args.facing, args.heading), cuts=tuple(sorted(cuts)), band=band,
              floor=args.floor, eye=eye)
    guessed = not (args.hfov or trace.f_source in ("exif", "arg"))
    return ask, guessed


def _scorer(args, trace: Trace, ask: Ask, ground) -> Scorer:
    traces = [trace] + ([trace.mirrored()] if args.selfie else [])
    lo = ask.lens_bounds[0]
    rough_step = float(np.clip(lo / 120, 0.1, 0.5))
    fine_step = float(np.clip(lo / 400, 0.02, 0.15))
    return Scorer(ground, traces, ask, Layout("rough", rough_step, 0.025), Layout("fine", fine_step, 0.008),
                  rough_step, max(fine_step, float(np.clip(lo / 240, 0.05, 0.5))))


def _view_cmd(p: dict, cut: float) -> str:
    return (f"terrain.py view --at {p['lat']:.5f},{p['lon']:.5f} --heading {p['heading']:.1f} --hfov {p['hfov']:.1f} "
            f"--pitch {p['tilt']:.1f} --roll {p['roll']:.1f} --range {cut:.0f} --zoom 13 --peaks --photo <photo> --overlay --out view.png")


# ------------------------------------------------------------------------------------------------- workers

_W: dict = {}


def _worker_init(spec: dict) -> None:
    ground = Ground(spec["bbox"], spec["reach"], spec["fetch"], spec["cache"], spec["proxy"], spec["zoom"], log=lambda *a: None)
    _W["sc"] = Scorer(ground, spec["traces"], spec["ask"], spec["rough_layout"], spec["fine_layout"],
                      spec["rough_step"], spec["fine_step"])


def _work(sc: Scorer, kind: str, p) -> object:
    if kind == "rough":
        return sc.rough(*p)
    if kind == "full":
        fits, tn = sc.full(*p, most=2)
        return fits, tn.ground
    return sc.decoy(*p)


def _worker(job):
    kind, points = job
    return [_work(_W["sc"], kind, p) for p in points]


class Pool:
    """Scores points on worker processes, each with its own copy of the terrain read from the tile cache the main
    process filled (the work holds Python's lock, so threads would not help), or here with one worker."""

    def __init__(self, sc: Scorer, spec: dict | None, workers: int):
        self.sc, self.ex, self.n = sc, None, 1
        if workers > 1 and spec is not None:
            import multiprocessing as mp
            import os
            from concurrent.futures import ProcessPoolExecutor
            if spec.get("dem"):
                os.environ["GEOINT_DEM"] = spec["dem"]                    # the workers read the same terrain source
            self.ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init, initargs=(spec,))
            self.n = workers

    def map(self, kind: str, points: list, progress=None) -> list:
        if not self.ex:
            out = []
            for i, p in enumerate(points):
                out.append(_work(self.sc, kind, p))
                if progress:
                    progress(i + 1)
            return out
        size = max(1, min(24 if kind == "rough" else 2, len(points) // (self.n * 3) or 1))
        out = []
        for r in self.ex.map(_worker, [(kind, points[i:i + size]) for i in range(0, len(points), size)]):
            out += r
            if progress:
                progress(len(out))
        return out

    def close(self) -> None:
        if self.ex:
            self.ex.shutdown(cancel_futures=True)


def _workers(args) -> int:
    import os
    n = getattr(args, "workers", 0) or 0
    return n if n > 0 else max(1, min(8, (os.cpu_count() or 2) - 2))


# ------------------------------------------------------------------------------------------------- commands

def cmd_find(args, fetch, ground=None) -> None:
    log = (lambda *a: print(*a, file=sys.stderr))
    trace = load_trace(args.ridge, args.f35, args.key)
    ask, guessed = _ask(args, trace)
    if args.bbox:
        b = [float(v) for v in args.bbox.split(",")]
        s, w, n, e = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
        center = ((s + n) / 2, (w + e) / 2)
        half_x = (e - w) / 2 * 111320 * math.cos(math.radians(center[0]))
        half_y = (n - s) / 2 * 110540

        def inside(la, lo):
            return s <= la <= n and w <= lo <= e
    else:
        if not (args.at and args.radius):
            sys.exit("give the area: --bbox s,w,n,e, or --at lat,lon with --radius m")
        center = tuple(float(v) for v in args.at.split(","))
        half_x = half_y = args.radius
        s, n = center[0] - args.radius / 110540, center[0] + args.radius / 110540
        kx = 111320 * math.cos(math.radians(center[0]))
        w, e = center[1] - args.radius / kx, center[1] + args.radius / kx

        def inside(la, lo):
            return dist_m(center, (la, lo)) <= args.radius
    area_km2 = 4 * half_x * half_y / 1e6 * (1 if args.bbox else math.pi / 4)
    # a long lens reads the parallax between near and far ridges in more pixels, so its score peak is narrower: a
    # 20° lens has the Matterhorn and the ranges behind it slide apart by tens of pixels over a few hundred metres
    zoom = max(1.0, (LENS_WIDE / ask.lens_bounds[0]) ** 2)
    budget = min(GRID_MAX, (900 if args.quick else 1600) * zoom)
    spacing = args.spacing or float(np.clip(math.sqrt(area_km2 * 1e6 / budget), 60, 2000))
    if ask.band and not args.spacing:
        # near ridges change over little ground: lay the grid at a tenth of their distance, within a budget of points
        # (the filling-in and the refinement go finer round the best)
        spacing = min(spacing, max(40.0, 0.1 * ask.band[0], math.sqrt(area_km2 * 1e6 / GRID_MAX)))
    shp, nt = trace.shape_px(), trace.turns()
    log(f"trace: {trace.x.size} points, {nt} turns, shape {shp:.1f} px (of a {PLANE_W:.0f}-px frame); lens "
        f"{ask.lens_bounds[0]:.0f}–{ask.lens_bounds[1]:.0f}° ({'guessed' if guessed else 'given'}); tilt {ask.tilts[0]:.0f}…{ask.tilts[1]:.0f}°; "
        f"reach {', '.join(f'{c / 1000:g}' for c in ask.cuts)} km; area {area_km2:.0f} km², grid {spacing:.0f} m")
    if shp < FLAT_PX:
        log("warning: the trace is nearly straight: no place can be told from another by it (verdict will be 'flat')")
    t0 = time.time()
    bbox = (s, w, n, e)
    spec = None
    if ground is None:
        ground = Ground(bbox, max(ask.cuts), fetch, args.cache, args.proxy, args.zoom, log)
        spec = {"bbox": bbox, "reach": max(ask.cuts), "fetch": fetch, "cache": args.cache, "proxy": args.proxy, "zoom": args.zoom,
                "dem": getattr(args, "dem", None)}
    fine = max(25.0, 2 * ground.detail)
    sc = _scorer(args, trace, ask, ground)
    if spec:
        spec.update(traces=sc.traces, ask=ask, rough_layout=sc.rough_layout, fine_layout=sc.fine_layout,
                    rough_step=sc.rough_step, fine_step=sc.fine_step)
    nw = _workers(args) if spec else 1
    pool = Pool(sc, spec, nw)
    try:
        _find(args, log, trace, ask, guessed, ground, sc, pool, nw, bbox, center, half_x, half_y, inside, spacing, fine,
              area_km2, shp, nt, t0)
    finally:
        pool.close()


def _find(args, log, trace, ask, guessed, ground, sc, pool, nw, bbox, center, half_x, half_y, inside, spacing, fine,
          area_km2, shp, nt, t0) -> None:
    s, w, n, e = bbox
    pts = hex_grid(center, half_x, half_y, spacing, inside)
    kinds = landform(ground, np.array([p[0] for p in pts]), np.array([p[1] for p in pts])) if pts else np.array([])
    allowed = set((args.stands or "summit,slope,valley,plain").split(","))
    bad = allowed - set(LANDFORMS)
    if bad:
        sys.exit(f"--stands takes {','.join(LANDFORMS)} (got {','.join(sorted(bad))})")
    pts = [p for p, k in zip(pts, kinds) if k in allowed]
    log(f"grid: {len(pts)} points to score ({', '.join(f'{k} {int((kinds == k).sum())}' for k in LANDFORMS if (kinds == k).any())}), "
        f"{nw} worker{'s' if nw > 1 else ''}")
    if not pts:
        sys.exit("no point of the area is left to search (check --stands and the area)")

    scored: dict[tuple, dict] = {}

    def key(p):
        return (round(p[0] * 110540 / (fine / 2)), round(p[1] * 111320 * math.cos(math.radians(p[0])) / (fine / 2)))

    def run(points, spc, stage):
        todo, seen = [], set()
        for p in points:
            k = key(p)
            if k not in scored and k not in seen:
                seen.add(k)
                todo.append(p)
        if not todo:
            return
        t = time.time()
        told = [False]

        def progress(done):
            if not told[0] and done >= max(25, nw * 24) and len(todo) > 4 * done:
                told[0] = True
                each = (time.time() - t) / done
                log(f"  {stage}: {each * 1000:.0f} ms a point, about {each * (len(todo) - done) / 60:.1f} min left")

        for p, r in zip(todo, pool.map("rough", todo, progress)):
            scored[key(p)] = {"lat": p[0], "lon": p[1], "spacing": spc, "stage": stage,
                              "explained": r.explained if r else 0.0, "fit": r}
        log(f"  {stage}: {len(todo)} points in {time.time() - t:.0f} s")

    run(pts, spacing, "grid")
    base = list(scored.values())
    chance = float(np.quantile([v["explained"] for v in base], 0.9)) if len(base) >= 10 else None
    # cells whose fit lies on near ridges are filled in finer: there the skyline changes over little ground
    med = float(np.median([v["explained"] for v in base]))
    want = sorted((v for v in base if v["fit"] and v["explained"] >= med and np.isfinite(v["fit"].ridge_m)
                   and FILL_SHARE * v["fit"].ridge_m < spacing / 2), key=lambda v: v["fit"].ridge_m)
    fill, budget = [], int(FILL_BUDGET * len(base))
    for v in want:
        sp = max(fine, FILL_SHARE * v["fit"].ridge_m, spacing / 3)
        fill += [q for q in patch((v["lat"], v["lon"]), sp, spacing * 0.6) if inside(*q)]
        if len(fill) >= budget:
            break
    if fill:
        run(fill[:budget], spacing / 3, "fill")
    spc = spacing
    keep = KEEP_FIRST if not args.quick else KEEP_FIRST // 2
    while spc > fine * 1.01:
        nxt = max(fine, spc / 3)
        top = sorted(scored.values(), key=lambda v: -v["explained"])[:keep]
        run([q for v in top for q in patch((v["lat"], v["lon"]), nxt, spc * 0.6) if inside(*q)], nxt, f"refine {nxt:.0f} m")
        spc, keep = nxt, KEEP_NEXT
        if args.quick:
            break

    # places apart, polished in full: every heading finely, then exactly, the best few on a patch about them
    apart = max(300.0, 3 * fine)
    heads = []
    for v in sorted(scored.values(), key=lambda v: -v["explained"]):
        if all(dist_m((v["lat"], v["lon"]), (h["lat"], h["lon"])) >= apart for h in heads):
            heads.append(v)
            if len(heads) >= max(args.top, POLISH_PLACES) * 2:
                break
    jobs = []
    for i, v in enumerate(heads):
        jobs.append((i, (v["lat"], v["lon"])))
        if i < POLISH_PLACES:
            jobs += [(i, q) for q in patch((v["lat"], v["lon"]), fine / 2, fine) if inside(*q)]
    t = time.time()
    log(f"polishing {len(heads)} places ({len(jobs)} spots, every heading, finely, then exactly)")
    res = pool.map("full", [q for _, q in jobs])
    best_of: dict[int, tuple] = {}
    for (i, q), (fits, g0) in zip(jobs, res):
        if fits and (i not in best_of or fits[0].explained > best_of[i][0][0].explained):
            best_of[i] = (fits, g0, q)
    places = []
    for i, (fits, g0, q) in best_of.items():
        row = {"lat": round(q[0], 6), "lon": round(q[1], 6), **fits[0].row(), "ground_m": round(g0, 1),
               "second_heading": fits[1].row() if len(fits) > 1 else None, "rough": round(heads[i]["explained"], 4)}
        places.append((row, fits[0]))
    places.sort(key=lambda r: -r[0]["explained"])
    if not places:
        sys.exit("no place could be scored")
    log(f"  polished in {time.time() - t:.0f} s; decoys at the best {min(POLISH_PLACES, len(places))}")
    best = places[0][0]
    for row, _ in places:
        row["from_best_m"] = round(dist_m((best["lat"], best["lon"]), (row["lat"], row["lon"])))
    for (row, _), d in zip(places[:POLISH_PLACES], pool.map("decoy", [(r["lat"], r["lon"]) for r, _ in places[:POLISH_PLACES]])):
        row["decoy"] = round(d, 4)
    # the lone-spike test: where the place is real, the ground round it explains nearly as much; a chance fit stands alone.
    # Scored in full like the best place itself (rough scores are too coarse for it)
    # the skyline changes over ground in proportion to how far its ridges stand: read the ring that much nearer
    spot = float(np.clip(SPOT_SHARE_OF_RIDGE * (best["ridge_km"] or 3.0) * 1000, SPOT_NEAR_MIN, SPOT_NEAR_M))
    ring = [q for q in patch((best["lat"], best["lon"]), spot / 2.5, spot) if inside(*q)]
    if ring:
        # only ground one could stand on beside it: past a cliff's edge the ground is the valley below, another view
        g = ground.point(np.array([q[0] for q in ring]), np.array([q[1] for q in ring]))
        ring = [q for q, h in zip(ring, g) if abs(float(h) - best["ground_m"]) <= SPOT_RISE_M]
        # … and that sees out the same way: a few steps back from a viewpoint's rail or a summit's top, the rock in
        # front hides the view, which says nothing against the viewpoint
        ring = [q for q in ring if not _blocked(sc, q, places[0][1])]
    around = None
    if len(ring) >= SPOT_POINTS:
        got = [fits[0].explained for fits, _ in pool.map("full", ring) if fits]
        if len(got) >= SPOT_POINTS and best["explained"] > 0:
            around = float(np.median(got)) / best["explained"]
    rows = [r for r, _ in places]
    verdict, level, reasons = trust(best, rows, chance, around, trace, ask, guessed, args.mode, spot)
    truth = None
    if args.truth:
        tl = tuple(float(v) for v in args.truth.split(","))
        (tf, _), tr = pool.map("full", [tl])[0], sc.rough(*tl)
        te, tre = (tf[0].explained if tf else 0.0), (tr.explained if tr else 0.0)
        truth = {"lat": tl[0], "lon": tl[1], "explained": round(te, 4), "rough": round(tre, 4),
                 "fit": tf[0].row() if tf else None,
                 "best_from_truth_m": round(dist_m(tl, (best["lat"], best["lon"]))),
                 "nearest_listed_m": round(min(dist_m(tl, (r["lat"], r["lon"])) for r in rows[:args.top])),
                 "rough_points_above_truth": int(sum(v["explained"] > tre for v in scored.values())),
                 "listed_above_truth": int(sum(r["explained"] > te for r in rows))}
    out = {"params": {"ridge": str(args.ridge), "area": {"bbox": [s, w, n, e], "at": args.at, "radius_m": args.radius},
                      "spacing_m": spacing, "finest_m": fine, "lens_deg": [round(x, 2) for x in ask.lens_bounds],
                      "lens_guessed": guessed, "tilt_deg": list(ask.tilts), "roll_max": ask.roll_max,
                      "sectors": ask.sectors, "reach_km": [c / 1000 for c in ask.cuts],
                      "band_km": [b / 1000 for b in ask.band] if ask.band else None,
                      "floor_m": ask.floor, "eye_m": ask.eye, "stands": sorted(allowed), "selfie": bool(args.selfie),
                      "mode": args.mode, "dem_zoom": args.zoom},
           "trace": {"points": int(trace.x.size), "turns": nt, "shape_px": round(shp, 2), "image_size": [trace.W, trace.H],
                     "f0": trace.f0, "f0_source": trace.f_source},
           "scored_points": len(scored), "seconds": round(time.time() - t0, 1),
           "chance": round(chance, 4) if chance is not None else None,
           "around_best": round(around, 4) if around is not None else None, "around_m": round(spot),
           "verdict": verdict, "trust": level, "reasons": reasons,
           "places": rows[:args.top], "truth": truth,
           "view": _view_cmd(best, places[0][1].cut)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(scored)} points scored in {time.time() - t0:.0f} s → {args.out}")
    print(f"verdict {verdict}, trust {level.upper()}" + (": " + "; ".join(reasons) if reasons else ""))
    print(f"chance fit of the area {chance:.0%}" if chance is not None else "chance fit: too few points")
    for i, r in enumerate(rows[:args.top]):
        dc = f"  decoy {r['decoy']:.0%}" if r.get("decoy") is not None else ""
        print(f"{i + 1:>2}. {r['lat']:.5f},{r['lon']:.5f}  explains {r['explained']:.0%}{dc}  heading {r['heading']:.1f}°  "
              f"lens {r['hfov']:.0f}°  tilt {r['tilt']:+.1f}°  roll {r['roll']:+.1f}°  reach {r['reach_km']:g} km  "
              f"ridges ~{r['ridge_km']} km  {r['from_best_m'] / 1000:.1f} km from #1")
    if truth:
        print(f"truth {truth['lat']:.5f},{truth['lon']:.5f}: explains {truth['explained']:.0%} (rough {truth['rough']:.0%}); best place "
              f"{truth['best_from_truth_m']} m away, nearest listed {truth['nearest_listed_m']} m; {truth['listed_above_truth']} listed and "
              f"{truth['rough_points_above_truth']} scored points above it")
    print("look from the best place: " + out["view"])
    photo = _photo(args, trace)
    if args.sheet:
        tiles = []
        for i, (r, ft) in enumerate(places[:args.sheet_n]):
            tn = sc.turn(r["lat"], r["lon"], True)
            dt = f"  truth {dist_m((truth['lat'], truth['lon']), (r['lat'], r['lon'])) / 1000:.1f} km" if truth else ""
            title = (f"#{i + 1} {r['lat']:.5f},{r['lon']:.5f}  {r['explained']:.0%}" + (f" (decoy {r['decoy']:.0%})" if r.get('decoy') is not None else "")
                     + f"\nH {r['heading']:.1f}° lens {r['hfov']:.0f}° tilt {r['tilt']:+.1f} roll {r['roll']:+.1f} reach {r['reach_km']:g} km{dt}")
            tiles.append(overlay(photo, trace if ft.trace != "mirrored" else trace.mirrored(), ft, tn, ask, args.sheet_width, title))
        cols = min(len(tiles), 3)
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (tw * cols, th * math.ceil(len(tiles) / cols)), "white")
        for i, t_ in enumerate(tiles):
            sheet.paste(t_, ((i % cols) * tw, (i // cols) * th))
        sheet.save(args.sheet, quality=88)
        print(f"overlays → {args.sheet} (red = the place's skyline through the fitted camera; green = trace points on it, orange = off)")
    if args.heat:
        heatmap(ground, list(scored.values()), (s, w, n, e), rows, (truth["lat"], truth["lon"]) if truth else None, Path(args.heat), chance)
        print(f"map of scores → {args.heat}")


def _blocked(sc: Scorer, q, fit: Fit) -> bool:
    """Whether ground nearer than the floor fills most of the view a fit looks out on from q."""
    tn = sc.turn(q[0], q[1], False)
    az = fit.heading + np.linspace(-fit.hfov / 2, fit.hfov / 2, 60)
    _, far = tn.at(az, tn.cuts.index(fit.cut))
    return float(np.mean(far < max(sc.ask.floor, FLOOR_M))) > SPOT_BLOCKED


def _photo(args, trace: Trace) -> Path | None:
    photo = Path(args.photo) if args.photo else (Path(trace.photo) if trace.photo else None)
    if photo and not photo.exists() and trace.photo:
        alt = Path(args.ridge).parent / Path(trace.photo).name
        photo = alt if alt.exists() else photo
    return photo


def cmd_check(args, fetch, ground=None) -> None:
    log = (lambda *a: print(*a, file=sys.stderr))
    trace = load_trace(args.ridge, args.f35, args.key)
    ask, guessed = _ask(args, trace)
    lat, lon = (float(v) for v in args.at.split(","))
    r = max(args.radius or 0.0, 0.0)
    kx = 111320 * math.cos(math.radians(lat))
    box = (lat - (r + 50) / 110540, lon - (r + 50) / kx, lat + (r + 50) / 110540, lon + (r + 50) / kx)
    spec = None
    if ground is None:
        ground = Ground(box, max(ask.cuts), fetch, args.cache, args.proxy, args.zoom, log)
        spec = {"bbox": box, "reach": max(ask.cuts), "fetch": fetch, "cache": args.cache, "proxy": args.proxy, "zoom": args.zoom,
                "dem": getattr(args, "dem", None)}
    sc = _scorer(args, trace, ask, ground)
    cands = [(lat, lon)] + (patch((lat, lon), args.spacing or max(25.0, r / 6), r) if r > 0 else [])
    pool = None
    if spec and len(cands) > 4:
        spec.update(traces=sc.traces, ask=ask, rough_layout=sc.rough_layout, fine_layout=sc.fine_layout,
                    rough_step=sc.rough_step, fine_step=sc.fine_step)
        pool = Pool(sc, spec, min(_workers(args), len(cands)))
    try:
        res = pool.map("full", cands) if pool else [_work(sc, "full", q) for q in cands]
    finally:
        if pool:
            pool.close()
    scored = [(fits, q) for (fits, _), q in zip(res, cands) if fits]
    if not scored:
        sys.exit("nothing to score here")
    fits, q = max(scored, key=lambda x: x[0][0].explained)
    tn = sc.turn(q[0], q[1], True)
    dec = sc.decoy(*q, tn=tn)
    e = fits[0].explained
    if trace.shape_px() < FLAT_PX:
        verdict = "flat"
    elif e < NONE_BELOW:
        verdict = "none"
    elif len(fits) > 1 and fits[1].explained >= AMBIGUOUS_SHARE * e:
        verdict = "ambiguous"
    elif e < LOOSE_BELOW:
        verdict = "loose"
    else:
        verdict = "match"
    notes = []
    if dec >= e - DECOY_MARGIN:
        notes.append(f"a decoy of the trace fits here nearly as well ({dec:.0%})")
    if trace.turns() < TURNS_MIN:
        notes.append(f"the trace turns only {trace.turns()} times: a plain shape fits many places")
    if abs(fits[0].roll) >= ask.roll_max - 0.2:
        notes.append("the roll is at its limit")
    if r > 0 and len(scored) >= SPOT_POINTS:
        others = sorted(f[0].explained for f, _ in scored)[:-1]
        notes.append(f"the {len(others)} other spots within {r:.0f} m explain {np.median(others):.0%} in the middle")
    rows = [f.row() for f in fits]
    out = {"at": [round(q[0], 6), round(q[1], 6)], "asked": [lat, lon], "moved_m": round(dist_m((lat, lon), q)),
           "ground_m": round(tn.ground, 1), "verdict": verdict, "decoy": round(dec, 4), "notes": notes, "fits": rows,
           "view": _view_cmd({"lat": q[0], "lon": q[1], **rows[0]}, fits[0].cut)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{q[0]:.5f},{q[1]:.5f} ({out['moved_m']} m from the point asked): verdict {verdict}, decoy {dec:.0%}"
          + (" — " + "; ".join(notes) if notes else ""))
    for i, r_ in enumerate(rows):
        print(f"  heading {i + 1}: explains {r_['explained']:.0%}  heading {r_['heading']:.1f}°  lens {r_['hfov']:.0f}°  "
              f"tilt {r_['tilt']:+.1f}°  roll {r_['roll']:+.1f}°  reach {r_['reach_km']:g} km  ridges ~{r_['ridge_km']} km")
    print("look: " + out["view"])
    if args.overlay_out:
        f0 = fits[0]
        im = overlay(_photo(args, trace), trace if f0.trace != "mirrored" else trace.mirrored(), f0, tn, ask, args.sheet_width,
                     f"{q[0]:.5f},{q[1]:.5f}  explains {f0.explained:.0%} (decoy {dec:.0%})\nH {f0.heading:.1f}° lens {f0.hfov:.0f}° "
                     f"tilt {f0.tilt:+.1f} roll {f0.roll:+.1f} reach {f0.cut / 1000:g} km")
        im.save(args.overlay_out, quality=90)
        print(f"overlay → {args.overlay_out}")
