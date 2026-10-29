#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "pillow-heif", "numpy", "torch", "torchvision", "transformers", "opencv-python-headless", "socksio", "pysocks", "requests", "streetlevel", "huggingface_hub", "safetensors"]
# ///
"""Compare every reverse-search result with the photo: merge all saved lists, rank them with MegaLoc, and have Gemini
check the top ones one image at a time. Also ranks any folder of reference photos (a venue's gallery, social posts).

Engines order "similar images" by look, not by place, and a contact sheet shows only the first 24 of each list: the
right place often sits further down, or in the list of a crop nobody opened. compare.py reads all of them at once:

  1. merge  every Baidu `similar` list in the session (the original, crops, flips and masked copies, from recon and
            every later intake.py run): one candidate per distinct image, with every list and position it came from
  2. rank   download the thumbnails and score each against the query photo(s) with MegaLoc (place recognition); with
            several queries (the original, crops, masked copies) their per-query z-scores are averaged
  3. check  send the top --check candidates to Gemini with the query, one image per candidate, asking only "same
            physical place?"; verdicts are cached, so a re-run after a new search pays only for new candidates

intake.py runs it after every reverse search and recon.py at the end of its first pass, so compare/compare.md always
covers every list saved so far. Run it yourself to add a crop as a query or to rank a venue's photos (--images).
Outputs in --out-dir (default <session>/compare; with --images, <session>/compare/images_<folder>): compare.md (read
this), compare.json (every candidate with its score, lists and verdict), sheet.jpg (the top candidates; Gemini's
matches framed green), thumbs/, verdicts.json (Gemini's cached answers).

  compare.py                                          # in the session folder: every saved list, query = the original
  compare.py --query crops/building.jpg               # also rank against a crop of what is distinctive
  compare.py --images web/hotel_gallery --no-lists    # rank a folder of reference photos instead of the lists
  compare.py --check 48                               # Gemini checks the top 48 instead of 24
  compare.py --no-check                               # MegaLoc ranking only

Measured on 25 archived photos with reverse-search lists: Gemini flagged a match in 3, all of them right, and raised
no false alarm in 600 checks; in one of them MegaLoc put the answer 1st of 517, where the agent had reached it after
an hour. On a folder of near-identical venues, Gemini also called three photos of a lookalike "same" (85–95%). A
flag is a lead, not a confirmation: open the full-size image and its page (triage.py), then match fixed features. Never judge candidates on a contact sheet: shown a 24-tile sheet, a vision model called three wrong
buildings "same" at 99% and missed the right one; shown one image per candidate, it picked only the right one.

Gemini key: GEMINI_API_KEY (or GOOGLE_API_KEY), else the env file named by "gemini_env_file" in config.local.json;
without one, compare.py only ranks. Model: --model, else GEOINT_GEMINI_MODEL, else gemini-3.8-flash.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import _gemini  # noqa: E402
from _net import PROXY_HELP, fetch_bytes, model_proxy_env  # noqa: E402

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC photos (common in Chinese travel albums) then fail to open, nothing else
    pass

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif", ".bmp", ".tif", ".tiff"}
VARIANT = re.compile(r"_(flip|gray|wb|up)$")      # imgprep.py variants of a query image: rank against the image itself
MAX_QUERIES = 6
PROMPT = """You are checking image-search results for a photo geolocation task.
QUERY images (first {nq}) come from the user's photo; ignore any people in them.
Then come CANDIDATE images, each preceded by its id. An image search engine returned them because they look alike;
almost all are lookalikes (same style, same kind of building or venue, same region) and NOT the same place.

Task: identify any candidate that shows the SAME PHYSICAL PLACE (the same building, room, wall, street or view, possibly
from another angle, distance, season, light or year), judged ONLY on fixed structure: architecture, window and balcony
patterns, railings, walls, fixed fittings, signage, skyline and terrain. Ignore people, furniture that moves, plants,
weather and image quality. Do not list mere lookalikes.

Answer ONLY with JSON: {{"matches":[{{"id":"<id>","verdict":"same"|"possible","confidence":0-100,"features":"which fixed features agree / disagree"}}]}}
List at most 5 candidates, best first; use an empty list when none qualifies."""


# ---------------------------------------------------------------- candidates

def session_root(start: Path) -> Path | None:
    """The photo session folder (the one holding session.json) at or above start."""
    start = start.resolve()
    for p in [start, *start.parents][:5]:
        if (p / "session.json").is_file():
            return p
    return None


def find_original(root: Path) -> Path | None:
    """The session's untouched photo: session.json's original, else the first image in original/."""
    try:
        rel = json.loads((root / "session.json").read_text(encoding="utf-8"))["original"]["relative_path"]
        if (root / rel).is_file():
            return root / rel
    except (OSError, ValueError, KeyError, TypeError):
        pass
    found = sorted(p for p in (root / "original").glob("*") if p.suffix.lower() in IMAGE_EXT)
    return found[0] if found else None


def _site(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    for pre in ("www.", "m.", "wap.", "mobile."):
        if host.startswith(pre) and host.count(".") > 1:
            return host[len(pre):]
    return host


def page_key(url: str) -> str:
    """A source page without scheme, www./m./wap. prefix or trailing slash, so two lists' copies of a page merge."""
    p = urlparse(url.strip())
    return _site(url) + p.path.rstrip("/") + ("?" + p.query if p.query else "")


def image_key(thumb: str) -> str:
    """Baidu thumbnails carry the image's id as u=<digits>,<digits> (whatever size parameters follow); other
    thumbnail URLs are keyed by host and path."""
    m = re.search(r"\bu=(\d+,\d+)", thumb)
    if m:
        return m.group(1)
    p = urlparse(thumb)
    return p.netloc + p.path


def _resolve(path: str, root: Path) -> Path | None:
    """A list's query image: as written, under the session, or by file name (absolute paths break when a session is renamed)."""
    p = Path(path)
    for cand in (p, root / p):
        if cand.is_file():
            return cand
    hits = [h for h in root.rglob(p.name) if h.is_file() and not any(s.startswith(".") for s in h.relative_to(root).parts)]
    return hits[0] if hits else None


def harvest(root: Path, skip: list[Path]) -> tuple[list[dict], list[str], int, list[Path]]:
    """Every Baidu similar list under root → (candidates, lists read, results read, the base images they searched with)."""
    cands: dict[str, dict] = {}
    lists, n_results, bases = [], 0, []
    skip = [s.resolve() for s in skip]
    for f in sorted(root.rglob("*_baidu.json")):
        rel = f.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or any(f.resolve().is_relative_to(s) for s in skip):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or "engine" not in d and "similar" not in d:
            continue
        lists.append(str(rel))
        img = d.get("image")
        if img and not VARIANT.search(Path(img).stem):
            q = _resolve(img, root)
            if q and q.resolve() not in {b.resolve() for b in bases}:
                bases.append(q)
        for i, x in enumerate(d.get("similar") or []):
            thumb = (x or {}).get("thumb") or ""
            if not thumb:
                continue
            n_results += 1
            page = x.get("from") or ""
            key = page_key(page) + "|" + image_key(thumb)
            c = cands.setdefault(key, {"key": key, "page": page, "site": x.get("site") or _site(page), "thumb": thumb, "hits": []})
            c["hits"].append([str(rel), i])
    return list(cands.values()), lists, n_results, bases


def local_candidates(patterns: list[str], root: Path) -> list[dict]:
    """Reference photos from folders or globs (a venue's gallery, saved social posts) as candidates."""
    out, seen = [], set()
    for pat in patterns:
        p = Path(pat)
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else \
            sorted(Path(f) for f in glob.glob(pat, recursive=True) if Path(f).suffix.lower() in IMAGE_EXT)
        for i, f in enumerate(files):
            if f.resolve() in seen:
                continue
            seen.add(f.resolve())
            rel = os.path.relpath(f.resolve(), root.resolve())
            out.append({"key": "file:" + rel, "page": "", "site": "local", "thumb": "", "file": rel, "hits": [[pat, i]]})
    return out


def download(cands: list[dict], thumbs: Path, root: Path, proxy: str | None, workers: int = 24) -> int:
    """Fetch every thumbnail once (named by a hash of its URL); sets c["file"]. Returns how many failed."""
    thumbs.mkdir(parents=True, exist_ok=True)

    def one(thumb: str) -> str | None:
        f = thumbs / (hashlib.sha1(thumb.encode()).hexdigest()[:16] + ".jpg")
        if not f.exists():
            for url in dict.fromkeys([thumb, thumb.replace("http://", "https://", 1)]):
                try:
                    data = fetch_bytes(url, proxy, 15, {"User-Agent": UA, "Referer": "https://graph.baidu.com/"})
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    im.thumbnail((512, 512))                  # what Gemini is shown; MegaLoc looks at 322 px
                    im.save(f.with_suffix(".part"), "JPEG", quality=88)
                    f.with_suffix(".part").replace(f)
                    break
                except Exception:  # noqa: BLE001
                    continue
        return os.path.relpath(f, root) if f.exists() else None

    todo = [c for c in cands if c.get("thumb") and not c.get("file")]
    urls = list(dict.fromkeys(c["thumb"] for c in todo))          # the same picture on two pages is fetched once
    with ThreadPoolExecutor(workers) as ex:
        got = dict(zip(urls, ex.map(one, urls)))
    for c in todo:
        c["file"] = got[c["thumb"]]
    return sum(1 for c in todo if not c["file"])


# ---------------------------------------------------------------- ranking

def _load(path: Path, max_side: int = 1024) -> Image.Image | None:
    try:
        with Image.open(path) as f:
            im = ImageOps.exif_transpose(f).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    im.thumbnail((max_side, max_side))
    return im


class Ranker:
    """MegaLoc vectors for image files, cached in an .npz by file name, size and modification time."""

    def __init__(self, cache: Path | None, proxy: str | None = None):
        self.cache, self.proxy, self.model = cache, proxy, None
        self.store: dict[str, np.ndarray] = {}
        if cache and cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as z:
                    self.store = dict(zip(z["keys"].tolist(), z["vecs"]))
            except Exception:  # noqa: BLE001
                self.store = {}

    @staticmethod
    def _key(p: Path) -> str:
        st = p.stat()
        return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"

    def embed(self, paths: list[Path]) -> np.ndarray:
        """(N, V, D) float32 for the files; unreadable files get zero vectors (they score 0)."""
        keys = [self._key(p) for p in paths]
        todo = [(k, p) for k, p in zip(keys, paths) if k not in self.store]
        if todo:
            if self.model is None:
                model_proxy_env(self.proxy)
                import match
                self.model = match.Embedder("megaloc")
            ims = [(k, _load(p)) for k, p in todo]
            good = [(k, im) for k, im in ims if im is not None]
            for i in range(0, len(good), 64):
                chunk = good[i:i + 64]
                vecs = self.model.embed([im for _, im in chunk], multi=True)
                for (k, _), v in zip(chunk, vecs):
                    self.store[k] = v.astype(np.float16)
            if self.cache:
                self._save()
        shape = next(iter(self.store.values())).shape if self.store else (3, 8448)
        return np.stack([self.store.get(k, np.zeros(shape, np.float16)) for k in keys]).astype(np.float32)

    def _save(self) -> None:
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache.with_name(self.cache.stem + ".part.npz")
        np.savez(tmp, keys=np.array(list(self.store)), vecs=np.stack(list(self.store.values())))
        tmp.replace(self.cache)


def score(C: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """(N, V, D) candidates × (M, W, D) queries → (N, M): the best cosine over view pairs, per query."""
    return np.einsum("nvd,mwd->nmvw", C, Q).max(axis=(2, 3))


def fuse(S: np.ndarray) -> np.ndarray:
    """One number per candidate: the cosine for a single query; the mean of per-query z-scores for several.
    (On a case with four queries the max over queries let a lookalike pass the answer; the mean of z-scores did not.)"""
    if S.shape[1] == 1:
        return S[:, 0]
    return ((S - S.mean(0)) / (S.std(0) + 1e-9)).mean(1)


# ---------------------------------------------------------------- Gemini check

def _qhash(queries: list[Path]) -> str:
    h = hashlib.sha1()
    for q in queries:
        h.update(q.read_bytes())
    return h.hexdigest()[:12]


def check(cands: list[dict], queries: list[Path], root: Path, out: Path, n: int, model: str, proxy: str | None,
          batch: int = 24, workers: int = 4) -> dict:
    """Gemini judges the top n candidates (one image each, with the query images); sets c["gemini"] and caches verdicts."""
    cache_f = out / "verdicts.json"
    try:
        cache = json.loads(cache_f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    qh = _qhash(queries)
    top = [c for c in cands if c.get("file")][:n]
    todo = []
    for c in top:
        hit = cache.get(f"{model}|{qh}|{c['key']}")
        if hit:
            c["gemini"] = hit
        else:
            todo.append(c)
    info = {"model": model, "checked": len(top), "new": len(todo), "tokens": 0, "errors": []}
    if not todo:
        return info
    key = _gemini.api_key()
    if not key:
        info["errors"].append("no Gemini key (GEMINI_API_KEY, or gemini_env_file in config.local.json): ranked only")
        info["checked"] = len(top) - len(todo)
        return info
    qparts = []
    for i, q in enumerate(queries):
        qparts += [{"text": f"QUERY {i + 1}:"}, _gemini.image_part(q, 1024)]

    def run(group: list[dict]) -> tuple[list[dict], dict | None, str]:
        parts = [{"text": PROMPT.format(nq=len(queries))}, *qparts]
        for c in group:
            parts += [{"text": f"CANDIDATE id={c['cid']}:"}, _gemini.image_part(root / c["file"], 512)]
        try:
            text, meta = _gemini.ask(parts, model, proxy, key=key)
        except RuntimeError as e:
            return group, None, str(e)
        return group, {"reply": _gemini.parse_json(text), "meta": meta}, ""

    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with ThreadPoolExecutor(min(workers, len(groups))) as ex:
        results = list(ex.map(run, groups))
    for group, res, err in results:
        if res is None:
            info["errors"].append(err[:200])
            continue
        info["tokens"] += res["meta"].get("tokens") or 0
        found = {}
        for m in res["reply"].get("matches") or []:
            if isinstance(m, dict) and str(m.get("id", "")).strip():
                found[str(m["id"]).strip()] = m
        for c in group:
            m = found.get(c["cid"])
            v = {"verdict": "different", "model": model}
            if m and m.get("verdict") in ("same", "possible"):
                v = {"verdict": m["verdict"], "confidence": m.get("confidence"), "features": str(m.get("features", ""))[:300],
                     "model": model}
            c["gemini"] = v
            cache[f"{model}|{qh}|{c['key']}"] = v
    info["checked"] = sum(1 for c in top if c.get("gemini"))
    tmp = cache_f.with_suffix(".part")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(cache_f)
    return info


# ---------------------------------------------------------------- outputs

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def sheet(cands: list[dict], query: Path | None, root: Path, out: Path, n: int, cols: int = 6, tile: int = 260) -> None:
    """The query and the top n candidates, labelled with rank and site; Gemini 'same' framed green, 'possible' orange."""
    rows = [c for c in cands if c.get("file")][:n]
    cells = [(query, "query", "cyan", None)] + [
        (root / c["file"], f"#{c['rank']} {c['cos']:.3f} {c['site']}"[:34], "yellow", (c.get("gemini") or {}).get("verdict")) for c in rows]
    bar = 22
    S = Image.new("RGB", (cols * tile, ((len(cells) + cols - 1) // cols) * (tile + bar)), (40, 40, 40))
    d, f = ImageDraw.Draw(S), _font(15)
    for k, (path, label, color, verdict) in enumerate(cells):
        x, y = (k % cols) * tile, (k // cols) * (tile + bar)
        d.rectangle([x, y, x + tile - 1, y + bar - 1], fill="black")
        d.text((x + 4, y + 3), label, fill=color, font=f)
        im = _load(path, tile - 8) if path else None
        if im is not None:
            S.paste(im, (x + (tile - im.width) // 2, y + bar + (tile - im.height) // 2))
        if verdict in ("same", "possible"):
            d.rectangle([x + 1, y + 1, x + tile - 2, y + bar + tile - 2], outline="lime" if verdict == "same" else "orange", width=5)
    S.save(out, quality=88)


def _hits(c: dict, k: int = 3) -> str:
    def name(h):
        return re.sub(r"_baidu\.json$", "", h[0]) if h[0].endswith(".json") else h[0]
    s = "; ".join(f"{name(h)} #{h[1]:02d}" for h in sorted(c["hits"], key=lambda h: h[1])[:k])
    return s + (f" (+{len(c['hits']) - k})" if len(c["hits"]) > k else "")


def report(cands: list[dict], meta: dict, out: Path, root: Path, top: int) -> str:
    rel = lambda p: os.path.relpath(p, root)  # noqa: E731
    g = meta.get("gemini") or {}
    flagged = sorted((c for c in cands if (c.get("gemini") or {}).get("verdict") in ("same", "possible")),
                     key=lambda c: (c["gemini"]["verdict"] != "same", -(c["gemini"].get("confidence") or 0), c["rank"]))
    src = " and ".join(([f"{meta['n_lists']} reverse-search lists ({meta['n_results']} results)"] if meta["n_lists"] else [])
                       + ([f"{meta['n_local']} reference photos"] if meta["n_local"] else [])) or "nothing"
    L = [f"# Compare: {len(cands)} candidates from {src}", "",
         "Query: " + ", ".join(f"`{rel(q)}`" for q in meta["queries"]) + ". Ranked with MegaLoc"
         + (" (mean of per-query z-scores)" if len(meta["queries"]) > 1 else "") + "."]
    if g.get("checked") or g.get("errors"):
        L.append(f"Gemini ({g.get('model')}) checked the top {g.get('checked', 0)}, one image per candidate"
                 + (f" ({g['new']} new, {g['tokens']} tokens)" if g.get("new") else " (cached verdicts)") + "."
                 + ("".join(f" **{e}**" for e in g.get("errors", []))))
    elif meta.get("no_check"):
        L.append("Gemini check not run (--no-check).")
    if meta["failed"]:
        L.append(f"{meta['failed']} thumbnails could not be downloaded (left out).")
    L += ["", "## Gemini's matches: open these first", ""]
    if flagged:
        for c in flagged:
            v = c["gemini"]
            L.append(f"{c['rank']}. **{v['verdict']} {v.get('confidence', '?')}%**: {v.get('features', '')}")
            L.append(f"   - page: {c['page'] or c['file']} ({c['site']}); thumbnail `{c['file']}`; found in {_hits(c)}")
            if c["page"]:
                L.append(f"   - next: `triage.py \"{c['page']}\"` (every photo and caption on that page, ranked)")
        L += ["", "A Gemini verdict is a lead, never a confirmation: open the full-size image and its page, then match "
              "fixed features (SKILL.md, step 7)."]
    elif g.get("checked"):
        L.append(f"None of the top {g['checked']} shows the same place according to Gemini. That is not proof of absence: "
                 "read the table below, add crops (`intake.py --box`), masked copies or other engines' results, then re-run.")
    else:
        L.append("No Gemini check: read the table below and open the most promising candidates at full size.")
    multi = len(meta["queries"]) > 1
    L += ["", f"## Top {min(top, len(cands))} by MegaLoc", "",
          "| # | " + ("score (mean z) | best cos" if multi else "cos") + " | Gemini | site | page | found in (list #position) |",
          "|---|---|" + ("---|" if multi else "") + "---|---|---|---|"]
    for c in cands[:top]:
        v = c.get("gemini") or {}
        verdict = f"{v['verdict']} {v.get('confidence', '')}".strip() if v.get("verdict") in ("same", "possible") else \
            ("–" if v else "")
        sc = f"{c['score']:.2f} | {c['cos']:.3f}" if multi else f"{c['cos']:.3f}"
        L.append(f"| {c['rank']} | {sc} | {verdict} | {c['site']} | {(c['page'] or c['file'] or '')[:90]} | {_hits(c, 2)} |")
    L += ["", f"Sheet: `{rel(out / 'sheet.jpg')}` (top {min(top, len(cands))}, Gemini's matches framed green) — for scanning "
          "only: thumbnails cannot confirm or exclude a place. Scores order the list; the 1st and 2nd can differ by "
          "0.005, so rank alone is not a match. Every candidate is in `compare.json`.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="where to look for reverse-search lists (default: the session folder you are in or under)")
    ap.add_argument("--query", action="append", type=Path, default=[],
                    help="extra query image, repeatable; the session's original and every list's own query image are always used")
    ap.add_argument("--images", action="append", default=[], help="folder or glob of reference photos to rank too, repeatable")
    ap.add_argument("--no-lists", action="store_true", help="leave the reverse-search lists out (rank only --images)")
    ap.add_argument("--out-dir", type=Path, help="default: <session>/compare")
    ap.add_argument("--check", type=int, default=24, help="how many top candidates Gemini checks, one image each (default 24)")
    ap.add_argument("--no-check", action="store_true", help="MegaLoc ranking only")
    ap.add_argument("--model", help=f"Gemini model (default: GEOINT_GEMINI_MODEL or {_gemini.DEFAULT_MODEL})")
    ap.add_argument("--top", type=int, default=24, help="candidates listed in compare.md and drawn on the sheet")
    ap.add_argument("--proxy", default=os.environ.get("GEO_PROXY"), help=PROXY_HELP)
    args = ap.parse_args()
    t0 = time.time()
    root = session_root(args.root) or args.root.resolve()
    default_out = root / "compare"
    if args.images:                       # a venue's photos get their own page; compare/compare.md stays the lists' page
        default_out = default_out / ("images_" + re.sub(r"[^A-Za-z0-9._-]+", "-", Path(args.images[0].rstrip("/*")).name or "photos")[:40])
    out = (args.out_dir or default_out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cands, lists, n_results, bases = ([], [], 0, []) if args.no_lists else harvest(root, [out])
    local = local_candidates(args.images, root)
    cands += local
    original = find_original(root)
    queries = ([original] if original else []) + list(args.query) + bases
    queries = list(dict.fromkeys(q.resolve() for q in queries if q.is_file()))[:MAX_QUERIES]
    if not queries:
        sys.exit("no query image: run inside a photo session (original/) or pass --query")
    if not cands:
        msg = f"compare: no candidates (no reverse-search lists under {root}" + (" and no --images)" if not args.images else ")")
        (out / "compare.md").write_text(f"# Compare\n\n{msg}.\n", encoding="utf-8")
        print(msg)
        return

    timings = {}
    t = time.time()
    failed = download([c for c in cands if c.get("thumb")], out / "thumbs", root, args.proxy)
    timings["download"] = round(time.time() - t, 1)
    usable = [c for c in cands if c.get("file")]
    t = time.time()
    ranker = Ranker(root / ".geo-cache" / "compare_megaloc.npz", args.proxy)      # per-photo cache: re-runs embed only new images
    C = ranker.embed([root / c["file"] for c in usable])
    Q = ranker.embed(queries)
    timings["rank"] = round(time.time() - t, 1)
    S = score(C, Q)
    fused = fuse(S)
    order = np.argsort(-fused, kind="stable")
    ranked = []
    for r, i in enumerate(order, 1):
        c = usable[int(i)]
        c.update(rank=r, score=round(float(fused[i]), 4), cos=round(float(S[i].max()), 4),
                 cos_by_query=[round(float(x), 4) for x in S[i]], cid=f"{r:04d}")
        ranked.append(c)

    model = _gemini.model_name(args.model)
    shown = {p.resolve() for p in ([original] if original else []) + list(args.query)}      # Gemini sees the photo and explicit crops
    gem_queries = [q for q in queries if q in shown][:3] or queries[:1]
    t = time.time()
    ginfo = {} if args.no_check or args.check <= 0 else check(ranked, gem_queries, root, out, args.check, model, args.proxy)
    timings["check"] = round(time.time() - t, 1)
    meta = {"root": str(root), "queries": queries, "n_lists": len(lists), "n_results": n_results, "n_local": len(local),
            "failed": failed, "gemini": ginfo, "no_check": args.no_check or args.check <= 0, "timings": timings}
    sheet(ranked, queries[0], root, out / "sheet.jpg", min(args.top, 30))
    md = report(ranked, meta, out, root, args.top)
    (out / "compare.md").write_text(md, encoding="utf-8")
    dump = {**meta, "queries": [os.path.relpath(q, root) for q in queries], "lists": lists, "seconds": round(time.time() - t0, 1),
            "candidates": [{k: v for k, v in c.items() if k != "cid"} for c in ranked]}
    tmp = out / "compare.json.part"
    tmp.write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out / "compare.json")
    flagged = [c for c in ranked if (c.get("gemini") or {}).get("verdict") in ("same", "possible")]
    print(f"compare: {len(ranked)} candidates from {len(lists)} lists" + (f" + {len(local)} photos" if local else "")
          + f" in {time.time() - t0:.0f}s (" + ", ".join(f"{k} {v}s" for k, v in timings.items()) + "); Gemini "
          + (f"checked {ginfo.get('checked', 0)}, flagged {len(flagged)}" if ginfo else "not run")
          + "".join(f"; {e}" for e in ginfo.get("errors", [])))
    for c in flagged:
        print(f"  #{c['rank']} {c['gemini']['verdict']} {c['gemini'].get('confidence', '?')}%  {c['page'] or c['file']}")
    print(f"-> {out / 'compare.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
