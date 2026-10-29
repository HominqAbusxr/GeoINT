#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "pillow-heif", "numpy", "torch", "torchvision", "transformers", "opencv-python-headless", "socksio", "pysocks", "requests", "streetlevel", "huggingface_hub", "safetensors"]
# ///
"""Open the page behind a lead and pull every picture and caption from it, ranked against the photo.

A reverse-search hit is usually one picture from an album, a travel diary or a post. The place's name sits in a caption
a few pictures away, in the title, or on a sign in another picture of the set. triage.py downloads the page, extracts
every picture with the text around it (a reader for Meipian albums, plain HTML for other sites), ranks the pictures
against the photo with MegaLoc, reads the text in the best ones with ocr.py, and writes one page to read.

  triage.py <url> [<url> ...]                       # in the session folder: query = the original
  triage.py <url> --query crops/building.jpg        # rank against a crop as well
  triage.py <url> --check                           # Gemini also checks the 6 best pictures, one at a time
  triage.py saved.html --base-url <url>             # a page saved from the browser (login walls, script-rendered sites)

Outputs in --out-dir (default <session>/triage)/<site>_<page id>/: triage.md (title, date, the best-matching pictures
with their captions and OCR, the text around the best one, the page's opening text), images/ (numbered in page order),
sheet.jpg (the best matches), sources.json (each local file → picture URL, page, caption).

Measured: on a 100-picture travel album reached from a reverse-search hit, it ranked the matching picture first (cosine
0.306, next 0.198) and surfaced the caption naming the region and the kind of hotel, in about a minute; on a 27-picture
album it read the building's street number and name off the neighbouring picture.
Captions and titles are leads: a travel diary names the hotel it stayed at, not always the place in each picture;
confirm on fixed features and an official address. A page that renders with scripts or needs a login gives few or no
pictures: open it in the browser and save it (pass the file) or its photos (rank them with compare.py --images).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import _gemini  # noqa: E402
import compare  # noqa: E402  (session helpers, MegaLoc ranking, Gemini check, sheets)
from _net import PROXY_HELP, fetch_bytes  # noqa: E402

HERE = Path(__file__).parent
UV = os.environ.get("UV") or "uv"
SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
LAZY = ("data-original", "data-src", "data-lazy-src", "data-actualsrc", "data-url", "data-echo", "data-lazyload")
META = ("og:title", "og:description", "description", "og:image", "article:published_time", "datepublished", "pubdate")
JUNK = re.compile(r"(logo|icon|avatar|emoji|sprite|qrcode|qr[-_]code|badge|loading|placeholder|blank\.|spacer|1x1|/ad[s/_])", re.I)
IMG_LINK = re.compile(r"\.(jpe?g|png|webp|heic)(\?|$)", re.I)


# ---------------------------------------------------------------- page → sequence of text and pictures

def _width(desc: str) -> float:
    m = re.match(r"([\d.]+)([wx])", desc.strip())
    return 0 if not m else float(m.group(1)) * (1000 if m.group(2) == "x" else 1)


def _src(a: dict) -> str:
    """The best picture URL of an <img>: lazy-loading attributes first, then the widest srcset entry, then src."""
    for k in LAZY:
        v = a.get(k, "").strip()
        if v and not v.startswith("data:"):
            return v
    for k in ("data-srcset", "srcset"):
        parts = [p.strip().split() for p in a.get(k, "").split(",") if p.strip()]
        if parts:
            return max(parts, key=lambda t: _width(t[1]) if len(t) > 1 else 0)[0]
    v = a.get("src", "").strip()
    return "" if v.startswith("data:") else v


class Page(HTMLParser):
    """The page in reading order: ("text", str) and ("img", {"url", "alt"}) entries, plus its title and meta tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.seq: list[tuple[str, object]] = []
        self.title, self.meta = "", {}
        self._skip, self._in_title = 0, False

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            k = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if k in META and a.get("content"):
                self.meta.setdefault(k, a["content"].strip())
        elif tag == "img" and not self._skip:
            url = _src(a)
            if url:
                self.seq.append(("img", {"url": url, "alt": (a.get("alt") or a.get("title") or "").strip()}))
        elif tag == "a" and IMG_LINK.search(a.get("href", "")):
            self.seq.append(("img", {"url": a["href"], "alt": a.get("title", "").strip()}))

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        t = " ".join(data.split())
        if not t or self._skip:
            return
        if self._in_title:
            self.title = (self.title + " " + t).strip()
        else:
            self.seq.append(("text", t))


def _plain(s: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())


def meipian(text: str) -> dict | None:
    """Meipian albums keep their pictures and captions in a JSON blob (window.ARTICLE_DETAIL), not in the HTML."""
    m = re.search(r"ARTICLE_DETAIL\s*=\s*", text)
    if not m:
        return None
    try:
        d = json.JSONDecoder().raw_decode(text[m.end():])[0]
    except ValueError:
        return None
    art = d.get("article") or d
    content = art.get("content")
    items = content.get("content") if isinstance(content, dict) else content
    seq: list = []
    for x in items or []:
        if not isinstance(x, dict):
            continue
        if x.get("img_url"):
            seq.append(("img", {"url": x["img_url"], "alt": ""}))
        if _plain(x.get("text") or ""):
            seq.append(("text", _plain(x["text"])))
    date = art.get("create_time") or ""
    if isinstance(date, (int, float)) or str(date).isdigit():
        date = dt.datetime.fromtimestamp(int(date) / (1000 if int(date) > 10 ** 11 else 1), dt.timezone.utc).strftime("%Y-%m-%d")
    return {"title": art.get("title") or "", "date": str(date), "seq": seq, "reader": "meipian"}


def decode(raw: bytes) -> str:
    """Page bytes → text, honouring a declared charset (GBK/GB2312 pages are read as GB18030)."""
    m = re.search(rb'charset=["\']?([A-Za-z0-9_-]+)', raw[:4096])
    enc = m.group(1).decode().lower() if m else "utf-8"
    enc = "gb18030" if enc in ("gb2312", "gbk") else enc
    try:
        return raw.decode(enc)
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", "replace")
        return text if text.count("�") < 50 else raw.decode("gb18030", "replace")


def parse(text: str, base: str) -> dict:
    """Title, date, opening text and every picture with the text just before and after it, in page order."""
    page = meipian(text)
    if page is None:
        p = Page()
        p.feed(text)
        page = {"title": p.meta.get("og:title") or p.title, "reader": "html", "seq": p.seq,
                "date": p.meta.get("article:published_time") or p.meta.get("datepublished") or p.meta.get("pubdate") or "",
                "description": p.meta.get("og:description") or p.meta.get("description") or ""}
        if p.meta.get("og:image"):             # the preview picture: often the post's main photo or a video's cover
            page["seq"] = page["seq"] + [("img", {"url": p.meta["og:image"], "alt": "", "og": True})]
    seq, pics, seen = page["seq"], [], set()
    for i, (kind, v) in enumerate(seq):
        if kind != "img":
            continue
        url = urljoin(base, v["url"].strip())
        if url in seen or not url.startswith("http") or JUNK.search(url) or url.lower().split("?")[0].endswith((".svg", ".gif", ".ico")):
            continue
        seen.add(url)

        def near(step: int) -> str:
            j = i + step
            while 0 <= j < len(seq) and seq[j][0] != "img":
                if len(seq[j][1]) >= 2:
                    return seq[j][1][:400]
                j += step
            return ""

        if v.get("og"):
            pics.append({"url": url, "alt": "the page's preview picture", "before": "", "after": page.get("description", ""), "seq": i, "og": True})
        else:
            pics.append({"url": url, "alt": v.get("alt", ""), "before": near(-1), "after": near(1), "seq": i})
    texts = [v for k, v in seq if k == "text" and len(v) >= 8]
    page.update(pictures=pics, opening=" ".join(texts)[:900])
    return page


# ---------------------------------------------------------------- pictures

def fetch_pictures(pics: list[dict], folder: Path, referer: str, proxy: str | None, limit: int) -> list[dict]:
    """Download up to limit pictures (HEIC included), skip tiny ones, save images/NNN.jpg in page order."""
    folder.mkdir(parents=True, exist_ok=True)

    def one(t: tuple[int, dict]) -> dict | None:
        k, p = t
        f = folder / f"{k:03d}.jpg"
        if not f.exists():
            try:
                raw = fetch_bytes(p["url"], proxy, 25, {"User-Agent": compare.UA, "Referer": referer})
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
            if min(im.size) < 150:
                return None
            im.thumbnail((2048, 2048))
            im.save(f, "JPEG", quality=90)
        return {**p, "n": k, "path": f}

    with ThreadPoolExecutor(12) as ex:
        got = list(ex.map(one, enumerate(pics[:limit], 1)))
    return [g for g in got if g]


def ocr(paths: list[Path]) -> dict[Path, str]:
    """Text read in each picture by ocr.py (Apple Vision on macOS, RapidOCR elsewhere)."""
    def one(p: Path) -> tuple[Path, str]:
        out = p.with_suffix(".ocr.json")
        try:
            subprocess.run([UV, "run", "-q", str(HERE / "ocr.py"), str(p), "--out", str(out)], capture_output=True,
                           timeout=300, env={**os.environ, "PYTHONUTF8": "1"})
            items = json.loads(out.read_text(encoding="utf-8")).get("items", [])
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return p, ""
        return p, " / ".join(it["text"] for it in items if it.get("conf", 0) >= 0.5)[:400]

    with ThreadPoolExecutor(3) as ex:
        return dict(ex.map(one, paths))


def slug(url: str) -> str:
    p = urlparse(url)
    tail = [s for s in p.path.split("/") if s][-1:] or [hashlib.sha1(url.encode()).hexdigest()[:8]]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{compare._site(url) or 'page'}_{tail[0]}")[:70]


def report(url: str, page: dict, ranked: list[dict], found: int, out: Path, gemini: dict, ocr_text: dict) -> str:
    best = ranked[0] if ranked else None
    L = [f"# Triage: {page.get('title') or url}", "",
         f"{url} · {compare._site(url)}" + (f" · published {page['date']}" if page.get("date") else "")
         + f" · {found} pictures on the page, {len(ranked)} read ({page['reader']} reader)", ""]
    if page.get("description"):
        L += [f"Description: {page['description'][:300]}", ""]
    if not ranked:
        L += ["No pictures could be read. The page may render with scripts or need a login: open it in the browser, "
              "save the page (pass the file with --base-url) or its photos (rank them with compare.py --images).", ""]
    else:
        L += ["## Best matches with the photo", ""]
        for r, p in enumerate(ranked[:6], 1):
            v = p.get("gemini") or {}
            tag = f" — Gemini: **{v['verdict']} {v.get('confidence', '?')}%** ({v.get('features', '')[:160]})" \
                if v.get("verdict") in ("same", "possible") else (" — Gemini: different" if v else "")
            L.append(f"{r}. `{os.path.relpath(p['path'], out)}` (picture {p['n']} of {found}) — cos {p['cos']:.3f}{tag}")
            for label, key in (("text after", "after"), ("text before", "before"), ("alt", "alt")):
                if p.get(key):
                    L.append(f"   - {label}: “{p[key][:300]}”")
            if ocr_text.get(p["path"]):
                L.append(f"   - OCR: {ocr_text[p['path']]}")
        gap = ranked[0]["cos"] - ranked[1]["cos"] if len(ranked) > 1 else 0
        L += ["", f"The best picture scores {ranked[0]['cos']:.3f}" + (f", the next {ranked[1]['cos']:.3f}" if len(ranked) > 1 else "")
              + (": a clear gap." if gap >= 0.05 else ": no clear winner, so look at the top few.") + " Scores order pictures; "
              "they do not prove the place."]
        if gemini.get("errors"):
            L.append("Gemini: " + "; ".join(gemini["errors"]))
        L += ["", f"## Around the best match (page order, picture {best['n']})", ""]
        seq, idx = page["seq"], best["seq"]
        imgs = [i for i, (k, v) in enumerate(seq) if k == "img" and not v.get("og")]
        if best.get("og") or idx not in imgs:
            L.append("- The best match is the page's preview picture, outside the text: read the title and description above.")
        else:
            pos = imgs.index(idx)
            lo, hi = imgs[max(0, pos - 3)], imgs[pos + 3] if pos + 3 < len(imgs) else len(seq)
            n_of = {p["seq"]: p["n"] for p in ranked}
            lines = 0
            for i in range(lo, hi):
                kind, v = seq[i]
                if kind == "img":
                    L.append(f"- [picture {n_of.get(i, '–')}]" + (" ← best match" if i == idx else ""))
                elif len(v) >= 4 and lines < 16:              # shorter strings are mostly menus and buttons
                    L.append(f"  “{v[:300]}”")
                    lines += 1
    if page.get("opening"):
        L += ["", "## Page text (opening)", "", "> " + page["opening"][:900], ""]
    L += ["Next: names in the captions, title and OCR are leads. Find the place on a map and an official source, then "
          "compare fixed features on the full-size picture with the photo. `sheet.jpg` shows the best matches; every "
          "picture is in `images/`, with its source in `sources.json`.", ""]
    return "\n".join(L)


def triage(src: str, base_url: str | None, queries: list[Path], root: Path, out_root: Path, args) -> str:
    t0 = time.time()
    if Path(src).is_file():
        raw, url = Path(src).read_bytes(), base_url or ""
    else:
        url = src
        raw = fetch_bytes(src, args.proxy, 25, {"User-Agent": compare.UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})
    page = parse(decode(raw), base_url or url)
    out = out_root / slug(url or src)
    out.mkdir(parents=True, exist_ok=True)
    pics = fetch_pictures(page["pictures"], out / "images", url, args.proxy, args.max_images)
    ranked: list[dict] = []
    if pics:
        ranker = compare.Ranker(root / ".geo-cache" / "triage_megaloc.npz", args.proxy)
        S = compare.score(ranker.embed([p["path"] for p in pics]), ranker.embed(queries))
        fused = compare.fuse(S)
        for r, i in enumerate(np.argsort(-fused, kind="stable"), 1):
            p = pics[int(i)]
            p.update(rank=r, cos=float(S[i].max()), score=float(fused[i]), cid=f"{p['n']:04d}", key=f"triage|{p['url']}",
                     site=compare._site(url), file=os.path.relpath(p["path"], root))
            ranked.append(p)
    gemini = compare.check(ranked, queries[:3], root, out, 6, _gemini.model_name(args.model), args.proxy) \
        if ranked and args.check else {}
    ocr_text = ocr([p["path"] for p in ranked[:args.ocr]]) if args.ocr and ranked else {}
    if ranked:
        compare.sheet(ranked, queries[0], root, out / "sheet.jpg", 12, cols=4)
    md = report(url or src, page, ranked, len(page["pictures"]), out, gemini, ocr_text)
    (out / "triage.md").write_text(md, encoding="utf-8")
    (out / "sources.json").write_text(json.dumps(
        {"page": url, "title": page.get("title"), "date": page.get("date"), "reader": page["reader"],
         "pictures": [{"file": os.path.relpath(p["path"], out), "url": p["url"], "picture": p["n"], "rank": p["rank"],
                       "cos": round(p["cos"], 4), "before": p["before"], "after": p["after"], "alt": p["alt"],
                       "gemini": p.get("gemini")} for p in sorted(ranked, key=lambda p: p["n"])]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    best = ranked[0] if ranked else None
    return (f"{url or src}: {len(page['pictures'])} pictures, {len(ranked)} read in {time.time() - t0:.0f}s"
            + (f"; best picture {best['n']} (cos {best['cos']:.3f})" if best else "; none readable (login or script page?)")
            + f" -> {out / 'triage.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", help="page URLs, or saved HTML files (with --base-url)")
    ap.add_argument("--query", action="append", type=Path, default=[], help="extra query image (the session's original is always used)")
    ap.add_argument("--base-url", help="the address a saved HTML file came from (resolves its picture links)")
    ap.add_argument("--out-dir", type=Path, help="default: <session>/triage")
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--ocr", type=int, default=3, help="OCR the best N pictures (0 = none)")
    ap.add_argument("--check", action="store_true", help="Gemini checks the 6 best pictures, one image each")
    ap.add_argument("--model", help=f"Gemini model (default: GEOINT_GEMINI_MODEL or {_gemini.DEFAULT_MODEL})")
    ap.add_argument("--proxy", default=os.environ.get("GEO_PROXY"), help=PROXY_HELP)
    args = ap.parse_args()
    root = compare.session_root(Path(".")) or Path(".").resolve()
    original = compare.find_original(root)
    queries = list(dict.fromkeys(q.resolve() for q in ([original] if original else []) + args.query if q.is_file()))
    if not queries:
        sys.exit("no query image: run inside a photo session (original/) or pass --query")
    out_root = (args.out_dir or root / "triage").resolve()
    for src in args.pages:
        try:
            print(triage(src, args.base_url, queries, root, out_root, args))
        except Exception as e:  # noqa: BLE001
            print(f"{src}: failed ({type(e).__name__}: {str(e)[:200]}); open it in the browser instead")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
