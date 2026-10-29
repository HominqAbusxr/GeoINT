"""Gemini REST client shared by compare.py and triage.py: images and text in, text out.

The key comes from GEMINI_API_KEY (or GOOGLE_API_KEY), else from the env file named by "gemini_env_file" in this skill's
config.local.json (git-ignored). It is sent only in the request header, never printed, logged or written to a file.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from _net import resolve_proxy

DEFAULT_MODEL = "gemini-3.8-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}"
SKILL_DIR = Path(__file__).resolve().parent.parent
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def model_name(model: str | None = None) -> str:
    return model or os.environ.get("GEOINT_GEMINI_MODEL") or DEFAULT_MODEL


def api_key() -> str | None:
    """The key from the environment, else from the env file that config.local.json names; None when there is none."""
    for name in KEY_NAMES:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    try:
        cfg = json.loads((SKILL_DIR / "config.local.json").read_text(encoding="utf-8"))
        env_file = Path(os.path.expanduser(str(cfg.get("gemini_env_file") or "")))
        lines = env_file.read_text(encoding="utf-8").splitlines() if cfg.get("gemini_env_file") else []
    except (OSError, ValueError, AttributeError):
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() in KEY_NAMES and value.strip().strip("\"'"):
            return value.strip().strip("\"'")
    return None


def key_source() -> str:
    """Where api_key() finds the key, without revealing it: 'environment', 'config.local.json' or ''."""
    if any(os.environ.get(n, "").strip() for n in KEY_NAMES):
        return "environment"
    return "config.local.json" if api_key() else ""


def image_part(im, max_side: int) -> dict:
    """A PIL image or an image path as an inline JPEG part, longest side at most max_side."""
    from PIL import Image

    if not hasattr(im, "size"):
        with Image.open(im) as f:
            im = f.convert("RGB")
    im = im.convert("RGB")
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}}


def _opener(proxy: str | None):
    p = resolve_proxy(proxy)
    return build_opener(ProxyHandler({"http": p, "https": p} if p else {}))


def ask(parts: list[dict], model: str | None = None, proxy: str | None = None, json_mode: bool = True,
        timeout: int = 300, key: str | None = None) -> tuple[str, dict]:
    """One generateContent call, retried on 429/5xx and network errors. Returns (text, meta); raises RuntimeError."""
    key = key or api_key()
    if not key:
        raise RuntimeError("no Gemini key: set GEMINI_API_KEY, or gemini_env_file in config.local.json")
    model = model_name(model)
    config: dict = {"temperature": 0.2}
    if json_mode:
        config["responseMimeType"] = "application/json"
    body = json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": config}).encode()
    req = Request(ENDPOINT.format(model=model) + ":generateContent", data=body,
                  headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    last = ""
    for attempt in range(3):
        t0 = time.time()
        try:
            with _opener(proxy).open(req, timeout=timeout) as r:
                d = json.load(r)
            c = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []) if not p.get("thought"))
            return text, {"model": model, "seconds": round(time.time() - t0, 1),
                          "tokens": (d.get("usageMetadata") or {}).get("totalTokenCount"), "finish": c.get("finishReason")}
        except HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, OSError, ValueError) as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
        if attempt < 2:
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(last.replace(key, "[key]"))


def probe(model: str | None = None, proxy: str | None = None, timeout: int = 15) -> tuple[bool, str]:
    """Check that the key works and the model exists (a metadata request: no tokens are spent)."""
    key = api_key()
    if not key:
        return False, "no key"
    model = model_name(model)
    try:
        with _opener(proxy).open(Request(ENDPOINT.format(model=model), headers={"x-goog-api-key": key}), timeout=timeout) as r:
            return True, f"{model} available (HTTP {r.status})"
    except HTTPError as e:
        return False, f"{model}: HTTP {e.code}"
    except (URLError, TimeoutError, OSError) as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def parse_json(text: str) -> dict:
    """The JSON object in a model reply, tolerating code fences and stray text; {} when there is none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}
