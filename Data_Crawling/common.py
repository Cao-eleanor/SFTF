"""Shared local paths, table IO, authentication, and collection manifests."""

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def parse_link(value):
    """Validate a post URL and return (uid, bid, normalized URL)."""
    parsed = urlparse(str(value).strip())
    parts = parsed.path.strip('/').split('/')
    if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('weibo.com', 'www.weibo.com') or len(parts) != 2:
        raise ValueError('Expected a URL of the form https://weibo.com/<uid>/<bid>')
    uid, bid = parts
    if not uid.isdigit() or not bid.isascii() or not bid.isalnum():
        raise ValueError('Invalid account or post identifier in URL')
    return uid, bid, f'https://weibo.com/{uid}/{bid}'


def read_table(path):
    """Read identifiers as text; retain missing cells as empty strings."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError("Input must be a .csv or .xlsx file")


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def write_table(frame, path):
    """Create parent directories and atomically replace a table file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    if path.suffix.lower() == ".csv":
        frame.to_csv(temp, index=False, encoding="utf-8-sig")
    elif path.suffix.lower() == ".xlsx":
        frame.to_excel(temp, index=False, engine="openpyxl")
    else:
        raise ValueError("Output must be a .csv or .xlsx file")
    os.replace(temp, path)


def load_cookie(path=None):
    """Read a private cookie file or WEIBO_COOKIE; never log its contents."""
    value = Path(path).read_text(encoding="utf-8-sig").strip() if path else os.getenv("WEIBO_COOKIE", "").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Set WEIBO_COOKIE or provide a single-line --cookie-file")
    return value


def make_session(cookie):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://weibo.com/",
        "Cookie": cookie,
    })
    for item in cookie.split(";"):
        key, sep, value = item.strip().partition("=")
        if sep and key == "XSRF-TOKEN":
            session.headers["X-XSRF-TOKEN"] = value
    return session


def get_json(session, url, params):
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected API response; check the login session")
    if data.get("ok") == 0:
        raise ValueError("Weibo API rejected the request; check login and post availability")
    return data


def manifest(directory, script, settings, files):
    """Record non-secret settings and SHA-256 hashes of generated files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in map(Path, files):
        if path.is_file():
            hashes[os.path.relpath(path, directory)] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"script": script, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "python_version": platform.python_version(), "settings": settings, "sha256": hashes,
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(ROOT.glob('*.py'))}}
    target = directory / f"{Path(script).stem}.manifest.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)
