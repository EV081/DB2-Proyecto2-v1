"""Descarga jam-alt (Jamendo lyrics) desde HuggingFace y materializa
audio MP3 + letras .txt a disco para alimentar `etl_music.py`.

Va directo contra el endpoint /api/datasets/{repo}/parquet y lee los
Parquet auto-generados con pyarrow. Asi evitamos el paquete `datasets`,
que en versiones recientes exige `torch` + `torchcodec` solo para tocar
la columna `audio` (aunque uses streaming + decode=False).

Uso:
    python scripts/etl_jamalt.py --out-dir data/jamalt --language en
    python scripts/etl_music.py \
        --lyrics-dir data/jamalt/lyrics \
        --audio-dir  data/jamalt/audio \
        --metadata-csv data/jamalt/metadata.csv \
        --reset
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_DATASET = os.environ.get("JAMALT_HF_DATASET", "jamendolyrics/jam-alt")
DEFAULT_LANGUAGE = os.environ.get("JAMALT_LANGUAGE", "en")
DEFAULT_SPLIT = os.environ.get("JAMALT_SPLIT", "test")

PARQUET_API = "https://huggingface.co/api/datasets/{repo}/parquet"


def _safe_stem(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (raw or "").strip())
    return s.strip("_") or "song"


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def _fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def _resolve_parquet_urls(repo: str, config: str, split: str) -> list[str]:
    """Resuelve URLs de parquet via API publica de HF datasets.

    El response tiene forma {config: {split: [urls]}}.
    """
    api = PARQUET_API.format(repo=repo)
    data = _fetch_json(api)
    if config not in data:
        sys.exit(
            f"Config {config!r} no existe en {repo}. "
            f"Disponibles: {sorted(data.keys())}"
        )
    if split not in data[config]:
        sys.exit(
            f"Split {split!r} no existe en {repo}/{config}. "
            f"Disponibles: {sorted(data[config].keys())}"
        )
    return data[config][split]


def run(out_dir: Path, language: str | None, split: str) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        sys.exit(f"Falta pyarrow: pip install pyarrow ({e})")

    audio_dir = out_dir / "audio"
    lyrics_dir = out_dir / "lyrics"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lyrics_dir.mkdir(parents=True, exist_ok=True)

    config = language or "default"
    print(f"Resolviendo parquets de {HF_DATASET} (config={config}, split={split})...")
    urls = _resolve_parquet_urls(HF_DATASET, config, split)
    print(f"  {len(urls)} parquet(s) a procesar")

    seen: set[str] = set()
    meta_rows: list[dict] = []
    written = 0

    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] bajando {url.rsplit('/', 1)[-1]}...")
        raw = _fetch_bytes(url)
        table = pq.read_table(io.BytesIO(raw))
        rows = table.to_pylist()
        for row in rows:
            stem = _safe_stem(row.get("name") or row.get("title") or f"song_{written:03d}")
            base = stem
            k = 1
            while stem in seen:
                stem = f"{base}_{k}"
                k += 1
            seen.add(stem)

            audio = row.get("audio") or {}
            audio_bytes = audio.get("bytes")
            if not audio_bytes:
                ap = audio.get("path")
                if ap and Path(ap).exists():
                    audio_bytes = Path(ap).read_bytes()
            if not audio_bytes:
                print(f"    [SKIP audio] {stem}: sin bytes")
                continue
            (audio_dir / f"{stem}.mp3").write_bytes(audio_bytes)

            text = (row.get("text") or "").strip()
            if not text:
                print(f"    [SKIP lyrics] {stem}: text vacio")
                continue
            (lyrics_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

            meta_rows.append({
                "stem": stem,
                "title": row.get("title") or stem,
                "artist": row.get("artist") or "",
                "genre": row.get("genre") or "",
            })
            written += 1

    meta_path = out_dir / "metadata.csv"
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "title", "artist", "genre"])
        w.writeheader()
        w.writerows(meta_rows)

    print(f"OK: {written} canciones en {out_dir}")
    print(f"  audio  -> {audio_dir}")
    print(f"  lyrics -> {lyrics_dir}")
    print(f"  meta   -> {meta_path}")
    return {"written": written, "out_dir": str(out_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL jam-alt (HF) -> archivos en disco")
    parser.add_argument("--out-dir", type=Path, default=Path("data/jamalt"))
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help="Filtrar por idioma (en/fr/de/es). Vacio = config 'default'.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    args = parser.parse_args()
    if args.language and args.language.lower() in {"", "all", "any"}:
        args.language = None
    run(out_dir=args.out_dir, language=args.language, split=args.split)
