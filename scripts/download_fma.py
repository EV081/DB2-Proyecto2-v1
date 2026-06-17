"""Descarga FMA small (~8GB) desde Switch cloud, valida SHA1, descomprime
y aplana la estructura anidada `fma_small/000/000002.mp3` en
`data/fma_small_flat/000002.mp3` para que `etl_music.py` la consuma.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

URL = os.environ.get("FMA_SMALL_URL", "https://os.unil.cloud.switch.ch/fma/fma_small.zip")
SHA1 = os.environ.get("FMA_SMALL_SHA1", "ade154f733639d52e35e32f5593efe5be76c6d70")


def _sha1(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def _curl(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["curl", "-L", "--fail", "-C", "-", "-o", str(dst), url]
    )


def run(out_dir: Path, skip_sha1: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "fma_small.zip"
    extract_root = out_dir / "fma_small"
    flat_dir = out_dir / "fma_small_flat"

    if not zip_path.exists():
        print(f"Descargando {URL} -> {zip_path} ...")
        _curl(URL, zip_path)
    else:
        print(f"[skip] zip ya presente: {zip_path}")

    if not skip_sha1:
        print("Validando SHA1 (puede tomar ~30s sobre 8GB)...")
        got = _sha1(zip_path)
        if got != SHA1:
            raise SystemExit(f"SHA1 mismatch: got {got}, expected {SHA1}")
        print("  OK")

    if not extract_root.exists():
        print(f"Descomprimiendo en {out_dir} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    else:
        print(f"[skip] ya descomprimido: {extract_root}")

    flat_dir.mkdir(exist_ok=True)
    n_linked = 0
    for mp3 in extract_root.rglob("*.mp3"):
        link = flat_dir / mp3.name
        if not link.exists():
            link.symlink_to(mp3.resolve())
            n_linked += 1
    n_total = sum(1 for _ in flat_dir.glob("*.mp3"))
    print(f"Aplanado: {n_linked} symlinks nuevos, {n_total} total en {flat_dir}")
    return {"audio_dir": str(flat_dir), "n_tracks": n_total}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Descarga FMA small -> data/")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--skip-sha1", action="store_true",
                   help="No verificar SHA1 (ahorra ~30s)")
    args = p.parse_args()
    if shutil.which("curl") is None:
        sys.exit("Necesitas `curl` instalado.")
    run(out_dir=args.out_dir, skip_sha1=args.skip_sha1)
