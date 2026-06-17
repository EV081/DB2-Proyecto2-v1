from __future__ import annotations

import argparse
import csv
from pathlib import Path


def run(metadata_csv: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with metadata_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = (row.get("stem") or "").strip()
            if not stem:
                continue
            parts = [
                row.get("name") or "",
                f"Category: {row.get('category', '')}",
                f"Subcategory: {row.get('subcategory', '')}",
            ]
            desc = ". ".join(p.strip() for p in parts if p.strip().rstrip(":"))
            (out_dir / f"{stem}.txt").write_text(desc, encoding="utf-8")
            n += 1
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Regenera fashion/descs/ desde metadata.csv")
    p.add_argument("--metadata-csv", type=Path, default=Path("data/fashion/metadata.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/fashion/descs"))
    args = p.parse_args()
    n = run(args.metadata_csv, args.out_dir)
    print(f"OK: {n} descripciones en {args.out_dir}")
