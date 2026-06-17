from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from src.api.search_service import (
    clear_caches,
    search_fashion_desc,
    search_fashion_image,
    search_music_audio,
    search_music_lyrics,
)
from src.db.database import get_session


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarize(latencies: list[float], successes: int) -> dict:
    if not latencies:
        return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "throughput_qps": 0,
                "queries_with_results": 0, "n": 0}
    avg = statistics.mean(latencies)
    return {
        "avg_ms": round(avg, 4),
        "p50_ms": round(_percentile(latencies, 50), 4),
        "p95_ms": round(_percentile(latencies, 95), 4),
        "throughput_qps": round(1000.0 / avg, 2) if avg > 0 else 0,
        "queries_with_results": successes,
        "n": len(latencies),
    }


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def _sample_lyrics(n: int, seed: int) -> list[str]:
    with get_session() as session:
        rows = session.execute(text(
            "SELECT lyrics_text FROM songs WHERE lyrics_text IS NOT NULL"
        )).all()
    texts = [r[0] for r in rows if r[0]]
    if not texts:
        return []
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        txt = rng.choice(texts)
        words = txt.split()
        if len(words) <= 4:
            queries.append(txt)
        else:
            start = rng.randint(0, len(words) - 4)
            queries.append(" ".join(words[start:start + rng.randint(2, 5)]))
    return queries


def _sample_descriptions(n: int, seed: int) -> list[str]:
    with get_session() as session:
        rows = session.execute(text(
            "SELECT description FROM products WHERE description IS NOT NULL"
        )).all()
    texts = [r[0] for r in rows if r[0]]
    if not texts:
        return []
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        txt = rng.choice(texts)
        words = txt.split()
        if len(words) <= 4:
            queries.append(txt)
        else:
            start = rng.randint(0, len(words) - 4)
            queries.append(" ".join(words[start:start + rng.randint(2, 5)]))
    return queries


def _sample_audio_paths(n: int, seed: int) -> list[Path]:
    with get_session() as session:
        rows = session.execute(text(
            "SELECT audio_path FROM songs WHERE audio_path IS NOT NULL"
        )).all()
    paths = [Path(r[0]) for r in rows if r[0] and Path(r[0]).exists()]
    if not paths:
        return []
    rng = random.Random(seed)
    return [rng.choice(paths) for _ in range(n)]


def _sample_image_paths(n: int, seed: int) -> list[Path]:
    with get_session() as session:
        rows = session.execute(text(
            "SELECT image_path FROM products WHERE image_path IS NOT NULL"
        )).all()
    paths = [Path(r[0]) for r in rows if r[0] and Path(r[0]).exists()]
    if not paths:
        return []
    rng = random.Random(seed)
    return [rng.choice(paths) for _ in range(n)]


# ----------------------------------------------------------------------------
# Runner por (app, modalidad)
# ----------------------------------------------------------------------------
def _run_text(queries: list[str], search_fn, engines: list[str], k: int) -> dict:
    out = {}
    for engine in engines:
        latencies = []
        successes = 0
        for q in queries:
            t0 = time.perf_counter()
            try:
                r = search_fn(q, engine, k)
                if r.get("results"):
                    successes += 1
            except Exception as e:
                print(f"  [WARN] {engine}/{q[:30]!r}: {e}")
                continue
            latencies.append((time.perf_counter() - t0) * 1000)
        out[engine] = _summarize(latencies, successes)
    return out


def _run_file(queries: list[Path], search_fn, engines: list[str], k: int) -> dict:
    out = {}
    for engine in engines:
        latencies = []
        successes = 0
        for p in queries:
            t0 = time.perf_counter()
            try:
                r = search_fn(p, engine, k)
                if r.get("results"):
                    successes += 1
            except Exception as e:
                print(f"  [WARN] {engine}/{p.name}: {e}")
                continue
            latencies.append((time.perf_counter() - t0) * 1000)
        out[engine] = _summarize(latencies, successes)
    return out


# ----------------------------------------------------------------------------
# Reporte markdown
# ----------------------------------------------------------------------------
def _markdown_report(report: dict) -> str:
    lines = ["# Benchmark Fase 4 — DB2 Proyecto 2\n"]
    lines.append(f"- n_queries por engine: **{report['n_queries']}**")
    lines.append(f"- k (top-K): **{report['k']}**")
    lines.append("")
    for section in ("music_lyrics", "music_audio", "fashion_desc", "fashion_image"):
        if section not in report:
            continue
        lines.append(f"## {section}\n")
        lines.append("| engine | avg ms | p50 ms | p95 ms | qps | con resultados | n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for engine, m in report[section].items():
            lines.append(
                f"| {engine} | {m['avg_ms']:.2f} | {m['p50_ms']:.2f} | "
                f"{m['p95_ms']:.2f} | {m['throughput_qps']:.1f} | "
                f"{m['queries_with_results']}/{m['n']} | {m['n']} |"
            )
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Fase 4")
    parser.add_argument("--queries", type=int, default=50,
                        help="Numero de queries por motor")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default="benchmark_fase4.json")
    parser.add_argument("--out-md", type=str, default="benchmark_fase4.md")
    parser.add_argument("--apps", default="music,fashion",
                        help="Apps a benchmarkear (coma-separadas)")
    args = parser.parse_args()
    apps = set(args.apps.split(","))

    report: dict = {"n_queries": args.queries, "k": args.k, "seed": args.seed}

    if "music" in apps:
        print("\n[music/lyrics]")
        qs = _sample_lyrics(args.queries, args.seed)
        if qs:
            report["music_lyrics"] = _run_text(
                qs, search_music_lyrics,
                ["spimi", "gin", "gist", "pgvector"], args.k,
            )
        else:
            print("  (skip) no hay songs con lyrics_text")

        print("\n[music/audio]")
        qs = _sample_audio_paths(args.queries, args.seed)
        if qs:
            report["music_audio"] = _run_file(
                qs, search_music_audio,
                ["spimi", "pgvector"], args.k,
            )
        else:
            print("  (skip) no hay songs con audio_path existente")

    if "fashion" in apps:
        print("\n[fashion/description]")
        qs = _sample_descriptions(args.queries, args.seed)
        if qs:
            report["fashion_desc"] = _run_text(
                qs, search_fashion_desc,
                ["spimi", "gin", "gist", "pgvector"], args.k,
            )
        else:
            print("  (skip) no hay products con description")

        print("\n[fashion/image]")
        qs = _sample_image_paths(args.queries, args.seed)
        if qs:
            report["fashion_image"] = _run_file(
                qs, search_fashion_image,
                ["spimi", "pgvector"], args.k,
            )
        else:
            print("  (skip) no hay products con image_path existente")

    clear_caches()

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(_markdown_report(report), encoding="utf-8")
    print(f"\nReporte JSON: {args.out_json}")
    print(f"Reporte MD:   {args.out_md}")


if __name__ == "__main__":
    main()