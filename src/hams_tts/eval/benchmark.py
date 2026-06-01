"""Reproducible TTFA / RTF / peak-VRAM benchmark against the target hardware.

Runs the eval set (or a fixed prompt set) N times per utterance, reports:
  * TTFA p50 / p95 (ms)   — time to first audio chunk (the streaming latency that matters)
  * RTF  mean / p95       — synthesis_time / audio_duration
  * peak VRAM (MB)        — torch.cuda.max_memory_allocated on GPU; nvidia-smi fallback;
                            CPU RSS clearly labelled as a proxy when no GPU is present

and prints a Markdown table comparing against the assessment KPIs, plus a JSON dump.

The script is backend-agnostic — point it at the deployable engine on the L4:
    python -m hams_tts.eval.benchmark --backend tensorrt --model-path hams_vits.plan
or validate the plumbing anywhere:
    python -m hams_tts.eval.benchmark --backend espeak --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import List, Optional

from ..inference.engine import build_engine

KPI = {"vram_mb": 3072, "ttfa_ms": 300, "rtf": 0.3}

DEFAULT_PROMPTS = [
    "مَرحَبا، كِيفَك؟ إن شاء الله مْنيح.",
    "Hello, how are you doing today?",
    "بَدّي إحجِز flight من بيروت to London بُكرا الساعة 9.",
]


def _peak_vram_mb() -> tuple[Optional[float], str]:
    """Return (peak_mb, source). Prefers CUDA; falls back to nvidia-smi; else CPU RSS."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6, "cuda.max_memory_allocated"
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return float(out.strip().splitlines()[0]), "nvidia-smi(memory.used)"
    except Exception:
        pass
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        rss_mb = rss_kb / 1e6 if rss_kb > 1e7 else rss_kb / 1e3
        return rss_mb, "CPU-RSS-proxy(NOT-gpu-vram)"
    except Exception:
        return None, "unavailable"


def _reset_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def benchmark(backend: str, prompts: List[str], runs: int = 5, warmup: int = 2,
              output_sr: int = 24000, **kw) -> dict:
    engine = build_engine(backend, output_sample_rate=output_sr, **kw)
    engine.warmup()
    for _ in range(warmup):
        engine.synthesize(prompts[0])

    _reset_vram()
    ttfa_ms: List[float] = []
    rtf: List[float] = []
    audio_total = 0.0
    wall_total = 0.0

    for r in range(runs):
        for text in prompts:
            t0 = time.perf_counter()
            first_t: Optional[float] = None
            n_samples = 0
            for ch in engine.stream(text):
                if first_t is None:
                    first_t = time.perf_counter() - t0
                n_samples += ch.pcm.shape[0]
            wall = time.perf_counter() - t0
            audio_s = n_samples / output_sr if n_samples else 0.0
            ttfa_ms.append((first_t or wall) * 1000)
            rtf.append(wall / max(audio_s, 1e-9))
            audio_total += audio_s
            wall_total += wall

    peak_mb, vram_src = _peak_vram_mb()

    def p(vals, q):
        return statistics.quantiles(vals, n=100)[q - 1] if len(vals) >= 2 else vals[0]

    results = {
        "backend": backend,
        "n_measurements": len(ttfa_ms),
        "ttfa_ms_p50": round(statistics.median(ttfa_ms), 1),
        "ttfa_ms_p95": round(p(ttfa_ms, 95), 1),
        "rtf_mean": round(statistics.mean(rtf), 4),
        "rtf_p95": round(p(rtf, 95), 4),
        "peak_vram_mb": round(peak_mb, 1) if peak_mb is not None else None,
        "vram_source": vram_src,
        "aggregate_rtf": round(wall_total / max(audio_total, 1e-9), 4),
        "output_sample_rate": output_sr,
    }
    return results


def render_table(r: dict) -> str:
    def verdict(val, target, lower_is_better=True):
        if val is None:
            return "—"
        ok = (val <= target) if lower_is_better else (val >= target)
        return "✅ PASS" if ok else "❌ FAIL"

    is_gpu = r["vram_source"].startswith(("cuda", "nvidia"))
    vram_note = "" if is_gpu else "  _(CPU RSS proxy — run on the L4 for the real VRAM number)_"
    rows = [
        "| Metric | Target | Measured | Verdict |",
        "|---|---|---|---|",
        f"| Peak VRAM (MB) | ≤ {KPI['vram_mb']} | {r['peak_vram_mb']} ({r['vram_source']}) | "
        f"{verdict(r['peak_vram_mb'], KPI['vram_mb']) if is_gpu else '—'+vram_note} |",
        f"| TTFA p50 (ms) | < {KPI['ttfa_ms']} | {r['ttfa_ms_p50']} | {verdict(r['ttfa_ms_p50'], KPI['ttfa_ms'])} |",
        f"| TTFA p95 (ms) | < {KPI['ttfa_ms']} | {r['ttfa_ms_p95']} | {verdict(r['ttfa_ms_p95'], KPI['ttfa_ms'])} |",
        f"| RTF mean | < {KPI['rtf']} | {r['rtf_mean']} | {verdict(r['rtf_mean'], KPI['rtf'])} |",
        f"| RTF p95 | < {KPI['rtf']} | {r['rtf_p95']} | {verdict(r['rtf_p95'], KPI['rtf'])} |",
    ]
    return "\n".join(rows)


def _load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [u["text"] for u in data["utterances"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="espeak", choices=["espeak", "onnx", "tensorrt", "torch"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--eval-set", default=None, help="path to eval_utterances.json (else 3 default prompts)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--output", default=None, help="write results JSON here")
    args = ap.parse_args()

    kw = {}
    if args.backend in ("onnx", "tensorrt", "torch"):
        key = {"onnx": "model_path", "tensorrt": "engine_path", "torch": "checkpoint"}[args.backend]
        kw[key] = args.model_path
    prompts = _load_prompts(args.eval_set)
    r = benchmark(args.backend, prompts, runs=args.runs, warmup=args.warmup,
                  output_sr=args.sample_rate, **kw)
    print(f"\n## Benchmark — backend={r['backend']}, {r['n_measurements']} measurements\n")
    print(render_table(r))
    print("\nRaw:", json.dumps(r, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
