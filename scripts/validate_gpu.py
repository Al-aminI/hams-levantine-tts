"""Validate the HamsVITS stack on THIS machine (RTX 3060 12 GB) before resuming training.

Three checks, all dataset-independent:
  1. Environment  — torch/CUDA/transformers, GPU, bf16.
  2. Inference    — load the resume checkpoint, synthesize a real code-switch sample to WAV
                    (exercises torch + HF VitsModel + our embeddings + the checkpoint).
  3. Train graph  — run ONE full VitsObjective GAN step on a synthetic batch and sweep
                    batch size to find the largest that fits 12 GB (their 3090 used batch 16
                    at ~14 GB, which will NOT fit here — this finds our number).

    python scripts/validate_gpu.py --ckpt checkpoints/resume_from_hf
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Windows console is cp1252 by default; model text is Arabic/IPA -> force UTF-8.
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402


def hr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def env_check():
    hr("1. ENVIRONMENT")
    import transformers
    print(f"python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — training needs the GPU.")
    p = torch.cuda.get_device_properties(0)
    print(f"gpu          : {p.name} (sm_{p.major}{p.minor})")
    print(f"vram         : {p.total_memory / 2**30:.1f} GB")
    print(f"bf16         : {torch.cuda.is_bf16_supported()}")


def inference_check(ckpt: str):
    hr("2. INFERENCE (resume checkpoint -> WAV)")
    from hams_tts.models.hams_vits import HamsVITS
    from hams_tts.text import phoneme_inventory as PI
    import soundfile as sf
    import json

    t0 = time.perf_counter()
    model = HamsVITS.from_checkpoint(ckpt).cuda().eval()
    print(f"loaded {ckpt} in {time.perf_counter()-t0:.1f}s | sample_rate={model.sample_rate} Hz")

    # Use a real precomputed code-switch eval sample; derive ids from its IPA.
    sample = REPO / "samples/neural/eval/cs_01.phonemes.json"
    meta = json.loads(sample.read_text(encoding="utf-8"))
    ids, syms = PI.encode(meta["ipa"], add_bos_eos=True)
    lang = meta.get("language_ids") or []
    if len(lang) != len(ids):  # rebuild a length-matched lang stream (BOS/EOS -> PAD)
        lang = [PI.Lang.PAD] + [PI.Lang.AR] * (len(ids) - 2) + [PI.Lang.PAD]
    print(f"text  : {meta['text']}")
    print(f"ipa   : {meta['ipa']}")
    print(f"tokens: {len(ids)} phonemes")

    pid = torch.tensor([ids], device="cuda")
    lid = torch.tensor([lang], device="cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wav = model.infer(pid, lid, length_scale=float(meta.get("length_scale", 5.0)))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1

    wav_np = wav.squeeze().cpu().numpy()
    dur = len(wav_np) / model.sample_rate
    out = REPO / "checkpoints" / "validate_infer.wav"
    sf.write(str(out), wav_np, model.sample_rate)
    rtf = dt / max(dur, 1e-6)
    print(f"synth : {dur:.2f}s audio in {dt*1000:.0f} ms  (RTF {rtf:.3f})")
    print(f"wrote : {out}")
    print(f"peak VRAM (infer): {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


def _synth_batch(bs, n_phon=48, spec_frames=220, n_fft=1024, hop=256, vocab=88):
    freq = n_fft // 2 + 1
    wav_len = spec_frames * hop
    return {
        "phoneme_ids": torch.randint(6, vocab, (bs, n_phon), device="cuda"),
        "phoneme_lengths": torch.full((bs,), n_phon, device="cuda"),
        "language_ids": torch.randint(1, 4, (bs, n_phon), device="cuda"),
        "spec": torch.rand(bs, freq, spec_frames, device="cuda") + 1e-4,
        "spec_lengths": torch.full((bs,), spec_frames, device="cuda"),
        "wav": torch.rand(bs, wav_len, device="cuda") * 2 - 1,
        "wav_lengths": torch.full((bs,), wav_len, device="cuda"),
        "speaker_id": torch.zeros(bs, dtype=torch.long, device="cuda"),
    }


def train_graph_check(ckpt: str, batches):
    hr("3. TRAIN GRAPH + VRAM SWEEP (full GAN step, synthetic batch)")
    from hams_tts.models.hams_vits import HamsVITS
    from hams_tts.training.objective import VitsObjective

    model = HamsVITS.from_checkpoint(ckpt).cuda().train()
    for p in model.parameters():
        p.requires_grad_(True)
    obj = VitsObjective(model, seg_size=8192, sample_rate=model.sample_rate).cuda()
    opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(obj.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99))
    g = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    d = sum(p.numel() for p in obj.discriminator.parameters()) / 1e6
    print(f"generator {g:.1f}M trainable | discriminator {d:.1f}M")

    best = 0
    for bs in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            for _ in range(3):  # a few steps so allocator reaches steady state
                logs = obj.step(_synth_batch(bs), opt_g, opt_d, 0)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 3
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  batch {bs:>2}: OK  peak {peak:5.2f} GB  {dt*1000:6.0f} ms/step  "
                  f"({1/dt:.2f} it/s)  g={logs['loss_g']:.1f} d={logs['loss_d']:.1f}")
            best = bs
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>2}: OOM (exceeds 12 GB)")
            break
        except Exception as e:
            print(f"  batch {bs:>2}: ERROR {type(e).__name__}: {str(e)[:200]}")
            break
    if best:
        print(f"\n-> Largest batch that fits: {best}. "
              f"Recipe: batch {best} + grad_accum {max(1, 16 // best)} = effective 16 (their 3090 setting).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "checkpoints/resume_from_hf"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--batches", default="4,8,12,16")
    args = ap.parse_args()

    env_check()
    inference_check(args.ckpt)
    if not args.skip_train:
        train_graph_check(args.ckpt, [int(x) for x in args.batches.split(",")])
    hr("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
