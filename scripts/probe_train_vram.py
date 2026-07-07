"""Measure REAL-data training VRAM + throughput on the 3060, warm-started from the
resume checkpoint, to pick a safe batch size before the long run. Sweeps batch sizes;
uses num_workers=0 (Windows-safe) and the actual dataset/objective."""
import argparse, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch
from torch.utils.data import DataLoader
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.training._torch_dataset import VitsCodeSwitchDataset, collate
from hams_tts.training.objective import VitsObjective

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(REPO / "checkpoints/resume_from_hf"))
ap.add_argument("--manifest", default=str(REPO / "data/manifests/train.phon.filtered.jsonl"))
ap.add_argument("--batches", default="8,12,16")
ap.add_argument("--iters", type=int, default=6)
args = ap.parse_args()

model = HamsVITS.from_checkpoint(args.ckpt).cuda().train()
for p in model.parameters():
    p.requires_grad_(True)
obj = VitsObjective(model, seg_size=8192, sample_rate=model.sample_rate).cuda()
opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, betas=(0.8, 0.99))
opt_d = torch.optim.AdamW(obj.discriminator.parameters(), lr=2e-4, betas=(0.8, 0.99))
ds = VitsCodeSwitchDataset(args.manifest, sample_rate=model.sample_rate)
print(f"dataset: {len(ds)} clips @ {model.sample_rate} Hz")

for bs in [int(x) for x in args.batches.split(",")]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate,
                    num_workers=0, drop_last=True)
    try:
        it = iter(dl); t0 = time.perf_counter(); n = 0
        for _ in range(args.iters):
            batch = next(it)
            batch = {k: v.cuda() for k, v in batch.items()}
            logs = obj.step(batch, opt_g, opt_d, 0)
            n += 1
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  batch {bs:>2}: OK  peak {peak:5.2f} GB  {dt*1000:6.0f} ms/step  "
              f"({1/dt:.2f} it/s)  mel={logs['mel']:.1f} kl={logs['kl']:.2f}")
    except torch.cuda.OutOfMemoryError:
        print(f"  batch {bs:>2}: OOM")
        break
    except Exception as e:
        print(f"  batch {bs:>2}: {type(e).__name__}: {str(e)[:160]}")
        break
