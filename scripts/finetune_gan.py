"""Full-objective VITS fine-tune: everything trainable + mel-reconstruction + adversarial
GAN (MPD/MSD) + KL + duration.  Warm-started from the reconstruction checkpoint.

This is the path to *intelligible* audio: the mel + adversarial losses give direct
audio-level supervision tying phonemes to sound (the reconstruction-only recipe lacked
that). bf16 autocast (Ampere-stable). Checkpoints frequently so intermediate models can
be auditioned.

    python scripts/finetune_gan.py --manifest /workspace/data/manifests/train.phon.jsonl \
        --ckpt /workspace/ckpt_ft2/final --out /workspace/ckpt_gan --steps 16000 --batch 16
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from hams_tts.models.hams_vits import HamsVITS, HamsVITSConfig
from hams_tts.training._torch_dataset import VitsCodeSwitchDataset, collate
from hams_tts.training.objective import VitsObjective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", default=None, help="warm-start HamsVITS checkpoint dir")
    ap.add_argument("--base", default="facebook/mms-tts-ara")
    ap.add_argument("--out", default="/workspace/ckpt_gan")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--c-mel", type=float, default=45.0)
    ap.add_argument("--c-kl", type=float, default=1.0)
    ap.add_argument("--c-dur", type=float, default=1.0)
    ap.add_argument("--c-fm", type=float, default=2.0)
    ap.add_argument("--seg-size", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    dev = "cuda"
    model = (HamsVITS.from_checkpoint(args.ckpt) if args.ckpt
             else HamsVITS(HamsVITSConfig(base_model_id=args.base, num_languages=4, num_speakers=1)))
    model = model.to(dev).train()
    for p in model.parameters():  # full objective: everything trainable
        p.requires_grad_(True)

    obj = VitsObjective(model, seg_size=args.seg_size, sample_rate=model.sample_rate,
                        c_mel=args.c_mel, c_kl=args.c_kl, c_dur=args.c_dur, c_fm=args.c_fm).to(dev)
    opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    opt_d = torch.optim.AdamW(obj.discriminator.parameters(), lr=args.lr, betas=(0.8, 0.99))

    ds = VitsCodeSwitchDataset(args.manifest, sample_rate=model.sample_rate)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                    num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True)
    g_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in obj.discriminator.parameters())
    print(f"[gan] G {g_params/1e6:.1f}M  D {d_params/1e6:.1f}M | {len(ds)} clips @ {model.sample_rate} Hz "
          f"| batch {args.batch} | warm-start {args.ckpt}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    step, t0 = 0, time.perf_counter()
    while step < args.steps:
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            logs = obj.step(batch, opt_g, opt_d, step)
            if step % args.log_every == 0:
                sps = (step + 1) / (time.perf_counter() - t0)
                print(f"[{step:>6}] g={logs['loss_g']:.2f} d={logs['loss_d']:.2f} "
                      f"mel={logs['mel']:.2f} kl={logs['kl']:.2f} dur={logs['dur']:.2f} "
                      f"adv={logs['adv']:.2f} fm={logs['fm']:.2f} {sps:.2f}it/s", flush=True)
            step += 1
            if step % args.save_every == 0:
                model.save_checkpoint(os.path.join(args.out, f"step_{step}"))
                print(f"[gan] saved step_{step}", flush=True)
            if step >= args.steps:
                break
    model.save_checkpoint(os.path.join(args.out, "final"))
    print(f"[gan] done -> {args.out}/final", flush=True)


if __name__ == "__main__":
    main()
