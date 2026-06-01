"""Reconstruction fine-tune: adapt HamsVITS's text front-end (unified-IPA phoneme +
language-ID embeddings + text encoder + duration + flow) to the *pretrained* MMS-TTS
acoustic latent space, with the HiFi-GAN decoder and posterior encoder FROZEN.

Why this objective. With the decoder + posterior frozen, the losses that actually teach
the new phoneme/language embeddings are the **KL** (text-encoder prior learning to match
the pretrained posterior's latent distribution) and the **duration** loss (alignment).
This is a fast, low-risk way to make a brand-new IPA front-end drive an existing,
high-quality vocoder — exactly what we need to get intelligible code-switched audio
without GPU-days of adversarial training.

    python scripts/finetune_recon.py --manifest /workspace/data/manifests/train.phon.jsonl \
        --out /workspace/ckpt_ft --steps 4000 --batch 16 --lr 2e-4
    # add --smoke to run a couple of steps and exit (validates the training graph)
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from hams_tts.models.hams_vits import HamsVITS, HamsVITSConfig
from hams_tts.training._torch_dataset import VitsCodeSwitchDataset, collate
from hams_tts.training.objective import maximum_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", default="facebook/mms-tts-ara")
    ap.add_argument("--ckpt", default=None, help="resume from a HamsVITS checkpoint dir")
    ap.add_argument("--out", default="/workspace/ckpt_ft")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--c-kl", type=float, default=1.0)
    ap.add_argument("--c-dur", type=float, default=1.0)
    ap.add_argument("--seg-size", type=int, default=8192)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    if args.ckpt:
        model = HamsVITS.from_checkpoint(args.ckpt)
    else:
        model = HamsVITS(HamsVITSConfig(base_model_id=args.base, num_languages=4, num_speakers=1))
    model = model.to(dev).train()

    # freeze the pretrained vocoder + posterior; adapt the front-end into their latent space
    for p in model.backbone.decoder.parameters():
        p.requires_grad_(False)
    for p in model.backbone.posterior_encoder.parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[ft] trainable {n_tr/1e6:.2f}M params | base {args.base}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)

    sr = model.sample_rate
    ds = VitsCodeSwitchDataset(args.manifest, sample_rate=sr)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                    num_workers=6, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    print(f"[ft] dataset {len(ds)} clips @ {sr} Hz | batch {args.batch}")

    os.makedirs(args.out, exist_ok=True)
    step, t0 = 0, time.perf_counter()
    while step < args.steps:
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            out = model.training_forward(
                phoneme_ids=batch["phoneme_ids"], phoneme_lengths=batch["phoneme_lengths"],
                language_ids=batch["language_ids"], spec=batch["spec"],
                spec_lengths=batch["spec_lengths"], speaker_id=batch["speaker_id"],
                wav=batch["wav"], seg_size=args.seg_size, hop=args.hop, maximum_path=maximum_path,
            )
            loss = args.c_kl * out["kl"] + args.c_dur * out["dur_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()

            if step % args.log_every == 0:
                sps = (step + 1) / (time.perf_counter() - t0)
                print(f"[step {step:>5}] loss={loss.item():.3f} kl={out['kl'].item():.3f} "
                      f"dur={out['dur_loss'].item():.3f} gnorm={float(gn):.2f} {sps:.2f} it/s", flush=True)

            if args.smoke and step >= 2:
                print("[smoke] y_hat", tuple(out["y_hat"].shape), "y_slice", tuple(out["y_slice"].shape),
                      "| kl finite:", bool(torch.isfinite(out["kl"])), "| dur finite:", bool(torch.isfinite(out["dur_loss"])))
                print("[smoke] OK — training graph runs end-to-end")
                return

            step += 1
            if step % args.save_every == 0:
                model.save_checkpoint(os.path.join(args.out, f"step_{step}"))
                print(f"[ft] saved step_{step}", flush=True)
            if step >= args.steps:
                break

    model.save_checkpoint(os.path.join(args.out, "final"))
    print(f"[ft] done -> {args.out}/final")


if __name__ == "__main__":
    main()
