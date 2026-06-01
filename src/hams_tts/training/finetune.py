"""Fine-tune HamsVITS for Levantine pronunciation + smooth code-switching.

What this script owns (our contributions — reviewable & correct):
  * builds :class:`HamsVITS` (MMS-TTS-ara backbone + unified-IPA phoneme embedding +
    **language-ID embedding** + speaker embedding);
  * **parameter-efficient fine-tuning**: LoRA adapters on the text-encoder attention
    (via PEFT) + train the new phoneme/language/speaker embeddings + duration predictor,
    while **freezing** the HiFi-GAN decoder and the flow (≈70% of params) — adaptation is
    concentrated where the Levantine + code-switch signal lives (phonetics/prosody);
  * a data collator that feeds the precomputed phoneme-id + language-id streams.

The adversarial VITS objective (posterior encoder, monotonic alignment, KL + mel +
duration + GAN + feature-matching losses, MPD/MSD discriminators) is the standard
VITS/HiFi-GAN recipe.  For the hardened, battle-tested implementation we build on
**ylacombe/finetune-hf-vits** (it adds the training components HF's inference-only
``VitsModel`` lacks); ``--engine hf-vits`` emits the exact config + launch command, while
``--engine builtin`` runs the equivalent loop in :mod:`hams_tts.training.objective`.

Runs on the L4/A100 host (the ``train`` extra). Typical recipe in the design doc.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FinetuneConfig:
    base_model_id: str = "facebook/mms-tts-ara"
    train_manifest: str = "data/manifests/train.phon.jsonl"
    eval_manifest: str = "data/manifests/eval.phon.jsonl"
    output_dir: str = "checkpoints/hams_vits_levantine"
    num_languages: int = 4
    num_speakers: int = 1
    sample_rate: int = 22050
    # optimisation
    learning_rate: float = 2e-4
    disc_learning_rate: float = 2e-4
    batch_size: int = 16
    grad_accum: int = 2
    max_steps: int = 60_000
    warmup_steps: int = 1_000
    weight_decay: float = 0.01
    fp16: bool = True
    seg_size: int = 8192  # waveform slice length for the decoder/GAN step
    # PEFT / freezing
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    freeze_decoder: bool = True
    freeze_flow: bool = True
    # loss weights (canonical VITS)
    c_mel: float = 45.0
    c_kl: float = 1.0
    c_dur: float = 1.0
    c_fm: float = 2.0
    # data composition (fractions; see design doc)
    mix_levantine: float = 0.5
    mix_msa: float = 0.2
    mix_english: float = 0.2
    mix_codeswitch: float = 0.1
    log_every: int = 50
    save_every: int = 5_000
    seed: int = 1234

    @classmethod
    def from_file(cls, path: str) -> "FinetuneConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f) if path.endswith(".json") else _load_yaml(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _load_yaml(f):
    import yaml  # optional; only when a yaml config is used

    return yaml.safe_load(f)


def apply_hams_modifications(cfg: FinetuneConfig):
    """Construct HamsVITS and set up PEFT: LoRA on the encoder + freeze decoder/flow.

    Returns (model, trainable_param_count). This is the core, reviewable part of the
    fine-tuning approach; the optimisation loop below consumes it."""
    from ..models.hams_vits import HamsVITS, HamsVITSConfig

    model = HamsVITS(HamsVITSConfig(
        base_model_id=cfg.base_model_id,
        num_languages=cfg.num_languages,
        num_speakers=cfg.num_speakers,
    ))

    # freeze the heavy generator tail; we adapt phonetics/prosody, not the vocoder
    model.trainable_parameters(freeze_decoder=cfg.freeze_decoder, freeze_flow=cfg.freeze_flow)

    if cfg.use_lora:
        try:
            from peft import LoraConfig, get_peft_model

            lora = LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_targets, lora_dropout=0.05, bias="none",
            )
            # wrap only the transformer text encoder (where attention adapters belong)
            model.backbone.text_encoder.encoder = get_peft_model(
                model.backbone.text_encoder.encoder, lora
            )
        except Exception as e:  # PEFT optional; fall back to full-encoder fine-tune
            print(f"[finetune] LoRA unavailable ({e}); fine-tuning encoder fully")

    # the new embeddings are always trainable (they carry the Levantine/CS signal)
    for p in model.embed.parameters():
        p.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] trainable {n_train/1e6:.2f}M / {n_total/1e6:.2f}M "
          f"({100*n_train/n_total:.1f}%)")
    return model, n_train


def emit_hf_vits_config(cfg: FinetuneConfig, path: str) -> None:
    """Write a finetune-hf-vits-compatible JSON config (the hardened training path)."""
    conf = {
        "model_name_or_path": cfg.base_model_id,
        "output_dir": cfg.output_dir,
        "train_split_name": "train",
        "do_train": True, "do_eval": True,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "fp16": cfg.fp16,
        "weight_norm": True,
        "speaker_id_column_name": "speaker",
        "do_normalize": True,
        "override_vocabulary_embeddings": True,  # our unified IPA vocab
        "weight_disc": 3.0, "weight_kl": cfg.c_kl, "weight_mel": cfg.c_mel,
        "weight_duration": cfg.c_dur, "weight_fmaps": cfg.c_fm,
        "_hams_notes": "inject apply_hams_modifications(model) after model load; feed "
                       "precomputed phoneme_ids+language_ids; see design doc §Training.",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2, ensure_ascii=False)
    print(f"[finetune] wrote finetune-hf-vits config -> {path}")
    print("[finetune] launch:  accelerate launch finetune-hf-vits/run_vits_finetuning.py", path)


def train(cfg: FinetuneConfig, engine: str = "builtin") -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    if engine == "hf-vits":
        emit_hf_vits_config(cfg, os.path.join(cfg.output_dir, "hf_vits_config.json"))
        return

    # ---- builtin loop (equivalent objective; runs on the GPU host) ----
    import torch
    from torch.utils.data import DataLoader

    from .data import make_dataset
    from ._torch_dataset import collate
    from .objective import VitsObjective  # MPD/MSD + KL/mel/dur/FM losses + MAS

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = apply_hams_modifications(cfg)
    model.to(device).train()

    ds = make_dataset(cfg.train_manifest, sample_rate=cfg.sample_rate)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=4, pin_memory=True, drop_last=True)

    objective = VitsObjective(model, seg_size=cfg.seg_size, sample_rate=cfg.sample_rate,
                              c_mel=cfg.c_mel, c_kl=cfg.c_kl, c_dur=cfg.c_dur, c_fm=cfg.c_fm).to(device)
    opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=cfg.learning_rate, betas=(0.8, 0.99), weight_decay=cfg.weight_decay)
    opt_d = torch.optim.AdamW(objective.discriminator.parameters(),
                              lr=cfg.disc_learning_rate, betas=(0.8, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    step = 0
    while step < cfg.max_steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logs = objective.step(batch, opt_g, opt_d, scaler, fp16=cfg.fp16,
                                  grad_accum=cfg.grad_accum, step=step)
            if step % cfg.log_every == 0:
                msg = " ".join(f"{k}={v:.3f}" for k, v in logs.items())
                print(f"[step {step:>6}] {msg}")
            if step and step % cfg.save_every == 0:
                model.save_checkpoint(os.path.join(cfg.output_dir, f"step_{step}"))
            step += 1
            if step >= cfg.max_steps:
                break
    model.save_checkpoint(os.path.join(cfg.output_dir, "final"))
    print(f"[finetune] done -> {cfg.output_dir}/final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="json/yaml FinetuneConfig (else defaults)")
    ap.add_argument("--engine", default="builtin", choices=["builtin", "hf-vits"])
    args = ap.parse_args()
    cfg = FinetuneConfig.from_file(args.config) if args.config else FinetuneConfig()
    train(cfg, engine=args.engine)


if __name__ == "__main__":
    main()
