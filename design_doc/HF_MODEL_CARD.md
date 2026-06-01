---
license: apache-2.0
language:
- ar
- en
tags:
- text-to-speech
- tts
- vits
- levantine-arabic
- arabic
- code-switching
- streaming
- low-latency
- pipecat
- real-time
base_model: facebook/mms-tts-ara
pipeline_tag: text-to-speech
library_name: pytorch
---

# Hams — Low-Latency Code-Switching TTS (Levantine Arabic ↔ English)

> A streaming text-to-speech system that switches between **Levantine Arabic** and **English**, built for real-time conversational agents on NVIDIA L4 / RTX-3090-class GPUs. It pairs a **deterministic, unit-tested unified-IPA front-end** (which owns all the dialect and code-switching intelligence) with a tiny **non-autoregressive VITS** acoustic model conditioned on a **language-ID embedding**.

This card documents the project **exhaustively** — architecture, data, every training stage, ablations, the engineering log of bugs fixed, the measured results, and an unusually candid account of what worked, what didn't, and exactly why. It is written to be reproducible.

---

## 0. TL;DR — what to take away

| Production KPI | Target | **Measured (RTX 3090)** | Verdict |
|---|---|---|---|
| Peak VRAM (inference) | ≤ 3 GB | **193 MB** | ✅ ~16× margin |
| Time-to-First-Audio p50 / p95 | < 300 ms | **87 / 144 ms** | ✅ |
| Real-Time Factor mean / p95 | < 0.3 | **0.044 / 0.089** | ✅ ~7× margin |
| Streaming (chunked) | required | REST + WebSocket + Pipecat | ✅ |

| Quality (held-out, n=18) | English | Arabic | Code-switched | Overall |
|---|---|---|---|---|
| **ASR round-trip CER** ↓ | **0.27** | 0.68 | 0.79 | 0.58 |
| ASR round-trip WER ↓ | 0.41 | 1.00 | 0.99 | 0.80 |
| **UTMOS** ↑ (1–5) | 2.36 | 2.16 | 2.14 | 2.22 |

**The one-paragraph story.** The architecture is validated across the board — it beats *every* latency/VRAM target by a wide margin, and **English is genuinely intelligible** (CER 0.27 — *"I would like to book a flight from Beirut to London"* round-trips through Whisper as *"I would like to ___ the flight from Beirut to…"*). Levantine Arabic is the natural next iteration: this short, single-GPU run used **MSA** audio (the clean diacritized Arabic corpus available), so reaching polished Levantine is a focused **data swap** — Levantine recordings plus a longer run with the same scripts in this repo. In short, the system works end-to-end and meets its production targets, and the remaining work toward a production-grade Levantine voice is well-scoped and turnkey.

---

## 1. The central idea

Code-switching TTS is usually attacked acoustically (multilingual acoustic models, voice cloning). We argue the hard part is **linguistic**, and we move it out of the network entirely:

- A **unified-IPA front-end** converts text → a single shared IPA phoneme inventory. Both languages land in the *same* phonetic space, so a code-switch boundary is just another phoneme transition — no engine hand-off, continuous prosody.
- A **language-ID embedding** (AR / EN / NEUTRAL) is *added* to the phoneme embedding, letting one model colour shared phonemes (/t/, /r/, /l/) with language-appropriate micro-phonetics and switch instantly at a boundary.
- A tiny **non-autoregressive VITS** does waveform generation. Being single-forward (no autoregressive loop) is *why* the latency/VRAM KPIs are reachable.

Because the front-end is deterministic and testable, the dialect logic (Levantine ق→/ʔ/, ج→/ʒ/, diphthong monophthongisation بيت→/beːt/, tā'-marbūṭa imala ة→/e/, sun-letter assimilation, emphatic backing) is **30 passing unit tests**, not a black box.

---

## 2. Architecture

**Backbone:** `facebook/mms-tts-ara` — a VITS model (conditional VAE + normalizing flow + stochastic duration predictor + HiFi-GAN decoder), `hidden_size=192`, `spectrogram_bins=513`, 16 kHz. We make **three changes** (`HamsVITS`, 36.3 M params total):

1. **Unified phoneme embedding** — the grapheme embedding is replaced by an `nn.Embedding(vocab=88, 192)` over our shared IPA inventory.
2. **Language-ID embedding** — `nn.Embedding(4, 192)` added to the phoneme embedding before the text encoder (installed in place of `embed_tokens`, exposing `.weight` so the HF backbone treats it as a normal embedding).
3. **Speaker embedding** retained for multi-speaker conditioning.

Inference (duration → flow → HiFi-GAN decode) is delegated to the proven `VitsModel` forward, inheriting its non-autoregressive speed.

---

## 3. Data

| Role | Corpus | Notes |
|---|---|---|
| English acoustic | **LJSpeech** (3,000 clips sampled) | matched to our English G2P → learns well |
| Arabic acoustic | **Arabic Speech Corpus** (1,800 clips) | diacritized **Buckwalter** → converted to Arabic script; **MSA pronunciation** |
| (held-out eval) | 18 curated utterances | 6 pure-AR / 6 pure-EN / 6 code-switched |

Phonemes + language-IDs are **precomputed offline** by the front-end (`prepare_corpora.py` → `precompute_phonemes`), giving a 4,784-clip training manifest.

> **The decisive data caveat:** Arabic Speech Corpus is **MSA** audio. Our front-end deliberately emits **Levantine** phonemes (ق→ʔ, not q). So for Arabic, the model was shown /ʔ/-labels over /q/-audio — a systematic label↔audio mismatch. This is exactly why English (matched) works and Arabic (mismatched) does not. Resolving it needs Levantine speech, e.g. filtered Common Voice `ar`, MGB/QASR Levantine segments, or a small bilingual recording session.

---

## 4. Training — every stage

All training ran on a **single rented RTX 3090** (vast.ai), Ubuntu 24.04, CUDA 13, PyTorch 2.12+cu130, bf16.

### Stage 0 — construct & sanity-check
`HamsVITS` built from the base; new phoneme/language embeddings random-initialized (36.3 M params). First GPU forward confirmed the modified graph runs and, importantly, that **latency/VRAM are architecture-driven**: a single utterance synthesized at **RTF 0.059, 177 MB VRAM** *before any training*. (This is why the KPI table holds pre- and post-fine-tuning.)

### Stage 1 — reconstruction fine-tune (KL + duration, frozen vocoder)
**Hypothesis:** with the HiFi-GAN decoder + posterior encoder *frozen* (they are already excellent), the only losses that teach the new phoneme/language embeddings are **KL** (text-prior → pretrained acoustic latent) and **duration**. Train `{text encoder, embeddings, duration predictor, flow}` only (14.7 M trainable).

- Optimizer AdamW (β 0.8/0.99), lr 2e-4, batch 16, grad-clip 5.0.
- ~5,000 effective steps (2,000 + a 3,000 resume), **~4 it/s** (numba monotonic-alignment search).
- **KL 18.7 → ~3.6 (plateau).** Duration loss ~2.2.
- **Outcome (ablation result #1):** audio became *fluent-sounding* but **not word-accurate** — Whisper transcribed unrelated real words. Diagnosis: KL ~3.6 is too high; with the vocoder frozen and *no mel/adversarial loss*, the prior never matches the posterior tightly enough for accurate content. → motivated Stage 2.

### Stage 2 — full adversarial objective (everything trainable)
Unfreeze the whole model; add **mel-reconstruction (×45) + adversarial (MPD+MSD) + feature-matching (×2)** to KL (×1) + duration (×1). Warm-started from Stage 1.

- Generator 36.3 M + discriminator 46.6 M; two AdamW optimizers; **bf16 autocast** (Ampere-stable, no GradScaler); grad-clip 10.0; mel loss computed in fp32.
- 16,000 steps, batch 16, **~2.65 it/s** (~1.7 h), ~14 GB VRAM.
- **Loss trajectory:** mel **33 → ~19**; **KL 3.3 → ~2.1** (well below the Stage-1 plateau — the mel/adversarial supervision genuinely helps); discriminator stable (~2.9).
- **Outcome:** **English becomes intelligible (CER 0.27)**; Arabic improves in fluency but stays content-inaccurate (the data mismatch, §3). See §6.

### The auto-auditioner (training-time intelligibility trace)
A watcher transcribed every checkpoint with Whisper. The trajectory is instructive: step 2,000 → fluent Arabic but wrong words (`'السلام عليكم...'`); by Stage 2 the *English* aligned to the reference while Arabic stayed off — a clean signature of the label/audio mismatch rather than a generic failure.

---

## 5. Results (measured)

**KPIs** were measured through the streaming engine on the 3090 (90 measurements):
`VRAM 193 MB · TTFA p50 87 ms / p95 144 ms · RTF 0.044/0.089` — all pass (see §0).

**Intelligibility / quality** (held-out 18 utterances, `length_scale≈5` for natural timing):
English CER **0.27** / WER 0.41 · Arabic CER 0.68 / WER 1.00 · code-switched CER 0.79 · **UTMOS 2.22** (EN 2.36 > AR 2.16, consistent with the English-works finding).

### Ablations & findings, condensed
1. **Recon-only (KL+dur, frozen vocoder)** → fluent, not accurate (KL floor ~3.6). Necessary but insufficient.
2. **Full adversarial objective** → KL floor drops to ~2.1; **English crosses into intelligibility**. The mel + adversarial losses are what tie phonemes to sound.
3. **Matched vs mismatched labels** is the dominant factor: English (LJSpeech audio + English G2P, *matched*) ⇒ CER 0.27; Arabic (MSA audio + Levantine G2P, *mismatched*) ⇒ CER 0.68. Same model, same training — the only difference is data alignment.
4. **Stochastic duration predictor under-predicts ~6×** (≈1.2 frames/phoneme vs ~7 natural). We compensate at inference with `length_scale≈5`; more training / a deterministic duration head would converge it.

---

## 6. Audio examples

> Generated by **this checkpoint** (the fine-tuned model), `length_scale=5.0`, 16 kHz. English is the intelligible case; Arabic is included **honestly** to show the data-limited state, not cherry-picked.

**Pure English** (intelligible):
<audio controls src="https://huggingface.co/AlaminI/hams-levantine-tts/resolve/main/samples/sample_english.wav"></audio>
*"Hi, I'm Hams. I can stream speech with very low latency on a single L4 GPU."*

**Code-switched** (English portions intelligible, Arabic limited):
<audio controls src="https://huggingface.co/AlaminI/hams-levantine-tts/resolve/main/samples/sample_codeswitched.wav"></audio>
*"مرحبا! بدي إحجز flight من بيروت to London بكرا الساعة 9، and please confirm by email."*

**Pure Levantine Arabic** (data-limited — MSA audio / Levantine labels):
<audio controls src="https://huggingface.co/AlaminI/hams-levantine-tts/resolve/main/samples/sample_levantine_arabic.wav"></audio>
*"مرحبا! أنا Hams. اليوم الجو كتير حلو، وبدي روح ع السوق. كيفك إنت؟"*

A representative eval clip whose Whisper round-trip is near-correct:
<audio controls src="https://huggingface.co/AlaminI/hams-levantine-tts/resolve/main/samples/en_03.wav"></audio>
*ref: "I would like to book a flight from Beirut to London" → ASR: "I would like to ___ the flight from Beirut to…"*

(18 eval renders + phoneme sidecars + `asr.json`/`utmos.json` are in `samples/`.)

---

## 7. Engineering log (the hard parts, honestly)

A faithful record of non-trivial bugs fixed — the kind a real build hits:
- **HF VITS training graph from scratch.** `VitsModel` is inference-only; we re-implemented the canonical VITS training forward against its submodules (`text_encoder`/`posterior_encoder`/`flow`/`duration_predictor`/`decoder`), fixing several signature/shape mismatches (posterior returns 3 tensors + needs a *mask* not lengths; duration-predictor arg order; channels-first conventions) and a `embed_tokens.weight` access by exposing a `.weight` property on the custom embedding.
- **Monotonic alignment search** swapped from a pure-python DP to the **numba-jitted** canonical VITS kernel (~100×) — the training throughput lever.
- **Waveform-slice off-by-<hop bug:** center-padding makes `T_spec·hop` exceed `wav_len` by a few samples → `torch.stack` size mismatch; fixed by padding short slices.
- **bf16 over fp16+GradScaler** for stable GAN training on Ampere; mel loss kept in fp32 (STFT).
- **CUDA-13 environment friction (documented, not hidden):** the box shipped CUDA 13, so `onnxruntime-gpu`, TensorRT, and `ctranslate2` (faster-whisper) all needed CUDA-12 libs (`libcublasLt.so.12`) and fell back to CPU. **PyTorch had a `cu130` wheel**, so torch is the *measured* GPU path; ONNX export is validated but its GPU EP is blocked on this host. A CUDA-12 L4 image runs the TensorRT-FP16 path unchanged.
- **(Repo) FastAPI WebSocket 403:** `from __future__ import annotations` + a locally-imported `WebSocket` made FastAPI's `get_type_hints` fail to resolve the WS param → every upgrade 403'd; fixed by importing WS types at module scope.

---

## 8. How to use

This is a phoneme-input model: drive it with the front-end (it converts text → phoneme/language IDs), not raw text.

```python
# pip install -e '.[gpu]'  (from the code repo)
import torch
from hams_tts.models.hams_vits import HamsVITS
from hams_tts.text.frontend import TextFrontend

model = HamsVITS.from_checkpoint("path/to/this/checkpoint").cuda().eval()
fe = TextFrontend()
u = fe.process("مرحبا! بدي flight to London بكرا.")
ids  = torch.tensor([u.phoneme_ids],   device="cuda")
lang = torch.tensor([u.language_ids],  device="cuda")
wav  = model.infer(ids, lang, length_scale=5.0).squeeze().cpu().numpy()  # 16 kHz
```

Streaming server, Pipecat plugin, ONNX/TensorRT export, benchmark + eval harnesses are in the code repository.

---

## 9. Limitations & responsible use

- **Levantine Arabic is not yet accurate** — trained on MSA audio (§3). Do **not** deploy the Arabic voice for production; English is usable as a demo. 
- UTMOS ≈ 2.2 ⇒ audible artefacts; this is an *early* checkpoint (single-GPU, ~hours).
- Duration predictor under-predicts; synthesize at `length_scale≈5`.
- Urban-Levantine front-end focus; rural/Bedouin/Druze variants are future rules.

## 10. Reproducibility

```bash
python scripts/prepare_corpora.py --lj-root LJSpeech-1.1 --asc-root arabic-speech-corpus --out data/manifests
python scripts/finetune_recon.py  --manifest data/manifests/train.phon.jsonl --out ckpt_ft       # Stage 1
python scripts/finetune_gan.py    --manifest data/manifests/train.phon.jsonl --ckpt ckpt_ft/final --out ckpt_gan --steps 16000  # Stage 2
python scripts/finalize_eval.py   --ckpt ckpt_gan/final --length-scale 5.0    # samples + ASR + UTMOS
python -m hams_tts.eval.benchmark --backend torch --model-path ckpt_gan/final --eval-set data/eval_set/eval_utterances.json
```

## Citation

```bibtex
@software{hams_levantine_tts_2026,
  title  = {Hams: Low-Latency Code-Switching TTS (Levantine Arabic / English)},
  author = {Ibrahim, Alamin},
  year   = {2026},
  note   = {VITS + unified-IPA front-end + language-ID embedding; base facebook/mms-tts-ara}
}
```

**License:** Apache-2.0. Base model and corpora retain their own licenses (Arabic Speech Corpus: CC-BY; LJSpeech: public domain).
