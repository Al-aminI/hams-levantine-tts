"""HamsVITS — a VITS acoustic model adapted for Levantine/English code-switching.

We start from a strong, permissively-licensed Arabic VITS (``facebook/mms-tts-ara``,
which *is* VITS in 🤗 Transformers as ``VitsModel``) and make three deliberate
architectural changes:

  1. **Unified phoneme vocabulary.**  The text-encoder embedding is replaced with one
     sized to our shared IPA inventory (:mod:`hams_tts.text.phoneme_inventory`), so the
     model is driven by phonemes we control rather than raw graphemes.
  2. **Language-ID embedding.**  A small ``nn.Embedding(num_languages, hidden)`` is
     *added* to the phoneme embedding before the text encoder.  This lets one model
     colour shared phonemes (e.g. /t/, /r/, /l/) with language-appropriate micro-phonetics
     and switch instantly at a code-switch boundary — no engine hand-off, continuous
     prosody.  This is the single most important change for smooth code-switching.
  3. **Speaker embedding** retained/added for multi-speaker conditioning.

Inference (duration prediction → flow → HiFi-GAN decode) is delegated to the proven
``VitsModel`` forward, so we inherit its non-autoregressive speed (the property that
makes the <300 ms TTFA / <0.3 RTF KPIs reachable).  Training uses the established VITS
adversarial recipe (see :mod:`hams_tts.training.finetune`, which adapts
``ylacombe/finetune-hf-vits``).

This module requires PyTorch + Transformers (the ``gpu``/``train`` extras) and runs on
the GPU host; it is import-guarded so the CPU dev box can import the package without torch.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except Exception:  # keep package importable on the CPU dev box
    _TORCH = False
    torch = None  # type: ignore
    nn = object  # type: ignore

from ..text.phoneme_inventory import VOCAB_SIZE, Lang


@dataclass
class HamsVITSConfig:
    base_model_id: str = "facebook/mms-tts-ara"
    vocab_size: int = VOCAB_SIZE
    num_languages: int = len(Lang)  # PAD/AR/EN/NEUTRAL
    num_speakers: int = 1
    speaker_embedding_dim: int = 256
    # default inference scales (length_scale<1 => faster speech, helps RTF)
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    speaking_rate: float = 1.0


if _TORCH:

    def _sequence_mask(lengths, max_len=None):
        max_len = max_len or int(lengths.max())
        ids = torch.arange(max_len, device=lengths.device)
        return (ids[None, :] < lengths[:, None]).float()

    def _generate_path(duration, mask):
        """Expand integer durations into a hard (text→spec) alignment path.
        duration: (b,1,t_text), mask: (b,1,t_text,t_spec) -> path (b,1,t_text,t_spec)."""
        b, _, t_x, t_y = mask.shape
        cum = torch.cumsum(duration, -1)
        cum_flat = cum.view(b * t_x)
        path = _sequence_mask(cum_flat, t_y).view(b, t_x, t_y)
        path = path - F.pad(path, (0, 0, 1, 0))[:, :-1]
        return (path.unsqueeze(1) * mask)

    class _PhonemeLangEmbedding(nn.Module):
        """Phoneme embedding + additive language embedding.

        Installed in place of ``VitsTextEncoder.embed_tokens`` so the language signal is
        injected exactly where the grapheme embedding used to be, with no other change to
        the encoder.  ``cur_lang_ids`` is set by :meth:`HamsVITS._set_lang` immediately
        before each forward pass (the encoder calls ``embed_tokens(input_ids)`` with only
        the ids, so we stash the parallel language stream here)."""

        def __init__(self, vocab_size: int, num_languages: int, dim: int):
            super().__init__()
            self.phoneme = nn.Embedding(vocab_size, dim)
            self.language = nn.Embedding(num_languages, dim)
            nn.init.normal_(self.phoneme.weight, 0.0, dim ** -0.5)
            nn.init.normal_(self.language.weight, 0.0, dim ** -0.5)
            self.cur_lang_ids: Optional["torch.Tensor"] = None

        @property
        def weight(self):
            # transformers' VitsModel.forward reads `embed_tokens.weight.dtype`; expose
            # the phoneme table so the backbone treats us like a normal nn.Embedding.
            return self.phoneme.weight

        def num_embeddings_(self):
            return self.phoneme.num_embeddings

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            emb = self.phoneme(input_ids)
            if self.cur_lang_ids is not None:
                emb = emb + self.language(self.cur_lang_ids)
            return emb

    class HamsVITS(nn.Module):
        def __init__(self, config: HamsVITSConfig):
            super().__init__()
            from transformers import VitsModel  # local import (heavy)

            self.config = config
            self.backbone = VitsModel.from_pretrained(config.base_model_id)
            hidden = self.backbone.config.hidden_size

            # (1)+(2) swap in unified phoneme vocab + language embedding
            self.embed = _PhonemeLangEmbedding(config.vocab_size, config.num_languages, hidden)
            self.backbone.text_encoder.embed_tokens = self.embed

            # (3) ensure a speaker embedding exists for multi-speaker conditioning
            if config.num_speakers > 1:
                self.backbone.config.num_speakers = config.num_speakers
                self.backbone.embed_speaker = nn.Embedding(
                    config.num_speakers, self.backbone.config.speaker_embedding_size
                )

            # default sampling behaviour
            self.backbone.noise_scale = config.noise_scale
            self.backbone.noise_scale_duration = config.noise_scale_duration
            self.backbone.speaking_rate = config.speaking_rate

        # -- helpers ----------------------------------------------------------------
        def _set_lang(self, lang_ids: "torch.Tensor") -> None:
            self.embed.cur_lang_ids = lang_ids

        def trainable_parameters(self, freeze_decoder: bool = True, freeze_flow: bool = True):
            """Parameter-efficient fine-tuning: freeze the heavy HiFi-GAN decoder and the
            flow, train the text encoder, duration predictor and the new embeddings.

            Freezing the decoder/flow keeps ~70% of params fixed, slashes optimiser memory,
            and concentrates adaptation on phonetics/prosody — which is exactly where the
            Levantine + code-switching signal lives.  (LoRA adapters on the encoder
            attention are added separately in finetune.py via PEFT.)"""
            if freeze_decoder:
                for p in self.backbone.decoder.parameters():
                    p.requires_grad_(False)
            if freeze_flow:
                for p in self.backbone.flow.parameters():
                    p.requires_grad_(False)
            return [p for p in self.parameters() if p.requires_grad]

        # -- inference --------------------------------------------------------------
        @torch.inference_mode()
        def infer(
            self,
            phoneme_ids: "torch.Tensor",
            language_ids: "torch.Tensor",
            speaker_id: Optional["torch.Tensor"] = None,
            length_scale: float = 1.0,
            noise_scale: Optional[float] = None,
            attention_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            self._set_lang(language_ids)
            if noise_scale is not None:
                self.backbone.noise_scale = noise_scale
            # length_scale<1 speeds speech; VitsModel exposes speaking_rate = 1/length_scale
            self.backbone.speaking_rate = 1.0 / max(length_scale, 1e-3)
            if attention_mask is None:
                attention_mask = torch.ones_like(phoneme_ids)
            kwargs = {"input_ids": phoneme_ids, "attention_mask": attention_mask}
            if speaker_id is not None and self.config.num_speakers > 1:
                kwargs["speaker_id"] = speaker_id
            out = self.backbone(**kwargs)
            return out.waveform  # (batch, num_samples)

        def training_forward(self, phoneme_ids, phoneme_lengths, language_ids, spec,
                             spec_lengths, speaker_id, wav, seg_size, hop, maximum_path):
            """Canonical VITS training graph, returning the tensors the GAN objective needs:
            ``{y_hat, y_slice, kl, dur_loss}``.

            REFERENCE IMPLEMENTATION (GPU host): calls into the HF VitsModel submodules
            (text_encoder / posterior_encoder / flow / duration_predictor / decoder) are
            pinned to the installed ``transformers`` build; ``finetune-hf-vits`` is the
            hardened equivalent invoked by ``finetune.py --engine hf-vits``. The math
            (NCE → MAS alignment → duration expansion → decoder slice → KL) is the standard
            VITS recipe (Kim et al., 2021)."""
            bb = self.backbone
            self._set_lang(language_ids)
            g = bb.embed_speaker(speaker_id).unsqueeze(-1) if self.config.num_speakers > 1 else None

            # masks, channels-first (b, 1, T)
            t_mask = _sequence_mask(phoneme_lengths, phoneme_ids.shape[1]).unsqueeze(1)   # (b,1,seq)
            y_mask = _sequence_mask(spec_lengths, spec.shape[2]).unsqueeze(1)             # (b,1,T_spec)

            # text encoder: padding_mask is (b,seq,1); outputs (b,seq,C) -> channels-first
            enc = bb.text_encoder(phoneme_ids, t_mask.transpose(1, 2), attention_mask=t_mask.squeeze(1))
            x = enc.last_hidden_state.transpose(1, 2)         # (b,H,seq)  -> duration predictor
            m_p = enc.prior_means.transpose(1, 2)             # (b,F,seq)
            logs_p = enc.prior_log_variances.transpose(1, 2)  # (b,F,seq)

            # posterior encoder (linear spec -> z); flow maps posterior z -> prior space z_p
            z, m_q, logs_q = bb.posterior_encoder(spec, y_mask, g)   # each (b,F,T_spec)
            z_p = bb.flow(z, y_mask, g)                              # reverse=False

            # monotonic alignment search (negative cross-entropy) -> hard path (b,seq,T_spec)
            with torch.no_grad():
                s_r = torch.exp(-2 * logs_p)                                       # (b,F,seq)
                neg1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, 1, keepdim=True)  # (b,1,seq)
                neg2 = torch.matmul((-0.5 * z_p ** 2).transpose(1, 2), s_r)        # (b,T_spec,seq)
                neg3 = torch.matmul(z_p.transpose(1, 2), m_p * s_r)                # (b,T_spec,seq)
                neg4 = torch.sum(-0.5 * (m_p ** 2) * s_r, 1, keepdim=True)         # (b,1,seq)
                neg = (neg1 + neg2 + neg3 + neg4).transpose(1, 2)                  # (b,seq,T_spec)
                attn = maximum_path(neg, phoneme_lengths, spec_lengths)            # (b,seq,T_spec)

            # duration loss from the stochastic duration predictor (reverse=False, target durs)
            w = attn.sum(2).unsqueeze(1)                                            # (b,1,seq)
            dl = bb.duration_predictor(x, t_mask, global_conditioning=g, durations=w)
            dur_loss = torch.sum(dl) / torch.sum(t_mask)

            # expand prior to spec frames: m_p (b,F,seq) @ attn (b,seq,T_spec) -> (b,F,T_spec)
            m_p_e = torch.matmul(m_p, attn)
            logs_p_e = torch.matmul(logs_p, attn)

            # KL( posterior z_p || expanded prior )
            kl = (logs_p_e - logs_q - 0.5 + 0.5 * ((z_p - m_p_e) ** 2) * torch.exp(-2.0 * logs_p_e))
            kl = torch.sum(kl * y_mask) / torch.sum(y_mask)

            # random latent slice -> decode; matching waveform slice for the GAN/mel loss
            seg_frames = seg_size // hop
            max_start = (spec_lengths - seg_frames).clamp(min=0)
            starts = (torch.rand(z.shape[0], device=z.device) * (max_start.float() + 1)).long()
            z_slice = torch.stack([z[i, :, s:s + seg_frames] for i, s in enumerate(starts)])
            y_hat = bb.decoder(z_slice, g)                                          # (b,1,seg_size)
            # center-padding can make T_spec*hop exceed wav_len by <hop samples; pad short slices
            ys = []
            for i, s in enumerate(starts):
                seg = wav[i, s * hop:s * hop + seg_size]
                if seg.shape[0] < seg_size:
                    seg = F.pad(seg, (0, seg_size - seg.shape[0]))
                ys.append(seg)
            y_slice = torch.stack(ys).unsqueeze(1)
            return {"y_hat": y_hat if y_hat.dim() == 3 else y_hat.unsqueeze(1),
                    "y_slice": y_slice, "kl": kl, "dur_loss": dur_loss}

        @property
        def sample_rate(self) -> int:
            return self.backbone.config.sampling_rate

        # -- (de)serialisation ------------------------------------------------------
        def save_checkpoint(self, path: str) -> None:
            os.makedirs(path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(path, "hams_vits.pt"))
            with open(os.path.join(path, "hams_vits_config.json"), "w") as f:
                json.dump(asdict(self.config), f, indent=2)

        @classmethod
        def from_checkpoint(cls, path: str) -> "HamsVITS":
            if os.path.isdir(path):
                with open(os.path.join(path, "hams_vits_config.json")) as f:
                    cfg = HamsVITSConfig(**json.load(f))
                model = cls(cfg)
                state = torch.load(os.path.join(path, "hams_vits.pt"), map_location="cpu")
                model.load_state_dict(state, strict=False)
                return model
            raise FileNotFoundError(path)

else:

    class HamsVITS:  # pragma: no cover - placeholder on torch-less machines
        def __init__(self, *a, **k):
            raise RuntimeError(
                "HamsVITS requires PyTorch + Transformers. Install the 'gpu' or 'train' "
                "extra:  pip install -e '.[gpu]'   (runs on the L4 host, not the CPU dev box)."
            )
