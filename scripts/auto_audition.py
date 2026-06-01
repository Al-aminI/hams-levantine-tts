"""Watch a checkpoint dir and audition each new checkpoint with Whisper (intelligibility
trajectory during training). Appends to a log so the run can be monitored cheaply.

    python scripts/auto_audition.py --watch /workspace/ckpt_gan --log /workspace/audition.log
"""

import argparse
import glob
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", default="/workspace/ckpt_gan")
    ap.add_argument("--log", default="/workspace/audition.log")
    ap.add_argument("--whisper", default="small")
    args = ap.parse_args()

    import torch
    import soundfile as sf
    from faster_whisper import WhisperModel

    from hams_tts.models.hams_vits import HamsVITS
    from hams_tts.text.frontend import TextFrontend

    fe = TextFrontend()
    asr = WhisperModel(args.whisper, device="cpu", compute_type="int8")
    texts = [
        ("ar", "مَرحَبا، كِيفَك؟ إن شاء الله مْنيح."),
        ("en", "I would like to book a flight to London."),
        ("ar", "بَدّي إحجِز flight بُكرا الساعة 9."),
    ]
    seen = set()

    def log(msg):
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    while True:
        ckpts = sorted(glob.glob(os.path.join(args.watch, "step_*"))) + glob.glob(os.path.join(args.watch, "final"))
        for d in ckpts:
            if d in seen or not os.path.exists(os.path.join(d, "hams_vits.pt")):
                continue
            try:
                m = HamsVITS.from_checkpoint(d).cuda().eval()
                log(f"\n=== {os.path.basename(d)} ({time.strftime('%H:%M')}) ===")
                for lang, t in texts:
                    u = fe.process(t)
                    ids = torch.tensor([u.phoneme_ids], device="cuda")
                    lg = torch.tensor([u.language_ids], device="cuda")
                    for ls in (1.5, 2.5):
                        with torch.inference_mode():
                            w = m.infer(ids, lg, length_scale=ls).squeeze().float().cpu().numpy()
                        sf.write("/tmp/aud.wav", w, m.sample_rate)
                        segs, _ = asr.transcribe("/tmp/aud.wav", language=lang, beam_size=1)
                        hyp = "".join(s.text for s in segs).strip()
                        log(f"  [{lang} ls={ls}] {t[:26]:26} -> {hyp[:60]!r} ({len(w)/m.sample_rate:.1f}s)")
                del m
                torch.cuda.empty_cache()
                seen.add(d)
            except Exception as e:
                log(f"  audition error {os.path.basename(d)}: {e}")
                seen.add(d)
        time.sleep(20)


if __name__ == "__main__":
    main()
