"""Probe the lahgtna-levantine-tts parquet schema + content (run once, understand the data)."""
import sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = "mohammedaly22/lahgtna-levantine-tts"
p = hf_hub_download(REPO, "data/train-00000-of-00024.parquet", repo_type="dataset")
pf = pq.ParquetFile(p)
print("=== schema ===")
print(pf.schema_arrow)
print("rows in shard 0:", pf.metadata.num_rows)

tbl = pf.read_row_group(0)
cols = tbl.column_names
print("\n=== first row ===")
row = {k: tbl[k][0].as_py() for k in cols}
for k, v in row.items():
    if isinstance(v, dict):
        desc = {}
        for kk, vv in v.items():
            desc[kk] = ("<%d bytes>" % len(vv)) if isinstance(vv, (bytes, bytearray)) else vv
        print(" ", k, "= dict", desc)
    elif isinstance(v, (bytes, bytearray)):
        print(" ", k, "= <%d bytes>" % len(v))
    else:
        print(" ", k, "=", repr(v)[:160])

# distributions over the whole shard
n = min(pf.metadata.num_rows, tbl.num_rows)
def dist(col):
    if col in cols:
        return collections.Counter(tbl[col][i].as_py() for i in range(n))
print("\n=== distributions (shard 0, %d rows) ===" % n)
for c in ("speaker_id", "speaker_name", "gender", "sentence_type"):
    d = dist(c)
    if d:
        print(f"{c}: {dict(list(d.items())[:12])}{' ...' if len(d)>12 else ''}")

print("\n=== sample texts by sentence_type ===")
seen = set()
for i in range(n):
    st = tbl["sentence_type"][i].as_py() if "sentence_type" in cols else "?"
    if st not in seen:
        seen.add(st)
        print(f"[{st}] {tbl['text'][i].as_py()[:120]}")

# audio: inspect the embedded audio dict (HF Audio feature = {bytes, path})
a = row.get("audio")
if isinstance(a, dict):
    import io, soundfile as sf
    b = a.get("bytes")
    if b:
        w, sr = sf.read(io.BytesIO(b))
        print(f"\naudio[0]: {len(w)} samples @ {sr} Hz = {len(w)/sr:.2f}s | container decoded OK | dtype {w.dtype}")
