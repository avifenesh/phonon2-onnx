"""Build the parity test set manifest.

Public clips: the 73 LibriSpeech dev-clean utterances in hf-internal-testing/librispeech_asr_dummy
(LibriSpeech is CC-BY-4.0), written as 16 kHz mono PCM16 WAV under work/clips/.
Private clips: every 16 kHz WAV in $PHONON2_PRIVATE_CLIPS (your own speech), referenced by path only,
never copied into the repo. They have no ground-truth text.

Writes work/manifest.jsonl: {"id", "path", "ref" (str or null), "source"}.
"""
import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PRIVATE_DIR = os.environ.get("PHONON2_PRIVATE_CLIPS", "")


def main():
    (WORK / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    p = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet",
                        repo_type="dataset")
    for r in pq.read_table(p).to_pylist():
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000 and a.ndim == 1
        out = WORK / "clips" / f"ls_{r['id']}.wav"
        if not out.exists():
            sf.write(out, a, 16000, subtype="PCM_16")
        rows.append({"id": f"ls_{r['id']}", "path": str(out), "ref": r["text"], "source": "librispeech-dev-clean"})
    for f in sorted(glob.glob(f"{PRIVATE_DIR}/*.wav") if PRIVATE_DIR else []):
        info = sf.info(f)
        assert info.samplerate == 16000 and info.channels == 1, f
        rows.append({"id": "own_" + Path(f).stem, "path": f, "ref": None, "source": "private"})
    with open(WORK / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} clips -> {WORK / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
