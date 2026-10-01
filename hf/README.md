---
license: cc-by-4.0
language:
- en
base_model: FermionResearch/Phonon-2
pipeline_tag: automatic-speech-recognition
library_name: onnx
tags:
- onnx
- onnxruntime
- speech-recognition
- parakeet
- fastconformer
- tdt
- onnx-asr
---

# Phonon-2 ONNX

[Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) by Fermion Research, exported to ONNX
for ONNX Runtime on CPU with no Python at inference time. Phonon-2 is an English speech recognizer
derived from NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

The fp32 export reproduces Fermion's exact (fp32) runtime token for token on all 82 test clips.
The graphs follow the [onnx-asr](https://github.com/istupakov/onnx-asr) `nemo-conformer-tdt` layout,
plus one graph for Phonon-2's own front end. This is the speech recognizer in
[parlar](https://github.com/avifenesh/parlar), a voice mode for coding agents, which runs it from Rust.

## Files

| File | Size | What it is |
|---|---:|---|
| `preprocessor-model.onnx` | 1.2 MB | log-mel front end, 16 kHz audio in, 128 features out |
| `encoder-model.onnx` + `.onnx.data` | 2.44 GB | FastConformer encoder, fp32, exact |
| `encoder-model.exact4x2.onnx` | 662 MB | the same weights bit for bit, two ternary planes in 4-bit MatMulNBits |
| `encoder-model.int8.onnx` | 614 MB | int8 weights, dynamic uint8 activations, lossy |
| `decoder_joint-model.onnx` | 72.5 MB | prediction network and joint, fp32 |
| `decoder_joint-model.int8.onnx`, `.exact4x2.onnx` | | the same fp32 decoder, under the names onnx-asr looks for with each encoder |
| `vocab.txt`, `config.json` | | 8,192 SentencePiece pieces plus blank; onnx-asr config |

Pick one encoder. `exact4x2` needs an ONNX Runtime build with contrib ops (the standard builds have
them); the other two use standard ai.onnx ops only.

| Encoder | Same tokens as Fermion fp32 | LibriSpeech dev-clean WER | RAM after load / peak |
|---|---:|---:|---:|
| fp32 | 82/82 | 4.26% | 2.2 / 3.0 GB |
| exact4x2 | 82/82 | 4.26% | 1.1 / 1.6 GB |
| int8 | 72/82 | 4.35% | 0.9 / 1.3 GB |

The 82 clips are 73 from LibriSpeech dev-clean and 9 of conversational speech, all 35 s or shorter.
WER is on uppercase text without punctuation over 1,150 words, so the 0.09-point gap is one word.
Fermion's own two CPU tiers disagree with each other on 11 of the 82 clips.

## Use

With onnx-asr, pointing it at this folder (its built-in NeMo preprocessor works, with punctuation
differences on 2 of 82 clips; `preprocessor-model.onnx` is the exact one):

```python
import onnx_asr
model = onnx_asr.load_model("nemo-conformer-tdt", "path/to/Phonon-2-ONNX", quantization="int8")  # or "exact4x2", or None for fp32
print(model.recognize("speech.wav"))
```

From any language, the contract is:

- `preprocessor-model.onnx`: `waveforms` f32 [B, N] at 16 kHz and `waveforms_lens` i64 [B] in;
  `features` f32 [B, 128, F] and `features_lens` i64 [B] out, F = N // 160 + 1.
- Encoder: `audio_signal` f32 [B, 128, T] and `length` i64 [B] in; `outputs` f32 [B, 1024, T'] and
  `encoded_lengths` i64 [B] out, T' = T subsampled 8x.
- `decoder_joint-model.onnx`: `encoder_outputs` f32 [B, 1024, 1], `targets` i32 [B, 1],
  `target_length` i32 [B], `input_states_1` and `input_states_2` f32 [2, B, 640] in;
  `outputs` f32 [B, 1, 1, 8198], `output_states_1`, `output_states_2` out.

Greedy TDT decoding: start with zero states and previous token = blank (8192), t = 0. At each step
take the argmax of the first 8,193 outputs (the token) and of the last 5 (duration 0 to 4). On a
non-blank token keep it and the new states. Advance t by the duration; a blank with duration 0
advances by 1, and at most 10 tokens are taken on one frame. To make text, drop `<unk>`, `<pad>` and
`<|...|>` pieces, join the rest, replace `▁` with a space and trim. Audio longer than about 35 s
should be cut at pauses, as Fermion's runtime does.

## How it was made

1. Fermion's loader expands the five-value, int6 and fp16 tensors to fp32; the transformers
   `ParakeetForTDT` module is traced to ONNX (opset 17). The fp32 weights equal Fermion's
   expansion exactly.
2. `preprocessor-model.onnx` re-implements Fermion's front end: pre-emphasis 0.97, 512-point STFT
   with a symmetric 400-sample Hann window, hop 160, 128 slaney mel bands, log(x + 2^-24), and
   per-band normalization over all frames.
3. `exact4x2`: each five-value row {-hi, -lo, 0, lo, hi} is written as lo*A + (hi-lo)*B with two
   ternary planes A and B, stored in 4-bit MatMulNBits. Only 0.63% of rows fit a single uniform
   4-bit grid, and ONNX Runtime has no 2-bit CPU kernel, so the 2.1-bit container cannot be matched.
4. `int8`: per-channel int8 MatMul weights, after rewriting the pointwise convolutions as MatMul.

## License and attribution

CC-BY-4.0, as Phonon-2 and Parakeet TDT 0.6B v3.

Parakeet TDT 0.6B v3 by NVIDIA; Phonon-2 by Fermion Research; ONNX conversion by Tiyuvta.
The changes made here are listed under "How it was made". Fermion's NOTICE, including the training
data it lists, is reproduced in `NOTICE`.
