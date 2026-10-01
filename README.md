# phonon2-onnx

ONNX export of Fermion Research's Phonon-2 speech recognizer, for ONNX Runtime 1.23 on CPU with no Python at inference time.
The graphs follow the onnx-asr `nemo-conformer-tdt` layout (the same layout as the onnx-asr export of parakeet-tdt-0.6b-v3), plus one extra graph for the front end.
Parity is checked against Fermion's own runtime (`fermion-research` 0.2.4) on 82 clips.

Nothing here is uploaded or published. The ONNX files live in `dist/`, which is gitignored. Rebuild them with the commands under "Reproduce".

## Provenance and license

- Base model: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), by NVIDIA, CC-BY-4.0.
- Phonon-2: `FermionResearch/Phonon-2`, by Fermion Research, a derivative of the above. The container's manifest says `license: cc-by-4.0`, `base_model: nvidia/parakeet-tdt-0.6b-v3`, and `training: C4c: 6,000 h targeted diet on the five-value encoder (step 45,001); int6 dense tables`.
  The input was the local download `model_phonon2_c4c_int6/`: `model.fermion` sha256 `4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468` (177,438,361 bytes). The distribution tarball `phonon-2.bps.tar.zst` has sha256 `98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`.
- This export is a derivative of both, also under CC-BY-4.0. Attribution: "Parakeet-TDT-0.6B-v3 by NVIDIA; Phonon-2 by Fermion Research; ONNX conversion by Avi Fenesh."
- Changes made here:
  1. The five-value, int6 and fp16 tensors are expanded to fp32 by Fermion's own loader, then traced into ONNX graphs. The fp32 weights equal Fermion's dense expansion exactly.
  2. A new preprocessor graph re-implements Fermion's log-mel front end.
  3. `encoder-model.exact4x2.onnx` re-encodes the 264 five-value linears as two ternary planes in 4-bit MatMulNBits. The weights are bit-exact.
  4. `encoder-model.int8.onnx` is lossy: per-channel int8 weights and dynamic uint8 activations.
  5. `vocab.txt` and `config.json` are derived from the model's `config.json`.
- The test-set clips come from LibriSpeech dev-clean (CC-BY-4.0) via `hf-internal-testing/librispeech_asr_dummy`. The private clips are referenced by path only and are not in this repo.

## Facts about Phonon-2, from Fermion's source

Everything marked VERIFIED was read in `fermion-research` 0.2.4 (`fermion/_speech/engine_phonon2_cpu.py`, `_engine_phonon2/fermion_container.py`, `_engine_phonon2_cpu/__init__.py`), and then confirmed by a check or measurement in this repo.

**Container (`fermion-five-value-parakeet-v1`), VERIFIED.** The file starts with an 8-byte little-endian header length, then a JSON index of `{n, k, shape, b}` records in order, then the raw blobs.
There are 264 `five_value`, 35 `int6` and 424 `fp16` records, with HF `ParakeetForTDT` tensor names.

- `five_value [O, I]` blobs hold three parts:
  1. `O * ceil(I/5)` bytes of base-3 trits, 5 per byte, least-significant trit first. Code 0/1/2 means sign -1/0/+1.
  2. A little-endian bitmask with one bit per nonzero weight, in row-major order. A set bit means the weight uses `hi`.
  3. `lo[O]` and `hi[O]` as fp16.

  Each weight is `sign * (is_hi ? hi[row] : lo[row])`, so a row holds five values: {-hi, -lo, 0, lo, hi}. This format covers all 264 encoder linears: 11 per block (ff1.linear1/2, q/k/v/o, relative_k, pointwise_conv1/2, ff2.linear1/2), 603,979,776 weights in total, at 2.11 bits per weight.
- `int6 [O, ...]` packs 4 values into 3 little-endian bytes. Each value is `u - 32`, and the blob ends with one fp16 scale per row, so `w = q * scale`. This covers the subsampling pointwise convs and linear, the depthwise convs, `encoder_projector`, the decoder embedding, the LSTM, the decoder projector and the joint head.
- `fp16`: everything else (conv biases, norms, `bias_u`/`bias_v`, BatchNorm stats, LSTM biases).

**Census of the five-value levels, MEASURED** (`tools/inspect_weights.py`, `work/weights_census.json`).

- Level occupancy: -hi 9.1%, -lo 15.3%, 0 51.3%, lo 15.3%, hi 9.1%.
- `hi/lo` percentiles per row: p1 1.19, p10 1.76, p50 2.13, p90 3.31, p99 101.
- Only 779 of 442,368 rows have `hi == 2*lo`, and only 2,778 rows (0.63%) fit any uniform grid a single 4-bit weight could hold exactly (`hi/lo = b/a`, `b <= 7`).

The levels are not a uniform grid. See "Smaller encoders" for what follows from that.

**Runtime tiers, VERIFIED.** `engine_phonon2_cpu.load` has two CPU tiers:

- **fp32 tier-1** (`FERMION_P2_CPU=fp32`, `FERMION_P2_CPU_TDT=torch`): the container is expanded to dense fp32 inside the stock transformers `ParakeetForTDT`, with a torch greedy TDT loop.
- **default** on this x86 box: one-dot packed int8 GEMMs, a C encoder and a C TDT loop. This tier re-quantizes the activations to int8 per row, and the TDT loop also runs int8 activations over the int6 tables.

The two tiers do not give identical transcripts: 71 of 82 match on this test set. Fermion's docstring says the same ("same-accuracy class ... NOT word-identical"). The ONNX fp32 graph is traced from tier-1, so tier-1 is the reference for exact parity.
bf16 autocast only turns on with AMX-bf16, which this CPU (Core Ultra 9 275HX) lacks, so tier-1 here is pure fp32.

**Front end, VERIFIED** (`_log_mel`, the HF ParakeetFeatureExtractor arithmetic).

1. Input is mono float32 at 16 kHz (soundfile, channels averaged, `resample_poly` if the rate differs).
2. Pre-emphasis: `y[0] = x[0]`, then `y[n] = x[n] - 0.97 x[n-1]`.
3. `torch.stft` with n_fft 512, hop 160, win 400, a symmetric Hann window (`periodic=False`, zero-padded to 512 and centred), `center=True` and zero padding. That gives **`frames = len // 160 + 1`**.
4. Power spectrum, computed as `sqrt(re^2 + im^2)^2`.
5. A 128-band slaney mel filterbank (librosa `norm="slaney"`, fmin 0, fmax 8000).
6. `log(x + 2^-24)`.
7. Per-feature normalisation over **all** frames: `(x - mean) / (sqrt(sum((x - mean)^2) / (n - 1)) + 1e-5)`.
8. No dither.

The onnx-asr `nemo128` preprocessor (NeMo's) differs in one respect. It uses `len // 160` frames and leaves the last frame out of the statistics. The filterbank and window are the same (max abs filterbank diff 3e-8).
With stock `nemo128`, 80 of 82 transcripts match tier-1. The two misses are punctuation only (0 word errors). This is why the export ships `preprocessor-model.onnx`.

**Encoder, VERIFIED.** FastConformer: dw_striding subsampling x8 to `T' = floor((T-1)/2+1)` applied three times, then 24 blocks, d_model 1024, 8 heads, rel-pos attention, conv kernel 9 with BatchNorm, no xscaling. The output is 1024-d. The projection to 640 (`encoder_projector`) lives in `decoder_joint`, as in NeMo's export.

**Prediction net and joint, VERIFIED.** The prediction net is an embedding of 8193 x 640, where row 8192 (blank) is all zeros (checked), followed by a 2-layer LSTM (hidden 640) and `decoder_projector`, a linear layer from 640 to 640.
The joint computes `head(relu(encoder_projector(enc) + decoder_projector(lstm_out)))`, and the head is a linear layer from 640 to 8198. The 8198 outputs are 8193 token logits (ids 0..8191 are SentencePiece pieces, 8192 is blank) followed by 5 duration logits for durations `[0, 1, 2, 3, 4]`.
No log-softmax is applied. Greedy decoding only takes the argmax of each slice, so softmax would change nothing.

**Tokenizer, VERIFIED.** The vocabulary is `config.json["joint"]["vocabulary"]`, 8192 SentencePiece pieces. 264 of them are special pieces (`<unk>`, `<pad>`, `<|...|>`).
Fermion builds the text in three steps:
1. Drop pieces matching `<|...|>`, `<unk>` and `<pad>`.
2. Concatenate the rest and replace `▁` (U+2581) with a space.
3. Strip leading and trailing whitespace.

**Long audio, VERIFIED, not exported.** Fermion decodes up to 35 s in one call. Longer audio is cut into 25-35 s windows at pauses (`_speech/segment.py`), each window is decoded alone, and the results are joined with spaces. Every clip in this test set is 35 s or shorter, so a single call covers all of them. The daemon must window long audio itself.

## Files (`dist/`)

| file | bytes | sha256 |
|---|---:|---|
| `preprocessor-model.onnx` | 1,193,996 | `8184d564f7d34d1daf04e4b35a0222fb72c54668b38dcd2b8ddda3517676614b` |
| `encoder-model.onnx` | 789,941 | `799506536ea9ab6933174bae2bb00177284f79419ceb5efa7e35a16bb9d4133f` |
| `encoder-model.onnx.data` | 2,435,420,160 | `4fa6441ccbed2c8d1242bd5cf7a9ee522b57148fadb34a016fa661e8d540fab3` |
| `encoder-model.exact4x2.onnx` | 662,190,977 | `abfdefaa1c74d6d3ca367a7ed358732a6140fb26a312b650ee57e46f1a9849ec` |
| `encoder-model.int8.onnx` | 614,486,780 | `3c100e38ca2e70623c928ab5c5414c62848603ea3d943a7f2e1c3ce1d92fc04b` |
| `decoder_joint-model.onnx` | 72,518,934 | `420125e0e13596692320c35ef648eee9bf4583718c7896c8732ebf6f50b9ca0d` |
| `vocab.txt` | 93,939 | `d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d` |
| `config.json` | 121 | `db59e29a3c1fde6a081bf04965e72bba26cd65be1aee65b064360df8aef468e5` |

- `decoder_joint-model.{exact4x2,int8}.onnx` are symlinks to the fp32 decoder. onnx-asr resolves both files from the quantization suffix, and the decoder is never quantized.
- All graphs use IR 8 and opset 17. `exact4x2` also imports `com.microsoft` v1 for MatMulNBits, which ships in the standard ORT builds.
- `config.json` contains `{"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8, "max_tokens_per_step": 10}`.
- `vocab.txt` has one `"<piece> <id>"` line per id from 0 to 8192. The last line is `<blk> 8192`.

### Inputs and outputs

| model | name | dtype | shape |
|---|---|---|---|
| preprocessor | in `waveforms` | float32 | [B, N] samples, 16 kHz, in [-1, 1] |
| | in `waveforms_lens` | int64 | [B] |
| | out `features` | float32 | [B, 128, F], F = N // 160 + 1 |
| | out `features_lens` | int64 | [B] = lens // 160 + 1 |
| encoder (all three) | in `audio_signal` | float32 | [B, 128, T] |
| | in `length` | int64 | [B] |
| | out `outputs` | float32 | [B, 1024, T'] |
| | out `encoded_lengths` | int64 | [B], T' = floor((T-1)/2+1) three times |
| decoder_joint | in `encoder_outputs` | float32 | [B, 1024, 1] (one encoder frame) |
| | in `targets` | int32 | [B, 1] previous token (8192 at the start) |
| | in `target_length` | int32 | [B] (pass 1; it is passed straight through) |
| | in `input_states_1` | float32 | [2, B, 640] LSTM h |
| | in `input_states_2` | float32 | [2, B, 640] LSTM c |
| | out `outputs` | float32 | [B, 1, 1, 8198] |
| | out `prednet_lengths` | int32 | [B] |
| | out `output_states_1` | float32 | [2, B, 640] |
| | out `output_states_2` | float32 | [2, B, 640] |

Padded batches work. The preprocessor zeroes samples past `waveforms_lens` after pre-emphasis, and the encoder masks frames past `length`. A padded batch of 5 clips gave the same 5 texts as single runs. B = 1 is the measured path.

### Decoding (greedy TDT, as in Fermion's torch loop and onnx-asr)

```
enc = encoder(features)[0]            # [1024, T']
h = c = zeros(2, 1, 640); last = 8192; t = 0; nsym = 0; ids = []
while t < T':
    o, _, h2, c2 = decoder_joint(enc[:, t:t+1], [[last]], [1], h, c)
    tok = argmax(o[0,0,0,0:8193]);  dur = [0,1,2,3,4][argmax(o[0,0,0,8193:8198])]
    if tok == 8192 and dur == 0: dur = 1
    if tok != 8192: ids.append(tok); last = tok; h, c = h2, c2   # state advances only on a real token
    if dur == 0:
        nsym += 1
        if nsym >= 10: dur = 1; nsym = 0                          # max 10 symbols per frame
    else: nsym = 0
    t += dur
text = "".join(vocab[i] for i in ids if not special(vocab[i])).replace("▁", " ").strip()
```

`tools/parity.py:ref_decode` is this loop, and on every clip its ids equal those of onnx-asr's own loop. Rust can skip the decoder_joint call after a blank token, because `h`, `c` and `last` do not change. Only the encoder frame changes.
onnx-asr builds text with a different rule: it keeps special pieces and uses a regex that removes a space before punctuation. On this set its text matched the Fermion rule on 82 of 82 clips, but use the Fermion rule above.

## Parity (82 clips: 73 LibriSpeech dev-clean, 9 private, all 35 s or shorter)

The ONNX models run through onnx-asr 0.12.0 (`NemoConformerTdt`), with its preprocessor swapped for `preprocessor-model.onnx`. The table compares them with Fermion's fp32 tier-1 and default tiers.

| system | token ids = Fermion fp32 | = Fermion default | LibriSpeech WER (1,150 words) |
|---|---:|---:|---:|
| Fermion fp32 tier-1 | 82/82 | 71/82 | 4.261% |
| Fermion default (packed int8 + C) | 71/82 | 82/82 | 4.348% |
| **ONNX fp32** | **82/82** | 71/82 | 4.261% |
| **ONNX exact4x2** | **82/82** | 71/82 | 4.261% |
| ONNX int8 | 72/82 | 74/82 | 4.348% |
| ONNX fp32 + stock nemo128 | 80/82 (0 word diffs) | 73/82 | 4.261% |
| ONNX exact4x2, accuracy_level 4 (not shipped) | 77/82 | 71/82 | 4.348% |

WER uses uppercase text with punctuation stripped. The WER differences are 1 word in 1,150, so they are not significant on this set.

Encoder outputs against Fermion fp32 (`work/parity/*.summary.json`). "End to end" means our preprocessor feeding our encoder.

| clip | frames | ONNX fp32 max abs (end to end / encoder alone) | exact4x2 max abs | int8 max abs (rel L2) |
|---|---:|---|---|---|
| ls_1272-141231-0000 (4.65 s) | 466 | 1.6e-7 / 1.8e-7 | 1.7e-7 | 1.2e-2 (8.2%) |
| ls_1272-128104-0004 (29.4 s) | 2941 | 1.7e-6 / 1.3e-6 | 2.1e-6 | 1.1e-1 (14.2%) |
| own_1790825456375 (30.0 s) | 3001 | 1.9e-6 / 6.4e-7 | 2.5e-6 | 6.9e-2 (11.9%) |

- The encoder output magnitude is about 0.2-0.27.
- Preprocessor features against Fermion's differ by at most 3e-4. The ONNX graph uses a conv DFT and Fermion uses an FFT, and the gap shows up in log-mel bins near the 2^-24 floor.
- For comparison, Fermion's own default tier is 6-10% rel L2 away from its fp32 tier.
- Edge cases gave the same output as Fermion fp32 (an empty transcript in all four): 1 s of zeros, 1 s of 1e-3 noise, 0.3 s of speech and 0.05 s of speech (one encoder frame).

## Smaller encoders

The levels are not a uniform grid. A single 4-bit MatMulNBits cannot hold {-hi, -lo, 0, lo, hi} exactly, because only 0.63% of rows fit. ORT 1.23's CPU MatMulNBits supports only 4 and 8 bits; 2-bit fails at session creation with "Only 4b and 8b quantization is supported". Two encoders were built (`tools/quantize.py`):

- **exact4x2.** Each five-value linear is rewritten as `W = lo*A + (hi-lo)*B`, where `A = sign`, `B = sign*is_hi`, and both planes are in {-1, 0, 1}.
  - Both planes are stacked along N into one `MatMulNBits` (bits 4, block 128, default zero point 8, fp32 scales `lo` and `hi-lo`), followed by Split and Add. Pointwise convs get a Transpose on each side.
  - The dequantized weights equal the fp32 weights exactly, with max abs diff 0.0 over all 264 tensors. `hi-lo` was checked to be exact in fp32.
  - The tensors not stored as five-value stay fp32.
  - Cost: 8 bits per weight. ORT has no 2-bit kernel, so the 2.1-bit container cannot be matched.
  - File size 662 MB, against 2.44 GB for fp32.
- **int8.** `quantize_dynamic` with `per_channel=True`, QInt8 weights, uint8 per-tensor dynamic activations, applied to MatMul only. The 48 pointwise convs are first rewritten to Transpose, MatMul, Transpose, because ORT 1.23 CPU has no ConvInteger kernel for int8 weights. File size 614 MB. It is lossy: 72 of 82 clips give the same ids as fp32.

## CPU timing and RAM

The table shows medians over 5 separate process launches, each running 1 warm-up plus 3 timed runs (`tools/bench.py`, `tools/bench_summary.py`). Setup: ORT 1.23.2 CPU EP, `intra_op_num_threads=4`, `inter_op 1`, B = 1, `nice -n 19 taskset -c 0-3`.

**The measurements were taken on 4 cores while another CPU-heavy job (a ColBERT run) and other nice-0 processes shared those same cores. Load average was about 40 on 24 cores.** Treat the absolute numbers as pessimistic upper bounds and re-measure on an idle machine. Short clips suffer most, because thread barriers wait on descheduled threads.

The total covers preprocessor, encoder, and the Python greedy loop over decoder_joint. In milliseconds, with min..max across launches:

| variant | 4.93 s clip | 29.4 s clip | 30.0 s clip | RSS after load | peak RSS |
|---|---:|---:|---:|---:|---:|
| ONNX fp32 | 1261 (1034..1658), enc 1089 | 2974 (2891..3045), enc 2420 | 3156 (2929..3224) | 2.20 GB | 2.96 GB |
| ONNX exact4x2 | 1770 (1713..1854), enc 1596 | 4401 (4201..4536), enc 3670 | 4380 (4235..4787) | 1.11 GB | 1.57 GB |
| ONNX int8 | 650 (567..1243), enc 464 | 2435 (2219..2761), enc 1687 | 2254 (2107..2279) | 0.88 GB | 1.34 GB |
| Fermion default, for context | 591 (344..921) | 2324 (1777..2783) | 2213 (1964..2420) | 1.40 GB | 1.63 GB |

- The decode loop accounts for 0.5-0.7 s of the 30 s totals. It makes one ORT call per decode step from Python, so most of that time is per-call overhead.
- Under contention, exact4x2 is slower than fp32. Its MatMulNBits fp32 path dequantizes on every call, and the stacked planes double N. Its value is a 3.7x smaller file and half the RAM with exact weights.
- Load time is 1.7-2.7 s for the ONNX variants and 8.6 s for Fermion.

## Reproduce

Prerequisites: Phonon-2 is downloaded to `~/.cache/fermion/speech/FermionResearch__Phonon-2/model_phonon2_c4c_int6/` (`fermion transcribe phonon-2 --download-only`), or `PHONON2_MODEL_DIR` points at it.

```
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match "torch==2.14.1+cpu" "transformers==5.18.0" "fermion-research==0.2.4" \
  onnx==1.23.1 onnxscript "onnxruntime==1.23.*" onnx-asr==0.12.0 numpy soundfile scipy pyarrow
R="nice -n 19 taskset -c 0-3 env OMP_NUM_THREADS=4"
$R .venv/bin/python tools/fetch_clips.py                 # work/manifest.jsonl (PHONON2_PRIVATE_CLIPS=/tmp/parley-turns)
$R .venv/bin/python tools/fermion_ref.py fp32 --dump ls_1272-141231-0000,ls_1272-128104-0004,own_1790825456375
$R .venv/bin/python tools/fermion_ref.py default --dump ls_1272-141231-0000,ls_1272-128104-0004,own_1790825456375
$R .venv/bin/python tools/export.py                      # dist/ fp32 graphs + vocab.txt + config.json (~5.4 GB peak RSS)
$R .venv/bin/python tools/quantize.py --mode exact4x2
$R .venv/bin/python tools/quantize.py --mode int8
$R .venv/bin/python tools/parity.py; $R .venv/bin/python tools/parity.py --suffix exact4x2; $R .venv/bin/python tools/parity.py --suffix int8 --no-ref-loop
.venv/bin/python tools/wer.py
$R .venv/bin/python tools/inspect_weights.py
for i in 1 2 3 4 5; do $R .venv/bin/python tools/bench.py --suffix exact4x2 >> work/bench/onnx-exact4x2.jsonl; done   # and "", int8, --fermion default
.venv/bin/python tools/bench_summary.py
```

The export traces the transformers module code with the legacy TorchScript exporter (`dynamo=False`) at opset 17, with `_attn_implementation = "eager"`. torch folds the BatchNorm into the depthwise conv in eval mode.

## Not verified

- No Rust run. The I/O contract was exercised through ORT 1.23.2's Python binding and onnx-asr only. A Rust `ort` build must include the contrib ops (MatMulNBits) to load `exact4x2`. The fp32 and int8 encoders use standard ai.onnx ops only, including MatMulInteger and DynamicQuantizeLinear.
- Only x86-64 (Core Ultra 9 275HX) was tested. ARM and other ORT builds or execution providers were not.
- Timing was measured only under contention (see above). There is no idle-machine number.
- Audio over 35 s was not tested and has no exported windowing.
- Exact parity was checked against Fermion's fp32 tier only. No ONNX variant reproduces the default packed int8 tier exactly; Fermion's own tiers disagree with each other on 11 of 82 clips.
- The test set is small: 1,150 reference words. A WER difference of 0.09 points is a single word.
