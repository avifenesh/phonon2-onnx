"""Smaller encoders from dist/encoder-model.onnx.

  --mode exact4x2  The 264 five-value linears (11 per block: ff1.linear1/2, q/k/v/o, relative_k, pointwise_conv1/2,
                   ff2.linear1/2) become ONE com.microsoft MatMulNBits (bits=4, block 128, default zero point 8) over the
                   two planes stacked along N, then Split + Add:
                       W = lo * A + (hi - lo) * B,   A = sign,  B = sign * is_hi,  A, B in {-1, 0, 1}
                   Both planes and both fp32 row scales (lo, hi - lo) are exact, so every dequantized weight equals the
                   fp32 export's weight bit for bit (checked here for every tensor). Cost: 8 bits per weight, because
                   ORT 1.23's CPU MatMulNBits only implements 4 and 8 bits (2-bit planes fail at session creation).
                   Everything else (subsampling, depthwise convs, norms, the int6 subsampling linear) stays fp32.
  --mode exact4x2a4  Same weights, accuracy_level=4: MLAS quantizes the activations to int8 inside MatMulNBits
                   (faster, no longer exact arithmetic; the weights are still exact).
  --mode int8      onnxruntime.quantization.quantize_dynamic, per-channel QInt8 weights, uint8 dynamic activations.

Writes dist/encoder-model.<mode>.onnx and a dist/decoder_joint-model.<mode>.onnx symlink to the fp32 decoder (onnx-asr
resolves both files from the quantization suffix; the decoder is not quantized).
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
FV_RE = re.compile(r"^/enc/layers\.(\d+)/(feed_forward[12]/linear[12]|self_attn/(q|k|v|o|relative_k)_proj|conv/pointwise_conv[12])/(MatMul|Conv)$")
BLOCK = 128


def planes(W):
    """W [N, K] fp32 five-value matrix -> (A, B, lo, d) with W == lo*A + d*B exactly (asserted)."""
    a = np.abs(W)
    hi = a.max(1)
    lo = np.where(a > 0, a, np.inf).min(1)
    lo = np.where(np.isfinite(lo), lo, 0).astype(np.float32)
    levels = [np.unique(a[i][a[i] > 0]) for i in (0, len(a) // 2, len(a) - 1)]
    assert all(len(l) <= 2 for l in levels), "not a five-value row"
    A = np.sign(W).astype(np.int8)
    isd = (a == hi[:, None]) & (a > 0) & (hi[:, None] > lo[:, None])
    B = (A * isd).astype(np.int8)
    d = (hi - lo).astype(np.float32)
    assert np.array_equal(hi.astype(np.float64) - lo.astype(np.float64), d.astype(np.float64)), "hi-lo not exact in fp32"
    rec = lo[:, None] * A.astype(np.float32) + d[:, None] * B.astype(np.float32)
    assert np.array_equal(rec, W), "two-plane reconstruction is not exact"
    assert set(np.unique(a)) <= set(np.unique(np.concatenate([[0], lo, hi]))), "levels outside {0, lo, hi}"
    return A, B, lo, d


def pack4(codes, K):
    """codes [N, K] uint8 in 0..15 -> MatMulNBits B [N, K/BLOCK, BLOCK/2], element k in the low nibble when k is even."""
    N = codes.shape[0]
    c = codes.reshape(N, K // 2, 2)
    p = (c[:, :, 0] | (c[:, :, 1] << 4)).astype(np.uint8)
    return p.reshape(N, K // BLOCK, BLOCK // 2)


def exact4x2(src, dst, accuracy_level=0):
    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    new_nodes, drop, add_inits, n_done, max_err = [], set(), [], 0, 0.0
    for node in m.graph.node:
        mt = FV_RE.match(node.name)
        if not mt:
            new_nodes.append(node); continue
        wname = node.input[1]
        Wt = numpy_helper.to_array(inits[wname]).astype(np.float32)
        if node.op_type == "MatMul":                    # x[..., K] @ Wt[K, N]
            W = np.ascontiguousarray(Wt.T)
        else:                                           # Conv1d k=1 on [B, C, T]: weight [N, K, 1], no bias
            assert len(node.input) == 2 and Wt.shape[2] == 1
            W = np.ascontiguousarray(Wt[:, :, 0])
        N, K = W.shape
        assert K % BLOCK == 0
        A, B, lo, d = planes(W)
        codes = np.concatenate([A, B], 0).astype(np.int16) + 8
        packed = pack4(codes.astype(np.uint8), K)
        scales = np.repeat(np.concatenate([lo, d])[:, None], K // BLOCK, 1).reshape(-1).astype(np.float32)
        # dequant check, MatMulNBits semantics: w = (q - 8) * scale
        deq = (codes.astype(np.float32) - 8) * np.concatenate([lo, d])[:, None]
        rec = deq[:N] + deq[N:]
        max_err = max(max_err, float(np.abs(rec - W).max()))
        base = node.name.strip("/").replace("/", "_")
        bn, sn = f"{base}_q4planes", f"{base}_scales"
        add_inits += [numpy_helper.from_array(packed, bn), numpy_helper.from_array(scales, sn)]
        drop.add(wname)
        x, y = node.input[0], node.output[0]
        pre, post = [], []
        if node.op_type == "Conv":
            xt = f"{base}_xT"; pre.append(helper.make_node("Transpose", [x], [xt], perm=[0, 2, 1], name=f"{base}_inT")); x = xt
        y2, ya, yb = f"{base}_y2", f"{base}_ya", f"{base}_yb"
        mm = helper.make_node("MatMulNBits", [x, bn, sn], [y2], domain="com.microsoft", K=K, N=2 * N, bits=4,
                              block_size=BLOCK, accuracy_level=accuracy_level, name=f"{base}_mmnb")
        split = helper.make_node("Split", [y2, f"{base}_splitsz"], [ya, yb], axis=-1, name=f"{base}_split")
        add_inits.append(numpy_helper.from_array(np.array([N, N], np.int64), f"{base}_splitsz"))
        ysum = f"{base}_ysum" if node.op_type == "Conv" else y
        add = helper.make_node("Add", [ya, yb], [ysum], name=f"{base}_add")
        if node.op_type == "Conv":
            post.append(helper.make_node("Transpose", [ysum], [y], perm=[0, 2, 1], name=f"{base}_outT"))
        new_nodes += pre + [mm, split, add] + post
        n_done += 1
    assert n_done == 264, n_done
    still_used = {i for n in new_nodes for i in n.input}
    keep = [i for i in m.graph.initializer if i.name not in drop or i.name in still_used]
    del m.graph.node[:]; m.graph.node.extend(new_nodes)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep + add_inits)
    m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    print(f"replaced {n_done} five-value linears, max |dequant - fp32| = {max_err}", file=sys.stderr)
    onnx.save(m, str(dst), save_as_external_data=False)


def conv1x1_to_matmul(m):
    """The 48 pointwise Conv1d (k=1, no bias) -> Transpose, MatMul, Transpose, so quantize_dynamic can reach them
    (ORT 1.23 CPU has no ConvInteger kernel for int8 weights)."""
    inits = {i.name: i for i in m.graph.initializer}
    out, n = [], 0
    for node in m.graph.node:
        if node.op_type == "Conv" and re.search(r"/conv/pointwise_conv[12]/Conv$", node.name):
            w = inits[node.input[1]]
            W = numpy_helper.to_array(w)
            assert len(node.input) == 2 and W.shape[2] == 1
            base = node.name.strip("/").replace("/", "_")
            wt = numpy_helper.from_array(np.ascontiguousarray(W[:, :, 0].T), f"{base}_Wt")
            m.graph.initializer.remove(w); m.graph.initializer.append(wt)
            out += [helper.make_node("Transpose", [node.input[0]], [f"{base}_xT"], perm=[0, 2, 1], name=f"{base}_inT"),
                    helper.make_node("MatMul", [f"{base}_xT", wt.name], [f"{base}_y"], name=f"{base}_MatMul"),
                    helper.make_node("Transpose", [f"{base}_y"], [node.output[0]], perm=[0, 2, 1], name=f"{base}_outT")]
            n += 1
        else:
            out.append(node)
    assert n == 48, n
    del m.graph.node[:]; m.graph.node.extend(out)
    return m


def int8(src, dst):
    """Per-channel QInt8 weights on every MatMul with a constant weight (the 264 five-value linears, the int6
    subsampling linear); activations are quantized per tensor to uint8 at run time (DynamicQuantizeLinear)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    m = conv1x1_to_matmul(onnx.load(str(src)))
    quantize_dynamic(m, str(dst), per_channel=True, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     use_external_data_format=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact4x2", "exact4x2a4", "int8"], required=True)
    ap.add_argument("--dist", default=str(ROOT / "dist"))
    a = ap.parse_args()
    dist = Path(a.dist)
    src = dist / "encoder-model.onnx"; dst = dist / f"encoder-model.{a.mode}.onnx"
    {"exact4x2": exact4x2, "exact4x2a4": lambda s_, d_: exact4x2(s_, d_, accuracy_level=4), "int8": int8}[a.mode](src, dst)
    link = dist / f"decoder_joint-model.{a.mode}.onnx"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink("decoder_joint-model.onnx", link)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
