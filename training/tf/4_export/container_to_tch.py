#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  The container formats this script reads (magic FX2TFWC3 and
# FX2TFWC4), the range coder and the bit trees are `weights_compress_v3.py`
# next to this file, which is itself a fork of `pysrc/weights_compress.py` from
# fx2-cmix-transformer by Vladimir Ivanov (GPL-3.0).  The export arithmetic
# that is inverted and replayed below (bf16 row scale, IEEE division, round
# half to even, clamp to [-7, 7]), the tensor naming and the reference weight
# format (magic FX2TFW01) are his, in `pysrc/export_weights.py`.  Our addition
# is this script.  See ../../ATTRIBUTION.md.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""turn a weight container back into a training checkpoint (state_dict)

The export chain (tft/export_weights_poc.py, then export.sh: wpack2_bf16.py,
weights_compress_v3.py compress3, prune_dead_neurons.py,
convert_container_v4.py) goes from a checkpoint to the container.  The
checkpoints of the shipped transformer are not part of this entry; this script
goes the other way, so that training can continue from the shipped weights:

  X.weight.q (int4 rows) and X.weight.scale (bf16 row scale s)
      -> X.weight = q * s, X.quantize_weight.scale = s (fp32).  The product is
         exact (|q| <= 7 adds three bits to the eight of a bf16 scale), so the
         exporter's q = round(X.weight / s) gives q back.
  X.quantize_activation.scale, attention.quantize_{queries,keys,values}.scale
      -> the fp32 value of the bf16 bits.
  the raw f32 tensors -> copied (bf16 rounded already, wpack2_bf16.py).
  rope.*, config.* -> not in a state_dict (the exporter derives them from the
      geometry); config.ints and config.kimi go to the sidecar.

Two things the chain removed cannot come back.  They are rebuilt in a form the
chain removes again:

  * the rows of collapsed attention channels (row and scale zero in v4, 86 in
    the shipped container): the weight row stays 0, the scale becomes
    2^ZERO_ROW_SCALE_LOG2 instead of 0 (the exporter divides by the scale),
    which is below the limit of zero_collapsed_rows;
  * the structurally dead MLP neurons (70): appended as zero rows of mlp.up
    (scale 2^ZERO_ROW_SCALE_LOG2) and zero columns of mlp.down at the END of
    each layer, up to the trained width config.ints[5].  An MLP does not change
    under a permutation of its hidden neurons, and the original positions are
    not in the container.  prune_dead_neurons.py finds exactly these dead.

The int4 noise of those rows and the latent offsets inside each quantization
bin are lost: the checkpoint is a QAT state with every latent weight on the
centre of its bin, not the latent state of the training run.

Before anything is written the script proves the conversion: it replays the
export arithmetic of tft/export_weights_poc.py on the checkpoint in numpy and
then the four steps of export.sh (the v3 step is lossless and is skipped), and
the resulting v4 container must be BYTE IDENTICAL to the input.  The
deadness test must find exactly the appended neurons, the row zeroing exactly
the substituted rows.  If anything differs, nothing is written.

Outputs:
  <out.npz>          the state_dict, key -> float32 array, in export order (an
                     uncompressed zip with fixed time stamps, so the same input
                     gives the same bytes; np.load reads it)
  <out.npz>.json     sidecar: source and sha256, geometry, MLP widths, appended
                     neurons, substituted rows, the export counts
  --bin <path>       also the reference weight file (FX2TFW01) the exporter
                     writes from this checkpoint, the input of export.sh
  --tch <path>       also torch.save(state_dict), when torch is importable
  --manifest <json>  the export manifest of the container (.bin.json); its
                     geometry is checked against the container and its arm
                     fields are copied into the sidecar

Using the checkpoint in the training tree (GPU host): it has the keys of the
quantized model of its arm (for the shipped weights ZCTX_ARM=pkd, the arm
fields are in the manifest), so it loads with strict=True into the qat and fp32
configurations.  Two checks before training on it:
  1. as models/<geometry>-pkd-q4-fp32.tch, tft/export_weights_poc.py and then
     export.sh must give the input container again (step 4 sha256);
  2. a loss parity run against the coder on a slice.
To continue the QAT phase from it, place it where that phase loads from
(tftcfg.model_path("bf16")) and run with initialize_scales_steps None.  With an
integer, even 0 (TFT_SCALE_STEPS), pysrc/train.py initialize_scales() runs one
weight step, scale = max(scale, max|w|/7) (scaled_max): on the bin centres that
keeps every positive scale (to an fp32 ulp, below the bf16 rounding), but flips
the 18 negative row scales of the shipped weights.  tft/config.py poc_schedule() has no switch for None yet.

Deterministic, no randomness, no network, no environment, torch free unless
--tch is given.

Usage, from the tree root:

  python3 tf/4_export/container_to_tch.py <container.c4> <out.npz>
          [--bin <out.bin>] [--tch <out.tch>] [--manifest <in.bin.json>]
  python3 tf/4_export/container_to_tch.py --check <container.c4>
"""

import hashlib
import json
import os
import struct
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True   # leave no __pycache__ in the package

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# installs the torch free stand in for pysrc.export_weights when needed
import prune_dead_neurons as pdn  # noqa: E402
import weights_compress_v3 as wc3  # noqa: E402

DTYPE_INT8 = 0
DTYPE_BF16_BITS = 1
DTYPE_FLOAT32 = 2
DTYPE_INT32 = 3
REFERENCE_MAGIC = b"FX2TFW01"
QMIN, QMAX = -7, 7

# the scale of a row the chain set to zero and of an appended dead neuron:
# never 0 (the exporter divides by it) and below the limit of
# zero_collapsed_rows, so that the chain removes the row again
ZERO_ROW_SCALE_LOG2 = -24
assert ZERO_ROW_SCALE_LOG2 < wc3.ZERO_ROWS_LOG2

# _ACTIVATION_SCALE_SUFFIXES and _UNQUANTIZED_FP32_SUFFIXES of
# pysrc/export_weights.py, repeated because that module needs torch
ACTIVATION_SCALE_SUFFIXES = (
    ".quantize_activation.scale",
    ".quantize_queries.scale",
    ".quantize_keys.scale",
    ".quantize_values.scale",
)
UNQUANTIZED_FP32_SUFFIXES = (
    ".query_convolution.weight",
    ".key_convolution.weight",
    ".value_convolution.weight",
    ".beta_projection.weight",
    ".dt_bias",
    ".log_baseline_decay_rate",
    ".output_fused_norm_gate.weight",
    ".residual_stream_coefficient.value",
    ".token_embedding_coefficient.value",
    "skip_connection_weights.value",
)
# what tft/export_weights_poc.py appends after the state_dict, in its order
DERIVED = ("rope.inv_freq", "rope.sin", "rope.cos", "config.ints", "config.kimi")
CONFIG_FIELDS = (
    "vocabulary_size", "d_model", "n_layers", "d_head", "n_query_heads",
    "d_mlp", "window_size", "kimi_linear_d_head", "kimi_linear_n_heads",
    "kimi_linear_convolution_size", "rope_base",
)
D_MLP = CONFIG_FIELDS.index("d_mlp")
N_LAYERS = CONFIG_FIELDS.index("n_layers")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def bf16_round(x):
    """float32 rounded to bfloat16 precision, ties to even (the rule of
    torch.bfloat16 and of wpack2_bf16.py), kept as float32"""
    a = np.ascontiguousarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    bias = np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))
    r = ((u + bias) & np.uint64(0xFFFF0000)).astype(np.uint32)
    return r.view(np.float32).reshape(a.shape)


def bf16_bits(x):
    """the uint16 bfloat16 bits of x, round to nearest even"""
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def bf16_value(bits):
    """float32 values of raw bfloat16 bits, same shape"""
    b = np.asarray(bits, dtype=np.uint16)
    return pdn.bf16_to_f32(b.ravel()).reshape(b.shape)


def wpack2_class(name):
    """the class of a raw f32 tensor in wpack2_bf16.py ('-' = not rounded);
    repeated because that module imports the torch side of pysrc"""
    if name.endswith("_convolution.weight"):
        return "conv"
    if name.endswith(".beta_projection.weight"):
        return "beta"
    if name.endswith(".attention.dt_bias"):
        return "dt_bias"
    if name.endswith(".output_fused_norm_gate.weight"):
        return "norm"
    if name.endswith(".log_baseline_decay_rate"):
        return "decay"
    if name.endswith("_coefficient.value") or name == "skip_connection_weights.value":
        return "scale"
    return "-"


def expected_counts(n_layers, kimi, paket_channels):
    """expected_counts() of tft/export_weights_poc.py, repeated (torch)"""
    n_kimi = sum(1 for k in kimi if k)
    n_attn = n_layers - n_kimi
    n_quantized = 3 + 2 * n_layers + 4 * n_attn + 8 * n_kimi
    n_raw_fp32 = 7 * n_kimi + 2 * n_layers + 1
    n_activation_scales = (n_quantized - 1) + 3 * n_attn
    n_quantized += 2 * paket_channels
    n_activation_scales += paket_channels
    return [n_quantized, n_activation_scales, n_raw_fp32]


# ---------------------------------------------------------------------------
# container -> checkpoint
# ---------------------------------------------------------------------------
def checkpoint_from_container(tensors):
    """(state_dict in export order, derived entries, substituted rows,
    appended neurons per layer)"""
    config = np.asarray(tensors["config.ints"][1])
    width = int(config[D_MLP])
    tiny = np.float32(2.0 ** ZERO_ROW_SCALE_LOG2)
    state = {}
    zero_rows = []
    appended = {}
    for name, (code, array) in tensors.items():
        if name in DERIVED or name.endswith(".weight.scale"):
            continue                      # derived, or consumed with its .q
        if name.endswith(".weight.q"):
            assert code == DTYPE_INT8, name
            base = name[:-len(".q")]
            prefix = base[:-len(".weight")]
            scode, sbits = tensors[base + ".scale"]
            assert scode == DTYPE_BF16_BITS, name
            assert not name.endswith(".mlp.up.weight.q") or (
                prefix + ".bias" not in tensors), "an MLP bias is not handled"
            q = np.asarray(array, dtype=np.int8)
            assert q.ndim == 2, name
            s = bf16_value(sbits).ravel().copy()
            assert s.shape == (q.shape[0],), name
            zero = np.flatnonzero(s == 0)
            if len(zero):
                assert ".mlp." not in name, name + ": zero scale inside the MLP"
                assert not (q[zero] != 0).any(), name + ": zero scale, nonzero row"
                s[zero] = tiny
                zero_rows += [(name, int(r)) for r in zero]
            if name.endswith(".mlp.up.weight.q") and q.shape[0] < width:
                layer = int(name.split(".")[1])
                n = width - q.shape[0]
                appended[layer] = list(range(q.shape[0], width))
                q = np.concatenate([q, np.zeros((n, q.shape[1]), np.int8)])
                s = np.concatenate([s, np.full(n, tiny, np.float32)])
            elif name.endswith(".mlp.down.weight.q") and q.shape[1] < width:
                n = width - q.shape[1]
                q = np.concatenate([q, np.zeros((q.shape[0], n), np.int8)], axis=1)
            w = q.astype(np.float32) * s[:, None]
            state[base] = w
            state[prefix + ".quantize_weight.scale"] = s
        elif code == DTYPE_BF16_BITS:
            assert name.endswith(ACTIVATION_SCALE_SUFFIXES), name
            state[name] = bf16_value(array)
        else:
            assert code == DTYPE_FLOAT32, name
            assert name.endswith(UNQUANTIZED_FP32_SUFFIXES), name
            state[name] = np.array(array, dtype=np.float32)
    derived = [(n, tensors[n][0], tensors[n][1]) for n in DERIVED]
    return state, derived, zero_rows, appended


# ---------------------------------------------------------------------------
# checkpoint -> container: the export arithmetic and export.sh, in numpy
# ---------------------------------------------------------------------------
def export_entries(state, derived):
    """the entries tft/export_weights_poc.py writes for this state_dict, in its
    order: quantize_weight_rows and bf16_bits of pysrc/export_weights.py"""
    entries = []
    counts = [0, 0, 0]               # quantized, activation scales, raw fp32
    negative = 0
    for key, value in state.items():
        prefix = key[:-len(".weight")] if key.endswith(".weight") else None
        if prefix is not None and prefix + ".quantize_weight.scale" in state:
            scale_fp32 = state[prefix + ".quantize_weight.scale"]
            scale = bf16_round(scale_fp32)
            with np.errstate(divide="raise", invalid="raise"):
                scaled = value / scale[:, None]       # IEEE fp32 division per row
            ints = np.clip(np.rint(scaled), QMIN, QMAX).astype(np.int8)
            negative += int((scale < 0).sum())
            entries.append((key + ".q", DTYPE_INT8, ints))
            entries.append((key + ".scale", DTYPE_BF16_BITS, bf16_bits(scale_fp32)))
            counts[0] += 1
        elif key.endswith(".quantize_weight.scale"):
            assert key[:-len(".quantize_weight.scale")] + ".weight" in state, key
        elif key.endswith(ACTIVATION_SCALE_SUFFIXES):
            assert bool((bf16_round(value) > 0).all()), "not positive: " + key
            entries.append((key, DTYPE_BF16_BITS, bf16_bits(value)))
            counts[1] += 1
        else:
            assert key.endswith(UNQUANTIZED_FP32_SUFFIXES), key
            assert value.dtype == np.float32, key
            entries.append((key, DTYPE_FLOAT32, value))
            counts[2] += 1
    entries += derived
    return entries, counts, negative


def replay_export_chain(entries):
    """export.sh on the reference entries: bf16 rounding (wpack2_bf16.py),
    [v3, lossless], dead neurons (prune_dead_neurons.py), collapsed rows
    (convert_container_v4.py).  Returns the v4 entries, dead, zeroed rows."""
    rounded = [(n, c, bf16_round(a) if c == DTYPE_FLOAT32 and wpack2_class(n) != "-"
                else a) for n, c, a in entries]
    dead, _stats = pdn.dead_neurons({n: (c, a) for n, c, a in rounded})
    pruned, _rows, _cols = pdn.prune(rounded, dead)
    zeroed, report = wc3.zero_collapsed_rows(pruned, wc3.ZERO_ROWS_LOG2)
    return zeroed, dead, report


def same_bytes(path, entries):
    """does the v4 writer, given these entries, write exactly `path`?"""
    fd, tmp = tempfile.mkstemp(prefix="c2t_roundtrip_", suffix=".c4")
    os.close(fd)
    try:
        wc3.write_tensor_file_v4(tmp, entries)
        with open(path, "rb") as a, open(tmp, "rb") as b:
            return a.read() == b.read()
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def write_npz(path, state):
    """np.savez without its time stamps: one stored .npy member per key"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for key, value in state.items():
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            with z.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(value),
                                          allow_pickle=False)


def write_reference_file(path, entries):
    """the reference weight format FX2TFW01 (write_tensor_file of
    pysrc/export_weights.py)"""
    numpy_of = {DTYPE_INT8: np.int8, DTYPE_BF16_BITS: np.uint16,
                DTYPE_FLOAT32: np.float32, DTYPE_INT32: np.int32}
    with open(path, "wb") as f:
        f.write(REFERENCE_MAGIC)
        f.write(struct.pack("<I", len(entries)))
        for name, code, array in entries:
            data = np.ascontiguousarray(array)
            assert data.dtype == numpy_of[code], name
            encoded = name.encode()
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<B", code))
            f.write(struct.pack("<I", data.ndim))
            f.write(struct.pack("<%dI" % data.ndim, *data.shape))
            f.write(data.tobytes())


def write_tch(path, state):
    import collections

    import torch

    torch.save(collections.OrderedDict(
        (k, torch.from_numpy(np.array(v, copy=True))) for k, v in state.items()), path)


# ---------------------------------------------------------------------------
def main(argv):
    args = list(argv[1:])
    opts = {}
    for flag in ("--bin", "--tch", "--manifest"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i:i + 2]
    check_only = "--check" in args
    if check_only:
        args.remove("--check")
    if len(args) != (1 if check_only else 2):
        print(__doc__, file=sys.stderr)
        return 2
    src = args[0]
    outs = [] if check_only else [args[1], args[1] + ".json"] + [
        opts[f] for f in ("--bin", "--tch") if f in opts]
    for path in outs:
        if os.path.exists(path):
            print("refusing to overwrite %s" % path, file=sys.stderr)
            return 3
    if "--tch" in opts:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("--tch needs torch; without it write the .npz and convert on "
                  "the GPU host", file=sys.stderr)
            return 2
    with open(src, "rb") as f:
        magic = f.read(8)
    if magic != wc3.MAGIC_V4:
        print("STOP: %s is not a v4 container (magic %r); the roundtrip proof "
              "replays the chain that ends in v4" % (src, magic), file=sys.stderr)
        return 4
    src_sha = sha256(src)
    print("input  %s: %d B, sha256 %s" % (src, os.path.getsize(src), src_sha))

    tensors = wc3.read_tensor_file_v3(src)
    config = [int(v) for v in np.asarray(tensors["config.ints"][1])]
    kimi = [int(v) for v in np.asarray(tensors["config.kimi"][1])]
    state, derived, zero_rows, appended = checkpoint_from_container(tensors)
    n_params = sum(int(v.size) for v in state.values())
    print("checkpoint: %d tensors, %d values" % (len(state), n_params))
    print("collapsed rows given scale 2^%d: %d" % (ZERO_ROW_SCALE_LOG2, len(zero_rows)))
    print("dead neurons appended (zero rows/columns): %d in %d layers"
          % (sum(len(v) for v in appended.values()), len(appended)))

    entries, counts, negative = export_entries(state, derived)
    paket_channels = int("embedding.last_table.weight" in state) + int(
        "embedding.match_table.weight" in state)
    want = expected_counts(config[N_LAYERS], kimi, paket_channels)
    assert counts == want, "export counts %s != expected %s" % (counts, want)
    assert len(entries) == 2 * counts[0] + counts[1] + counts[2] + len(DERIVED)

    print("roundtrip: export arithmetic, bf16 rounding, dead neurons, collapsed "
          "rows, v4 ...")
    v4_entries, dead, report = replay_export_chain(entries)
    want_dead = {layer: appended.get(layer, []) for layer in dead}
    ok_dead = dead == want_dead
    ok_rows = sorted((r[0], r[1]) for r in report) == sorted(zero_rows)
    ok_bytes = ok_dead and ok_rows and same_bytes(src, v4_entries)
    print("  dead neurons found = appended: %s" % ok_dead)
    print("  zeroed rows = substituted rows: %s" % ok_rows)
    print("  v4 container: %s" % ("BYTE IDENTICAL" if ok_bytes else "DIFFERENT"))
    if not ok_bytes:
        print("STOP: the checkpoint does not export back to %s.  Nothing was "
              "written." % src, file=sys.stderr)
        return 5
    if check_only:
        return 0

    sidecar = {
        "source": os.path.basename(src),
        "source_bytes": os.path.getsize(src),
        "source_sha256": src_sha,
        "config_ints": dict(zip(CONFIG_FIELDS, config)),
        "config_kimi": kimi,
        "tensors": len(state),
        "values": n_params,
        "export_counts": dict(zip(("n_quantized", "n_activation_scales",
                                   "n_raw_fp32"), counts)),
        "negative_row_scales": negative,
        "zero_row_scale_log2": ZERO_ROW_SCALE_LOG2,
        "collapsed_rows": [[n, r] for n, r in zero_rows],
        "mlp_width_container": {str(l): int(state["blocks.%d.mlp.up.weight" % l]
                                            .shape[0]) - len(appended.get(l, []))
                                for l in range(config[N_LAYERS])},
        "appended_dead_neurons": {str(l): v for l, v in sorted(appended.items())},
        "roundtrip": "export arithmetic + export.sh replayed in numpy: v4 "
                     "container byte identical to the source",
        "note": "QAT state: latent weights on the bin centres (q * s); the "
                "int4 noise of the removed rows is not recoverable",
    }
    if "--manifest" in opts:
        with open(opts["--manifest"], encoding="utf-8") as f:
            manifest = json.load(f)
        for key, value in (("n_layers", config[N_LAYERS]),
                           ("d_model", config[1]),
                           ("vocabulary_size", config[0]),
                           ("window_size", config[6]),
                           ("embedding_rows", state["embedding.weight"].shape[0])):
            assert manifest.get(key) == value, "manifest %s: %r != container %r" % (
                key, manifest.get(key), value)
        sidecar["manifest"] = os.path.basename(opts["--manifest"])
        sidecar["manifest_sha256"] = sha256(opts["--manifest"])
        for key in ("geometry", "paket", "arme", "prior0_sha", "fenster",
                    "redirmerge", "x0row", "vocab_bitmap_hex"):
            if key in manifest:
                sidecar[key] = manifest[key]

    out = args[1]
    write_npz(out, state)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
        f.write("\n")
    print("wrote %s: %d B, sha256 %s" % (out, os.path.getsize(out), sha256(out)))
    print("wrote %s" % (out + ".json"))
    if "--bin" in opts:
        write_reference_file(opts["--bin"], entries)
        back = wc3.read_tensor_file(opts["--bin"])   # pysrc.export_weights
        assert list(back) == [n for n, _c, _a in entries]
        assert all(back[n][0] == c and np.array_equal(back[n][1], a)
                   for n, c, a in entries), "reference file read back differs"
        print("wrote %s: %d B, sha256 %s (reference file, input of export.sh)"
              % (opts["--bin"], os.path.getsize(opts["--bin"]), sha256(opts["--bin"])))
    if "--tch" in opts:
        write_tch(opts["--tch"], state)
        print("wrote %s: %d B" % (opts["--tch"], os.path.getsize(opts["--tch"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
