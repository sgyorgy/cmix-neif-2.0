# dev/ - measurement tools of the development work

Nothing in this directory is part of the submission. `build.sh` does not read
it, and the submitted `comp9` is byte for byte the same with or without it
(sha256 `8b07d85f…`, checked by rebuilding after the hooks went in).

## The development build

`src/neif_dev.h` holds measurement hooks. They are compiled in only by
`make cmix_dev` (`-DNEIF_DEV=1`) and `make cmix_devtime` (in addition
`-DNEIF_DEV_TIME=1`). In the submitted build every hook expands to nothing or
to exactly the call it wraps.

`cmix_dev` has hooks in the cold unit `src/runner.cpp` only. They read the
state of the range coder and of the predictor between two coded bits and do
not touch the coder or the models. Switches, from the environment of the
`-e` run:

| variable | effect |
|---|---|
| `NEIF_DEV_STOP=<n>` | code only the first `n` bytes of the stream, print the result, exit. The coded prefix is the prefix of a full run. |
| `NEIF_DEV_COST=<path>` | per stream byte 8 × u16: the coded cost of each bit in 1/2048 bit, from the coder state (the real cost, rounding included) |
| `NEIF_DEV_STREAM=<path>` | a copy of the stream the coder codes (552,750,198 B for enwik9) |
| `NEIF_DEV_TFDUMP=<path>`, `_FROM`, `_LEN` | per stream byte in the range: the byte, the gate zone, the transformer row the mixers read, the final hidden state, the PPMd row; 1224 B per byte (`tfdump.py`) |

At the end of the coding loop the binary prints a line
`NEIF_DEV_RESULT stream_bytes=… coded_bytes=… seconds=… peak_rss_kb=…` on
stdout. `cmix_devtime` adds cycle counters for the sections of
`Predictor::Predict`/`Perceive` and the BITLSTM head. They sit in hot code and
can change the order of float operations, so that build is for time only.

## Scripts

* `build_dev.sh <tree> <comp9> <out> [time]` builds the development binary of
  a source tree and appends the four data blocks of a built `comp9` (hints,
  article order, weight container, header), so that it runs `-e enwik9`.
* `ab_prefix.sh <enwik9> <bytes> <workdir> <comp9> <arm>...` builds every
  arm and codes the first `<bytes>` bytes of the stream with each, one run
  after the other. It prints the coded bytes against the first arm and the
  per-window difference of the cost files. An arm is a source tree,
  `<tree>=<defines>` for experiment defines, and `...@<container>` for another
  weight container (`build_dev.sh` stamps its FNV-1a-64 into the work copy of
  `src/transformer.h`).
* `cost_report.py` costs per window, per bit position and per byte value of
  one run, or the difference of two.
* `tfdump.py` reads a transformer dump: cross entropy of the transformer slot
  and of PPMd, and with the cost file of the same run that of the coder.
* `fetch_enwik9.sh <dir> [<url>]` fetches `enwik9.zip` (default: the official
  location) and keeps `enwik9` only if its sha256 is `159b8535…`.

## How to use them

1. `bash build.sh` in a scratch directory gives `comp9` (sha256 `8b07d85f…`).
2. For each arm a source tree: `git worktree add ../arm-x <commit>`.
3. `bash dev/ab_prefix.sh enwik9 40000000 work comp9 . ../arm-x` codes the
   first 40 MB of the stream (7.2 % of it) with both; about 3.5 h per arm on
   one core of a c2d machine, judged by the full run (48.7 h for the whole
   stream).
4. A change that must not change the archive (a speed change): run comp9 and
   the arm's `cmix_dev` on the same prefix and compare the coded output. The
   coder writes the archive stream in 1 MiB pieces, so the first bytes of
   `archive9` of a running `comp9 -e` are the coded prefix as well.

The stream prefix is not a random sample: the metadata block comes first, and
the rate along the stream is in `doc/stream_rate.tsv`. A decision rests on a
full run.

## Experiment arms

An arm is code behind a define that the submitted build does not set; it is
built with `DEV_DEFS` (`ab_prefix.sh` takes `<tree>=<defines>`). Arms in this
tree:

| define | idea | files |
|---|---|---|
| `NEIF_ARM_TFCTX=1` | the final mixer selects its weight row, outside the gated zones, by the transformer's most probable byte: still possible after the bits coded so far, and its probability in 8 steps (today one row serves all those bytes); the new rows start seeded like the head row | `src/tf_sharpen.h`, `src/mixer/byte-mixer.cpp`, `src/predictor.cpp`, `src/context-manager.cpp` |
| `NEIF_ARM_TFCTX=2` | the 24th layer-0 mixer selects by the state of the byte LSTM (as today) times that state of the transformer | same |
| `NEIF_ARM_TFCTX=3` | both | same |

Neither changes the geometry of the mixers, so BITLSTM reads the same
columns. Example, 40 MB of the stream, reference and the three arms:

```
bash dev/ab_prefix.sh enwik9 40000000 work comp9 . .=-DNEIF_ARM_TFCTX=1 \
     .=-DNEIF_ARM_TFCTX=2 .=-DNEIF_ARM_TFCTX=3
```

## Memory

A coder run allocates its tables in full whatever the prefix, about 9.75 GiB.
Two runs at once need about 20 GiB.
