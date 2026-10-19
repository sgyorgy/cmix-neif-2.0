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
  one run, or the difference of two (with `--diff` also split into the
  metadata block, the redirect block and the article text of the prefix).
* `tfdump.py` reads a transformer dump: cross entropy of the transformer slot
  and of PPMd, and with the cost file of the same run that of the coder.
* `fetch_enwik9.sh <dir> [<url>]` fetches `enwik9.zip` (default: the official
  location) and keeps `enwik9` only if its sha256 is `159b8535…`.
* `sim_online.py <tfdump.bin> <cost.u16>` simulates online adaptation on the
  transformer state (an online head adapter, LSH contexts of the hidden state)
  and estimates the gain for the coder with a per-bit mixer proxy
  (PLAN_10PCT.md 2.1).
* `sim_final.py <tfdump.bin> <cost.u16>` tests the coder's final stage on the
  same data: an APM behind the coder keyed by the transformer state, an online
  re-mix of the coder with PPMd and the transformer under confidence-selected
  weight sets, and the slot under a temperature or a linear PPMd mix
  (PLAN_10PCT.md 1.2, section 10).
* `meta_report.py <stream.bin> <cost.u16>` splits the cost of the metadata
  block by field (page-id delta, revid, restrictions, timestamps, names, uids).
* `prefix_identity.sh <archive9> <archive9>` checks that two runs coded the
  same bits on their common prefix (a running `comp9 -e` or a stopped
  development run).
* `prof_sample.sh <pid> <s> <run.err> <out>` and `prof_report.py <map> <out>`:
  a sampling time profile of a running coder (gdb reads its program counter
  every few seconds; the linker map of the same build, which `ab_prefix.sh`
  keeps as `cmix_dev.map`, names the function), split by object file and
  function, and by the progress of the run (`--from 44.22`: the article text
  of a 20 MB prefix). It needs no extra run and does not change the coding.

`ab_prefix.sh` runs up to `AB_JOBS` arms at once (default 1).

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
| `NEIF_ARM_TFCTX=3` | both (20 MB prefix: +0.22 %; the 9 x 5 states times the partial byte overflow the 10,000 rows a mixer keeps) | same |
| `NEIF_ARM_TFCTX=4` | as 2 with three states of the transformer (top byte ruled out, below 0.6, from 0.6); combines with 1 (`=5`), not with 2 | `src/tf_sharpen.h`, `src/context-manager.cpp` |
| `NEIF_ARM_TFFX=1` | the transformer slot's prediction of the current bit as an internal input of the 18 first-layer mixers of fxcm (today only the byte LSTM's is) | `src/models/fxcmv1.cpp`, `src/predictor.cpp` |
| `NEIF_ARM_TFFX=2` | the same for the second-layer mixers of fxcm | same |
| `NEIF_ARM_TFFX=4` | an APM on that prediction, in the context of the partial byte and the last byte, as one more internal input of the first-layer mixers of fxcm (as an fxcm output it did not fit: fxcm fills 555 of its 557 columns, the predictor needs two free) | same |
| `NEIF_ARM_ROWLR=1` | a per-row learning rate in every mixer: the update of a weight row is scaled by 1.5 - (its updates) / (updates of the busiest row), the per-row factor of the original cmix mixer that this lineage dropped; rare contexts learn faster | `src/mixer/mixer.h`, `src/mixer/mixer.cpp` |

The bits of `NEIF_ARM_TFFX` combine (`=7`: all three). The first layer of
fxcm has room for 560 inputs, the second for 64 (32 used by its mixers); the
development build prints the most it saw at the end of a run, and the main
predictor stops with a message if fxcm's outputs leave fewer than two free
columns.

None of them changes the geometry of the main mixers, so BITLSTM reads the
same columns. Example, 40 MB of the stream, reference and the three arms:

```
bash dev/ab_prefix.sh enwik9 40000000 work comp9 . .=-DNEIF_ARM_TFCTX=1 \
     .=-DNEIF_ARM_TFCTX=2 .=-DNEIF_ARM_TFCTX=3
```

## Memory

A coder run allocates its tables in full whatever the prefix, about 9.75 GiB.
Two runs at once need about 20 GiB. About 9.8 GB of a run cannot be reclaimed
(5.65 GB anonymous, 4.19 GB shared memory); in a memory cgroup the rest of the
limit is all that other jobs get, and when they cross it the kernel kills the
largest process, the coder. On the 13.36 GiB cgroup of the development
container that left 3.5 GB: start other jobs during a run with
`sh -c 'echo 1000 > /proc/self/oom_score_adj; exec ...'` so that they go
first.
