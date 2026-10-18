#ifndef NEIF_DEV_H_
#define NEIF_DEV_H_

// neif_dev.h - measurement hooks of the DEVELOPMENT build (make cmix_dev,
// make cmix_devtime; see dev/README.md).  Not part of the submitted program.
//
// The submitted build defines neither NEIF_DEV nor NEIF_DEV_TIME.  Then every
// NEIF_DEV_* macro below expands to nothing, or to exactly the call it wraps,
// no other line of this file is compiled, and comp9 stays byte for byte the
// same.
//
// NEIF_DEV=1 (make cmix_dev): hooks in the cold unit runner.cpp only.  The
// coder and the models are not touched; the hooks read the state of the range
// coder and of the predictor between two coded bits.  They act in the main
// process of an -e run and read their switches from its environment (the
// self-invocations that code the dictionary inherit the environment, but run
// with -c and never read it):
//
//   NEIF_DEV_STOP=<n>       code only the first n bytes of the stream, print
//                           the result and exit (the coded prefix is the same
//                           as in a run over the whole stream)
//   NEIF_DEV_COST=<path>    per stream byte 8 x u16: the coded cost of each of
//                           its bits, high bit first, in 1/2048 bit (0 for a bit
//                           the grammar automaton forces); taken from the state
//                           of the range coder, so it is the real cost
//   NEIF_DEV_STREAM=<path>  a copy of the stream the coder codes
//   NEIF_DEV_TFDUMP=<path>  one record per stream byte in [NEIF_DEV_TFDUMP_FROM,
//                           NEIF_DEV_TFDUMP_FROM + NEIF_DEV_TFDUMP_LEN), taken
//                           before the byte is coded, little endian:
//                             u32 position, u8 byte, u8 gate zone, u16 0,
//                             f16 slot[208]    the transformer row the mixers
//                                              read, in set order (uniform in a
//                                              gated zone),
//                             f16 hidden[192]  the transformer's final hidden
//                                              state (zero without transformer),
//                             f16 ppmd[208]    the PPMd byte distribution, in
//                                              set order
//                           = 1224 B per byte
//
// NEIF_DEV_TIME=1 (make cmix_devtime, needs NEIF_DEV): in addition the cycle
// counters of the sections of Predictor::Predict/Perceive and of the BITLSTM
// head (NEIF_DEV_T0/T1), printed at the end of the coding loop.  They sit in
// the hot units, where they may change how the compiler orders float
// arithmetic, so this build can code other bits: use it for time only.

#ifndef NEIF_DEV
#define NEIF_DEV 0
#endif
#ifndef NEIF_DEV_TIME
#define NEIF_DEV_TIME 0
#endif
#if NEIF_DEV_TIME && !NEIF_DEV
#error "NEIF_DEV_TIME needs NEIF_DEV"
#endif

#if NEIF_DEV

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#include <string>

#include "tf1/tf1_engine.h"
#include "transformer.h"

namespace neif_dev {

// ---- sections of the cycle counters (NEIF_DEV_TIME) ----
enum Section {
  kFxcmPredict,      // fxcm PredictHalf
  kInputs,           // contexts, model outputs into the layer-0 row
  kMix,              // adaptive depth, mixers, SSE, final probability
  kBitl,             // BITLSTM correction and its observation
  kWordLabel,        // word-label model, update
  kModelsBit,        // bit updates of bracket, direct, match, indirect, grammar
  kByteModelsBit,    // bit updates of PPMd, byte LSTM and transformer slot
  kMixTrain,         // mixer, SSE and adaptive-depth training
  kContexts,         // context manager update
  kModelsByte,       // byte updates of bracket, direct, match, indirect, grammar
  kPpmd,             // PPMd byte update
  kTransformer,      // transformer steps (coded and observed bytes, prompts)
  kByteLstm,         // byte LSTM byte update and its outputs
  kFxcmPerceive,     // fxcm update
  kSections
};
inline const char* const kSectionName[kSections] = {
  "fxcm predict", "inputs", "mixers+sse", "bitlstm", "word label",
  "models bit", "byte models bit", "mixer training", "contexts",
  "models byte", "ppmd byte", "transformer", "byte lstm", "fxcm perceive",
};
inline unsigned long long g_cyc[kSections] = {0};

inline unsigned long long Tsc() { return __rdtsc(); }

// ---- switches and state of the -e run ----
inline bool on = false;
inline bool truncated = false;
inline unsigned long long stop_at = 0;
inline FILE* cost_file = nullptr;
inline std::string stream_path;
inline FILE* tf_file = nullptr;
inline unsigned long long tf_from = 0;
inline unsigned long long tf_len = 0;
inline uint16_t bit_cost[8] = {0};
inline double t0 = 0;
inline unsigned long long tsc0 = 0;

inline double Now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

inline unsigned long long EnvU(const char* name, unsigned long long def) {
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = nullptr;
  const unsigned long long x = strtoull(v, &end, 10);
  if (!end || *end) {
    fprintf(stderr, "dev build: %s=%s is not a number\n", name, v);
    exit(2);
  }
  return x;
}

inline FILE* OpenOut(const char* name) {
  const char* path = getenv(name);
  if (!path || !*path) return nullptr;
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "dev build: cannot write %s=%s\n", name, path);
    exit(2);
  }
  setvbuf(f, nullptr, _IOFBF, 1 << 20);
  return f;
}

// main(): only the -e run switches the hooks on.
inline void Init(bool e_run) {
  if (!e_run) return;
  on = true;
  stop_at = EnvU("NEIF_DEV_STOP", 0);
  cost_file = OpenOut("NEIF_DEV_COST");
  if (const char* s = getenv("NEIF_DEV_STREAM")) stream_path = s;
  tf_file = OpenOut("NEIF_DEV_TFDUMP");
  tf_from = EnvU("NEIF_DEV_TFDUMP_FROM", 0);
  tf_len = EnvU("NEIF_DEV_TFDUMP_LEN", 1000000);
  fprintf(stderr, "dev build: stop %llu, cost %s, stream %s, tf dump %s",
          stop_at, cost_file ? "on" : "off",
          stream_path.empty() ? "off" : stream_path.c_str(),
          tf_file ? "on" : "off");
  if (tf_file) fprintf(stderr, " [%llu, +%llu)", tf_from, tf_len);
  fprintf(stderr, "%s\n", NEIF_DEV_TIME ? ", cycle counters on" : "");
}

// RunCompression(): the stream, once it is complete.
inline void StreamCopy(const std::string& temp_path) {
  if (!on || stream_path.empty()) return;
  FILE* in = fopen(temp_path.c_str(), "rb");
  FILE* out = fopen(stream_path.c_str(), "wb");
  if (!in || !out) {
    fprintf(stderr, "dev build: cannot copy the stream to %s\n",
            stream_path.c_str());
    exit(2);
  }
  static char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) fwrite(buf, 1, n, out);
  fclose(in);
  fclose(out);
}

// Compress(), after its progress step is computed: shortens the loop.
inline void Limit(unsigned long long* input_bytes) {
  if (!on) return;
  for (int k = 0; k < kSections; ++k) g_cyc[k] = 0;
  if (stop_at && stop_at < *input_bytes) {
    *input_bytes = stop_at;
    truncated = true;
  }
  t0 = Now();
  tsc0 = Tsc();
}

template <class E>
inline void EncodeBit(E& e, int bit, int j) {
  if (!cost_file) {
    e.Encode(bit);
    return;
  }
  const double before = e.DevBits();
  e.Encode(bit);
  double c = (e.DevBits() - before) * 2048.0 + 0.5;
  if (c < 0) c = 0;
  if (c > 65535) c = 65535;
  bit_cost[7 - j] = (uint16_t)c;
}

template <class P>
inline void WriteTfRecord(P* p, unsigned long long pos, unsigned int byte) {
  uint8_t head[8] = {(uint8_t)pos, (uint8_t)(pos >> 8), (uint8_t)(pos >> 16),
                     (uint8_t)(pos >> 24), (uint8_t)byte,
                     (uint8_t)(p->bitl_zone_ ? *p->bitl_zone_ : 0), 0, 0};
  fwrite(head, 1, 8, tf_file);
  float row[transformer::kSetSize];
  uint16_t half[transformer::kSetSize];
  const float* slot = p->DevTfRow();
  const float* ppmd = p->DevPpmdRow();
  for (int pass = 0; pass < 2; ++pass) {
    const float* src = pass == 0 ? slot : ppmd;
    for (int i = 0; i < transformer::kSetSize; ++i) row[i] = 0.0f;
    if (src) {
      for (int b = 0; b < 256; ++b) {
        if (transformer::InSet(b)) row[transformer::kByteToIndex[b]] = src[b];
      }
    }
    transformer::FloatsToHalves(row, half, transformer::kSetSize);
    fwrite(half, 2, transformer::kSetSize, tf_file);
    if (pass == 0) {
      float hid[192] = {0.0f};
      if (p->bitl_hid_ && p->bitl_n_hid_ == 192) {
        for (int i = 0; i < 192; ++i) hid[i] = p->bitl_hid_[i];
      }
      uint16_t hhalf[192];
      transformer::FloatsToHalves(hid, hhalf, 192);
      fwrite(hhalf, 2, 192, tf_file);
    }
  }
}

// Compress(): before the bits of a stream byte are coded ...
template <class E, class P>
inline void ByteBegin(E& e, P* p, unsigned long long pos, unsigned int byte) {
  (void)e;
  if (!on) return;
  for (int j = 0; j < 8; ++j) bit_cost[j] = 0;
  if (tf_file && pos >= tf_from && pos - tf_from < tf_len)
    WriteTfRecord(p, pos, byte);
}

// ... and after the byte and everything it triggers are done.
inline void ByteEnd() {
  if (on && cost_file) fwrite(bit_cost, 2, 8, cost_file);
}

inline void PrintCycles(unsigned long long bytes) {
#if NEIF_DEV_TIME
  const unsigned long long total = Tsc() - tsc0;
  unsigned long long sum = 0;
  fprintf(stderr, "dev build: cycles of the coding loop, %llu stream bytes\n",
          bytes);
  for (int k = 0; k < kSections; ++k) {
    sum += g_cyc[k];
    fprintf(stderr, "  %-16s %16llu  %6.2f %%  %10.1f per byte\n",
            kSectionName[k], g_cyc[k], 100.0 * g_cyc[k] / (double)total,
            bytes ? g_cyc[k] / (double)bytes : 0.0);
  }
  const unsigned long long rest = total > sum ? total - sum : 0;
  fprintf(stderr, "  %-16s %16llu  %6.2f %%  %10.1f per byte\n", "rest", rest,
          100.0 * rest / (double)total, bytes ? rest / (double)bytes : 0.0);
  fprintf(stderr, "  %-16s %16llu\n", "total", total);
#else
  (void)bytes;
#endif
}

// Compress(), after Flush(): the result.  A shortened run ends here.
template <class O>
inline void AfterFlush(O* os, unsigned long long pos) {
  if (!on) return;
  const double secs = Now() - t0;
  const unsigned long long out = (unsigned long long)os->tellp();
  // the 5-byte header of the stream is part of the output file
  fprintf(stderr,
          "dev build: coded %llu stream bytes -> %llu bytes (with the 5-byte "
          "header), %.6f bits per stream byte, %.1f s in the coding loop\n",
          pos, out, pos ? 8.0 * (double)(out - 5) / (double)pos : 0.0, secs);
  unsigned long long hwm_kb = 0;
  if (FILE* st = fopen("/proc/self/status", "r")) {
    char line[256];
    while (fgets(line, sizeof(line), st)) {
      if (sscanf(line, "VmHWM: %llu kB", &hwm_kb) == 1) break;
    }
    fclose(st);
  }
  printf("NEIF_DEV_RESULT stream_bytes=%llu coded_bytes=%llu seconds=%.1f "
         "peak_rss_kb=%llu\n", pos, out, secs, hwm_kb);
  PrintCycles(pos);
  if (cost_file) fclose(cost_file);
  if (tf_file) fclose(tf_file);
  cost_file = tf_file = nullptr;
  on = false;
  if (truncated) {
    fflush(nullptr);
    _exit(0);
  }
}

}  // namespace neif_dev

#define NEIF_DEV_INIT(e_run) neif_dev::Init(e_run)
#define NEIF_DEV_STREAM_COPY(path) neif_dev::StreamCopy(path)
#define NEIF_DEV_LIMIT(n) neif_dev::Limit(&(n))
#define NEIF_DEV_ENCODE(enc, bit, j) neif_dev::EncodeBit(enc, bit, j)
#define NEIF_DEV_BYTE_BEGIN(enc, p, pos, c) neif_dev::ByteBegin(enc, p, pos, c)
#define NEIF_DEV_BYTE_END() neif_dev::ByteEnd()
#define NEIF_DEV_AFTER_FLUSH(os, pos) neif_dev::AfterFlush(os, pos)

#else  // the submitted build

#define NEIF_DEV_INIT(e_run)
#define NEIF_DEV_STREAM_COPY(path)
#define NEIF_DEV_LIMIT(n)
#define NEIF_DEV_ENCODE(enc, bit, j) enc.Encode(bit)
#define NEIF_DEV_BYTE_BEGIN(enc, p, pos, c)
#define NEIF_DEV_BYTE_END()
#define NEIF_DEV_AFTER_FLUSH(os, pos)

#endif

#if NEIF_DEV_TIME
#define NEIF_DEV_T0(v) const unsigned long long v = neif_dev::Tsc()
#define NEIF_DEV_T1(v, sec) neif_dev::g_cyc[neif_dev::sec] += neif_dev::Tsc() - (v)
#else
#define NEIF_DEV_T0(v)
#define NEIF_DEV_T1(v, sec)
#endif

#endif  // NEIF_DEV_H_
