#ifndef TF_SHARPEN_H_
#define TF_SHARPEN_H_

// tf_sharpen.h - experiment arm NEIF_ARM_TFCTX (dev/README.md): the
// transformer's most probable byte, whether it is still possible after the
// bits of the current byte coded so far, and its probability, as a selector of
// mixer weight rows.  The same idea as lstm_sharpen.h for the small byte LSTM
// (the context of the 24th layer-0 mixer), applied to the strongest model.
//
//   NEIF_ARM_TFCTX & 1  the final mixer selects its row by this state outside
//                       the gated zones (today: one row for every byte there)
//   NEIF_ARM_TFCTX & 2  the 24th layer-0 mixer selects by the LSTM state times
//                       this state
//   NEIF_ARM_TFCTX & 4  the same with three states (Q3) instead of nine.  With
//                       nine the mixer needs up to ~13,000 rows (state x the
//                       partial byte), past the 10,000 a mixer keeps
//                       (Mixer::GetContextData); every row beyond shares one.
//                       TFCTX=3 lost 0.22 % on a 20 MB prefix.
//
// Not compiled into the submitted program (NEIF_ARM_TFCTX undefined).

#ifndef NEIF_ARM_TFCTX
#define NEIF_ARM_TFCTX 0
#endif

#if NEIF_ARM_TFCTX

namespace tf_sharpen {

struct State {
  int ex = -1;          // most probable byte of the slot row, -1 = no row
  float best = 0.0f;    // its probability
  int alive = 0;        // ex is still possible after the bits coded so far
};

inline State state;

// after the slot row is final (ByteModel::ByteUpdate zeroes blocked bytes)
template <class Row>
inline void Top(const Row& probs) {
  int ex = 0;
  float best = probs[0];
  float sum = probs[0];
  for (int i = 1; i < 256; ++i) {
    sum += probs[i];
    if (probs[i] > best) { best = probs[i]; ex = i; }
  }
  state.ex = ex;
  state.best = sum > 0.0f ? best / sum : 0.0f;
  state.alive = 1;
}

inline void Clear() {
  state.ex = -1;
  state.best = 0.0f;
  state.alive = 0;
}

// 0: no row or the top byte is ruled out; 1..8: the top byte's probability
inline unsigned int Q() {
  if (!state.alive) return 0;
  const float b = state.best;
  if (b < 0.20f) return 1;
  if (b < 0.40f) return 2;
  if (b < 0.60f) return 3;
  if (b < 0.75f) return 4;
  if (b < 0.85f) return 5;
  if (b < 0.92f) return 6;
  if (b < 0.97f) return 7;
  return 8;
}
constexpr unsigned int kStates = 9;

// 0: no row or the top byte is ruled out; 1: below 0.6; 2: from 0.6
inline unsigned int Q3() {
  const unsigned int q = Q();
  return q == 0 ? 0u : (q <= 3 ? 1u : 2u);
}
constexpr unsigned int kStates3 = 3;

static_assert((NEIF_ARM_TFCTX & 6) != 6, "NEIF_ARM_TFCTX: bits 2 and 4 exclude each other");

}  // namespace tf_sharpen

#endif  // NEIF_ARM_TFCTX

#endif  // TF_SHARPEN_H_
