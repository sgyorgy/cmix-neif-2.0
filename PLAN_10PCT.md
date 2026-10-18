<!-- Terv a cmix-neif-2.0-hoz: a teljes repo átnézése alapján (2026-10-18). A számok forrása a kód, a README, doc/stream_rate.tsv, training/tf/tree_overlay/zctx/eval_zctx.py és training/tf/4_export/container_stats.py. -->

# cmix-neif-2.0 → −10 % S: megvalósíthatósági ítélet és végrehajtási terv

## Context
Cél: a repo jelenlegi Hutter-beadásához (cmix-neif-1.0) képest 10 %-kal kisebb
S = comp9 + archive9 + comp9.args, a Hutter-szabályok betartásával.
- Bázis: S = 4,000,689 + 91,913,343 + 19 = **95,914,051 B** → cél **≤ 86,322,646 B (−9.59 MB)**.
- Szabályok (README „The Hutter Prize” alapján; a prize.hutter1.net ebből a sessionből nem
  volt elérhető, beadás előtt újraellenőrizni): 1 CPU mag, GPU nélkül, ≤ 70,000/T óra
  programonként (T = Geekbench 5 single-core), ≤ 10 GiB RAM, ≤ 100 GB temp disk; S-be minden
  bájt beszámít; ≥ 1 % javulás a rekordhoz képest.
- Erőforrások (felhasználói válasz): GPU = Colab TPU v5e-1 (40 óra) + helyi RTX 5070 Ti 12 GB
  (óraszám tisztázandó); elérhető: BITLSTM trace-író, BITLSTM súlyok, PPMd prior-generátor,
  tréningkód; **a transformer bf16/QAT checkpointjai NINCSENEK meg**; célgép: csak AVX2 (c2d/Zen 3).

## 1. Hova megy S (a kód alapján mért/levezetett)
| tétel | bájt | megjegyzés |
|---|---:|---|
| payload (kódolt stream) | ≈ 87.90 M | doc/stream_rate.tsv; 552.75 M stream-bájt, átlag 1.27 bpc |
| ebből metaadat-blokk | ≈ 1.40 M | part A ≈0.69 M (page-id delta + revid), part B ≈0.71 M |
| transformer-konténer ×2 | 7,433,334 | 7,785,008 int4 súly × **3.73 bit** = 3.63 MB/példány (a konténer 98 %-a) |
| kód + .dict.comp + hints + order | ≈ 0.58 M | decomp_bin ≈0.16 M ×(archív) + packed bináris ≈0.18 M + dict.comp ≈0.14 M + 57 K + 44 K |

Kulcs-átváltás (training/tf/tree_overlay/zctx/eval_zctx.py, a projekt saját mérése):
**0.001 nat/byte transformer-veszteség ≈ 33/41/51 KB archive9**, súly ≈ 0.478 B/param **×2**.
Időkeret: 48.72 h / 53.60 h (≈9 % tartalék); RAM 9.75 / 10 GiB.

## 2. Megvalósíthatósági ítélet (őszintén)
- A −9.59 MB-hoz egyedül a transformerből kb. −0.23 nat/byte javulás kellene (41 KB/millinat
  mellett; a mai veszteség ~0.75–0.9 nat/byte nagyságrendű), ami irreális; csak több nagy kar
  együttes felső becslésével jön ki. A legnagyobb karok (nagyobb/jobb transformer,
  kvantálás újratanítása nulláról) a szállított lánc nagyságrendjét igénylik (**~2,800 RTX 5090
  GPU-óra**; egy új teljes modell ~460 5090-óra ≈ ~1,100 óra egy 5070 Ti-n).
- A megadott erőforrásokkal (5070 Ti + 40 h TPU, checkpointok nélkül) **reálisan −1…−4 %
  (közép ≈ −2.2 %, −2.1 MB)** érhető el (lásd 4. táblázat). **A 10 %-hoz a 5. fázis (nagy GPU-track, kb.
  2,000–3,000 5090-ekvivalens GPU-óra bérelve) szükséges**, és ott is a felső becslések kellenek.
- A TPU v5e-1 tréningre gyakorlatilag nem használható (PyTorch + Triton/fla KDA-kernelek, nincs
  TPU-port); legfeljebb JAX-os offline szimulációkra (2. fázis) opcionális.
- A terv ezért döntési kapukkal halad: először a biztos, olcsó karok (0–3. fázis), utána mért
  adatok alapján döntés a 4–5. fázisról.

## 3. Fázisok

### 0. fázis — Bázis és mérőinfrastruktúra (CPU + GPU bring-up)
1. Reprodukció: `install.sh`, `build.sh` Ubuntu 24.04-en; comp9 sha256 = 8b07d85f…;
   egy teljes `-e` futás c2d-highcpu-8-on (bázis archive9 + időprofil).
2. Fejlesztői build-variáns (nem szállított, `-D` kapcsoló, mint `AUX_BUILD`):
   - bitenkénti kódköltség-dump régiócímkével (meta A/B1/B2, redirect, cikk, naptár, census,
     image, `Wikipedia:`, egyértelműsítő) — `Encoder::Encode` (src/coder/encoder.cpp);
   - bájtonkénti transformer-dump: `TransformerOpt::last_hidden()` (192×fp16) + kimeneti
     eloszlás + kódolt bájt — `Tf1Engine::Step` (src/tf1/tf1_engine.cpp);
   - rdtsc-szekciók komponensenként (transformer, fxcm, PPMd+lapozás, mixerek, byte-LSTM,
     BITLSTM, word-label) — `Predictor::Predict/Perceive` (src/predictor.cpp).
3. A/B harness: rögzített szeletek (pl. 0–40 MB prefix, egy 40 MB-os cikk-szelet, census/végszelet),
   minden kar ugyanazon a gépen/binárison; döntéshez teljes futás.
4. GPU bring-up (5070 Ti): stream → dump → priorok (228 GB, `training/tf/1_dump`, a
   felhasználó prior-generátora) → package D (`training/tf/2_package/build_package.sh`);
   micro-batch 2^17 token + gradient checkpointing (12 GB elég kell legyen; mérni).
5. **Új: konténer → checkpoint konverter** (`training/tf/4_export/container_to_tch.py`):
   `weights_compress_v3.read_tensor_file_v3` a torch-mentes stubbal
   (`prune_dead_neurons._install_export_weights_stub` mintájára) + az
   `tft/export_weights_poc.py` névleképezésének inverze; latens súly = q·scale, LSQ-scale =
   konténer-scale. Paritás: GPU fp32 eval vs C++ engine valószínűségek egy szeleten.
6. Zónatérkép a transformer hozzájárulásáról: `zctx/zonemap.py`, `zonemap2.py`, `zone_eval.py`.

### 1. fázis — Kódoló-oldali gyors nyereségek (CPU, újratanítás nélkül)
Mindegyik külön A/B kar; a mixer-/stage-1 geometria változása után a BITLSTM-et újra kell
tanítani (1.6), mert a stage-1 sor oszlopait olvassa (`bit_head.cpp` Head: n1=28, drop 7–20, 27).
1. Transformer-konfidencia mint mixer-kontextus: `Tf1Slot::SetProbs`
   (src/mixer/byte-mixer.cpp) számolja az argmax-bájtot, top-p-t, entrópia-vödröt
   (a `ByteMixer::SetProbs`/`lstm_sharpen::state` mintájára); `ContextManager`-ben
   `UpdateTf1Agree()` (a `UpdateLx1` mintájára: „az argmax még kompatibilis-e a c0 bitekkel” ×
   konfidencia) → új layer-0 mixer (25.) és gazdagabb final-mixer kontextus (jelenleg csak a
   gate-zóna: `manager_.tf1_gate_ctx1`, src/predictor.cpp AddMixers). Build-ellenőrzések:
   `Mixer` ctor N0=588/N1=27, `mixer_chain::kMaxMixers`.
2. APM/SSE a transformer bit-valószínűségén (c0 × order-1 × kvantált stretch(p_tf)) extra
   layer-0 bemenetként (fxcm `APM` vagy `SSE_sh` újrahasznosítva).
3. p_tf betáplálása fxcm belső mixereibe és APM-láncába (ma csak a kis LSTM `lstmpr` megy be):
   `x.mxInputs1.add_internal / mxInputs2.add_internal` (src/models/fxcmv1.cpp ~6270, ~6291),
   `apmA0` kontextus (`predict_half`, ~6374); új `tf_fx_link` a `src/lstm_fx_link.h` mintájára.
4. Mixer-optimalizáló: normalizált/adaptív lépés (RMS- vagy Adam-szerű) a layer-0 és final
   mixerekben (src/mixer/mixer.cpp `Perceive`, src/mixer_bank.h `TrainRows`), determinisztikus
   osztás/gyök a `mixer_chain::RmsScale` mintájára.
5. Elavult, fx2-cmix-streamre hangolt pozíció-küszöbök az fxcm-ben: `x.blpos<448131719`
   (fxcmv1.cpp 4964, 4983), `x.blpos<451531986` (3462) → a mostani stream régióhatáraihoz
   kötni (stampelt konstans), mérni.
6. BITLSTM teljes streamen: a dev-fa trace-íróval 100 %-os trace, replay-tanítás
   (`training/bitl/tools/train_blob.sh`, `cut_trace.sh`), új súlyok `src/bit_head_weights.h`,
   a fagyasztási pont `src/bit_head.cpp:1198` (418837176) átállítása.
7. Metaadat-blokk (a transformer gate-elve van ott → **nem kell újratanítás**): page-id kódolása
   rangként a még szabad id-k halmazában (a `id_exclude.h` Fenwick-fáját újrahasználva),
   revid-kontextusok; fájlok: `src/metablock_form.h`, `src/metablock.h`, `src/id_exclude.h`,
   `src/readalike_prepr/header_sort.h`; roundtrip: `src/preprocess_roundtrip.cpp`.
   A frozen 208-as ábécén belül kell maradni (`src/vocab_frozen.h`).

### 2. fázis — Online adaptáció a transformer állapotán (CPU)
1. Offline szimuláció előbb (numpy; opcionálisan JAX/TPU): a 0.2-es dumpból
   (a) LSH-vázlat (pl. 12/16/20 bit, fix seed-ű random-projekció előjelei h-ból) + order-0/1/2
   bájt-kontextus → állapotszámlálós prediktorok; (b) online fej-adapter ΔW (208×192, SGD)
   a frozen unembedding mellé. Mérőszám: a transformer-slot CE-javulása és egy 2-bemenetes
   keverék-proxy. **Go/no-go: ≥ 0.004 nat/byte proxy-javulás (≈ ≥ 150 KB várható).**
2. C++: a nyertes(ek) szigorú FP-egységben (`bit_head.cpp`/`adaptive_depth.cpp` mintája,
   makefile-szabály `-ffp-model=precise -ffp-contract=off`); LSH-kontextusok új context-map
   modellként a layer-0-ba; az adapter új bájt-slotként (`Tf1Slot` mintájára).
3. Időmérleg: ha az adapter dominálja a 80 cellás byte-LSTM-et (~8–10 % futásidő, kis
   nyereség), annak kivétele/szűkítése; cél ≤ 50 h a c2d-n (tartalék a 70,000/T limithez).

### 3. fázis — Súly-oldali ráta–torzítás (GPU-light, 5070 Ti)
Indok: a szintek közel i.i.d.-k (az upstream mérés szerint order-1 kontextus nem segít), a
hisztogram harang alakú, de **6.7 % a ±7 vágási széleken** → a fix-rátás, levágó LSQ-kvantálás
entrópiakódolt tárolásra nem optimális. Becslés: azonos MSE-n ~0.3–0.4 bit/súly, RD-optimális λ-val
~0.7–0.9 bit/súly is (≈ 0.7–0.9 MB/példány, ×2 S-ben), a veszteség-növekedés ellenében.
1. ECQ-szerű QAT: a kerekítés `argmin_k (w/s−k)² + λ·I_k` (I_k = −log2 P_k a tenzorcsalád
   futó hisztogramjából), rátatag logolva; implementáció overlay-alosztályként
   (`pysrc/quantization.py` `QuantizeFunction` érintetlen marad), `tft/config.py`
   `make_quantizations`, `tft/runner.py` (warm start a konverter kimenetéből, `TFT_WARMSTART`).
2. Finomhangolás a konvertált checkpointból: ~8 QAT + 2 fp32 epoch (becslés ~3.5 h/QAT-epoch
   a 5070 Ti-n → ~35 h/λ), KD-veszteséggel a jelenlegi modell kimenetére (drift ellen).
3. λ-söprés (2–3 érték), választás a ΔS = 2·Δkonténer + Δarchive (41 KB/millinat) szerint,
   C++ engine-nel szeleteken mérve.
4. Konténer v5: a tenzorcsaládonkénti fák a mérés szerint legfeljebb ~4 KB-ot érnek
   példányonként (`training/tf/4_export/container_stats.py`: 3,631,529 B vs 3,635,528 B), tehát a
   v5 lényege az RD-kvantáló új szinteloszlásának kódolása (családonkénti fa akkor számít, ha a
   λ családonként eltér) + a konténer maradékának (≈81–85 KB/példány: scale-ek, f32 tenzorok,
   nevek) olcsóbb kódolása — `training/tf/4_export/weights_compress_v3.py` (író) és
   `src/tf1/weights_io_compressed.cpp` (olvasó) szinkronban; bitazonos roundtrip-teszt.
5. Opcionális bővítés ±7-en túli szintekre: a dense kernel int16-akkumulációja (d_in=192: 6 chunk
   × 2×255×7 = 21,420) ±15-nél túlcsordul → szélesítés 4 chunkonként (`src/tf1/opt/qmat_dense.cpp`),
   sparse int4 oszlopok → int8 (`qmat_sparse.cpp`, `arena_build.cpp`); sebességmérés kötelező.
6. Új konténer bekötése: fájlnév/fingerprint `src/transformer.h` (`WeightsPath`,
   `WeightsFingerprint`), `src/runner.cpp`, `build_and_construct_comp.sh`; holt neuronok/collapsed
   sorok újraellenőrzése (`prune_dead_neurons.py`, `convert_container_v4.py`); BITLSTM újratanítás.

### 4. fázis (opcionális, mérsékelt GPU) — CM-komplementer finomhangolás
CM-only coder-variánssal bitenkénti stretch-predikciók dumpja (~8.8 GB); a transformer
finomhangolása bit-szintű keverék-veszteséggel (p_tf bitjei logisztikusan keverve a CM bitjeivel),
hogy a ma ~39 %-os „átengedési” arány nőjön. 5–10 QAT-epoch. Kutatási tét, kapu: szeleten mért ΔS.

### 5. fázis — A 10 %-hoz szükséges nagy GPU-track (bérelt GPU nélkül nem kivitelezhető)
Nulláról tanított modell a teljes lánccal (`training/tf/3_train`), benne:
RD/ECQ-QAT a kezdetektől + hosszabb QAT (ma csak 32 epoch, a kvantálási adó +2.6 % ≈ 0.8 MB);
méret/bitszám újraoptimalizálása 0.478 B/param helyett ~0.3 B/param mellett; KDA-állapot
átvitele cikkhatáron (tréningben a meglévő `max_merge_tokens` + kauzális reset-szabály a coderben);
súlymegosztott mélység (`attention/mlp_weight_sharing_pattern` már támogatott a tréningkódban);
az időkeret fedezése: byte-LSTM kiváltása (2. fázis), packed-int4 dense arénák a CM-mel közös
folyamatban (L3 ~22 B/c vs DRAM ~6 B/c; a kernelek megvannak: `qpacked_build`,
`qgemv_packed_f32/i32` a `src/tf1/opt/qmat_dense.cpp`-ben; egzakt egész skalárszorzat → az archívnak
bitazonosnak kell maradnia, tisztán sebesség-kísérlet, akár már a 0./2. fázisban), KDA/attn-hangolás.
Igény: ~2,000–3,000 RTX-5090-ekvivalens GPU-óra (pl. 4–8 kártya 3–6 hétig).

### 6. fázis — Integráció és beadás
Teljes compress + decompress c2d-highcpu-8-on és a Judging Assistantban; `cmp enwik9`;
idő ≤ 70,000/T ≥5 % tartalékkal mindkét programra; RAM-csúcs a folyamatfán < 10 GiB;
disk; determinizmus (két gép/két build bitazonos archívum, `rcpps/rsqrtps/vrcp*` szkennelés,
új FP-kód csak szigorú egységekben, nincs CPUID-elágazás); reprodukálható build (sha256);
README/CHANGES/NOTICE/ATTRIBUTION frissítése (GPL-3).

## 4. Nyereség-költségvetés (MB S; alacsony / közép / magas)
| kar | erőforrás | low | mid | high |
|---|---|---:|---:|---:|
| 3. súly RD (ECQ) + konténer v5 | ~100–150 GPU-h (5070 Ti) | −0.6 | −1.2 | −2.0 |
| 1.1–1.4 transformer-tudatos keverés/SSE/fxcm-betáp/mixer-opt. | CPU | −0.1 | −0.3 | −0.6 |
| 2. online adaptáció (LSH-kontextus, adapter, LSTM-csere) | CPU | 0.0 | −0.35 | −0.9 |
| 1.6 BITLSTM 100 %-on | CPU ~90 h | −0.02 | −0.05 | −0.12 |
| 1.7 metaadat (rang-kódolt page-id, revid) | CPU | −0.03 | −0.1 | −0.3 |
| 1.5 + apró CM-tételek, csomagolás, dict.comp | CPU | −0.02 | −0.08 | −0.2 |
| **részösszeg a mostani erőforrásokkal** | | **−0.8** | **−2.1** | **−4.1** |
| 4. CM-komplementer finomhangolás | ~50–100 GPU-h | 0.0 | −0.3 | −0.8 |
| 5. nagy GPU-track | ~2–3 k 5090-h | −2.0 | −4.0 | −6.5 |
| **mindennel** | | **−2.8 (2.9 %)** | **−6.4 (6.7 %)** | **−11.4 (11.9 %)** |

## 5. Erőforrásigény
- CPU: minden döntő mérés teljes futás (~49 h compress + ~49 h decompress, 10 GiB, ~21 GB disk);
  a terv kb. 1,500–3,000 core-óra (szeletek 2–6 h/kar). Párhuzamos futtatás 1 mag/futás.
- GPU: 0.4/3. fázis ~150–250 óra az 5070 Ti-n (az óraszám tisztázása után ütemezni);
  ~250 GB NVMe a priorokhoz; 5. fázis csak bérelt GPU-val.

## 6. Verifikáció
- Minden transzformnál bitazonos roundtrip (`src/preprocess_roundtrip.cpp` mintájára).
- Minden kar: szelet-A/B (azonos gép/bináris) → teljes futás → teljes dekompresszió + `cmp`.
- Konténer: író↔olvasó bitazonosság (Python ↔ `weights_io_compressed.cpp`), fingerprint-ellenőrzés.
- Transformer-változás: C++ engine vs PyTorch fp32 loss-paritás szeleten (upstream `test_e2e_opt`
  mintájára), utána BITLSTM-újratanítás.
- S = `wc -c comp9 archive9` + 19; idő/RAM a Judging Assistant mérése szerint.

## 7. Döntési kapuk és kockázatok
- Kapu A (0. fázis után): valós időprofil és régió-költségek → a 2./5. fázis prioritásai.
- Kapu B (1–3. fázis után): ha a mért összeg < −3 MB, a 10 % csak az 5. fázissal lehetséges →
  GPU-bérlés döntése; különben beadható köztes verzió (≥1 % a rekordhoz képest).
- Kockázatok: időkeret (minden új modell + idő), determinizmus (FP), a 12 GB VRAM a 2^17-es
  dokumentumokhoz, checkpoint hiánya (ECQ csak a kvantált pontból indulhat), BITLSTM-függés
  minden upstream-változástól.

## Függelék — a kódátnézés kulcsmegállapításai
- A transformer egyetlen bemenet (tf1_slot) a 24 layer-0 mixerben és a final mixerben; nincs
  rá szabott mixer-kontextus, SSE, és fxcm belső mixerei nem látják (csak a kis LSTM-et).
- Final mixer kontextusa csak a gate-zóna (`tf1_gate_ctx1`).
- Layer-0 mixerek: sima SGD fix ütemezéssel (1.0→0.7→0.3→0.1 @ 1M/5M/25M lépés).
- Konténer: egyetlen globális számláló fa minden int4 tenzorra; családonként 3.53–3.79 bit, de a
  családonkénti fák csak ~4 KB-ot érnének; 86 csupa-nulla sor; a nem-int4 maradék ≈81–85 KB.
- QAT csak 32 epoch (1000 bf16 mellett); upstream szerint +2.6 % veszteség a kvantálástól.
- fxcm elavult pozíció-küszöbök (448131719, 451531986).
- Part A ~22.7 bit/rekord (redirectek ~14.8, cikkek ~26.5); a sorrend klaszterenként csökkenő page id.
- A transformer cikkhatáron resetel (KV + KDA); a tréning `max_merge_tokens` opciója nincs használva.
- Upstream: L3 ~22 B/c, DRAM ~6 B/c; a CM-mel közös folyamatban a súlyfolyam kiszorulhat L3-ból →
  packed int4 (meglévő kernel) in-process mérése indokolt.

## 8. Állapot és mért számok (2026-10-18)

### Kész
| lépés | eredmény | hol |
|---|---|---|
| 0.1 reprodukció | `build.sh` → comp9 sha256 `8b07d85f…`, 4,000,689 B (4 perc, 4 vCPU) | — |
| 0.2 fejlesztői build | `make cmix_dev` / `cmix_devtime`: prefix-leállítás, bitenkénti valós kódköltség a coder állapotából, stream-másolat, transformer-dump (slot-sor, hidden, PPMd), ciklusszámlálók; a szállított comp9 a hookokkal is bájtazonos (újrafordítással ellenőrizve) | `src/neif_dev.h`, `dev/` |
| 0.3 A/B harness | karok: forrásfa, `fa=-D…` kísérleti define-ok, `…@konténer` másik súlykonténer (ujjlenyomat-bélyegzéssel) | `dev/ab_prefix.sh`, `dev/build_dev.sh`, `dev/cost_report.py`, `dev/tfdump.py`, `dev/fetch_enwik9.sh` |
| 0.5 konténer → checkpoint | bájtazonos roundtrip a teljes export-láncon; CPU-n (torch 2.14.1) `strict=True` betöltés a pkd-ág qat/fp32 modelljébe, és az eredeti torch-export ugyanazokat a tenzorokat írja (a CPU-n számolt rope-táblák kivételével) | `training/tf/4_export/container_to_tch.py` |
| 1.1 kar | `NEIF_ARM_TFCTX=1/2/3` (transformer top-bájt + konfidencia mint final-/layer-0-mixer kontextus), alapból ki, mérés nélkül | `src/tf_sharpen.h` |
| 3.1 ECQ-QAT | entrópia-korlátos kerekítés, a skála-gradiensben az S árfolyamán árazott ráta-tag (μ = 6.1e-9 nat/token/bit), „bake” a szintközepekre; `TFT_SCALE_STEPS=none`; CPU self-test 5/5; recept a `training/tf/3_train/README.md`-ben | `training/tf/tree_overlay/tft/ecq.py` |

### Mért számok (enwik9 nélkül is mérhetők)
- **int4 szintek kontextusos kódolása** (oszlop-/sorátlag, előző szint): legfeljebb ~8.8 KB
  konténerenként (statikus felső becslés) → alacsony prioritás.
- **ECQ szimuláció** a szállított súlyokon (latens = szintközép + egyenletes zaj a binben):
  λ = 0.1: −5 KB, λ = 0.3: −29 KB (+6.8 % torzítás), λ = 0.5: −206 KB (+66 %) konténerenként;
  a lépésköz növelése: ×1.1: −88 KB (+19 %), ×1.25: −266 KB (+57 %), ×1.5: −512 KB (+126 %).
  A szomszéd-választás önmagában kevés; a ráta fő karja a lépésköz, ezért kapott a skála
  gradiense ráta-tagot.
- **Durva veszteségmodell** (veszteség ∝ torzítás; a 4 bites kvantálás adója ≈ 2.6 % ≈ 0.02
  nat/token): az egyenletesen durvább lépésköz nagyjából nullszaldós (×1.25: −531 KB S a két
  konténeren, +467 KB archívum). Nyereség csak abból jöhet, hogy a QAT a ráta-taggal ott
  durvít, ahol a veszteség nem érzékeny → **a 3. fázis becslése lefelé módosul: 0 / −0.3 /
  −1.0 MB** (korábban −0.6 / −1.2 / −2.0).
- **Módosított részösszeg a mostani erőforrásokkal: −0.2 / −1.2 / −3.1 MB (−0.2 / −1.2 /
  −3.3 %)**; mindennel (4. és 5. fázis is): −2.2 / −5.5 / −10.4 MB (−2.3 / −5.7 / −10.8 %).
  A 10 % így csak a nagy GPU-track felső becslésével jön ki.

### Blokkolók
- **enwik9 nem érhető el ebből a környezetből** (a hálózati szabály tiltja a `mattmahoney.net`-et);
  enélkül egyetlen kódolási mérés sem futtatható (a coder csak a saját stream-formáján fut).
  Megoldás: a környezet hálózati beállításaiban az *Allowed domains* közé `mattmahoney.net` és
  `www.mattmahoney.net`, utána `bash dev/fetch_enwik9.sh <dir>`.
- A GPU-oldalhoz (λ/μ-söprés) a GPU-gépen kell a D-csomag (stream-dump + priorok a
  felhasználó prior-generátorával, ~228 GB).

### Következő lépések
1. enwik9 után: 40 MB-os prefix A/B (referencia vs. `NEIF_ARM_TFCTX=1/2/3`), időprofil
   `cmix_devtime`-mal, transformer-dump egy 4 MB-os szeletről → 2. fázis szimulációi.
2. GPU: konténer → checkpoint → QAT+ECQ (μ = 6.1e-9; λ ∈ {0, 0.1, 0.3}), 8+2 epoch karonként,
   aztán konténer + prefix A/B `…@új.c4`-gyel.
3. Az első mérések után az 1.2–1.5-ös karok.
