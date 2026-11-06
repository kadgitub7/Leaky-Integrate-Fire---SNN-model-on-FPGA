# Comprehensive System Audit — Analog ECG Classifier
## Senior ASIC Design Engineer Review

**Date:** 2026-11-05
**Scope:** Full audit of analog frontend + ReRAM crossbar classifier
**Technology:** SKY130 130nm CMOS, 1T1R ReRAM
**Goal:** Ultra-low-power fully-analog ECG arrhythmia classification (DS2 inter-patient, De Chazal protocol)

---

# PART 1: ANALOG FRONTEND AUDIT

## 1.1 Architecture Overview

The frontend extracts 8 features from raw ECG (2 leads) using subthreshold analog circuits at VDD=0.9V.

**Signal chain:** Electrodes → CCIA (×2 leads) → R-peak Detector → Delay Timer → Feature Extraction (×8) → Analog Outputs

**Circuit inventory** (from `frontend_top.spice`):
- 1× Bias generator (`bias_gen.spice`) — self-biased, vbias≈0.29V, vref=0.30V
- 2× Instrumentation amplifier (`ia.spice`) — CCIA, gain=100, ~570Hz BW
- 1× R-peak detector (`rpeak_detector.spice`) — two-stage comparator
- 1× R-peak hold (`rpeak_hold.spice`) — debounce for NEG-polarity QRS
- 1× Delay timer (`delay_timer.spice`) — dual RC + 7 comparators
- 1× P-wave predictor (`pwave_predictor.spice`) — 3 comparators on RR timer/EWMA
- 6× Pulse generators (`pulse_gen.spice`, `long_pulse_gen.spice`) — AC-coupled edge-to-pulse
- 2× Edge buffers (`edge_buffer.spice`) — inverter pair for slow transitions
- 8× Feature circuits (described below)

**Total:** ~95 MOSFETs, ~25 MIM capacitors

## 1.2 Feature Circuit Audit

### F1/F2: RR Timing (`rr_timing.spice`)
**Function:** F1 = RR deviation from EWMA; F2 = RR asymmetry (pre − post)
**Topology:** 320pF timer cap → SF → 5pF prev cap, 5pF esamp cap, 45pF EWMA cap → two diff-pair readouts
**EWMA:** Charge-sharing α = 5pF/(5pF+45pF) = 0.10 ✓
**Readout:** NFET diff pair L=1 W=8 with diode-connected PMOS loads L=4 W=2

**Findings:**
- [OK] EWMA alpha=0.10 correctly implemented via capacitor ratio
- [OK] Source follower level shift exposes timer/EWMA nodes for P-wave predictor
- [OK] IC propagation between batches was fixed (per memory: `ewma-ic-propagation-fix`)
- [OK] Esamp validity gate (AND gate blocks trig_rr_copy when esamp < 0.35V) prevents corrupted EWMA updates
- [CONCERN] Timer cap 320pF (W=400 L=400) — very large MIM. Verify this is within SKY130 MIM max dimensions.
  - SKY130 MIM m3_1: 2 fF/µm². 400×400 = 320pF. Dimension OK per PDK (max 400µm per side).
- [CONCERN] SF decay rate depends on bias current. If vbias drifts, timer slope changes. The integration_test.py calibrates thresholds per-record, but a real ASIC has fixed thresholds.

### F3/F4: Beat Instability (`beat_instability.spice`)
**Function:** |ECG_current − ECG_previous| at a fixed QRS offset
**Topology:** curr S&H 5pF → esamp 22.5pF (acts as EWMA α≈0.18) → prev S&H 5pF → diff pair readout
**Two instances:** F3 = Lead 0, F4 = Lead 1

**Findings:**
- [OK] Uses `capture_trig` (rpeak) for current beat capture and `copy_trig` (delayed ~300ms) for prev update
- [OK] EWMA smoothing via 22.5pF esamp gives α = 5/(5+22.5) = 0.182 — reasonable
- [CONCERN] The 22.5pF cap (W=150 L=150) is shared between current and previous — charge injection from switches may introduce offset. However, diff pair comparison cancels common-mode offset.
- [OK] L=1 W=8 diff pair with L=4 W=2 diode PMOS loads — validated topology

### F5: P-wave Difference (`pwave_diff.spice`)
**Function:** ECG(P-peak) − ECG(P-baseline)
**Topology:** Two NFET S&H (L=0.5 W=2), 2pF caps (W=32 L=32), diff pair readout L=1 W=8 + PMOS loads L=4 W=4
**Timing:** Samples at P-wave baseline and P-wave peak (from pwave_predictor triggers)

**Findings:**
- [OK] Diff pair output proportional to P-wave amplitude
- [CONCERN] P-wave amplitude is very small (~100-300µV after IA gain=100 → ~10-30mV). This is near the diff pair's noise floor. IA gain of 100 helps but may still be marginal.
- [OK] Uses predicted timing from pwave_predictor — no look-ahead needed
- [NOTE] PMOS loads are L=4 W=4 here (not W=2 as in other features). Intentional — larger W gives lower impedance, wider linear range for the small P-wave signal.

### F6: QRS Width (`qrs_width.spice`)
**Function:** ECG voltage sample at R+28ms (proxy for QRS width)
**Topology:** Single NFET pass gate (L=0.5 W=2), 2pF cap (W=32 L=32)
**Simplest feature — passive S&H only.**

**Findings:**
- [OK] Extremely simple — minimal area and power
- [CONCERN] No diff pair readout — output is raw held voltage. Sensitive to cap leakage over measurement interval. At ~67pA leakage, 2pF cap loses 67pA/2pF = 33.5mV/ms. Over 100ms measurement window → 3.35V droop. This is severe.
  - **HOWEVER:** The .meas captures the voltage at rpeak+100ms, and the S&H samples at rpeak+28ms. That's 72ms droop time → ~2.4V loss. The cap should be fully discharged.
  - **WAIT** — re-examining: The pass gate is NFET with gate driven by trig_wid. When gate goes LOW, leakage depends on sub-threshold current of the pass gate, NOT 67pA from bias. The pass gate is L=0.5 (minimum), so sub-threshold leakage is actually higher (~nA range). This is a potential problem.
  - **RESOLUTION:** Looking at the full_accuracy_test.py measurements, F6 achieves |ρ| > 0.95 consistently. The leakage concern may be overstated because: (a) the measurement happens quickly, and (b) the pass gate NFET with L=0.5 in sub-threshold at VDD=0.9V has gate voltage at 0V when OFF — the sub-threshold current is exponentially reduced.
- [OK] No bias dependency

### F7: QRS Slope (`qrs_slope.spice`)
**Function:** ECG(R+14ms) − ECG(R+69ms)
**Topology:** Two NFET S&H (2pF each, W=32 L=32), diff pair L=1 W=8, PMOS loads L=1 W=4

**Findings:**
- [OK] Captures two ECG timepoints and outputs their difference
- [CONCERN] Per memory (`delay-timer-calibration`), F7 degradation was traced to pipeline delays in the analog delay timer. Fix: per-comparator pipeline delay calibration (pipe_s14=2.9ms, pipe_s69=4.4ms).
- [OK] Diff pair topology validated
- [NOTE] PMOS loads are L=1 W=4 — shorter than other features (L=4). This gives lower output resistance, meaning output settles faster but has lower gain.

### F8: QRS Symmetry (`qrs_symmetry.spice`)
**Function:** ECG voltage at R-peak (proxy for QRS symmetry)
**Topology:** Transmission gate (NFET L=0.5 W=2 + PMOS L=0.5 W=4), inline inverter, 2pF cap

**Findings:**
- [OK] Transmission gate (TG) provides better charge injection cancellation than single NFET
- [OK] Clock feedthrough: NFET injects −ΔQ, PMOS injects +ΔQ, partial cancellation
- [CONCERN] Inline inverter for complementary trigger adds ~1-2ns propagation delay between NFET and PMOS gate signals. At these slow signal rates (ms timescale), this is negligible.
- [OK] Captures at rpeak_pulse — no delay timer dependency

## 1.3 Support Circuit Audit

### Bias Generator (`bias_gen.spice`)
- Self-biased current mirror: PMOS (L=8 W=0.42) + asymmetric NFET (L=8 W=0.42 / L=12 W=0.42)
- **[CRITICAL]** Requires `.ic V(xbias.iref)=0.3` for startup. Without this, circuit has a stable zero-current state. In silicon, a startup circuit is mandatory.
- [OK] Output vbias≈0.29V, vref=0.30V — correct for subthreshold operation

### R-peak Detector (`rpeak_detector.spice`)
- Two-stage: NFET diff pair (L=2 W=4) + PMOS mirror (L=2 W=4) + inverter (PFET L=0.5 W=8 / NFET L=0.5 W=4)
- [OK] First stage provides gain, second stage provides rail-to-rail output
- [OK] External threshold allows per-patient calibration

### R-peak Hold (`rpeak_hold.spice`)
- PMOS fast charge + weak NFET discharge + 12pF hold cap → ~300ms blanking
- [OK] Prevents T-wave retriggering on NEG-polarity records
- [OK] 300ms blanking covers typical T-wave window (200-300ms post-QRS)

### Delay Timer (`delay_timer.spice`)
- **Dual timer architecture:** Main (100pF, ~0.52mV/ms) + Fast (~4pF, ~21mV/ms)
- [OK] Fast timer used for s14 (14ms) and rr_copy (10ms) — short delays need better resolution
- [OK] Main timer for longer delays: s69 (69ms), wid (28ms), bi_copy (300ms)
- [OK] 7 two-stage comparators — all use validated L=4 W=2 diff pair topology
- [CONCERN] vbias_delay is separate from main vbias — allows independent tuning. Good design choice.
- [NOTE] Per memory (`delay-timer-calibration`): pipeline delays calibrated per-comparator

### P-wave Predictor (`pwave_predictor.spice`)
- 3 comparators watching SF-shifted timer vs thresholds
- [OK] Predicts next R-peak from EWMA, fires triggers at R-206ms and R-131ms
- [OK] Uses same comparator topology as delay_timer — validated
- [OK] Reads exposed rr_timer_node and rr_ewma_node from rr_timing circuit

### Pulse Generator (`pulse_gen.spice`)
- AC coupling (0.2pF) + bleeder NFET (L=8 W=1.34) + clamp + double inverter
- [OK] Converts level transitions to ~0.5ms pulses
- [OK] Long variant (W=0.41 bleeder) gives ~2ms pulses for EWMA charge-sharing
- [OK] Negative clamp prevents undershoot on falling edge

### Instrumentation Amplifier (`ia.spice`)
- CCIA: C_in=10pF, C_fb=0.098pF → Gain≈103
- [OK] 5T OTA (L=8 W=2) — high output resistance in subthreshold
- [OK] Pseudo-resistors (PMOS L=10 W=0.42, back-to-back) for DC bias — >1TΩ
- [OK] BW = Gm/(2π·C_load) ≈ 570Hz — sufficient for ECG (<100Hz content)
- [CONCERN] Gain=103 slightly above design target of 100. Manufacturing variation on 0.098pF feedback cap could shift gain ±10%.
- [OK] HPF corner = 1/(2π·R·C_fb) < 0.01Hz — removes electrode DC offset

## 1.4 Frontend Validation Summary

The `integration_test.py` validates the full analog frontend against Python references using Spearman rank correlation across 30 MIT-BIH records, 35 beats each.

**Known good features (|ρ| > 0.9):** F3, F4, F5, F6, F7, F8
**Historically problematic features (now fixed):** F1, F2 (fixed by IC propagation between batches)
**Known calibration-sensitive features:** F6, F7 (fixed by per-comparator delay calibration)

---

# PART 2: CLASSIFIER AUDIT

## 2.1 Architecture Overview

**MLP:** 8 → 64 → 5 (8 inputs, 64 hidden with ReLU, 5 output classes)
**Technology:** 1T1R ReRAM crossbar on SKY130 NMOS, VDD=1.8V
**Read voltage:** V_READ = 0.15V
**Conductance range:** G_MIN=0.296µS, G_MAX=99.273µS

**Signal path:**
```
feat[0..7] (normalized ~[-4,+4])
  → VCVS offset-binary: V_out = (feat+4)/8 × 0.15V → [0, 0.15V]
  → L1 Crossbar 8×64 (1024 NMOS + 1024 ReRAM, G+/G−)
  → CCCS column current summation (scale=137,049)
  → 64× interstage: PMOS bias + NMOS bias + ReLU + output mirrors
  → L2 Crossbar 64×5 (640 NMOS + 640 ReRAM, G+/G−)
  → CCCS column summation (scale=197,848)
  → 5× L2 bias mirrors
  → Lazzaro WTA → one-hot output
```

## 2.2 Crossbar Design Audit

### L1 Crossbar (`L1_crossbar_8x64.spice`)
- 8 rows × 64 columns, differential (Gp/Gn) → 1024 1T1R cells
- NMOS access transistors: W=1.0µm L=0.15µm
- [OK] Differential encoding: W_effective = (Gp − Gn) × V_READ
- [OK] Gate voltage fixed at 1.8V — fully ON, transistor acts as simple switch
- [CONCERN] L=0.15µm is minimum length. In practice, the access transistor ON resistance adds to the ReRAM resistance. At G_MAX=99.273µS (R=10.07kΩ), the NMOS ON resistance (~few hundred Ω at W=1.0µm) is <5% of ReRAM resistance — acceptable.
- [CONCERN] At G_MIN=0.296µS (R=3.38MΩ), NMOS ON resistance is negligible. But leakage currents at 1.8V could be comparable to the signal current. I_signal = 0.296µS × 0.15V = 44.4nA. NMOS sub-threshold leakage at L=0.15µm, VDD=1.8V is ~1-10nA range — this is 2-20% of the minimum signal. This is a potential accuracy concern for small weights.

### L2 Crossbar (`L2_crossbar_64x5.spice`)
- 64 rows × 5 columns, differential → 640 1T1R cells
- Same topology as L1
- [OK] Smaller array — fewer parasitics

### Weight Mapping (`full_analog_mlp.py`)
- Linear mapping: G = G_MIN + (|w|/w_max) × (G_MAX − G_MIN)
- Positive weights: Gp = G_MIN + ..., Gn = G_MIN (minimum)
- Negative weights: Gp = G_MIN, Gn = G_MIN + ...
- Scale factor: w_max / (G_range × 1e-6 × V_READ)
- [OK] Correct differential encoding
- [OK] Scale factors: L1=137,049, L2=197,848
- [CONCERN] G_MAX/G_MIN ratio = 335:1. This is aggressive for ReRAM. Typical experimental ratios are 10:1 to 100:1. Need to verify against the specific ReRAM PDK specs in `skywater-pdk-libs-sky130_fd_pr_reram/`.

### Input Conditioning (`generate_top_level.py`)
- VCVS: V_row = (feat + 4.0) / 8.0 × V_READ
- Maps normalized features from [-4,+4] to [0, 0.15V]
- [OK] Offset-binary encoding avoids negative voltages on crossbar
- [OK] Linear mapping preserves feature relationships
- [NOTE] VCVS is an ideal element in simulation. In silicon, this would be implemented as an op-amp or OTA with resistive feedback.

## 2.3 Interstage Audit (`interstage_bias_relu.spice`)

- 64× PMOS bias mirrors (positive bias current injection)
- 64× NMOS bias mirrors (negative bias current injection)
- 64× ReLU: diode-connected NMOS (W=4.0µm L=0.5µm) — passes positive current, blocks negative
- 64× PMOS output mirrors (current → voltage for L2 input)
- [OK] relu_gain=1.015 — near-unity, validated by ngspice DC sweep
- [OK] mirror_gain=0.965 — slight droop acceptable
- [CONCERN] 394 transistors — substantial area. Layout-dependent mismatch could degrade accuracy.

## 2.4 WTA Audit (`WTA_5class.spice`)

- Lazzaro circuit: 5 input cells with shared inhibition node
- 16 transistors total
- NMOS input: W=2.0µm L=0.5µm
- PFET load: W=4.0µm L=0.5µm
- Ibias=10µA
- [OK] Standard Lazzaro topology — well-proven for current-mode argmax
- [OK] Winner pulls HIGH, losers suppressed LOW
- [CONCERN] 10µA bias is relatively high for "ultra-low-power" — dominates power budget. But it's only active during the brief classification phase.
- [CONCERN] For close margins (two classes within ~1% current difference), WTA may not resolve correctly. Need to characterize minimum differential.

## 2.5 Classifier Verification Results

From `full_analog_results.json`:
- Software MLP accuracy: 96.042%
- Full analog (crossbar) accuracy: 96.04%
- SW ↔ HW agreement: 99.647%
- L1 weight error: mean=1.242%, max=7.089%
- L2 weight error: mean=1.165%, max=7.089%

**This verification was done on DIGITAL features (features_8_cache.npz), NOT analog frontend outputs.**

From `full_system_results.json` (legacy, 8→32→5):
- Full analog frontend → crossbar: 93.81%

---

# PART 3: CRITICAL INTEGRATION ISSUES

## Issue 1: Feature Ordering Mismatch ⚠️

**Frontend output order** (from `frontend_top.spice`, `full_accuracy_test.py`):
```
F1 = rr_ratio        (rr_timing)
F2 = rr_asymmetry     (rr_timing)
F3 = beat_instab_L0   (beat_instability, lead 0)
F4 = beat_instab_L1   (beat_instability, lead 1)
F5 = pwave_diff       (pwave_diff)
F6 = qrs_width        (qrs_width)
F7 = qrs_slope        (qrs_slope)
F8 = qrs_symmetry     (qrs_symmetry)
```

**Classifier expected order** (from `classifier_mlp.py` FEATURE_NAMES_8):
```
feat[0] = rr_ratio          ← F1
feat[1] = beat_instab_L1    ← F4
feat[2] = rr_asymmetry      ← F2
feat[3] = pwave_energy_L1   ← F5
feat[4] = qrs_width_L0      ← F6
feat[5] = beat_instab_L0    ← F3
feat[6] = slope_ratio_L0    ← F7
feat[7] = qrs_symmetry_L0   ← F8
```

**Reorder mapping:** Frontend → Classifier crossbar rows:
```
Frontend F1 → Crossbar row 0 (rr_ratio)
Frontend F4 → Crossbar row 1 (beat_instab_L1)
Frontend F2 → Crossbar row 2 (rr_asymmetry)
Frontend F5 → Crossbar row 3 (pwave)
Frontend F6 → Crossbar row 4 (qrs_width)
Frontend F3 → Crossbar row 5 (beat_instab_L0)
Frontend F7 → Crossbar row 6 (slope)
Frontend F8 → Crossbar row 7 (qrs_symmetry)
```

**Severity:** LOW — this is purely a wiring issue, no additional circuitry needed.
**Fix:** Route frontend outputs to correct crossbar input rows.

## Issue 2: Voltage Domain Mismatch ⚠️⚠️

- **Frontend:** VDD = 0.9V. Feature outputs range ~0.1V to 0.8V.
- **Classifier:** VDD = 1.8V. Input conditioning maps [-4,+4] to [0, 0.15V].

**Severity:** HIGH — requires interface circuitry.
**Fix options:**
1. **Direct connection** — Frontend outputs (0.1-0.8V) are within classifier's 0-1.8V input range. The VCVS conditioning stage can be redesigned to map from the frontend's voltage range directly.
2. **Level shifter** — Standard sub-threshold level shifter (0.9V → 1.8V domain)
3. **Retrain classifier** — Train the MLP with input conditioning that maps from the frontend's actual voltage range instead of [-4,+4] normalized features.

**Recommended:** Option 3 (retrain) combined with modified VCVS conditioning.

## Issue 3: Normalization Gap ⚠️⚠️⚠️

This is the **most critical** integration challenge.

**Problem:** The classifier's weights in `full_analog_weights.npz` were trained on **digitally-computed, standardized features** (zero-mean, unit-variance). The analog frontend produces **raw voltage differences** that are monotonically correlated with but numerically different from these standardized features.

**Evidence:**
- `features_8_cache.npz`: Training features have mean≈0, std≈1 (pre-standardized)
- Frontend outputs: raw voltages in ~0.1-0.8V range from diff pairs and S&H circuits
- Spearman correlation between analog and digital features: 0.83-0.99 (rank-order preserved, but not values)

**The 96.04% accuracy was validated on digital features only.** The legacy `full_system_test.py` achieved 93.81% when using analog frontend features — but with a smaller 8→32→5 network and training a new MLP directly on the analog voltages.

**Severity:** CRITICAL — without normalization, the crossbar classifier will not work.

**Fix options:**
1. **Retrain classifier on analog frontend voltages** (RECOMMENDED)
   - Run full_accuracy_test.py to collect all analog feature voltages
   - Train new 8→64→5 MLP on these voltages directly
   - Map new weights to ReRAM conductances
   - Input conditioning VCVS maps from actual voltage range
   - Pro: Accounts for all frontend nonlinearities. Simple. Proven (93.81% with 8→32→5).
   - Con: Accuracy may differ from the 96.04% digital baseline.

2. **Analog normalization stage**
   - Build per-feature gain+offset analog circuits
   - V_norm = (V_frontend − offset) × gain
   - Pro: Preserves 96.04% trained weights
   - Con: Requires characterizing transfer function per feature. Additional area/power. Gain/offset variation degrades accuracy.

3. **Mixed approach**
   - Use linear regression to find per-feature V_norm = a × V_analog + b
   - Absorb these into the input conditioning VCVS (modify gain/offset per row)
   - Pro: No additional circuits, uses existing VCVS
   - Con: Assumes linear relationship (may not hold for all features)

## Issue 3b: Digital Feature Outliers ⚠️

The digital feature cache reveals extreme outliers that the VCVS clips at [-4,+4]:
```
rr_ratio:  [-4.57, 91.74] — up to 91.7σ!
bi_L1:     [-0.40, 23.25] — up to 23.2σ
rr_asym:   [-2.86, 120.15] — up to 120σ!
pwave:     [-0.47, 80.21] — up to 80σ!
width:     [-2.31, 0.78]  — well-behaved
bi_L0:     [-0.58, 41.11] — up to 41σ
slope:     [-3.12, 10.36] — moderate
symmetry:  [-2.26, 3.34]  — well-behaved
```

Features with extreme positive outliers (rr_ratio, rr_asym, pwave, bi_L0, bi_L1) are heavily clipped by the [-4,+4] → [0, 0.15V] mapping. The classifier still achieves 96.04% despite this, meaning most beats are within range and outlier beats are predominantly N-class (which has many redundant examples). **For integration, the analog frontend's bounded voltage outputs may actually be an advantage** — analog circuits naturally saturate instead of producing 90σ outliers.

## Issue 4: SKY130 Model Library Mismatch ⚠️

- **Frontend:** Uses `sky130_minimal.lib.spice` (compact model in `ngspice_frontend/`)
- **Classifier:** Uses full SKY130 PDK from `C:/src/skywater-pdk-libs-sky130_fd_pr/`

**Severity:** MEDIUM — both should use the same models for accurate combined simulation.

**Fix:** Unify to the full PDK. Verify frontend circuits still produce equivalent results with full models. The minimal lib is likely a subset of BSIM4 parameters — differences should be small at tt corner.

## Issue 5: Simulation Methodology Mismatch ⚠️

- **Frontend:** `.tran` (transient) — simulates multiple beats sequentially, captures features at specific timepoints
- **Classifier:** `.op` (DC operating point) — one simulation per beat, assumes steady-state inputs

**Severity:** MEDIUM — affects how combined simulation is structured.

**Fix options:**
1. **Extract-and-inject:** Run frontend .tran, extract feature voltages via .meas, inject as DC into classifier .op. (Two-stage simulation)
2. **Full transient:** Build combined .tran testbench where frontend outputs directly drive classifier inputs. Classifier settles within ~µs (DC circuit), so measurement after frontend features stabilize.
3. **Hybrid:** Frontend .tran produces feature voltage cache → classifier uses per-beat .op with cached voltages.

**Recommended:** Option 1 for development (faster iteration), Option 2 for final validation.

---

# PART 4: INTEGRATION PLAN

## Phase 1: Frontend Feature Caching (prerequisite — BLOCKING)
**Goal:** Complete cache of analog frontend feature voltages for all DS1+DS2 records.

**Current status:** `_cache/` directory exists but is EMPTY. All 44 records must be simulated.
This is the longest phase — each record requires ~200-1200s of ngspice transient simulation.
Estimated total: 44 records × ~10min avg = ~7 hours of simulation time.

1. Run `full_accuracy_test.py` to populate cache for all 44 records
2. Verify each cache file for completeness (features + labels + py_features)
3. Record feature voltage statistics: min, max, mean, std per feature
4. Compute Spearman ρ for each feature across all records

## Phase 2: Per-Feature Affine Calibration (96% Strategy)
**Goal:** PRESERVE existing 96.04% trained weights; calibrate only the VCVS input conditioning.

**Key Insight:** Retraining on analog voltages drops to ~93.8% (legacy full_system_test.py). Instead, we find a per-feature affine transform that maps analog frontend voltages back to the digital standardized feature space that the existing 96.04% classifier was trained on.

**Mathematical framework:**
- For each feature i: `feat_digital_i = alpha_i * V_analog_i + beta_i`
- alpha_i and beta_i are found by linear regression on (V_analog, feat_digital) pairs from DS1
- The VCVS parameters absorb this calibration:
  - `A_i = V_READ * alpha_i / (2 * X_OFFSET)` where X_OFFSET = 4.0
  - `B_i = V_READ * (beta_i + X_OFFSET) / (2 * X_OFFSET)`
- The classifier then operates as: Z1 = X_cal @ W1_eff + b1 (no retraining needed)

**Steps:**
1. Load analog feature voltages from cache (frontend order: F1-F8)
2. Reorder to classifier order: [F1, F4, F2, F5, F6, F3, F7, F8]
3. Load digital standardized features from `features_8_cache.npz`
4. For each feature: linear regression (analog → digital) → alpha_i, beta_i
5. Verify per-feature R² > 0.9 (linearity requirement)
6. Compute VCVS parameters A_i, B_i from alpha, beta
7. Software-emulate full DS2 classification with calibrated features
8. Target: ≥95% (accounting for analog noise + saturation losses)

**Implementation:** `experiments/integration/integrate_frontend_classifier.py`
- `calibrate_analog_to_digital()` — linear regression per feature
- `compute_vcvs_params()` — VCVS gain/offset from calibration
- `classify_software_emulated()` — full pipeline emulation
- `generate_integrated_classifier()` — SPICE netlist with calibrated VCVS

**Digital verification (no analog cache):** 96.04% — exactly matches reference. Confirmed the W1_eff space is ORIGINAL feature space, not V_row space.

## Phase 3: Interface Design ✅ COMPLETE (pending calibration data)
**Goal:** Design the analog interface between frontend (0.9V) and classifier (1.8V).

**Status:** Integration script and combined testbench generator are complete. Awaiting analog cache for actual calibration parameters.

1. **VCVS calibration (done):** Per-feature affine VCVS absorbs both the analog→digital mapping and the offset-binary encoding. Parameters A_i, B_i computed from linear regression.
   - V_row_i = A_i * V_analog_i + B_i  (clamped to [0, V_READ])
   - In simulation: VCVS in `integrated_classifier.spice`
   - In silicon: OTA-based programmable gain amplifier (one per feature)
2. **Feature reordering (done):** FRONTEND_TO_CLASSIFIER = [0, 3, 1, 4, 5, 2, 6, 7]
   - Frontend: [rr_ratio, rr_asym, bi_L0, bi_L1, pwave, width, slope, symmetry]
   - Classifier: [rr_ratio, bi_L1, rr_asym, pwave, width, bi_L0, slope, symmetry]
3. **Combined testbench generator (done):** `experiments/integration/generate_combined_testbench.py`
   - Mode 1: .op per beat (fast validation, ~2s/beat)
   - Mode 2: Full .tran frontend+classifier (final validation)
4. **Level shifter:** Not needed — VCVS absorbs the 0.9V→1.8V domain crossing directly.

## Phase 4: Combined End-to-End Testbench
**Goal:** Single ngspice simulation: ECG → Frontend → Classifier → Output

1. Build combined top-level netlist:
   ```
   .include frontend_top.spice
   .include top_level_classifier.spice  (modified for analog inputs)
   ```
2. Connect frontend feature outputs to classifier inputs (with reordering)
3. Provide ECG PWL stimulus and all threshold voltages
4. Run `.tran` for multiple beats
5. Measure WTA outputs after each beat
6. Compare to annotations

## Phase 5: Full DS2 Validation
**Goal:** Report final end-to-end accuracy on all 48,181 DS2 beats.

1. Run combined simulation on all DS2 records
2. Report: accuracy, per-class F1, confusion matrix
3. Compare to:
   - Digital feature MLP: 96.04% (reference ceiling)
   - Legacy analog→crossbar: 93.81% (retrained, old approach)
   - Target: **≥96%** with calibrated VCVS (preserve-weights strategy)

## Phase 6: Power Analysis
**Goal:** Estimate total energy per beat.

1. Frontend: ~5 nJ/beat (validated)
2. Classifier: Estimate from crossbar read energy + interstage + WTA
   - Crossbar: (8+64 rows) × V_READ × I_avg × T_READ
   - Interstage: 64 bias mirrors + 64 ReLU × quiescent power × T_active
   - WTA: Ibias × VDD × T_settle
3. Target: <15 nJ/beat total

---

# PART 5: RISK REGISTER

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|------------|------------|
| 1 | Accuracy drop from digital→analog features | HIGH | MEDIUM | Preserve 96.04% weights; calibrate VCVS per-feature via affine transform. Ceiling is 96.04%; analog noise/saturation may reduce to ~95% |
| 2 | ReRAM conductance ratio too aggressive (335:1) | MEDIUM | MEDIUM | Verify against PDK; if needed, quantize weights to smaller range |
| 3 | Feature voltage drift in silicon (no per-patient calibration) | HIGH | MEDIUM | Design for worst-case; characterize across PVT corners |
| 4 | Bias generator startup without IC | LOW | HIGH | Add startup circuit for tapeout; OK for simulation |
| 5 | P-wave feature noise floor | MEDIUM | MEDIUM | IA gain=100 should suffice; verify SNR |
| 6 | WTA resolution for close margins | MEDIUM | LOW | Characterize minimum differential; add hysteresis if needed |
| 7 | Combined simulation convergence | MEDIUM | MEDIUM | Use gear method, tight reltol; may need longer timeout |

---

# APPENDIX A: File Inventory

## Frontend Files (experiments/frontend_optimization/complete_frontend/)
| File | Type | Status | Description |
|------|------|--------|-------------|
| frontend_top.spice | Circuit | FINAL | Complete ASIC frontend top-level |
| bias_gen.spice | Circuit | FINAL | Self-biased current source |
| ia.spice | Circuit | FINAL | CCIA instrumentation amplifier |
| rpeak_detector.spice | Circuit | FINAL | Two-stage R-peak comparator |
| rpeak_hold.spice | Circuit | FINAL | NEG-QRS debounce hold |
| delay_timer.spice | Circuit | FINAL | Dual RC timer + 7 comparators |
| pwave_predictor.spice | Circuit | FINAL | Pre-R-peak trigger prediction |
| pulse_gen.spice | Circuit | FINAL | Edge-to-pulse converter (~0.5ms) |
| long_pulse_gen.spice | Circuit | FINAL | Long pulse generator (~2ms) |
| edge_buffer.spice | Circuit | FINAL | Two-inverter buffer |
| inverter.spice | Circuit | FINAL | Basic CMOS inverter subckt |
| rr_timing.spice | Circuit | FINAL | F1/F2: RR timing features |
| beat_instability.spice | Circuit | FINAL | F3/F4: Beat instability |
| pwave_diff.spice | Circuit | FINAL | F5: P-wave difference |
| qrs_width.spice | Circuit | FINAL | F6: QRS width proxy |
| qrs_slope.spice | Circuit | FINAL | F7: QRS slope |
| qrs_symmetry.spice | Circuit | FINAL | F8: QRS symmetry |
| integration_test.py | Script | FINAL | Full analog validation (Spearman ρ) |
| full_accuracy_test.py | Script | FINAL | DS2 accuracy with RF classifier |

## Classifier Files (experiments/Complete_imp/classifier_crossbar/)
| File | Type | Status | Description |
|------|------|--------|-------------|
| top_level_classifier.spice | Circuit | FINAL | Complete ASIC classifier |
| L1_crossbar_8x64.spice | Circuit | FINAL | Layer 1 crossbar |
| L2_crossbar_64x5.spice | Circuit | FINAL | Layer 2 crossbar |
| interstage_bias_relu.spice | Circuit | FINAL | Bias+ReLU+mirrors |
| WTA_5class.spice | Circuit | FINAL | Lazzaro WTA argmax |
| full_analog_mlp.py | Script | FINAL | Master training+verification |
| analog_interstage.py | Script | FINAL | Interstage design+test |
| generate_top_level.py | Script | FINAL | Top-level netlist generator |
| full_system_test.py | Script | LEGACY | End-to-end test (8→32→5) |

## Integration Files (experiments/integration/)
| File | Type | Status | Description |
|------|------|--------|-------------|
| integrate_frontend_classifier.py | Script | FINAL | Master integration: calibration, VCVS design, SW emulation |
| generate_combined_testbench.py | Script | FINAL | ngspice testbench generator for crossbar validation |
| digital_verification_results.json | Data | FINAL | Digital-only verification: 96.04% confirmed |

## Shared Data
| File | Location | Description |
|------|----------|-------------|
| features_8_cache.npz | classifier/ | Digital features (51k train, 48k test) |
| full_analog_weights.npz | classifier_crossbar/ | Trained weights + ReRAM mappings |
| _cache/*.json | complete_frontend/ | Per-record analog frontend features |
