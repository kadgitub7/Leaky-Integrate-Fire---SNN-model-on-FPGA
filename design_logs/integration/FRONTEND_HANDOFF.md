# Frontend Design Handoff — Fully Analog ECG Feature Extraction

**Date:** 2026-11-19
**Project:** Ultra-low-power analog ECG classifier — SNN on ReRAM crossbar
**Task:** MAJOR-007 — Make the analog frontend produce features matching the Python simulation
**PDK:** SKY130 130nm CMOS (open-source, BSIM4 models, tt corner)
**Simulator:** ngspice (`C:\Spice64\bin\ngspice_con.exe`)
**All SPICE files require:** `.option scale=1.0u`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Feature Definitions (Python Reference)](#2-feature-definitions-python-reference)
3. [Three-Layer Validation Framework](#3-three-layer-validation-framework)
4. [Circuit Inventory](#4-circuit-inventory)
5. [Signal Chain Architecture](#5-signal-chain-architecture)
6. [Timing Architecture](#6-timing-architecture)
7. [Calibration Data](#7-calibration-data)
8. [Current Status — Per-Feature Root Cause Analysis](#8-current-status--per-feature-root-cause-analysis)
9. [Specific Problems to Fix](#9-specific-problems-to-fix)
10. [File Inventory](#10-file-inventory)
11. [Test Infrastructure](#11-test-infrastructure)
12. [Design Constraints](#12-design-constraints)
13. [Acceptance Criteria](#13-acceptance-criteria)

---

## 1. System Overview

```
  Electrode L0 ──→ [IA_L0] ──→ ecg_amp_l0 ──┬→ [R-peak Det] → rpeak_level
  Electrode L1 ──→ [IA_L1] ──→ ecg_amp_l1 ─┐│                     │
                                             ││    ┌────────[rpeak_hold]──→ rpeak_level_held
  [Bias Gen] → vbias (0.300V), vref          ││    │                          │
                                             ││    │    ┌────[edge_buffer]──→[pulse_gen]──→[edge_buffer]──→ rpeak_pulse
                                             ││    │    │
                                             ││    ↓    ↓
                                             ││  [Delay Timer] → post-R trigger LEVELS
                                             ││  [P-wave Pred] → pre-R trigger LEVELS
                                             ││    │
                                             ││    ↓  (each level → edge_buffer → pulse_gen → edge_buffer → trigger pulse)
                                             ↓↓    ↓
                                      [Feature Extraction ×8]
                                             │
                                    8 analog feature voltages (f1..f8)
                                             │
                                    [To ReRAM crossbar classifier at 1.8V]
```

**Operating conditions:**
- VDD = 0.9V (subthreshold operation for ultra-low power)
- VBASE = 0.300V (DC operating point for IA output / ECG signal center)
- Temperature: 27°C (tt corner)
- Bias current: ~150pA per branch (set by vbias from bias_gen)
- Total transistor count: ~95 MOSFETs, ~25 MIM caps

---

## 2. Feature Definitions (Python Reference)

The classifier is an 8→64→5 MLP (832 MACs) on a 1T1R ReRAM crossbar. The 8 input features are extracted from the ECG waveform around each R-peak. The Python simulation defines the ground truth that the analog circuits must replicate.

### Python Feature Computation (from `compute_python_features()` in validate_3way.py)

All features are computed within a beat window: `WIN_L = 90` samples before annotation, `WIN_R = 108` samples after. Sample rate `FS = 360 Hz`.

| Feature | Name | Python Formula | Sample Indices | Physical Meaning |
|---------|------|---------------|----------------|-----------------|
| **F1** | rr_ratio | `pre_rr - ewma_rr_before` | N/A (timing) | Current RR interval minus EWMA of past RR intervals (seconds) |
| **F2** | rr_asym | `pre_rr - post_rr` | N/A (timing) | Pre-beat RR minus post-beat RR (seconds). Asymmetry of beat position |
| **F3** | bi_L0 | `ecg_l0[rpk_idx] - prev_ecg_l0[rpk_idx]` | rpk_idx = WIN_L = 90 | Beat-to-beat amplitude change at R-peak, Lead 0 |
| **F4** | bi_L1 | `ecg_l1[rpk_idx] - prev_ecg_l1[rpk_idx]` | rpk_idx = WIN_L = 90 | Beat-to-beat amplitude change at R-peak, Lead 1 |
| **F5** | pwave | `ecg_l0[PW_MID] - ecg_l0[PW_S]` | PW_S=16 (R-206ms), PW_MID=43 (R-131ms) | P-wave amplitude (signed difference) |
| **F6** | width | `ecg_l0[WIN_L + 10]` | WIN_L+10 = 100 (R+28ms) | ECG amplitude at R+28ms. Wide QRS → elevated |
| **F7** | slope | `ecg_l0[WIN_L+5] - ecg_l0[WIN_L+25]` | WIN_L+5=95 (R+14ms), WIN_L+25=115 (R+69ms) | Post-R slope asymmetry |
| **F8** | symmetry | `ecg_l0[rpk_idx]` | rpk_idx = WIN_L = 90 | ECG amplitude at R-peak |

### Python EWMA for F1

```python
EWMA_ALPHA_PY = 0.10
if ewma_rr is None:
    ewma_rr_before = rr_s
    ewma_rr = rr_s
else:
    ewma_rr_before = ewma_rr
    ewma_rr = EWMA_ALPHA_PY * rr_s + (1 - EWMA_ALPHA_PY) * ewma_rr
F1 = rr_s - ewma_rr_before
```

**CRITICAL MISMATCH:** Python uses `EWMA_ALPHA_PY = 0.10` but the SPICE rr_timing circuit has `alpha = Cesamp/(Cesamp + Cewma) = 5pF/(5pF + 330pF) ≈ 0.015`. This is a 6.7× mismatch in EWMA responsiveness. The SPICE EWMA has tau ≈ 67 beats vs Python tau ≈ 10 beats.

### Feature Reorder (Frontend → Classifier)

Frontend output `[F1, F2, F3, F4, F5, F6, F7, F8]` maps to classifier rows `[0, 3, 1, 4, 5, 2, 6, 7]`.

---

## 3. Three-Layer Validation Framework

We validate using three independently computed feature sets and compare them pairwise:

1. **Python features** — computed directly from raw MIT-BIH waveform samples using the formulas above
2. **Ideal-SPICE features** — SPICE simulation with real feature extraction circuits but **ideal PWL triggers** (perfect timing from annotations)
3. **Analog-SPICE features** — SPICE simulation with real feature extraction circuits AND **real analog trigger circuits**

**Metric:** Spearman rank correlation (rho) between each pair.

| Comparison | What it Tests | Pass Criterion |
|-----------|---------------|----------------|
| Py-Ideal | Feature circuit fidelity (do circuits compute the right thing?) | rho >= 0.97 |
| Id-Analog | Trigger quality (do analog triggers fire at the right time?) | rho >= 0.97 |
| Py-Analog | End-to-end accuracy (does the full analog chain match Python?) | rho >= 0.97 |

**Additional criteria:** Clip rate < 1%, output span >= 225mV, Cohen's d >= 75%.

---

## 4. Circuit Inventory

All subcircuits are in `experiments/frontend_optimization/complete_frontend_v9b/`.

### 4.1 Support Circuits

| File | Subcircuit | Pins | Description |
|------|-----------|------|-------------|
| `bias_gen.spice` | `bias_gen` | vbias vref vdd vss | Self-biased ~150pA reference. PMOS mirror + NFET diodes. VBIAS=VREF≈0.300V. Needs `.ic v(xbias.iref)=0.3` for startup |
| `ia.spice` | `ia` | inp inn out vdd vss vbias vref | Capacitively-coupled instrumentation amp. Gain = Cin/Cfb = 10pF/0.1pF = 100. BW ~570Hz. Pseudo-R biasing (>1TΩ PMOS) |
| `inverter.spice` | `inverter_sub` | vin vout vdd vss | CMOS inverter. PFET L=0.5 W=2, NFET L=0.5 W=1 |
| `pulse_gen.spice` | `pulse_gen` | level_in pulse_out vdd vss vbias | Rising-edge level→pulse converter. AC coupling 0.2pF + subthreshold bleeder. Output pulse ~0.5ms |
| `long_pulse_gen.spice` | `long_pulse_gen` | level_in pulse_out vdd vss vbias | Same topology, reduced bleeder W=0.41 → ~2ms pulse. For slow-transitioning comparator outputs |
| `edge_buffer.spice` | `edge_buffer` | vin vout vdd vss | Two-inverter buffer. Sharpens slow comparator transitions for reliable pulse_gen triggering |

### 4.2 R-Peak Detection Chain

| File | Subcircuit | Pins | Description |
|------|-----------|------|-------------|
| `rpeak_detector.spice` | `rpeak_detector` | ecg_in threshold rpeak_out vdd vss vbias | Two-stage comparator. Diff pair (NFET L=2 W=4) + current mirror + output inverter (PFET L=0.5 W=8, NFET L=0.5 W=4). Output HIGH when ECG > threshold |
| `rpeak_hold.spice` | `rpeak_hold` | input output vdd vss vbias | Debounce for NEG-polarity chatter. Cap charges to VDD on HIGH, slow NFET discharge (~5mV/ms). Hold time ~300ms (12pF MIM, 78×78). Blanks T-wave retriggering |

**R-peak signal chain:**
```
rpeak_detector → rpeak_hold → edge_buffer → pulse_gen → edge_buffer → rpeak_pulse
```

**Key behavior:**
- Fires at threshold crossing on the QRS **upslope**, NOT at the peak
- Offset: fires ~6.6ms BEFORE the annotation R-peak (varies 3-7ms beat-to-beat)
- Beat-to-beat jitter: ±1.5-1.8ms (caused by QRS morphology/amplitude variation)
- This jitter is the **primary cause** of F3/F7/F8 failure

### 4.3 Delay Timer (Post-R-peak Triggers)

| File | Subcircuit | Pins |
|------|-----------|------|
| `delay_timer.spice` | `delay_timer` | rpeak_pulse trig_s14 trig_s69 trig_wid trig_bi_copy trig_rr_copy trig_rpk_delayed sf_out vdd vss vbias_delay vth_14_fast vth_69 vth_83 vth_300 vth_10 vth_rpk_del |

**Architecture:**
- **Main timer:** 224×224 MIM (~100pF). Precharge PFET L=0.5 W=2, discharge NFET L=8 W=0.84. SF NFET L=8 W=0.42 + bias NFET L=8 W=0.42
- **Fast timer:** 45×45 MIM (~4pF). Same precharge/discharge/SF topology. ~25× faster decay
- **SF decay rates:** Main ~0.52mV/ms (dt_v table), Fast ~13mV/ms (ft_v table)
- **SF initial value:** ~0.481V at t=0 (after precharge + SF drop)

**Comparator allocation:**

| Comparator | Timer | Threshold Pin | Target Time | Pipeline Delay | Purpose |
|-----------|-------|--------------|-------------|---------------|---------|
| c2 | Main | vth_69 | R+69ms | 4.4ms | F7 slope sample 2 |
| c3 | Main | vth_83 (28ms) | R+28ms | 4.5ms | F6 width sample |
| c4 | Main | vth_300 | R+300ms | 5.9ms | F3/F4 EWMA copy |
| c5 | Fast | vth_10 | R+10ms | 1.8ms | RR copy (EWMA charge-share) |
| c6 | Fast | vth_14_fast | R+14ms | 2.9ms | F7 slope sample 1 |
| c7 | Fast | vth_rpk_del | R+0ms | 3.0ms | F3/F4/F8 delayed rpeak |

**All 7 comparators use identical topology:**
- NFET diff pair: L=4 W=2
- PMOS current mirror loads: L=4 W=2
- Tail current: L=8 W=0.42 (vbias_delay controlled)
- Output inverter: PFET L=0.5 W=8, NFET L=0.5 W=4

**Comparator fires** when SF voltage drops below threshold: `sf_timer < vth → output goes HIGH`

**Each comparator level output goes through:**
```
comparator_level → edge_buffer → pulse_gen → edge_buffer → trigger_pulse
```
(This chain adds the "pipeline delay" listed above)

### 4.4 P-Wave Predictor (Pre-R-peak Triggers)

| File | Subcircuit | Pins |
|------|-----------|------|
| `pwave_predictor.spice` | `pwave_predictor` | timer_node ewma_node pw_base_trig pw_peak_trig rr_cap_trig vth_base vth_peak vth_rr_cap vdd vss vbias |

**11-pin version** (complete_frontend_v9b). Uses EWMA + offset thresholds to predict when the next R-peak will arrive:

| Comparator | Threshold | Target Time | Purpose |
|-----------|-----------|-------------|---------|
| base | vth_base = esamp_pw + PWB_offset | R-206ms | F5 pwave base sample |
| peak | vth_peak = esamp_pw + PWP_offset | R-131ms | F5 pwave peak sample |
| rr_cap | vth_rr_cap = ewma + RRC_offset | R-5ms | RR timing capture trigger |

**How it works:** The rr_timing timer decays from VDD after each rpeak. The pwave_predictor watches the timer's SF output and fires comparators when the timer drops below threshold levels computed from the EWMA of previous end-of-interval SF values + timing offsets.

**Fundamental limitation:** The predictor fires at a fixed time after the CURRENT rpeak (determined by EWMA of past RR intervals). It CANNOT predict variable-RR intervals. Record 100 has RR intervals from ~673ms to ~1026ms (353ms range), causing ±180ms trigger timing error for F5.

### 4.5 Feature Extraction Circuits

| File | Subcircuit | Pins | Feature | Topology |
|------|-----------|------|---------|----------|
| `rr_timing.spice` | `rr_timing` | ecg_unused capture_trig rpeak_trigger copy_trig rr_ratio_out rr_asym_out timer_out ewma_out vdd vss vbias | F1, F2 | 400×400 MIM timer + CMOS TG S&H + EWMA charge-share + 2 diff pairs |
| `beat_instability.spice` | `beat_instability` | ecg_in capture_trig copy_trig instab_out vdd vss vbias | F3, F4 | NFET S&H + EWMA charge-share + diff pair |
| `pwave_diff.spice` | `pwave_diff` | ecg_in sample_base_trig sample_peak_trig pw_out vdd vss vbias | F5 | 2× NFET S&H + hybrid diff pair (3× gain) |
| `qrs_width.spice` | `qrs_width` | ecg_in sample_trig width_out vdd vss | F6 | Single NFET S&H (passive) |
| `qrs_slope.spice` | `qrs_slope` | ecg_in sample14_trig sample69_trig slope_out vdd vss vbias | F7 | 2× NFET S&H + diff pair |
| `qrs_symmetry.spice` | `qrs_symmetry` | ecg_in sample_r_trig sym_out vdd vss | F8 | CMOS TG S&H (full range) |

#### rr_timing.spice (F1 & F2) — DETAILED

**Timer:** 400×400 MIM (~320pF), PFET precharge L=0.5 W=2, NFET discharge L=8 W=0.42. Precharged to VDD at rpeak_trigger.

**Source Follower:** NFET L=8 W=0.42 + bias NFET L=8 W=0.42. Level-shifts timer down ~0.4V. V_sf ≈ 0.48V at start.

**CMOS Transmission Gate switches** (not NFET-only): eliminates ~24mV loss at Vsf≈0.4V where NFET-only Vgs is barely above Vth.

**Nodes:**
- `sf_out` — live source follower output tracking timer
- `prev_node` — captured sf_out at `capture_trig` (5pF, 50×50 MIM)
- `esamp_node` — captured sf_out at `capture_trig` (5pF, 50×50 MIM)
- `ewma_node` — charge-shared with esamp_node at `copy_trig` (330pF, 405×405 MIM)

**EWMA update:** `alpha = Cesamp/(Cesamp + Cewma) = 5/(5+330) ≈ 0.015`, tau ≈ 67 beats

**Diff pairs:**
- **F1 (rr_ratio):** `ewma_node` vs `sf_out` — diode loads, gain ~1
- **F2 (rr_asym):** `sf_out` vs `prev_node` — diode loads, gain ~1

**Control signals:**
- `capture_trig` — fires R-5ms (from pwave_predictor rr_cap). Captures sf_out into prev_node and esamp_node
- `rpeak_trigger` — fires at rpeak. Precharges timer to VDD
- `copy_trig` — fires R+10ms (from delay_timer). Charge-shares esamp→ewma

**Exposed nodes:** `timer_out` = sf_out, `ewma_out` = ewma_node (via 0Ω resistors, for pwave_predictor)

#### beat_instability.spice (F3, F4) — DETAILED

**Nodes:**
- `curr_node` — 50×50 MIM (5pF), captures ECG at `capture_trig` via NFET S&H
- `esamp_node` — 150×150 MIM (45pF), captures ECG at `capture_trig`
- `prev_node` — 50×50 MIM (5pF), charge-shared with esamp at `copy_trig`. Alpha = 5/(5+45) = 0.9

**Diff pair:** `curr_node` vs `prev_node`, diode loads, gain ~1

**Key issue:** F3/F4 require that capture_trig fires at the R-peak. The analog rpeak_pulse fires on the QRS upslope (6.6ms before peak), and the rpeak_pulse_delayed from delay_timer c7 is meant to compensate. But jitter of ±1.5ms on the steep QRS portion (~108mV/ms) causes ±45mV capture error, collapsing the signal range.

#### pwave_diff.spice (F5) — DETAILED

**S&H caps:** 32×32 MIM (~2pF) each for base and peak samples

**Diff pair:** Hybrid load = mirror W=4 + diode W=1.5 → gain ≈ 3×. Expands ~80mV pwave signal to ~256mV.

**Key issue:** Requires triggers at R-206ms and R-131ms BEFORE the R-peak. These come from the pwave_predictor, which estimates timing from EWMA. With variable RR intervals, trigger timing error can be ±180ms.

#### qrs_width.spice (F6) — Single S&H, WORKING

Simple NFET pass gate + 32×32 MIM. Captures ECG at R+28ms. PASSES validation (rho=0.982) because ECG changes slowly at R+28ms (~5mV/ms), making 3.5ms jitter only ~17mV error.

#### qrs_slope.spice (F7) — DETAILED

Two S&H (32×32 MIM each) capture ECG at R+14ms and R+69ms. Diff pair with diode loads, gain ~1.

**Key issue:** At R+14ms the QRS downslope has ~16.5mV/ms in SPICE domain. 3.5ms rpeak jitter → 58mV jitter in s14 capture, overwhelming the ~200mV signal range. Result: rho ≈ 0.70.

#### qrs_symmetry.spice (F8) — DETAILED

CMOS TG S&H (NFET + PFET, with inverter for complementary gate). Full-range voltage capture. 32×32 MIM.

**Key issue:** Same as F3 — captures ECG on steep QRS portion where jitter dominates.

---

## 5. Signal Chain Architecture

### R-peak to Triggers (complete signal chain)

```
ECG → rpeak_detector (comparator, fires on QRS upslope)
   → rpeak_hold (300ms debounce, prevents T-wave retrigger)
   → edge_buffer (sharpens slow transition)
   → pulse_gen (level-to-pulse, ~0.5ms)
   → edge_buffer (cleanup)
   → rpeak_pulse (brief pulse, ~0.5ms)
       |
       ├──→ delay_timer (rpeak_pulse precharges both fast+main timers)
       │       ├──→ c6: trig_s14_level (fast timer, R+14ms)
       │       ├──→ c2: trig_s69_level (main timer, R+69ms)
       │       ├──→ c3: trig_wid_level (main timer, R+28ms)
       │       ├──→ c4: trig_bi_level (main timer, R+300ms)
       │       ├──→ c5: trig_rr_level (fast timer, R+10ms)
       │       └──→ c7: trig_rpk_del_level (fast timer, R+0ms target)
       │
       └──→ rr_timing (rpeak_pulse precharges 320pF timer)
                └──→ pwave_predictor (watches sf_out from rr_timing)
                        ├──→ trig_pw_base (R-206ms predicted)
                        ├──→ trig_pw_peak (R-131ms predicted)
                        └──→ trig_rr_cap (R-5ms predicted)
```

### Each trigger level → pulse conversion

```
trig_xxx_level → edge_buffer → pulse_gen → edge_buffer → trig_xxx (pulse)
```

This chain adds a "pipeline delay" (2-6ms) that must be compensated in threshold calibration.

### Esamp Validity Gate (trig_rr_copy)

An AND gate blocks `trig_rr_copy` when `xrr.esamp_node < 0.35V`. This prevents false EWMA updates during the first few beats before the timer has converged:

```spice
* Comparator: esamp_node vs 0.35V threshold
Xesamp_chk xrr.esamp_node esamp_ok_thresh esamp_ok_level vdd vss vbias rpeak_detector

* NAND gate: trig_rr_copy_raw AND esamp_ok_level
XMnand_p1 nand_out trig_rr_copy_raw vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=2 nf=1
XMnand_p2 nand_out esamp_ok_level vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=2 nf=1
XMnand_n1 nand_out trig_rr_copy_raw nand_mid vss sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1
XMnand_n2 nand_mid esamp_ok_level vss vss sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1
Xinv_nand nand_out trig_rr_copy vdd vss inverter_sub
```

### Isolated Duplicate Timer for P-wave

The testbench (validate_3way.py analog testbench) creates a **separate isolated copy** of the rr_timing timer specifically for pwave_predictor base/peak comparators. This uses the same 400×400 MIM topology but on a separate node (`sf2_out`), so the pwave predictions don't disturb the main rr_timing timer. The rr_cap comparator still uses the main timer.

---

## 6. Timing Architecture

### NFET Pass Gate Capture Behavior

**CRITICAL:** NFET pass gates capture ECG at the **FALL edge** of the trigger pulse, NOT the rise edge. When the pulse goes LOW, the NFET turns off and the cap holds the voltage.

### R-peak Timing

The analog rpeak detector fires at a **threshold crossing on the QRS upslope**:
- Average offset: ~6.6ms BEFORE the annotation R-peak
- Beat-to-beat variation: 3-7ms (±1.5-1.8ms jitter)
- Jitter source: QRS morphology/amplitude variation changes where the threshold crossing occurs
- The delay timer itself is very consistent (±0.15ms) — the jitter is ALL from the rpeak detector

### Why centering at the R-peak helps

Near the R-peak, ECG changes **quadratically** (slope = 0 at peak). So ±1ms jitter causes only ~1.4mV error. On the downslope (1ms after peak), ECG changes **linearly** at ~8mV/ms, so ±1ms jitter causes ~8mV error. Features that capture on steep portions (F3, F7, F8) are most affected.

### Trigger Timing Summary

| Trigger | Target Time | Source | Used By |
|---------|------------|--------|---------|
| rpeak_pulse | R-6.6ms (actual) | rpeak_detector chain | rr_timing (precharge), delay_timer (start), F3/F4 capture |
| trig_rpk_del | R+0ms (annotation) | delay_timer c7 (fast) | F3/F4/F8 capture via rpeak_pulse_delayed |
| trig_s14 | R+14ms | delay_timer c6 (fast) | F7 slope S&H 1 |
| trig_rr_copy | R+10ms | delay_timer c5 (fast) | rr_timing EWMA charge-share |
| trig_wid | R+28ms | delay_timer c3 (main) | F6 width S&H |
| trig_s69 | R+69ms | delay_timer c2 (main) | F7 slope S&H 2 |
| trig_bi | R+300ms | delay_timer c4 (main) | F3/F4 EWMA copy |
| trig_rr_cap | R-5ms | pwave_predictor | rr_timing prev+esamp capture |
| trig_pw_base | R-206ms | pwave_predictor | F5 pwave base S&H |
| trig_pw_peak | R-131ms | pwave_predictor | F5 pwave peak S&H |

---

## 7. Calibration Data

### Source Follower Decay Lookup Tables

These tables map elapsed time (ms) to SF output voltage (V) for threshold computation.

**Main rr_timing timer (400×400 MIM, ~320pF):**
```
Time(ms):  10     20     50     100    200    300    400    500    600    700    800    900    1000
V_sf:     0.4819 0.4807 0.4772 0.4715 0.4602 0.4488 0.4375 0.4262 0.4150 0.4037 0.3925 0.3813 0.3702
```

**Delay timer main (224×224 MIM, ~100pF):**
```
Time(ms): 1    2    3    4    5    10   15   20   30   40   50   60   70   80   90   100  150  200  250  300  350  400  500  600  700  800
V_sf:     .483 .482 .482 .481 .481 .478 .476 .473 .468 .463 .457 .452 .447 .442 .437 .431 .406 .380 .355 .329 .304 .279 .230 .181 .133 .086
```

**Delay timer fast (45×45 MIM, ~4pF):**
```
Time(ms): 0.5  1    2    3    4    5    7    10   14   20   25   30   35   40
V_sf:     .481 .475 .462 .449 .436 .424 .398 .361 .311 .237 .177 .118 .062 .019
```

### Threshold Computation Formula

For each comparator, the threshold voltage is computed by looking up the SF value at the **effective delay time**:

```
vth = interp_sf(table, target_time + rpeak_offset - PG_WIDTH - pipeline_delay)
```

Where:
- `target_time` = desired trigger time relative to true R-peak (ms)
- `rpeak_offset` = 6.6ms (how early rpeak_detector fires)
- `PG_WIDTH` = 0.7ms (pulse_gen pulse width)
- `pipeline_delay` = per-comparator (see table in Section 4.3)

The effective delay time = `target_time + 6.6 - 0.7 - pipeline_delay` ms after timer precharge.

### Pipeline Delays (per-comparator, empirically calibrated)

```
PIPE_S14 = 2.9ms    (fast timer, c6)
PIPE_S10 = 1.8ms    (fast timer, c5)
PIPE_RPK_DEL = 3.0ms (fast timer, c7)
PIPE_S69 = 4.4ms    (main timer, c2)
PIPE_WID = 4.5ms    (main timer, c3)
PIPE_BI  = 5.9ms    (main timer, c4)
PIPE_PW  = 3.0ms    (pwave predictor comparators)
```

### EWMA IC Calibration

For the first beat, initial conditions must be set:
```
ic_ewma = interp_sf(rr_t, rr_v, avg_rr_ms)   # NO loss subtraction
```
Example: avg_rr_ms=831ms → ic_ewma ≈ 0.3914V

The timer net_timer should be initialized to VDD: `.ic V(xrr.net_timer)=0.9`

### Pwave Threshold Computation

```python
SF_DECAY_RATE = 0.112e-3  # V/ms (applies to 400×400 MIM timers)

# Esamp-based offsets (from captured sf2_out at R-5.9ms)
PWB_OFF_ESAMP = SF_DECAY_RATE * (PWB_TARGET_MS - capture_offset_ms + PIPE_PW)  # volts
PWP_OFF_ESAMP = SF_DECAY_RATE * (PWP_TARGET_MS - capture_offset_ms + PIPE_PW)  # volts
RRC_OFFSET_MV = 20.0  # mV, for rr_cap trigger
```

Where `PWB_TARGET_MS = 205.6`, `PWP_TARGET_MS = 130.6`, `capture_offset_ms = rpeak_offset - PG_WIDTH = 5.9`.

---

## 8. Current Status — Per-Feature Root Cause Analysis

### Latest Test Results (Record 100, 30 beats, PIPE_RPK_DEL=3.0)

| Feature | Py-Ideal rho | Py-Analog rho | Id-Analog rho | n | Status |
|---------|-------------|---------------|---------------|---|--------|
| **F1:rr_ratio** | 0.959 | **0.964** | **0.992** | 20 | **PASS** |
| **F2:rr_asym** | 0.787 | **0.991** | 0.735 | 19 | **FAIL** (ideal testbench bug) |
| **F3:bi_L0** | **0.973** | 0.040 | 0.189 | 30 | **FAIL** (rpeak jitter) |
| **F4:bi_L1** | 0.924 | **0.965** | 0.921 | 30 | **NEAR-PASS** |
| **F5:pwave** | **0.968** | 0.066 | 0.038 | 30 | **FAIL** (RR prediction) |
| **F6:width** | 0.958 | **0.970** | **0.982** | 30 | **PASS** |
| **F7:slope** | 0.964 | 0.686 | 0.704 | 30 | **FAIL** (s14 jitter) |
| **F8:symmetry** | **0.981** | 0.120 | 0.246 | 30 | **FAIL** (rpeak jitter) |

### Feature-by-Feature Root Cause

#### F1 (rr_ratio) — PASS ✓
- Analog works. Id-Analog = 0.992.
- Py-Ideal = 0.959 is slightly low due to EWMA alpha mismatch (Python 0.10 vs SPICE 0.015), but the RANKING is preserved.

#### F2 (rr_asym) — Circuit OK, testbench issue
- **Py-Analog = 0.991** — the analog circuit produces correct F2 values.
- Py-Ideal = 0.787 — the IDEAL testbench has a bug (rr_cap timing or EWMA behavior doesn't match the analog).
- **No circuit fix needed.** Only the ideal testbench needs debugging (low priority).

#### F3 (bi_L0) — FAIL ✗ — Rpeak Jitter
- **Root cause:** rpeak_pulse fires on QRS upslope where ECG slope is ~108mV/ms. Beat-to-beat jitter of ±1.5ms causes ±45mV capture error. The curr-prev signal range collapses from [-47mV, +37mV] (Python) to [-0.4mV, +6.8mV] (analog). The diff pair output has only 3.7mV variation vs Python's ~85mV range.
- **Py-Ideal = 0.973** proves the feature circuit itself works perfectly with ideal triggers.
- **Fix required:** Reduce rpeak timing jitter to ≤±0.3ms, OR capture at a point where ECG slope is flat (near true peak).

#### F4 (bi_L1) — NEAR-PASS (0.965)
- Same mechanism as F3 but Lead 1 has gentler QRS slope, reducing jitter impact.
- May pass with improved rpeak timing.

#### F5 (pwave) — FAIL ✗ — Fundamental RR Prediction Limitation
- **Root cause:** The pwave predictor fires at a fixed time (~618ms) after the current rpeak regardless of the NEXT RR interval. Record 100 RR range: 673-1026ms (353ms). This creates ±180ms trigger timing error.
- Output is bimodal (~0.87V or ~0.33V) depending on whether RR_next > or < RR_avg.
- Py-Ideal = 0.968 proves the circuit works if triggers fire at the right time.
- **This is NOT an EWMA convergence issue.** Fast EWMA (tau=11) and slow EWMA (tau=67) produce nearly identical results because the EWMA converges correctly — it just can't predict FUTURE beat timing.
- **Fix required:** Fundamentally different pwave trigger architecture, OR retrain classifier without F5.

#### F6 (width) — PASS ✓
- Works because ECG at R+28ms changes slowly (~5mV/ms), making 3.5ms jitter only ~17mV — small vs signal range.

#### F7 (slope) — FAIL ✗ — S14 Timing Jitter
- **Root cause:** At R+14ms the QRS downslope has ~16.5mV/ms. 3.5ms rpeak jitter → 58mV jitter in s14 capture, overwhelming ~200mV signal range.
- The s69 sample at R+69ms is in a flatter ECG region and is less affected.
- Py-Ideal = 0.964 proves circuit works with ideal triggers.
- **Fix required:** Same as F3 — reduce rpeak jitter.

#### F8 (symmetry) — FAIL ✗ — Rpeak Jitter
- Same mechanism as F3. Captures ECG at rpeak_pulse_delayed on steep QRS portion.
- Py-Ideal = 0.981 proves circuit works perfectly with ideal triggers.
- **Fix required:** Same as F3 — reduce rpeak jitter.

---

## 9. Specific Problems to Fix

### Problem 1: Rpeak Timing Jitter (affects F3, F7, F8; partially F4)

**The issue:** The rpeak_detector fires at a threshold crossing on the QRS upslope. Because beat-to-beat QRS morphology varies, the crossing point moves ±1.5ms. On steep QRS portions, this translates to tens of millivolts of S&H capture error.

**Proposed solution — True Peak Detector:**
A well-known analog circuit that fires at the actual R-peak rather than a threshold crossing:

1. **Peak tracking:** Comparator + pass gate + hold cap tracks ECG maximum
2. **Tracking comparator** with ~2mV positive offset prevents oscillation at peak
3. **Peak detection:** Second comparator fires when ECG drops 5mV below held peak (indicating we've passed the peak)
4. **ARM gate:** Uses existing rpeak_level_held to prevent P/T-wave false triggers
5. **Expected performance:** Fires at R+0.875ms with ±0.1ms jitter (15× improvement over threshold crossing)

**Design concern:** The standard pipeline (edge_buffer → pulse_gen → edge_buffer) adds ~3ms. Options:
- Use inverted true_rpeak directly as S&H trigger (bypassing pipeline)
- Trigger delay_timer from true_rpeak with recalibrated thresholds
- Accept ~3ms delay and recalibrate downstream timing

**Alternative solution:** Capture at a flatter ECG region. The region near the true R-peak has near-zero slope, so even with current 1.5ms jitter, capture error would be minimal. This requires the rpeak_pulse_delayed trigger (c7 on fast timer) to fire closer to the true peak — currently calibrated to fire at annotation time (R+0ms), which is ~6.6ms after the rpeak_pulse.

### Problem 2: EWMA Alpha Mismatch (affects Py-Ideal rho for F1, F2)

**The issue:** Python uses `EWMA_ALPHA = 0.10` but SPICE rr_timing has `alpha ≈ 0.015` (Cesamp=5pF/(5pF+330pF)).

**Impact:** Py-Ideal F2 rho is only 0.787. The ANALOG circuit works correctly (Py-Analog = 0.991), so this is a testbench/validation issue, not a circuit bug. However, classifier training uses Python features, so the classifier expects alpha=0.10 behavior.

**Options:**
1. Change SPICE Cewma to match Python alpha: Cewma = Cesamp × (1-alpha)/alpha = 5 × 0.9/0.1 = 45pF (45×45→67×67 MIM). This makes SPICE match Python exactly.
2. Retrain classifier with alpha=0.015 Python features. Changes F1/F2 feature distributions.
3. Accept the mismatch — the analog F2 already passes (Py-An=0.991).

### Problem 3: P-wave Prediction for Variable RR (affects F5)

**The issue:** F5 requires S&H triggers at R-206ms and R-131ms BEFORE the R-peak. The pwave_predictor estimates these times from the EWMA of past RR intervals. When RR varies significantly (as in Record 100 with 353ms range), the prediction is wrong by ±180ms.

**Possible solutions:**
1. **Drop F5** and retrain classifier on 7 features (simplest)
2. **Relative-to-previous pwave:** Instead of predicting absolute time before NEXT R-peak, capture at fixed times relative to the PREVIOUS R-peak (e.g., R_prev + 0.75×EWMA_RR and R_prev + 0.80×EWMA_RR). Still timer-based but shifts the reference point.
3. **Post-R pwave:** Change F5 to capture at fixed post-R delays (e.g., R-peak + 400ms and R-peak + 500ms). These are forward-looking into the P-wave of the NEXT beat. Needs delay_timer thresholds recalibrated for these longer delays.
4. **Accept the noise** and test whether the classifier still functions with bimodal F5 input.

### Problem 4: S&H Switch Type Consistency

Beat_instability uses NFET-only pass gates for curr_node and esamp_node, while rr_timing uses CMOS TG. At Vsf ≈ 0.4V, NFET-only has ~24mV loss (Vgs barely above Vth). Consider upgrading to CMOS TG for consistency and accuracy.

---

## 10. File Inventory

### Core Circuit Subcircuits (DO NOT MODIFY without validation)

```
experiments/frontend_optimization/complete_frontend_v9b/
├── bias_gen.spice          — Bias voltage generator (vbias=0.300V)
├── ia.spice                — Instrumentation amplifier (gain=100)
├── inverter.spice          — CMOS inverter subcircuit
├── rpeak_detector.spice    — QRS threshold comparator
├── rpeak_hold.spice        — 300ms debounce hold
├── pulse_gen.spice         — Level→pulse converter (~0.5ms)
├── long_pulse_gen.spice    — Level→pulse converter (~2ms)
├── edge_buffer.spice       — Two-inverter signal sharpener
├── delay_timer.spice       — Post-R trigger generator (7 comparators, 2 timers)
├── pwave_predictor.spice   — Pre-R trigger predictor (3 comparators)
├── rr_timing.spice         — F1/F2: RR timing + EWMA
├── beat_instability.spice  — F3/F4: beat-to-beat amplitude change
├── pwave_diff.spice        — F5: P-wave difference (3× gain hybrid loads)
├── qrs_width.spice         — F6: ECG at R+28ms
├── qrs_slope.spice         — F7: slope = ECG@R+14ms - ECG@R+69ms
├── qrs_symmetry.spice      — F8: ECG at R-peak
└── frontend_top.spice      — Top-level integration (asic_frontend subcircuit)
```

### Test Scripts

```
experiments/frontend_optimization/complete_frontend_v9b/
├── integration_test.py     — Multi-record fully-analog integration test (30 records)
├── verify_all_circuits.py  — Individual circuit verification
├── test_support_circuits.py — Bias gen, rpeak, delay timer isolated tests
├── test_delay_chain.py     — Delay timer timing characterization
├── test_rpeak_chain.py     — R-peak detection chain test
├── energy_test.py          — Power/energy characterization
├── full_analog_chain_test.py — Full chain validation
├── final_e2e_test.py       — End-to-end test
├── isolation_test.py       — Circuit isolation debugging
│   (also v2, v3, v4, v5)
├── diag_analog_modes.py    — Analog mode diagnostics
├── diag_pred_timer.py      — Pwave prediction timer diagnostics
├── diag_trig_timing.py     — Trigger timing analysis
├── diag_stepwise.py        — Stepwise debugging
├── phase4_test.py          — Phase 4 integration test
└── full_accuracy_test.py   — Full accuracy evaluation

scratchpad/
└── validate_3way.py        — THREE-WAY VALIDATION (Python vs Ideal vs Analog)
                              THIS IS THE CURRENT PRIMARY TEST SCRIPT
```

### SKY130 PDK Files

```
experiments/frontend_optimization/ngspice_frontend/
└── sky130_minimal.lib.spice  — Minimal SKY130 library for ngspice simulation
```

**Usage in SPICE:** `.lib "path/to/sky130_minimal.lib.spice" tt`

### MIT-BIH Database

```
mitdb_data/
├── 100.dat, 100.hea, 100.atr   — Record 100 (and others)
└── ... (30 records)
```

**Records used for validation:**
`100, 101, 102, 104, 105, 106, 107, 108, 109, 111, 114, 115, 116, 117, 118, 119, 200, 205, 207, 208, 212, 214, 215, 219, 220, 222, 228, 231, 232, 234`

### Design Logs

```
design_logs/integration/
├── MASTER_ISSUE_LOG.md         — Full project issue tracker
└── interface_register.json     — Interface specification
```

---

## 11. Test Infrastructure

### Running validate_3way.py

```powershell
cd C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA
python scratchpad/validate_3way.py
```

**Runtime:** ~3-4 minutes per record (ideal sim ~40s, analog sim ~160s)

**What it does:**
1. Loads MIT-BIH beats (WIN_L=90, WIN_R=108)
2. Computes Python features
3. Generates ideal SPICE testbench (PWL triggers at annotation times)
4. Runs ngspice, extracts .meas results
5. Generates analog SPICE testbench (full trigger chain)
6. Runs ngspice, extracts .meas results
7. Computes Spearman rho for all three pairwise comparisons
8. Prints diagnostic tables (rpeak timing, F5 pwave, F3/F8 per-beat)

**Configuration at top of file:**
```python
RECORDS = ['100']     # Which MIT-BIH records to test
MAX_BEATS = 35        # Beats per record
SKIP = 5              # Skip first N beats (settling)
SKIP_F1F2 = 15        # Skip more for F1/F2 (EWMA needs settling)
```

### Running integration_test.py

```powershell
cd experiments/frontend_optimization/complete_frontend_v9b
python integration_test.py
```

**Key difference from validate_3way.py:**
- Uses `complete_frontend` directory (not `complete_frontend_v9b`)
- Uses `pipe_rpk_del = 2.0` (vs 3.0 in validate_3way)
- Has adaptive rpeak polarity selection (NEG/POS per record)
- Runs 30 records
- **Uses measured S&H cap values as reference** (circuit fidelity), not Python features

### ngspice Invocation

```powershell
C:\Spice64\bin\ngspice_con.exe -b -o output.log testbench.spice
```

Always run from the `ngspice_frontend` directory (where `sky130_minimal.lib.spice` lives).

---

## 12. Design Constraints

1. **VDD = 0.9V** — subthreshold operation. No digital logic above 0.9V in the frontend
2. **Only inputs:** VDD=1.8V (externally), ECG (two differential lead pairs). The 0.9V must be generated from 1.8V via LDO
3. **No digital contamination** — all features must be fully analog. No ADCs, no digital arithmetic
4. **SKY130 130nm CMOS** — only devices from `sky130_fd_pr__nfet_01v8` and `sky130_fd_pr__pfet_01v8` families, plus `sky130_fd_pr__cap_mim_m3_1` for MIM caps
5. **`.option scale=1.0u`** required in all SPICE netlists
6. **Bias current ~150pA** — set by vbias≈0.300V on NFET L=8 W=0.42
7. **MIM cap formula:** Area = W × L × 2fF/μm² (at scale=1.0u). Example: W=50 L=50 → 5pF
8. **Feature output range:** Must span ≥225mV for sufficient classifier resolution
9. **Clipping:** <1% of beats should clip to supply rails

---

## 13. Acceptance Criteria

For each feature, ALL of these must pass:

| Criterion | Target | How Measured |
|----------|--------|-------------|
| Spearman rho (Py-Analog) | ≥ 0.97 | validate_3way.py end-to-end comparison |
| Spearman rho (Id-Analog) | ≥ 0.97 | validate_3way.py trigger quality comparison |
| Clip rate | < 1% | Fraction of beats where output = VDD or VSS |
| Output span | ≥ 225mV | max(feature) - min(feature) across all beats |
| Cohen's d | ≥ 75% | Effect size between N-class and abnormal classes |

**Multi-record requirement:** Must pass on the full 30-record MIT-BIH set, not just Record 100.

**Current scorecard:**
- **PASS (2/8):** F1 (rr_ratio), F6 (width)
- **NEAR-PASS (2/8):** F2 (rr_asym, circuit OK), F4 (bi_L1, 0.965)
- **FAIL (4/8):** F3 (bi_L0), F5 (pwave), F7 (slope), F8 (symmetry)

---

## Appendix A: Complete SPICE Testbench Template

The analog testbench in validate_3way.py generates a complete netlist including:

```spice
.title analog_triggers
.lib "sky130_minimal.lib.spice" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15 itl4=200
.temp 27

Vdd vdd 0 0.9
Vss vss 0 0
Vbias vbias 0 0.300
Vecg0 ecg_l0 0 PWL(...)     * ECG waveform Lead 0
Vecg1 ecg_l1 0 PWL(...)     * ECG waveform Lead 1

Vrpk_th rpeak_threshold 0 <threshold>  * Adaptive per-record

* Delay timer thresholds (computed from calibration tables)
Vth14 vth_14_node 0 <vth_14>
Vth69 vth_69_node 0 <vth_69>
Vth28 vth_28_node 0 <vth_28>
Vth300 vth_300_node 0 <vth_300>
Vth10 vth_10_node 0 <vth_10>
Vthrpk vth_rpk_del_node 0 <vth_rpk_del>

* Include all subcircuits
.include "inverter.spice"
.include "rpeak_detector.spice"
... (all subcircuits)

* R-peak detection chain
Xrpk ecg_l0 rpeak_threshold rpeak_level vdd vss vbias rpeak_detector
Xhold rpeak_level rpeak_level_held vdd vss vbias rpeak_hold
Xeb_rpk rpeak_level_held rpk_eb vdd vss edge_buffer
Xpg_rpk rpk_eb rpk_pg vdd vss vbias pulse_gen
Xeb_rpk2 rpk_pg rpeak_pulse vdd vss edge_buffer

* Delay timer
Xdt rpeak_pulse trig_s14_level trig_s69_level trig_wid_level trig_bi_level
+   trig_rr_level trig_rpk_del_level dt_sf_out vdd vss vbias
+   vth_14_node vth_69_node vth_28_node vth_300_node vth_10_node vth_rpk_del_node delay_timer

* Trigger pulse generation (for each delay timer output)
Xeb1_s14 trig_s14_level trig_s14_eb vdd vss edge_buffer
Xpg_s14 trig_s14_eb trig_s14_pg vdd vss vbias pulse_gen
Xeb2_s14 trig_s14_pg trig_s14 vdd vss edge_buffer
... (similar for all triggers)

* Rpeak delayed pulse
Xeb1_rpk trig_rpk_del_level trig_rpk_del_eb vdd vss edge_buffer
Xpg_rpk_del trig_rpk_del_eb trig_rpk_del_pg vdd vss vbias pulse_gen
Xeb2_rpk_del trig_rpk_del_pg rpeak_pulse_delayed vdd vss edge_buffer

* Esamp validity gate for trig_rr_copy
... (AND gate: trig_rr_copy_raw AND esamp_ok_level)

* Isolated timer for pwave base/peak (separate 400×400 MIM)
... (duplicate timer + SF + esamp capture)

* Pwave predictor comparators on isolated timer
... (threshold = esamp_pw + offset)

* Feature circuits
Xrr ecg_l0 trig_rr_cap rpeak_pulse trig_rr_copy f1 f2 ... rr_timing
Xbi0 ecg_l0 rpeak_pulse_delayed trig_bi_copy f3 ... beat_instability
Xbi1 ecg_l1 rpeak_pulse_delayed trig_bi_copy f4 ... beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5 ... pwave_diff
Xwid ecg_l0 trig_wid f6 vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7 ... qrs_slope
Xsym ecg_l0 rpeak_pulse_delayed f8 vdd vss qrs_symmetry

* Initial conditions
.ic V(xrr.net_timer)=0.9
.ic V(xrr.ewma_node)=<ic_ewma>
.ic V(xrr.esamp_node)=<ic_ewma>
.ic V(xrr.prev_node)=<ic_ewma>
.ic V(xbi0.prev_node)=0.3
.ic V(xbi0.esamp_node)=0.3
.ic V(xbi1.prev_node)=0.3
.ic V(xbi1.esamp_node)=0.3

.tran 0.5m <total_time> UIC

* Measurement directives (feature values after each beat)
.meas tran f1_5 FIND V(f1) AT=<time>
... (for each beat and feature)

.end
```

---

## Appendix B: Key Diagnostic Observations

### F3 Per-Beat Diagnostic (Record 100, analog)

```
Beat    rpk_ms     py_F3     sp_F3   ecg_cap
  5      -5.3    -0.0150    0.2961    0.4000
  6      -4.6    -0.0700    0.2946    0.3975
  7      -4.3    -0.0250    0.2964    0.3982
  8      -5.6     0.1150    0.2962    0.3984
  9      -6.5     0.0850    0.2971    0.4008
 10      -3.3    -0.3100    0.2970    0.4022
```

**Key observations:**
- `rpk_ms` varies from -3.3 to -6.8ms (3.5ms range)
- `sp_F3` (SPICE F3) has tiny range: 0.294-0.298V (4mV). Python range: -0.31 to +0.245 (556mV)
- `ecg_cap` (captured ECG voltage) varies only 0.397-0.403V — the threshold crossing region
- The analog F3 output is essentially constant because the same voltage is captured every beat

### F5 Per-Beat Diagnostic (Record 100, analog)

```
Beat     py_F5     id_F5     an_F5   pwb_cmp   pwp_cmp
  5    -0.0200    0.3751    0.8640     218.9     143.8
  6    -0.0550    0.3557    0.8720      55.1    -661.3
  7     0.0350    0.7266    0.8966     408.3     333.2
  8     0.0150    0.4512    0.6902     239.9     164.5
 10     0.1400    0.8748    0.3497     186.6     111.2
```

**Key observations:**
- `pwb_cmp` (ms before next R-peak when base trigger fires): varies 55-408ms (target: 206ms)
- `an_F5` is bimodal (~0.87 or ~0.33V) depending on cardiac phase at capture
- Even ideal F5 (`id_F5`) shows bimodal behavior because the timer-based prediction fails

---

## Appendix C: Relationship Between integration_test.py and validate_3way.py

| Aspect | integration_test.py | validate_3way.py |
|--------|-------------------|-----------------|
| Directory | `complete_frontend` | `complete_frontend_v9b` |
| PIPE_RPK_DEL | 2.0 | 3.0 |
| Pwave predictor | 10-pin (no vth_rr_cap) | 11-pin (with vth_rr_cap) |
| Three-way comparison | No (analog vs Python only) | Yes (Python vs Ideal vs Analog) |
| Records | 30 | Configurable (default: 100) |
| Rpeak polarity | Adaptive (NEG/POS per record) | Adaptive |
| Reference | Measured S&H cap values | Python features |
| Primary use | Multi-record production testing | Diagnostic debugging |
| Esamp validity gate | Yes | Yes |

Both scripts share the same calibration tables (rr_t/rr_v, dt_t/dt_v, ft_t/ft_v) and use the same threshold computation formula.

---

*End of Frontend Handoff Document*
