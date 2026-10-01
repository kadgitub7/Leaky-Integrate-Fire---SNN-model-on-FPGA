# ECG Arrhythmia Classifier Project Goals & Specifications

## 1. ACCURACY TARGETS

### Inter-Patient (AAMI EC57, de Chazal DS1/DS2 split)
- Overall accuracy: >= 93%
- **V sensitivity (ventricular): >= 90%**
- **S sensitivity (supraventricular): maximize (75%+ aspirational, literature ceiling ~40-50%)**
- Report per-class sensitivity AND positive predictivity for all 5 AAMI classes
- Primary metric: per-class recall, NOT overall accuracy (N dominates at 90%)

### Intra-Patient (stratified 80/20 all records)
- Overall accuracy: >= 97%

### Known Difficulty
- S-class inter-patient is the hardest problem. DS2 Patient 232 has 75% of S-beats
  with atypical barely-premature rhythm.
- **Literature ceiling**: purely global classifiers (no patient adaptation) top out
  at ~55-65% S recall under strict DS1/DS2 protocol. De Chazal (2004) original: ~36%.
  Deep residual CNN with focal loss + SMOTE: 50-65%.
- **Patient-adaptive methods** push to 75-79% (De Chazal 2006, Llamedo & Martinez 2011)
  but require expert-annotated beats from the test patient — not feasible in a
  fixed-weight analog classifier.
- **Our target**: maximize S recall without patient adaptation. 40-50% would already
  exceed most published global classifiers. 75%+ is aspirational.
- Track S recall from day one.

---

## 2. ENERGY TARGETS — FULL CHAIN

### Stretch target: < 100 nJ/classification
### Strong paper: < 150 nJ/classification

### Energy Budget Decomposition

The system has three power domains with different duty cycles:

| Domain | Blocks | Duty Cycle | Power Budget | Energy/Beat |
|--------|--------|-----------|-------------|-------------|
| **Always-on** | IA, bandpass filters, R-peak detector | 100% (0.833s at 72bpm) | < 80 nW total | < 66.7 nJ |
| **Event-gated** | Feature extraction (morphology) | ~12% (100ms QRS window) | < 100 nW | < 10 nJ |
| **Wake-on-event** | Classifier (crossbar) | ~0.1% (powers on at R-peak) | N/A (energy) | < 20 nJ |

#### Always-on blocks (~80 nW budget)
These must run continuously to detect the next heartbeat:
- Instrumentation Amplifier (IA): amplifies raw ECG (~30-50 nW target)
- Bandpass filter (0.05-100 Hz): removes baseline wander + HF noise (~10-20 nW)
- R-peak detector: comparator + threshold logic (~5-10 nW)
- Timing circuits: RR interval timers, counters (~5 nW)

#### Event-gated blocks (~100 nW for ~100ms)
Triggered by R-peak detection, active only during QRS analysis:
- QRS width measurement: comparator + timer (~5 nW)
- QRS area integration: gated integrator (~10 nW)
- Slope detection: differentiator + peak-hold (~15 nW)
- Template correlation: mini-crossbar (~30 nW)
- Adaptive reference: EWMA divider (~5 nW)
- Running RR std: variance circuit (~10 nW)

#### Classifier (~20 nJ budget)
Fully powered off between beats. ReRAM weights are nonvolatile — no refresh needed.
- Wakes on R-peak trigger after features are sampled
- Performs inference (single-pass MLP or multi-step SNN)
- Returns to full power-off

### Energy Calculation Formula

```
E_total = E_always_on + E_event_gated + E_classifier

E_always_on = P_always_on * T_beat     (nW * seconds = nJ)
E_event_gated = P_morph * T_qrs_window (nW * seconds = nJ)
E_classifier = E_cells + E_wake + P_periphery * t_window
```

#### Classifier energy breakdown (not just cell reads)
The classifier energy has three components:
- **E_cells**: total_MACs * E_per_MAC (crossbar array reads only)
- **E_wake**: wake-up transient settling charge = I_bias * V_supply * t_settle per block.
  Each block (input buffers, neuron circuits, WTA, bias generation) must settle
  before valid computation begins.
- **P_periphery * t_window**: steady-state periphery power during the inference
  window. Includes input buffers, current-mode neurons, winner-take-all (WTA),
  bias generation circuits.

In published analog classifiers the periphery often dominates the array.
Report with optimistic/nominal/pessimistic values per parameter, and present
full-chain energy as a range across seeds, not a single number.

#### Frontend energy (no AFE)
- `E_always_on = 5 nW * 0.833 s = 4.2 nJ` (timing circuits only)
- `E_event_gated = 95 nW * 0.100 s = 9.5 nJ` (morphology circuits)
- Frontend subtotal: 13.7 nJ

#### AFE energy (parametric, not validated)
The AFE (IA + filters + R-peak) energy must be added on top.
- Supervisor target: 80 nW always-on -> `80 * 0.833 = 66.6 nJ`
- **Two-lead concern**: the current feature set uses two leads, which requires
  two IAs. Two IAs at 30-50 nW each could alone exceed the 80 nW always-on
  budget. A single-lead ablation is included in the bakeoff to quantify the
  accuracy cost of dropping to one IA.
- AFE energy is reported parametrically until validated by circuit design.

---

## 3. HARDWARE PLATFORM

### Crossbar: SkyWater SKY130B ReRAM (sky130_fd_pr_reram, 1T1R)
- **Novelty claim**: First ECG classifier on a foundry ReRAM PDK
  (all prior memristor-ECG work uses lab devices or idealized models)
- Nonvolatile weights: classifier can fully power off between beats
- Energy per MAC: DO NOT hardcode. Compute from first principles:
  - E_cell = V_read^2 * G * t_read, where G drawn from LRS distribution (30-120 uS)
  - Sweep t_read from 100 ns to 1 us
  - **Validation**: netlist a 4x8 1T1R crossbar in ngspice using the actual
    sky130_fd_pr_reram cell model, apply read pulses, integrate supply current.
    This also gives read-disturb margins for free. (1-2 day task)
  - The validated E_per_MAC becomes the anchor for all energy estimates.
  - Preliminary estimate: ~2 pJ range, but must be confirmed by simulation.
  - SET energy: low pJ range per write (1.7V WL / 2.4V BL, 1us pulse)
  - Read energy: lower (0.1-0.2V read voltage)
- **Weight precision: 2-3 bits per cell max** (SKY130B HfO2 filamentary RRAM)
  - PDK documents up to 3 bits (8 levels) demonstrated
  - 4-bit (16 levels) NOT feasible with acceptable margins
  - LRS range: 30-120 uS; HRS: <0.1 uS (>10 MOhm)
  - Multi-level requires ISPVA (incremental step pulse with verify)
  - **Decision**: use 2-bit QAT (4 levels per cell), or 2 binary cells per
    4-bit weight. Bakeoff will test both 2-bit and 4-bit QAT.
- Key non-idealities to model:
  - Stuck-at faults: forming yield 92.7-99.8%, reset yield 90.8-99.4%
  - Conductance drift: intermediate states drift over hours
  - Write variability: device-to-device and cycle-to-cycle
  - Retention: challenges at high temperature with multi-bit storage

### Process: SkyWater 130nm (SKY130B)
- No external ADC — fully analog signal chain
- Analog frontend → analog feature extraction → crossbar classifier

### Closest competitor: Cao et al., BioCAS 2023
- Full citation: T. Cao, Z. Zhang, W. L. Goh, C. Liu, Y. Zhu, and Y. Gao,
  "ECG Classification using Binary CNN on RRAM Crossbar with Nonidealities-
  Aware Training, Readout Compensation and CWT Preprocessing," IEEE BioCAS
  2023, Toronto. DOI: 10.1109/BioCAS58349.2023.10389002
- Result: 98.9% on MIT-BIH (10-class), only 0.7% drop from FP32 baseline
- Key techniques: (1) non-ideality-aware training injection, (2) in-situ
  readout compensation for IR drop, (3) binary weights (1 cell/weight,
  half crossbar area), (4) CWT preprocessing
- Differences from our work: they use binary CNN (not SNN), CWT features
  (not cardiologist-curated), and unspecified RRAM (not foundry PDK)
- **Takeaway**: binary weights avoid multi-level non-ideality problems
  entirely. We should test binary/2-bit QAT in the bakeoff.

Sources:
- SKY130 ReRAM specs: sky130-fd-pr-reram.readthedocs.io
- GitHub: google/skywater-pdk-libs-sky130_fd_pr_reram
- Milo et al., TED 2021 (multilevel RRAM program/verify)
- Cao NTU thesis: dr.ntu.edu.sg

---

## 4. FEATURE SET — CARDIOLOGIST-DRIVEN

16 features from ~11 physical analog circuits:

| # | Feature | Circuit | Domain | Est. Power |
|---|---------|---------|--------|-----------|
| 1 | pre_rr | Timer (cap + comparator) | Always-on | ~1 nW (shared) |
| 2 | post_rr | Same timer | Always-on | — |
| 3 | rr_ratio | Analog divider | Always-on | — |
| 4 | rr_asymmetry | Same divider | Always-on | — |
| 5 | compensatory_ratio | Timer + divider | Always-on | — |
| 6 | rr_std_10 | Running variance (10-tap) | Always-on | ~4 nW |
| 7 | qrs_width_L0 | Comparator + timer | Event-gated | ~5 nW |
| 8 | qrs_width_L1 | Same, Lead II | Event-gated | — |
| 9 | qrs_area_L0 | Gated integrator | Event-gated | ~10 nW |
| 10 | qrs_area_L1 | Same, Lead II | Event-gated | — |
| 11 | max_slope_L0 | Differentiator + peak-hold | Event-gated | ~15 nW |
| 12 | max_slope_L1 | Same, Lead II | Event-gated | — |
| 13 | rel_area_L0 | Divider (area/EWMA) | Event-gated | ~5 nW |
| 14 | rel_area_L1 | Same, Lead II | Event-gated | — |
| 15 | templ_corr_L0 | Mini-crossbar correlator | Event-gated | ~30 nW |
| 16 | templ_corr_L1 | Same, Lead II | Event-gated | — |

Total: ~5 nW always-on (timing) + ~95 nW event-gated (morphology, active ~100ms/beat)

---

## 5. BEHAVIORAL MODEL REQUIREMENTS

Every behavioral model (SNN or MLP) MUST include:

### a) Quantization-Aware Training (QAT)
- Quantize weights during forward pass, restore for gradient (STE)
- Proven: 4-bit QAT is beneficial regularization (+2.8% vs 32-bit)
- **SKY130B supports 2-3 bits/cell max** bakeoff tests both:
  - 2-bit QAT (4 levels per cell, 1 cell/weight simplest)
  - 4-bit QAT (16 levels, requires 2 cells/weight doubles array area)
  - Binary QAT (2 levels, 1 cell/weight Cao et al. approach)

### b) Analog Non-Ideality Injection
Model the full analog chain imprecisions:
- **Weight noise**: multiplicative Gaussian on quantized weights (~5%)
- **Feature noise**: per-feature multiplicative + additive noise
  - Sample-and-hold offset (systematic per-channel)
  - Ramp-slope variation in timing measurements
  - Capacitor droop on EWMA template storage
  - Comparator offset in peak/width detection
- **Implementation**: noise injection layer after feature extraction,
  sweep noise levels 1%, 2%, 5%, 10% to find degradation curve

### c) Full-Chain Energy Reporting
Every experiment must report:
- **Classifier energy as three components**:
  - E_cells: MACs * E_per_MAC (from ngspice-validated crossbar simulation)
  - E_wake: wake-up transient (settling charge per block)
  - E_periphery: P_periphery * t_inference_window
- Frontend energy (5 nW * 0.833s + 95 nW * 0.100s = 13.7 nJ)
- AFE estimate (parametric until circuit design; default 80 nW * 0.833s = 66.6 nJ)
- **Total full-chain energy as a range** (optimistic/nominal/pessimistic)
  across the 5-seed evaluation, not a single number

### d) Per-Class Metrics
Always report for inter-patient:
- Per-class sensitivity (recall) for N, S, V, F, Q
- Per-class positive predictivity (precision)
- Confusion matrix
- Overall accuracy

### e) Reproducibility
- Fixed random seeds with multi-seed evaluation (5 seeds minimum)
- Report mean +/- std across seeds

---

## 6. ARCHITECTURE BAKEOFF — SNN vs MLP

### Purpose
Determine whether SNN temporal integration justifies the 12-20x crossbar
activation cost over a single-pass MLP, given that features already encode
timing information.

### Controlled Variables (same for both)
- 16 features (proven cardiologist set)
- AAMI 5-class, de Chazal DS1/DS2 inter-patient split
- SMOTE 0.33 oversampling
- Focal loss gamma=2.0
- 4-bit QAT with STE
- Feature noise injection (sweep 0%, 2%, 5%)
- Class weighting via cls_power sweep
- 200 epochs, batch_size=128, Adam lr=1e-3
- 5-seed evaluation

### SNN Configurations
Architecture: input -> BN -> h1 RLeaky -> h2 RLeaky -> 5 Leaky
- Width sweep: (h1,h2) in [(10,5), (20,10), (40,20), (48,24), (80,40)]
- Timestep sweep: [8, 12, 16, 20]
- With and without lateral inhibition (top_k = h1//2)
- Prediction: spike-count (sum over timesteps)

Energy: `fc1 + (rec1 + mid + rec2 + out) * steps * effective_fire_rate`
Each timestep activates the crossbar once.

### MLP Configurations
Architecture: input -> BN -> h1 ReLU -> h2 ReLU -> 5 Softmax
- Width sweep: (h1,h2) in [(10,5), (20,10), (40,20), (48,24), (80,40)]
- Single-pass (no timesteps)

Energy: `fc1 + mid + out` (each layer activated once, all neurons fire)
One crossbar activation total.

### Single-Lead Ablation
Both SNN and MLP must also be tested with single-lead features only
(L0 features: indices 0-5 timing + 6,8,10,12,14 morphology = 11 features).
Rationale: two leads require two IAs, which may alone exceed the 80 nW
always-on budget. The ablation quantifies the accuracy cost of dropping
to one IA and halving the morphology circuit count.

### EWMA Template Framing
The EWMA template (features 15-16: templ_corr) is unsupervised patient
adaptation: it updates a running template of the patient's normal QRS
without requiring expert annotation. This is allowed and may be our best
lever for S-recall (template correlation detects morphological deviations
that timing alone misses). However, we must frame this carefully in the
paper as unsupervised adaptation, distinct from the supervised patient-
adaptive methods (De Chazal 2006) that require labeled test-patient beats.

### Decision Criteria
1. Per-class recall (V >= 90%, S maximized)
2. Overall inter-patient accuracy (>= 93% target)
3. Full-chain energy as a range (< 100 nJ stretch, < 150 nJ strong)
4. Robustness to feature noise (2%, 5%)
5. Parameter count (affects crossbar area)
6. Single-lead vs dual-lead accuracy tradeoff

---

## 7. NEXT STEPS (ORDERED)

1. **ngspice crossbar validation** (1-2 days): netlist a 4x8 1T1R crossbar
   using sky130_fd_pr_reram cell models. Apply read pulses (V_read 0.1-0.2V,
   t_read 100ns-1us), integrate supply current to get validated E_per_MAC.
   Also extracts read-disturb margins. This anchors all energy estimates.
2. **Architecture bakeoff** SNN vs MLP, including single-lead ablation
3. **Periphery energy model**: estimate E_wake and P_periphery for input
   buffers, neurons, WTA, bias generation. Report classifier energy as
   E_cells + E_wake + P_periphery * t_window with opt/nom/pess values.
4. **Feature noise sweep** find accuracy degradation curve at 1-10%
5. **AFE power estimation** design IA/filter/R-peak blocks, validate 80 nW.
   Critical question: can two IAs fit in the 80 nW budget? If not,
   single-lead results from step 2 determine whether we can drop to one.
6. **Commit to one architecture** based on bakeoff results
7. **Circuit design** crossbar layout on SKY130B
8. **Pull Liu et al., JSSC 2025** (90 nJ, 96.6% inter, 5-class): check what
   their energy excludes. Primary accuracy competitor, ahead of Cao.
9. **Read Cao et al., BioCAS 2023** map their techniques to our architecture

---

## 8. BENCHMARK COMPETITORS

| Reference | Energy | Node | Method | Accuracy | Notes |
|-----------|--------|------|--------|----------|-------|
| Chollet et al., EMBC 2017 | 234 pJ cls | 65nm | Analog assoc. | 93.6% intra, 3-class | Cls only |
| Qin et al., ASPDAC 2025 | 0.37 nJ cls | 40nm | Sparse CNN | 99.95% intra, 2-class | IEGM, not ECG |
| Liu et al., JSSC 2025 | 90 nJ (cls+feat) | 55nm | Event CNN | 96.6% inter, 5-class | **Primary competitor**, no AFE |
| Janveja et al., TCAS-II 2022 | 0.785 uW | 180nm | DNN | 97.01% intra, 8-class | Power, not energy |
| Zhang et al., TBCAS 2024 | 150 nJ | 65nm | Multistage NN | 98.59% intra, 5-class | Cls only |
| Abubakar et al., TBCAS 2022 | 746 nW | 65nm | Ternary NN | 99.3% intra, 3-class | Power, not energy |
| Chu et al., TBCAS 2022 | 750 nJ | 40nm | SNN | 98.22% intra, 5-class | Cls only |
| **This work (target)** | **<100 nJ full** | **SKY130** | **Analog SNN/MLP** | **93%+ inter, 5-class** | **Full chain, ReRAM PDK** |

Our novelty: (1) full-chain energy including AFE, (2) foundry ReRAM PDK, (3) ADC-free analog pipeline.

### Liu et al., JSSC 2025 - Primary Accuracy Competitor
- Full citation: J. Liu et al., "A High-Accuracy and Ultra-Energy-Efficient
  Cardiac Arrhythmia Classification Processor for Wearable Intelligent ECG
  Monitoring," IEEE JSSC, 2025. DOI: 10.1109/JSSC.2025.3555512
- 55nm CMOS digital processor, 90 nJ per classification
- 98.7% intra-patient, 96.6% inter-patient, MIT-BIH 5-class
- Key techniques: (1) heartbeat difference-based classification,
  (2) event-driven NN with shared feature extraction (60.5% MAC reduction),
  (3) adaptive NN wake-up (skips normal beats)
- **Energy scope: 90 nJ covers the digital classification processor only
  (feature extraction + NN inference). Does NOT include analog front-end
  (IA, filters, ADC).** This is standard for JSSC processor papers.
- Ahead of Cao on accuracy (96.6% vs 98.9% but Cao is 10-class intra).
  Liu is the only competitor with inter-patient 5-class results.
- **Comparison to our work**: our full-chain energy (including AFE) should
  be compared to their 90 nJ processor-only + their unstated AFE cost.
  On a like-for-like basis (classifier + features only), our target is
  ~3-10 nJ (MLP) to ~10-20 nJ (SNN), well below their 90 nJ.
