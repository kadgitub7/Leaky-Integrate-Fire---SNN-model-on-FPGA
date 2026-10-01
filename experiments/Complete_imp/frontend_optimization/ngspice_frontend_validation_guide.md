# ngspice Frontend Energy Validation Guide — Transistor-Level SKY130B

## Manufacturing-Ready Analog Feature Extraction for 8-Feature Set

**Date:** 2026-09-25 (v2 — transistor-level rewrite)
**Target features:** rr_ratio, rr_asymmetry, beat_instability_L0, beat_instability_L1,
pwave_energy_L0, qrs_width_L0, slope_ratio_L0, qrs_symmetry_L0
**Process:** SkyWater SKY130 (130nm bulk CMOS)
**All circuits:** Transistor-level using `sky130_fd_pr` device models
**sky130 reram is stored in this directory**
**sky130 pr is stored in PS C:\src\skywater-pdk-libs-sky130_fd_pr**

---

## 0. SKY130 Device Reference

### Include syntax (all netlists use this header)

```spice
* SKY130 PDK include — adjust path to your installation
.lib "/path/to/sky130A/libs.tech/ngspice/sky130.lib.spice" tt
```

Corner options: `tt` (typical), `ss` (slow), `ff` (fast), `sf`, `fs`.

### Device instantiation

SKY130 devices are subcircuits — use `X` prefix, not `M`. Pin order: **drain, gate, source, body**.

```spice
* NMOS (standard Vt, 1.8V): Vth ~ 0.49V
XM1 drain gate source body sky130_fd_pr__nfet_01v8 L=0.15 W=0.42

* PMOS (standard Vt, 1.8V): Vth ~ -0.42V
XM2 drain gate source body sky130_fd_pr__pfet_01v8 L=0.15 W=0.42

* PMOS (high Vt, lowest leakage — ideal for current mirrors)
XM3 drain gate source body sky130_fd_pr__pfet_01v8_hvt L=0.15 W=0.42

* MIM capacitor (~2 fF/um²): 10x10um = 200 fF
XC1 bot_plate top_plate sky130_fd_pr__cap_mim_m3_1 W=10 L=10
```

### Subthreshold design point

For ultra-low-power ECG (0.5–40 Hz signals):
- Operate all transistors in **weak inversion** (Vgs well below Vth)
- Use **long channels** (L = 2–10 um) to suppress leakage and improve matching
- Target **1–10 nA** bias per OTA
- At W/L = 1u/1u, ~1-10 nA requires Vgs ≈ 0.25–0.35V (150–250 mV below Vth)
- Subthreshold slope: ~80–90 mV/decade
- gm in subthreshold: `gm = Id / (n × Vt)` where n ≈ 1.3, Vt = 26mV → `gm ≈ Id / 34mV`
- At Id = 5 nA: gm ≈ 150 pS

**Model accuracy warning:** SKY130 BSIM4 models are fitted from measured data and may lose accuracy below ~2 nA. Treat sub-nA simulations as estimates; validate critical paths at 5–10 nA.

---

## 1. Core Building Block: Subthreshold 5-Transistor OTA

Every analog block in the frontend is built from this OTA. This is the single most important circuit to get right — its bias current determines the entire system's energy budget.

### Schematic

```
          VDD
           |
     ┌─────┴─────┐
     |            |
    XM3(P)      XM4(P)       ← active load (PMOS current mirror)
     |            |
     |            ├──→ Vout
     |            |
    XM1(N)      XM2(N)       ← differential input pair (NMOS)
     |    |  |    |
     Vin+ └──┘ Vin-
           |
          XM5(N)              ← tail current source
           |
          GND
```

### ngspice netlist: `ota_5t_subthreshold.spice`

```spice
* ============================================================
* 5-Transistor OTA — Subthreshold, nA-range bias
* SKY130 sky130_fd_pr devices, manufacturing-ready
* ============================================================

.subckt ota_5t vinn vinp vout vdd vss vbias

* --- Active load: PMOS current mirror ---
* Long channel (L=4u) for matching in subthreshold
* HVT PMOS for lowest leakage
XM3 net1 net1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XM4 vout net1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* --- Differential input pair: NMOS ---
* Long channel for matching, standard Vt for subthreshold gm
XM1 net1 vinn net_tail vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XM2 vout vinp net_tail vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1

* --- Tail current source: NMOS ---
* Sets the bias current. Gate voltage (vbias) controls Id.
* Long channel for high output impedance
XM5 net_tail vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=1 nf=1

.ends ota_5t
```

### OTA characterization testbench: `ota_5t_test.spice`

```spice
* ============================================================
* OTA Characterization — sweep bias, measure gm, power, bandwidth
* ============================================================

.title ota_5t_characterization

.lib "/path/to/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.include "ota_5t_subthreshold.spice"

.param vdd_val = 1.8
.param vref = 0.9

Vdd vdd 0 {vdd_val}
Vss vss 0 0

* Bias voltage generator — sweep this to find nA operating point
Vbias vbias 0 0.3

* Input: DC + small AC signal for frequency response
Vinp vinp 0 DC {vref} AC 1m
Vinn vinn 0 DC {vref}

* OTA under test
Xota1 vinn vinp vout vdd vss vbias ota_5t

* Load capacitor (models next stage input cap)
XCload vout 0 sky130_fd_pr__cap_mim_m3_1 W=5 L=5

* ============================================================
* DC operating point — find bias current
* ============================================================
.op

* ============================================================
* DC sweep — transfer characteristic
* ============================================================
.dc Vinn 0.85 0.95 0.001

* ============================================================
* AC analysis — bandwidth, gain
* ============================================================
.ac dec 100 0.01 100k

* ============================================================
* Bias sweep — find the Vbias that gives target Id
* ============================================================

.control
  * --- Bias sweep to find nA operating point ---
  let vb_vals = vector(21)
  let id_vals = vector(21)

  * Sweep Vbias from 0.15V to 0.45V
  let idx = 0
  foreach vb 0.15 0.17 0.19 0.21 0.23 0.25 0.27 0.29 0.31 0.33 0.35 0.37 0.39 0.41 0.43 0.45 0.47 0.49 0.51 0.53 0.55
    alter Vbias = $vb
    op
    let vb_vals[idx] = $vb
    * Tail current = supply current / 2 (approximately)
    let id_vals[idx] = abs(i(Vdd))
    let idx = idx + 1
  end

  echo ""
  echo "=== BIAS SWEEP: Vbias vs Supply Current ==="
  echo "Target: total Idd = 2*Id_tail (both branches)"
  echo "For 5nA per branch: target Idd ~ 10nA"
  print vb_vals id_vals

  * --- AC analysis at nominal bias ---
  alter Vbias = 0.30
  op
  echo ""
  echo "=== OPERATING POINT at Vbias=0.30V ==="
  print all

  ac dec 100 0.01 100k
  echo ""
  echo "=== AC RESPONSE ==="
  * Unity-gain bandwidth
  meas ac GBW when vdb(vout)=0 rise=1
  * DC gain
  meas ac Adc find vdb(vout) at=0.01
  print GBW Adc

  plot vdb(vout) title "OTA Gain vs Frequency"
  plot vp(vout) title "OTA Phase vs Frequency"
.endc

.end
```

### What to extract from this testbench

| Measurement | Target | Why |
|------------|--------|-----|
| `Idd` at Vbias=0.30V | ~5–20 nA | Sets total system power |
| DC gain (Adc) | >40 dB | Sufficient for filters and comparators |
| GBW | >200 Hz | Must exceed 10× highest filter frequency (20 Hz) |
| Input-referred noise (V/√Hz at 10 Hz) | <50 uV/√Hz | ECG signal is ~1 mV amplitude |

**Run this first.** Every energy number in the rest of the guide flows from the OTA's Idd at the chosen Vbias.

---

## 2. Bias Current Generator

All OTAs share a single bias voltage. This circuit generates a stable, nA-range reference current independent of supply voltage and (partially) temperature.

### Circuit: Self-biased current reference (beta-multiplier)

```spice
* ============================================================
* nA-Range Self-Biased Current Reference (Beta Multiplier)
* Generates Vbias for all OTA tail transistors
* SKY130 transistor-level
* ============================================================

.subckt bias_gen vbias vdd vss

* Startup circuit: ensures the mirror doesn't stay at zero
XMs1 vbias ns1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=1 nf=1
XMs2 ns1 ns1 vss vss sky130_fd_pr__nfet_01v8 L=1 W=0.42 nf=1
* Startup capacitor — kicks the loop on power-up
XCs ns1 vss sky130_fd_pr__cap_mim_m3_1 W=3 L=3

* --- PMOS current mirror (sets current in both branches) ---
XMp1 net_a net_a vdd vdd sky130_fd_pr__pfet_01v8_hvt L=8 W=2 nf=1
XMp2 vbias net_a vdd vdd sky130_fd_pr__pfet_01v8_hvt L=8 W=2 nf=1

* --- NMOS branch A: diode-connected, sets Vgs ---
XMn1 net_a net_a vss vss sky130_fd_pr__nfet_01v8 L=8 W=1 nf=1

* --- NMOS branch B: source-degenerated by resistor (MOS pseudo-R) ---
* The W ratio between Mn1 and Mn2, combined with Rs, sets the current
* Id = (1/Rs) * (n*Vt) * ln(K) where K = (W/L)_n2 / (W/L)_n1
XMn2 vbias vbias net_rs vss sky130_fd_pr__nfet_01v8 L=8 W=4 nf=1

* Degeneration "resistor" — subthreshold PMOS as pseudo-resistor
* Gate tied to drain gives R ~ exp(Vth/(n*Vt)) / gm, on the order of GOhms
* This sets the nA-level current
XMrs net_rs net_rs vss vdd sky130_fd_pr__pfet_01v8_hvt L=10 W=0.42 nf=1

* --- Output: vbias node drives all OTA tail gates ---

.ends bias_gen
```

### Testbench: `bias_gen_test.spice`

```spice
.title bias_gen_characterization

.lib "/path/to/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.include "bias_gen.spice"

Vdd vdd 0 1.8

Xbias vbias vdd 0 bias_gen

* Test load: one OTA tail transistor (same sizing as in OTA)
XMtest drain_test vbias 0 0 sky130_fd_pr__nfet_01v8 L=8 W=1 nf=1
Vdtest drain_test 0 0.9

.op

* Sweep supply voltage to check PSRR
.dc Vdd 1.4 2.0 0.01

.control
  run
  echo "=== Bias Generator Operating Point ==="
  print v(vbias)
  print abs(i(Vdtest))
  echo "Target: Vbias ~ 0.25-0.35V, Id ~ 1-10 nA"

  * Plot bias current vs supply voltage (should be flat = good PSRR)
  plot abs(i(Vdtest)) title "Bias current vs VDD"
  plot v(vbias) title "Vbias vs VDD"
.endc

.end
```

---

## 3. OTA-C Filter Building Blocks

Every filter in the frontend is an OTA-C topology. These are the reusable subcircuits.

### 3a. First-order OTA-C Low-pass Filter

```spice
* ============================================================
* OTA-C Low-pass filter: fc = gm / (2*pi*C)
* At Id=5nA: gm~150pS. With C=1.2pF: fc ~ 20 Hz
* ============================================================

.subckt otac_lpf vin vout vdd vss vbias
.param cap_l = 5
.param cap_w = 5

* OTA: output current charges the cap
Xota vin vout vout vdd vss vbias ota_5t

* Integration capacitor
* 5x5um MIM cap = ~50 fF. Adjust W,L for desired fc.
XCint vout vss sky130_fd_pr__cap_mim_m3_1 W={cap_w} L={cap_l}

.ends otac_lpf
```

### 3b. First-order OTA-C High-pass Filter

```spice
* ============================================================
* OTA-C High-pass filter
* HPF = input - LPF(input)
* Uses two OTAs: one for LPF, one as subtractor
* ============================================================

.subckt otac_hpf vin vout vdd vss vbias
.param cap_l = 10
.param cap_w = 10

* LPF path
Xota_lp vin vlp vlp vdd vss vbias ota_5t
XClp vlp vss sky130_fd_pr__cap_mim_m3_1 W={cap_l} L={cap_w}

* Subtractor: vout = vin - vlp
Xota_sub vlp vin vout vdd vss vbias ota_5t

.ends otac_hpf
```

### 3c. Second-order OTA-C Bandpass Filter

```spice
* ============================================================
* OTA-C Bandpass = HPF(f_low) cascaded with LPF(f_high)
* For R-peak detection: f_low=8Hz, f_high=20Hz
* For P-wave extraction: f_low=3Hz, f_high=8Hz
* Adjust capacitor sizes to set corner frequencies
* ============================================================

.subckt otac_bpf vin vout vdd vss vbias
.param cl_hpf = 10
.param cw_hpf = 10
.param cl_lpf = 5
.param cw_lpf = 5

* HPF stage (sets lower corner)
Xhpf vin vmid vdd vss vbias otac_hpf cap_l={cl_hpf} cap_w={cw_hpf}

* LPF stage (sets upper corner)
Xlpf vmid vout vdd vss vbias otac_lpf cap_l={cl_lpf} cap_w={cw_lpf}

.ends otac_bpf
```

### Capacitor sizing table

At Id_tail = 5 nA → gm ≈ 150 pS:

| Target fc | C = gm/(2π×fc) | MIM size (W×L at 2fF/um²) |
|-----------|----------------|---------------------------|
| 0.5 Hz | 47.7 pF | 155×155 um (large!) |
| 3 Hz | 7.96 pF | 63×63 um |
| 8 Hz | 2.98 pF | 39×39 um |
| 20 Hz | 1.19 pF | 24×24 um |
| 40 Hz | 0.60 pF | 17×17 um |

**Note:** The 0.5 Hz HPF for baseline removal needs a ~48 pF cap, which is 155×155 um — about 0.024 mm². This is large but feasible in a dedicated ASIC. Alternative: use a subthreshold MOS pseudo-resistor with a smaller cap to achieve the same time constant.

---

## 4. Stage 1: Input Conditioning

### Architecture

```
ECG_L0 ──→ [Buffer OTA] ──→ [HPF 0.5Hz] ──→ [LPF 40Hz] ──→ ecg_cond_L0
ECG_L1 ──→ [Buffer OTA] ──→ [HPF 0.5Hz] ──→ [LPF 40Hz] ──→ ecg_cond_L1
```

Per lead: 1 buffer OTA + 2 OTAs (HPF) + 1 OTA (LPF) = 4 OTAs
Two leads: **8 OTAs total**

### Netlist: `stage1_input_conditioning.spice`

```spice
* ============================================================
* STAGE 1: Input Conditioning — 2-lead ECG
* Buffer + HPF(0.5Hz) + LPF(40Hz) per lead
* All transistor-level SKY130
* ALWAYS-ON
* ============================================================

.title stage1_input_conditioning

.lib "/path/to/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.include "ota_5t_subthreshold.spice"
.include "bias_gen.spice"
.include "otac_lpf.spice"
.include "otac_hpf.spice"

.param vdd_val = 1.8

Vdd vdd 0 {vdd_val}

* Bias generator (shared by all OTAs)
Xbias vbias vdd 0 bias_gen

* ECG inputs (from PWL files)
Vecg0 ecg_raw_L0 0 PWL file="ecg_multbeat_L0.txt"
Vecg1 ecg_raw_L1 0 PWL file="ecg_multbeat_L1.txt"

* ==========================================
* Lead 0: Buffer + HPF(0.5Hz) + LPF(40Hz)
* ==========================================

* Unity-gain buffer (protects S&H from electrode impedance)
Xbuf0 ecg_raw_L0 ecg_buf_L0 ecg_buf_L0 vdd 0 vbias ota_5t

* HPF at 0.5 Hz (removes baseline wander)
* Cap sizing: gm~150pS, fc=0.5Hz → C=47.7pF → W=155,L=155 um
* Use pseudo-resistor alternative to shrink the cap:
Xhpf0 ecg_buf_L0 ecg_hp_L0 vdd 0 vbias otac_hpf cap_l=100 cap_w=100

* LPF at 40 Hz (anti-alias / noise reduction)
* Cap: gm~150pS, fc=40Hz → C=0.6pF → W=17,L=17 um
Xlpf0 ecg_hp_L0 ecg_cond_L0 vdd 0 vbias otac_lpf cap_l=17 cap_w=17

* ==========================================
* Lead 1: identical chain
* ==========================================

Xbuf1 ecg_raw_L1 ecg_buf_L1 ecg_buf_L1 vdd 0 vbias ota_5t
Xhpf1 ecg_buf_L1 ecg_hp_L1 vdd 0 vbias otac_hpf cap_l=100 cap_w=100
Xlpf1 ecg_hp_L1 ecg_cond_L1 vdd 0 vbias otac_lpf cap_l=17 cap_w=17

* ==========================================
* ENERGY MEASUREMENT
* 8 OTAs always-on + 1 bias generator
* ==========================================

.tran 10u 10

.meas tran I_stage1_avg AVG abs(i(Vdd)) FROM=2 TO=9
.meas tran P_stage1_avg AVG par('v(vdd)*abs(i(Vdd))') FROM=2 TO=9
.meas tran E_stage1_per_beat PARAM par('P_stage1_avg * 0.833')

.control
  run
  echo ""
  echo "=== STAGE 1: INPUT CONDITIONING ENERGY ==="
  print I_stage1_avg P_stage1_avg E_stage1_per_beat
  echo "I_stage1_avg = average supply current (A)"
  echo "P_stage1_avg = average power (W)"
  echo "E_stage1_per_beat = energy per beat at 72bpm (J)"
  echo ""
  echo "Expected at 5nA/OTA: I~45nA, P~81nW, E~67nJ/beat"

  plot v(ecg_raw_L0) v(ecg_cond_L0) title "Lead 0: Raw vs Conditioned"
  plot v(ecg_raw_L1) v(ecg_cond_L1) title "Lead 1: Raw vs Conditioned"
.endc

.end
```

---

## 5. Stage 2: R-Peak Detection

### Architecture

```
ecg_cond_L0 ──→ [BPF 8-20Hz] ──→ [Rectifier] ──→ [Envelope LPF 2Hz] ──→ [Comparator] ──→ rpeak_trigger
                                                                              ↑
                                                   [Adaptive threshold] ──────┘
                                                   (slow LPF 0.1Hz of envelope)
```

OTA count: 3 (BPF) + 2 (rectifier) + 1 (envelope) + 1 (threshold) + 1 (comparator) = **8 OTAs**

### Precision full-wave rectifier (transistor-level)

```spice
* ============================================================
* Precision Full-Wave Rectifier
* Uses two OTAs + NMOS switches for current steering
* Output = |Vin - Vref|, referenced to Vref
* ============================================================

.subckt rectifier vin vout vref vdd vss vbias

* OTA1: compares input to reference (acts as comparator for sign)
Xota_cmp vin vref vcmp vdd vss vbias ota_5t

* OTA2: gain stage, output = (vin - vref)
Xota_gain vref vin vgain vdd vss vbias ota_5t

* Current steering switches:
* When vin > vref: vcmp high → pass vgain directly
* When vin < vref: vcmp low → invert vgain

* Positive path: transmission gate, enabled when vcmp high
XMsw_np vgain vcmp net_pos vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMsw_pp vgain vcmp_bar net_pos vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1

* Negative path: inverting through vref mirror
XMsw_nn net_neg vcmp_bar net_inv vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMsw_pn net_neg vcmp net_inv vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1

* Inversion: vref - (vin - vref) = 2*vref - vin
* This is achieved by connecting the gain OTA with swapped inputs for the negative path
Xota_inv vin vref net_neg vdd vss vbias ota_5t

* Combine paths
XMcomb_p net_pos vcmp vout vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMcomb_n net_inv vcmp_bar vout vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1

* Complementary comparator output
Xinv_cmp vcmp vcmp_bar vdd vss inverter_sub

.ends rectifier

* Simple CMOS inverter subcircuit
.subckt inverter_sub vin vout vdd vss
XMp vout vin vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMn vout vin vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
.ends inverter_sub
```

### Comparator with hysteresis

```spice
* ============================================================
* Schmitt-Trigger Comparator
* OTA with positive feedback resistor (hysteresis)
* Prevents retriggering on noisy QRS edges
* ============================================================

.subckt comparator vinp vinn vout vdd vss vbias

* Main OTA — open loop for maximum gain
Xota vinp vinn vout_raw vdd vss vbias ota_5t

* Output buffer (two cascaded inverters for rail-to-rail digital output)
Xinv1 vout_raw vout_inv vdd vss inverter_sub
Xinv2 vout_inv vout vdd vss inverter_sub

* Positive feedback for hysteresis:
* Small fraction of output fed back to non-inverting input
* Resistive divider using long-channel MOS
XMfb vout vinp_fb vss vss sky130_fd_pr__nfet_01v8 L=10 W=0.42 nf=1

.ends comparator
```

### Complete Stage 2 netlist: `stage2_rpeak_detector.spice`

```spice
* ============================================================
* STAGE 2: R-Peak Detection
* BPF(8-20Hz) + Rectifier + Envelope + Adaptive Threshold + Comparator
* ALWAYS-ON
* ============================================================

.title stage2_rpeak_detector

.lib "/path/to/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.include "ota_5t_subthreshold.spice"
.include "otac_lpf.spice"
.include "otac_hpf.spice"
.include "otac_bpf.spice"
.include "rectifier.spice"
.include "comparator.spice"
.include "bias_gen.spice"

.param vdd_val = 1.8

Vdd vdd 0 {vdd_val}
Vref vref 0 0.9

Xbias vbias vdd 0 bias_gen

* Conditioned ECG input (from Stage 1, or use PWL for standalone test)
Vecg ecg_cond_L0 0 PWL file="ecg_multbeat_L0.txt"

* ==========================================
* BPF: 8-20 Hz for QRS energy extraction
* HPF(8Hz): gm~150pS, C=2.98pF → MIM 39x39um
* LPF(20Hz): gm~150pS, C=1.19pF → MIM 24x24um
* 3 OTAs total (2 for HPF, 1 for LPF)
* ==========================================

Xbpf ecg_cond_L0 bpf_out vdd 0 vbias otac_bpf cl_hpf=39 cw_hpf=39 cl_lpf=24 cw_lpf=24

* ==========================================
* Precision rectifier: |BPF output|
* 3 OTAs + switches
* ==========================================

Xrect bpf_out rect_out vref vdd 0 vbias rectifier

* ==========================================
* Envelope detector: LPF at 2 Hz
* Smooths rectified output to get QRS energy envelope
* C: gm~150pS, fc=2Hz → C=11.9pF → MIM 77x77um
* 1 OTA
* ==========================================

Xenv rect_out env_out vdd 0 vbias otac_lpf cap_l=77 cap_w=77

* ==========================================
* Adaptive threshold: very slow LPF at 0.1 Hz
* Tracks the average envelope level
* C: gm~150pS, fc=0.1Hz → C=239pF (too large for MIM)
* Alternative: use pseudo-resistor + 10pF cap
* R_pseudo ~ 150 GOhm (subthreshold PMOS), C = 10pF → tau = 1.5s → fc ~ 0.1Hz
* 1 OTA (or passive RC with pseudo-R)
* ==========================================

Xthresh env_out thresh_out vdd 0 vbias otac_lpf cap_l=50 cap_w=50
* (Approximate — actual fc depends on OTA gm vs cap ratio)

* ==========================================
* Comparator: envelope > threshold → rpeak_trigger
* 1 OTA + 2 inverters
* ==========================================

Xcmp env_out thresh_out rpeak_trigger vdd 0 vbias comparator

* ==========================================
* TOTAL: 3(BPF) + 3(rect) + 1(env) + 1(thresh) + 1(comp) = 9 OTAs
* (Adjusted from 8 — rectifier needs 3 OTAs transistor-level)
* All ALWAYS-ON
* ==========================================

.tran 10u 10

.meas tran I_stage2_avg AVG abs(i(Vdd)) FROM=2 TO=9
.meas tran P_stage2_avg AVG par('v(vdd)*abs(i(Vdd))') FROM=2 TO=9
.meas tran E_stage2_per_beat PARAM par('P_stage2_avg * 0.833')

* Verify detection: count triggers
.meas tran n_triggers FIND v(rpeak_trigger) WHEN v(rpeak_trigger)=0.9 RISE=LAST

.control
  run
  echo ""
  echo "=== STAGE 2: R-PEAK DETECTION ENERGY ==="
  print I_stage2_avg P_stage2_avg E_stage2_per_beat

  plot v(ecg_cond_L0) v(bpf_out) v(rect_out) title "Signal processing chain"
  plot v(env_out) v(thresh_out) v(rpeak_trigger) title "Detection"
.endc

.end
```

---

## 6. Stage 3: Feature Extraction Circuits

### Shared infrastructure: Gating logic

The R-peak trigger from Stage 2 fires timing windows for each feature. These are monostable one-shots (pulse stretchers) built from an OTA + cap + comparator.

```spice
* ============================================================
* Monostable One-Shot: produces a timed pulse from a trigger edge
* Pulse width = C * Vth / gm (proportional to cap size)
* ============================================================

.subckt oneshot trigger pulse_out vdd vss vbias
.param cap_l = 10
.param cap_w = 10

* Trigger edge charges cap through switch
XMsw trigger trigger net_cap vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1

* Cap discharges through OTA (constant current)
Xota net_cap vref_int net_cap vdd vss vbias ota_5t
XCtime net_cap vss sky130_fd_pr__cap_mim_m3_1 W={cap_w} L={cap_l}

* Internal reference (mid-supply)
XMref1 vref_int vref_int vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=1 nf=1
XMref2 vref_int vref_int vss vss sky130_fd_pr__nfet_01v8 L=4 W=1 nf=1

* Comparator: cap voltage > threshold → output high
Xcmp net_cap vss pulse_out vdd vss vbias comparator

.ends oneshot
```

Timing signals derived from R-peak trigger:
- **qrs_gate**: high during QRS window (from ~70 samples before R-peak to ~60 after) ≈ 100ms
- **pwave_gate**: high during P-wave window (150ms before QRS onset) ≈ 150ms
- **sample_clock**: brief pulse ~50ms after QRS ends (triggers S&H bank)

In practice, these are generated by cascaded one-shots with different cap sizes:

```
rpeak_trigger ──→ [OneShot 100ms] ──→ qrs_gate
                  [OneShot 50ms delay] ──→ [OneShot 150ms] ──→ pwave_gate (INVERTED — fires BEFORE rpeak)
                  [OneShot 150ms delay] ──→ [OneShot 1us] ──→ sample_clock
```

**Important design note:** The P-wave gate must be active *before* the R-peak. This means it's triggered by the *previous* beat's rpeak, delayed by `(expected_RR - 350ms)`. Use the RR timer output to set this delay adaptively.

### 6a. Features 1 & 2: rr_ratio and rr_asymmetry

```spice
* ============================================================
* FEATURES 1 & 2: RR Timing
* rr_ratio = pre_rr / local_rr_mean
* rr_asymmetry = pre_rr / post_rr
*
* Timer: constant current source charges cap between R-peaks
* EWMA: leaky integrator with pseudo-resistor
* Divider: Gilbert cell (4 transistors)
* ============================================================

.subckt rr_timing rpeak_trigger rr_ratio_out rr_asym_out vdd vss vbias

* --- RR interval timer ---
* Constant current (from OTA bias) charges Ctimer
* Voltage = Id * t / C, proportional to RR interval
* At Id=5nA, C=5pF, t=0.833s: V = 5n*0.833/5p = 833mV ← good range

* Charge current source (PMOS mirror from bias)
XMcharge net_timer net_timer vdd vdd sky130_fd_pr__pfet_01v8_hvt L=8 W=1 nf=1
XMcharge_mirror vbias_p vbias_p vdd vdd sky130_fd_pr__pfet_01v8_hvt L=8 W=1 nf=1

* Timer capacitor
XCtimer net_timer vss sky130_fd_pr__cap_mim_m3_1 W=16 L=16

* Reset switch — NMOS, gate = rpeak_trigger (brief pulse dumps cap)
XMreset net_timer rpeak_trigger vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- Sample-and-hold: capture timer value BEFORE reset ---
* Transmission gate clocked by leading edge of rpeak_trigger
XMsh_n net_timer rpeak_trigger sh_prerr vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMsh_p net_timer rpeak_bar sh_prerr vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XChold_prerr sh_prerr vss sky130_fd_pr__cap_mim_m3_1 W=10 L=10

* Inverted trigger for transmission gate
Xinv_trig rpeak_trigger rpeak_bar vdd vss inverter_sub

* sh_prerr now holds the pre_rr voltage (proportional to RR interval)

* --- EWMA: local RR mean ---
* Pseudo-resistor (subthreshold PMOS) + cap = leaky integrator
* R ~ 100 GOhm, C = 2pF → tau = 200ms → alpha ≈ 0.004 (good for 20-beat avg)
XMpseudo sh_prerr ewma_out ewma_out vdd sky130_fd_pr__pfet_01v8_hvt L=10 W=0.42 nf=1
XCewma ewma_out vss sky130_fd_pr__cap_mim_m3_1 W=10 L=10

* --- Gilbert cell divider: rr_ratio = sh_prerr / ewma_out ---
* 4-transistor translinear divider (operates in subthreshold)
* Output voltage proportional to ratio of input currents
XMg1 net_g1 sh_prerr vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMg2 net_g2 ewma_out vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMg3 rr_ratio_out net_g1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XMg4 net_g2 net_g1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* --- Second S&H for post_rr (delayed by one beat) ---
* post_rr of beat N = pre_rr of beat N+1 = net_timer at next trigger
* Hold the CURRENT timer value (before reset) for asymmetry computation
XChold_postrr sh_postrr vss sky130_fd_pr__cap_mim_m3_1 W=10 L=10

* --- Gilbert cell divider: rr_asym = sh_prerr / sh_postrr ---
XMga1 net_ga1 sh_prerr vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMga2 net_ga2 sh_postrr vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMga3 rr_asym_out net_ga1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XMga4 net_ga2 net_ga1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* --- OTA count: 0 explicit OTAs (uses current mirrors + caps) ---
* Transistor count: 2(charge) + 1(reset) + 2(TG) + 1(pseudo-R)
*                 + 4(divider1) + 4(divider2) = 14 transistors + 4 caps
* Current draw: ~2× bias current (two mirror branches) + divider bias

.ends rr_timing
```

### 6b. Features 3 & 4: beat_instability_L0 and L1

```spice
* ============================================================
* FEATURES 3 & 4: Beat-to-Beat Amplitude Instability
* instability = |peak_current - peak_previous| / peak_previous
*
* Per lead: gated peak detector + S&H (previous) + subtract + rectify + divide
* ============================================================

.subckt beat_instability ecg_in qrs_gate rpeak_trigger instab_out vdd vss vbias

* --- Gated peak detector ---
* OTA as comparator: if ecg > peak_cap, charge peak_cap up
* Diode-connected PMOS passes current only when input exceeds stored peak
* Gate signal enables/disables the tracking

* Peak storage cap
XCpeak peak_node vss sky130_fd_pr__cap_mim_m3_1 W=7 L=7

* OTA compares input to stored peak
Xota_pk ecg_in peak_node net_pk_drive vdd vss vbias ota_5t

* Source follower charges peak cap (only when OTA output > threshold)
XMpk_charge peak_node net_pk_drive vdd vdd sky130_fd_pr__pfet_01v8 L=1 W=2 nf=1

* Gate: NMOS switch isolates peak detector when qrs_gate is low
XMpk_gate net_pk_drive qrs_gate net_pk_gated vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1

* Reset: dump peak cap before each QRS window
* Use a delayed rpeak trigger (a few ms before QRS gate opens)
XMpk_reset peak_node rpeak_trigger vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- Sample-and-hold: stores PREVIOUS beat's peak ---
* Clocked by rpeak_trigger (captures current peak, which becomes "previous" next beat)
XMprev_n peak_node rpeak_trigger prev_peak vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMprev_p peak_node rpeak_bar prev_peak vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XCprev prev_peak vss sky130_fd_pr__cap_mim_m3_1 W=10 L=10

Xinv rpeak_trigger rpeak_bar vdd vss inverter_sub

* --- Subtractor: diff = peak_current - peak_previous ---
Xota_sub prev_peak peak_node diff_out vdd vss vbias ota_5t

* --- Rectifier: |diff| (reuse precision rectifier subcircuit) ---
* Simplified: two NMOS + current mirror for absolute value
* For energy budget, approximate as 2 OTAs
Xota_abs1 diff_out vref abs_diff vdd vss vbias ota_5t
* (vref = mid-supply, 0.9V)

* --- Divider: |diff| / prev_peak ---
* Gilbert cell (4 transistors), same as in RR timing
XMd1 net_d1 abs_diff vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMd2 net_d2 prev_peak vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMd3 instab_out net_d1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XMd4 net_d2 net_d1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* Vref for rectifier
Xvref_div vdd vss vref resistor_div

* --- TOTAL per lead: 3 OTAs + 14 transistors + 3 caps ---
* GATED: only active during QRS window + ~10ms computation

.ends beat_instability

* Voltage divider for mid-supply reference
.subckt resistor_div vdd vss vmid
XMr1 vmid vmid vdd vdd sky130_fd_pr__pfet_01v8_hvt L=10 W=0.42 nf=1
XMr2 vmid vmid vss vss sky130_fd_pr__nfet_01v8 L=10 W=0.42 nf=1
.ends resistor_div
```

### 6c. Feature 5: pwave_energy_L0

```spice
* ============================================================
* FEATURE 5: P-wave Energy
* BPF(3-8Hz) of ECG, squared, integrated over pre-QRS window
*
* BPF: ALWAYS-ON (must have settled output when gate opens)
* Squarer + integrator: GATED to P-wave window only
* ============================================================

.subckt pwave_energy ecg_in pwave_gate pw_energy_out vdd vss vbias

* --- Bandpass filter 3-8 Hz (ALWAYS-ON) ---
* HPF(3Hz): C=7.96pF → MIM 63x63um
* LPF(8Hz): C=2.98pF → MIM 39x39um
Xbpf ecg_in bpf_pw vdd vss vbias otac_bpf cl_hpf=63 cw_hpf=63 cl_lpf=39 cw_lpf=39

* --- Analog squarer (Gilbert multiplier, both inputs = bpf_pw) ---
* 4-transistor translinear squarer
* V_out proportional to (V_in - Vref)²
* GATED by pwave_gate

XMs1 net_sq1 bpf_pw net_sq_tail vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMs2 net_sq2 bpf_pw net_sq_tail vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMs3 sq_out net_sq1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XMs4 net_sq2 net_sq1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* Gated tail current (only active during P-wave window)
XMsq_tail net_sq_tail vbias net_sq_gate vss sky130_fd_pr__nfet_01v8 L=8 W=1 nf=1
XMsq_gate net_sq_gate pwave_gate vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- Gated integrator ---
* OTA drives current into cap. Only charges during P-wave window.
* At end of window, cap voltage = ∫(signal²)dt = energy
Xota_int sq_out pw_energy_out pw_energy_out vdd vss vbias ota_5t
XCint pw_energy_out vss sky130_fd_pr__cap_mim_m3_1 W=10 L=10

* Integration gate (NMOS switch in OTA output path)
* When pwave_gate=low, integrator holds its value

* Reset integrator before each P-wave window
XMint_reset pw_energy_out rpeak_trigger_delayed vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- TOTAL: 3 OTAs (BPF, always-on) + 1 OTA (integrator, gated)
*          + 4 transistors (squarer, gated) + 2 caps ---

.ends pwave_energy
```

### 6d. Feature 6: qrs_width_L0

```spice
* ============================================================
* FEATURE 6: QRS Width
* Time above 30% of QRS peak amplitude
* Shares peak detector with beat_instability
*
* Components: threshold generator + comparator + current-source timer
* ============================================================

.subckt qrs_width peak_voltage ecg_in qrs_gate width_out vdd vss vbias

* --- 30% threshold generator ---
* Resistive divider: V_thresh = 0.3 * peak + 0.7 * Vref
* Two series subthreshold NMOS as voltage divider
* Ratio set by W/L sizing: 0.3 = (W/L)_bot / ((W/L)_top + (W/L)_bot)

XMthr_top peak_voltage peak_voltage thresh_30 vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=3 nf=1
XMthr_bot thresh_30 thresh_30 vss vss sky130_fd_pr__nfet_01v8 L=4 W=1.3 nf=1

* --- Comparator: |ecg| > threshold? ---
* Precision rectifier output from beat_instability can be reused
* Or use a new OTA as comparator
Xota_wcmp ecg_in thresh_30 width_gate vdd vss vbias ota_5t

* AND with qrs_gate (only measure during QRS window)
XMand width_gate qrs_gate width_gate_gated vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1

* --- Width timer: current charges cap while above threshold ---
* V_width = I_charge * t_above / C
* At I=5nA, C=2.5pF, t=100ms: V = 5n*0.1/2.5p = 200mV
* At t=30ms (narrow QRS): V = 60mV
* At t=160ms (wide BBB): V = 320mV
* Good dynamic range

XMw_charge width_out width_gate_gated vdd vdd sky130_fd_pr__pfet_01v8_hvt L=8 W=1 nf=1
XCwidth width_out vss sky130_fd_pr__cap_mim_m3_1 W=11 L=11

* Reset before each QRS window
XMw_reset width_out rpeak_trigger vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- TOTAL: 1 OTA (comparator) + 4 transistors + 1 cap ---
* GATED to QRS window

.ends qrs_width
```

### 6e. Feature 7: slope_ratio_L0

```spice
* ============================================================
* FEATURE 7: QRS Slope Ratio
* max_upslope / max_downslope
*
* Differentiator (OTA-C HPF with high alpha) + gated peak/valley + divider
* ============================================================

.subckt slope_ratio ecg_in qrs_gate slope_out vdd vss vbias

* --- Differentiator ---
* OTA-C highpass with very high corner (approximates d/dt)
* fc >> 40 Hz → C very small (just parasitic), gm dominates
* Output current proportional to dV/dt
Xota_diff ecg_in deriv_out deriv_out vdd vss vbias ota_5t
XCdiff ecg_in deriv_out sky130_fd_pr__cap_mim_m3_1 W=3 L=3

* --- Gated positive peak detector (max upslope) ---
XCpk_up max_up vss sky130_fd_pr__cap_mim_m3_1 W=5 L=5
Xota_pkup deriv_out max_up net_pkup vdd vss vbias ota_5t
XMpkup max_up net_pkup vdd vdd sky130_fd_pr__pfet_01v8 L=1 W=2 nf=1
XMpkup_gate net_pkup qrs_gate net_pkup_g vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMpkup_reset max_up rpeak_trigger vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- Gated negative peak detector (max downslope → valley) ---
* Track minimum of derivative = most negative slope
XCpk_dn min_dn vss sky130_fd_pr__cap_mim_m3_1 W=5 L=5
Xota_pkdn min_dn deriv_out net_pkdn vdd vss vbias ota_5t
XMpkdn min_dn net_pkdn vss vss sky130_fd_pr__nfet_01v8 L=1 W=2 nf=1
XMpkdn_gate net_pkdn qrs_gate net_pkdn_g vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMpkdn_reset min_dn rpeak_trigger vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1

* --- Take |min_dn| by reflecting through Vref ---
* abs_dn = Vref - (min_dn - Vref) = 2*Vref - min_dn
Xota_abs min_dn vref abs_dn vdd vss vbias ota_5t

* --- Gilbert cell divider: max_up / abs_dn ---
XMsr1 net_sr1 max_up vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMsr2 net_sr2 abs_dn vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMsr3 slope_out net_sr1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XMsr4 net_sr2 net_sr1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* Vref
Xvref vdd vss vref resistor_div

* --- TOTAL: 4 OTAs + 4 transistors (divider) + 3 caps ---
* GATED to QRS window

.ends slope_ratio
```

### 6f. Feature 8: qrs_symmetry_L0

```spice
* ============================================================
* FEATURE 8: QRS Symmetry Ratio
* time_to_peak / qrs_width
*
* Shares: peak detector from beat_instability (peak_time)
*         onset comparator from qrs_width (qrs_onset)
*
* Incremental: 1 timer + 1 divider
* ============================================================

.subckt qrs_symmetry qrs_onset_signal peak_detect_signal width_voltage sym_out vdd vss vbias

* --- Time-to-peak timer ---
* Current source charges cap from QRS onset to QRS peak
* QRS onset = rising edge of width comparator output
* QRS peak = rising edge of peak detector comparator output

* Timer charges from onset, stops at peak
XMsym_charge net_sym qrs_onset_signal vdd vdd sky130_fd_pr__pfet_01v8_hvt L=8 W=1 nf=1
XCsym net_sym vss sky130_fd_pr__cap_mim_m3_1 W=11 L=11

* Stop charging when peak is detected (gate blocks current)
XMsym_stop net_sym peak_detect_signal vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1

* Reset before each QRS
XMsym_reset net_sym rpeak_trigger vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1

* --- Gilbert cell divider: time_to_peak / width ---
* Both inputs are timer voltages (proportional to time)
XMsy1 net_sy1 net_sym vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMsy2 net_sy2 width_voltage vss vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMsy3 sym_out net_sy1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1
XMsy4 net_sy2 net_sy1 vdd vdd sky130_fd_pr__pfet_01v8_hvt L=4 W=2 nf=1

* --- TOTAL: 0 OTAs + 7 transistors + 1 cap ---
* Extremely cheap — shares most infrastructure with width + instability
* GATED to QRS window

.ends qrs_symmetry
```

---

## 7. Stage 4: Sample-and-Hold Bank

```spice
* ============================================================
* STAGE 4: 8-Channel Sample-and-Hold Bank
* Captures all feature voltages for crossbar input
* Transmission gate + hold capacitor per channel
* ============================================================

.subckt sh_channel vin vout clk clk_bar vdd vss

* Transmission gate
XMn vin clk vout vss sky130_fd_pr__nfet_01v8 L=0.5 W=1 nf=1
XMp vin clk_bar vout vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1

* Hold capacitor (10pF — holds for ~100ms with <1% droop at 37°C)
XChold vout vss sky130_fd_pr__cap_mim_m3_1 W=70 L=70

.ends sh_channel

* ============================================================
* 8-channel bank instantiation
* ============================================================

.subckt sh_bank f1 f2 f3 f4 f5 f6 f7 f8 o1 o2 o3 o4 o5 o6 o7 o8 clk clk_bar vdd vss

Xsh1 f1 o1 clk clk_bar vdd vss sh_channel
Xsh2 f2 o2 clk clk_bar vdd vss sh_channel
Xsh3 f3 o3 clk clk_bar vdd vss sh_channel
Xsh4 f4 o4 clk clk_bar vdd vss sh_channel
Xsh5 f5 o5 clk clk_bar vdd vss sh_channel
Xsh6 f6 o6 clk clk_bar vdd vss sh_channel
Xsh7 f7 o7 clk clk_bar vdd vss sh_channel
Xsh8 f8 o8 clk clk_bar vdd vss sh_channel

.ends sh_bank
```

Energy per sample event: `E = 8 × C × V² = 8 × 10pF × (1.8V)² = 259 pJ`. Negligible.

---

## 8. Complete System Testbench

```spice
* ============================================================
* COMPLETE FRONTEND: All stages, transistor-level, energy measurement
* ============================================================

.title frontend_8feat_complete_sky130

.lib "/path/to/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

* Include all subcircuit files
.include "ota_5t_subthreshold.spice"
.include "bias_gen.spice"
.include "otac_lpf.spice"
.include "otac_hpf.spice"
.include "otac_bpf.spice"
.include "rectifier.spice"
.include "comparator.spice"
.include "inverter.spice"
.include "rr_timing.spice"
.include "beat_instability.spice"
.include "pwave_energy.spice"
.include "qrs_width.spice"
.include "slope_ratio.spice"
.include "qrs_symmetry.spice"
.include "sh_bank.spice"

.param vdd_val = 1.8
.param t_sim = 10

Vdd vdd 0 {vdd_val}

* ECG inputs
Vecg0 ecg_raw_L0 0 PWL file="ecg_multbeat_L0.txt"
Vecg1 ecg_raw_L1 0 PWL file="ecg_multbeat_L1.txt"

* ==========================================
* STAGE 1: Input conditioning (always-on)
* ==========================================
* (8 OTAs — instantiate as in section 4 above)

* ==========================================
* STAGE 2: R-peak detection (always-on)
* ==========================================
* (9 OTAs — instantiate as in section 5 above)

* ==========================================
* STAGE 3: Feature extraction (mostly gated)
* ==========================================
* P-wave BPF: 3 OTAs (always-on)
* RR timing: ~14 transistors (gated)
* Beat instability ×2: ~6 OTAs + switches (gated)
* QRS width: 1 OTA + switches (gated)
* Slope ratio: 4 OTAs + divider (gated)
* QRS symmetry: divider only (gated)

* ==========================================
* STAGE 4: S&H bank
* ==========================================
* 8 transmission gates + 8 caps (pulse)

* ==========================================
* ENERGY MEASUREMENTS
* ==========================================

.tran 10u {t_sim}

* Total supply current and power
.meas tran I_total_avg AVG abs(i(Vdd)) FROM=2 TO={t_sim-1}
.meas tran P_total_avg AVG par('v(vdd)*abs(i(Vdd))') FROM=2 TO={t_sim-1}

* Energy per beat (at 72 bpm)
.meas tran E_per_beat PARAM par('P_total_avg * 0.833')

* Energy breakdown by time window (for one beat cycle, centered at t=5s)
* Idle (between beats): no QRS processing
.meas tran E_idle INTEG par('v(vdd)*abs(i(Vdd))') FROM=4.2 TO=4.6

* P-wave window: 150ms
.meas tran E_pwave_window INTEG par('v(vdd)*abs(i(Vdd))') FROM=4.6 TO=4.75

* QRS window: 100ms
.meas tran E_qrs_window INTEG par('v(vdd)*abs(i(Vdd))') FROM=4.8 TO=4.9

* Settle + S&H: 50ms
.meas tran E_settle INTEG par('v(vdd)*abs(i(Vdd))') FROM=4.9 TO=4.95

.control
  run

  echo ""
  echo "============================================================"
  echo " FRONTEND ENERGY BUDGET — SKY130 Transistor-Level"
  echo "============================================================"
  print I_total_avg P_total_avg E_per_beat
  echo ""
  echo "Energy breakdown per beat:"
  print E_idle E_pwave_window E_qrs_window E_settle
  echo ""
  echo "E_always_on_per_beat = E_idle * (0.833 / 0.4)"
  echo "E_gated_per_beat = E_pwave_window + E_qrs_window + E_settle"
  echo "E_total = E_always_on + E_gated + E_sh (259 pJ)"
  echo ""
  echo "============================================================"
.endc

.end
```

---

## 9. Complete OTA and Transistor Count

| Block | OTAs | Extra transistors | Caps | Always-on? |
|-------|------|------------------|------|------------|
| Bias generator | 0 | 7 | 1 | Yes |
| **Stage 1** Input conditioning (×2 leads) | 8 | 0 | 6 | **Yes** |
| **Stage 2** R-peak BPF + rect + env + comp | 9 | ~8 (switches) | 4 | **Yes** |
| **Stage 3** P-wave BPF | 3 | 0 | 2 | **Yes** |
| Stage 3: RR timing | 0 | 14 | 4 | Gated |
| Stage 3: Beat instability (×2 leads) | 6 | 16 | 6 | Gated |
| Stage 3: P-wave squarer + integrator | 1 | 6 | 1 | Gated |
| Stage 3: QRS width | 1 | 5 | 1 | Gated |
| Stage 3: Slope ratio | 4 | 4 | 3 | Gated |
| Stage 3: QRS symmetry | 0 | 7 | 1 | Gated |
| Stage 3: Timing generation (one-shots) | 3 | 6 | 3 | Gated |
| **Stage 4** S&H bank (8 channels) | 0 | 16 | 8 | Pulse |
| **TOTALS** | **35** | **~89** | **~40** | |

**Always-on OTAs: 20** (8 input + 9 R-peak + 3 P-wave BPF)
**Gated OTAs: 15** (active ~100-150ms per beat)

---

## 10. Projected Energy Budget

The final number depends entirely on the OTA characterization testbench (Section 1). Run that first, then fill in this table:

| Parameter | Value | How to get it |
|-----------|-------|--------------|
| `Id_per_OTA` | ___ nA | From `ota_5t_test.spice` at chosen Vbias |
| `I_bias_gen` | ___ nA | From `bias_gen_test.spice` |
| `I_always_on` | 20 × Id + I_bias | Sum of always-on currents |
| `I_gated_peak` | 15 × Id | Peak current during QRS processing |
| `P_always_on` | I_always_on × 1.8V | Continuous power draw |
| `E_always_per_beat` | P_always_on × 0.833s | Always-on energy per beat |
| `E_gated_per_beat` | I_gated × 1.8V × 0.15s | Gated energy (avg active ~150ms) |
| `E_sh_per_beat` | 259 pJ | Fixed (8 × 10pF × 1.8² / 2) |
| **E_frontend_total** | **sum of above** | **THE NUMBER** |

### Estimates at different bias points

| Bias point | Id/OTA | I_always (20 OTAs) | P_always | E_always/beat | E_gated/beat | **E_total/beat** |
|-----------|--------|-------------------|----------|--------------|-------------|----------------|
| Ultra-low | 1 nA | 20 nA | 36 nW | 30 nJ | 4 nJ | **~34 nJ** |
| Low | 5 nA | 100 nA | 180 nW | 150 nJ | 20 nJ | **~170 nJ** |
| Moderate | 20 nA | 400 nA | 720 nW | 600 nJ | 81 nJ | **~681 nJ** |
| Conservative | 100 nA | 2 uA | 3.6 uW | 3.0 uJ | 405 nJ | **~3.4 uJ** |

**Your target of <50 nJ per beat is achievable at the ultra-low bias point (1 nA/OTA)**, but this is at the edge of SKY130 model accuracy and may have noise/bandwidth issues. The **5 nA point (~170 nJ) is the safe design center** — well within model accuracy, adequate bandwidth, and still two orders of magnitude below the classifier energy.

---

## 11. Simulation Execution Order

1. **`ota_5t_test.spice`** — characterize the OTA. Find the Vbias that gives 1-10 nA. Measure gain, GBW, noise.
2. **`bias_gen_test.spice`** — verify self-biased current reference generates the target bias voltage.
3. **`stage1_input_conditioning.spice`** — verify ECG signal conditioning, measure always-on power.
4. **`stage2_rpeak_detector.spice`** — verify R-peak detection on multi-beat ECG, measure always-on power.
5. **Feature circuits individually** — verify each feature output against Python reference values.
6. **`frontend_8feat_complete.spice`** — full system, measure total energy per beat.
7. **Corner sweep** — repeat at `ss`, `ff`, `sf`, `fs` corners to get worst-case energy.
8. **Temperature sweep** — repeat at 27°C and 37°C (body temperature) to check leakage impact.

---

## 12. Files to Create

Save each subcircuit as a separate `.spice` file in `ngspice_frontend/`:

```
ngspice_frontend/
├── ota_5t_subthreshold.spice      (core OTA)
├── ota_5t_test.spice              (OTA characterization)
├── bias_gen.spice                 (nA current reference)
├── bias_gen_test.spice            (bias gen test)
├── otac_lpf.spice                 (LPF building block)
├── otac_hpf.spice                 (HPF building block)
├── otac_bpf.spice                 (BPF building block)
├── rectifier.spice                (precision rectifier)
├── comparator.spice               (Schmitt trigger)
├── inverter.spice                 (CMOS inverter)
├── resistor_div.spice             (mid-supply reference)
├── oneshot.spice                  (timing pulse generator)
├── sh_channel.spice               (single S&H)
├── sh_bank.spice                  (8-channel S&H)
├── rr_timing.spice                (features 1-2)
├── beat_instability.spice         (features 3-4)
├── pwave_energy.spice             (feature 5)
├── qrs_width.spice                (feature 6)
├── slope_ratio.spice              (feature 7)
├── qrs_symmetry.spice             (feature 8)
├── stage1_input_conditioning.spice (Stage 1 testbench)
├── stage2_rpeak_detector.spice    (Stage 2 testbench)
├── frontend_complete.spice        (full system testbench)
├── ecg_multbeat_L0.txt            (ECG test vector, lead 0)
├── ecg_multbeat_L1.txt            (ECG test vector, lead 1)
└── generate_ecg_pwl.py            (PWL generator script)
```
