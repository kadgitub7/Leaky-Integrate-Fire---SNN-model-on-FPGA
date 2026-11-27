"""
Generate SPICE testbenches for incremental testing of the R-peak detector.

Phase 1A: Test detection logic only (noise gate + threshold comp + AND + hold)
          using Python-generated env_fast and env_avg as inline PWL sources.

Phase 1B: Test fast envelope follower with known input.

Phase 2:  Test full chain (BPF + diff + rect + envelopes + detection).
"""

import numpy as np
import wfdb
import os
import sys

sys.path.insert(0, 'scratchpad')
from analog_detector_final import detect_final, get_intermediate_signals

DB_PATH = 'mitdb_data'
RECORD = '100'
VBASE = 0.300
FS = 360
TEST_DURATION_S = 10.0

SKY130_LIB = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/ngspice_frontend/sky130_minimal.lib.spice'
CIRCUIT_DIR = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/complete_frontend_v9b'
OUT_DIR = 'scratchpad'


def format_pwl(times, voltages, max_per_line=4):
    """Format time-voltage pairs as inline PWL string."""
    lines = []
    buf = []
    for t, v in zip(times, voltages):
        buf.append(f"{t:.7f} {v:.6f}")
        if len(buf) >= max_per_line:
            lines.append("+ " + " ".join(buf))
            buf = []
    if buf:
        lines.append("+ " + " ".join(buf))
    return "\n".join(lines)


def generate_phase1a():
    """Phase 1A: Test detection logic with Python envelope PWLs."""
    record = wfdb.rdrecord(os.path.join(DB_PATH, RECORD))
    ann = wfdb.rdann(os.path.join(DB_PATH, RECORD), 'atr')
    ecg = record.p_signal[:, 0]

    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    sigs = get_intermediate_signals(ecg, FS)
    detections = detect_final(ecg, FS)
    detections = [d for d in detections if d < n_samples]

    times = np.arange(n_samples) / FS
    rect = sigs['rectified']
    scale = 0.15 / (rect.max() + 1e-15)

    env_fast_v = VBASE + sigs['env_fast'] * scale
    env_avg_v = VBASE + sigs['env_avg'] * scale

    print(f"Phase 1A: Record {RECORD}, {TEST_DURATION_S}s, {len(detections)} Python detections")
    print(f"Env_fast range: {env_fast_v.min():.4f}V - {env_fast_v.max():.4f}V")
    print(f"Env_avg range:  {env_avg_v.min():.4f}V - {env_avg_v.max():.4f}V")

    # Downsample for inline PWL (every other sample to reduce file size)
    step = 1  # keep full resolution for accuracy
    t_ds = times[::step]
    ef_ds = env_fast_v[::step]
    ea_ds = env_avg_v[::step]

    env_fast_pwl = format_pwl(t_ds, ef_ds)
    env_avg_pwl = format_pwl(t_ds, ea_ds)

    det_comments = "\n".join([f"* Python det {i+1}: t={d/FS:.4f}s"
                               for i, d in enumerate(detections)])

    # Set threshold: a good fixed threshold for Record 100
    # Based on Python: spki ~ 0.44V, npki ~ 0.30V after warmup
    # threshold = 0.30 + 0.25*(0.44-0.30) = 0.335V
    # Use a lower value initially to catch all beats
    threshold_v = 0.315

    tb = f"""* ============================================================
* Phase 1A: Detection Logic Test
* ============================================================
* Tests noise gate + threshold comp + AND + hold
* with Python-computed env_fast and env_avg as inputs.
*
* Record {RECORD}, {TEST_DURATION_S}s, {len(detections)} Python detections
{det_comments}
* ============================================================

.lib {SKY130_LIB} tt

.include {CIRCUIT_DIR}/inverter.spice
.include {CIRCUIT_DIR}/rpeak_hold.spice
.include {CIRCUIT_DIR}/rpeak_detector_v2.spice

.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 0.300

* ---- Python-computed envelope signals as PWL ----
Vef env_fast 0 PWL(
{env_fast_pwl}
+ )

Vea env_avg 0 PWL(
{env_avg_pwl}
+ )

* ---- Threshold ----
Vthresh threshold 0 {threshold_v}

* ---- DUT: noise gate + threshold comp + AND + hold ----
* (Bypass envelope circuits — test only the detection path)

Xngate env_fast env_avg noise_gate_out vdd vss vbias noise_gate
Xtcomp env_fast threshold thr_gate_out vdd vss vbias threshold_comp
Xand noise_gate_out thr_gate_out rpeak_raw vdd vss and2_gate
Xhold rpeak_raw rpeak_out vdd vss vbias rpeak_hold

* ---- Simulation ----
.tran 50u {TEST_DURATION_S} 0 50u

* ---- Measurements ----
"""

    # Add detection time measurements
    for i in range(1, min(len(detections)+3, 15)):
        tb += f".meas tran det{i} when v(rpeak_out)=0.45 rise={i}\n"

    tb += f"""
.save v(env_fast) v(env_avg) v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

.control
set filetype=ascii
run

echo ""
echo "=== PHASE 1A: DETECTION LOGIC TEST ==="
echo "--- Detection times (SPICE vs Python) ---"
"""
    for i in range(1, min(len(detections)+3, 15)):
        tb += f"print det{i}\n"

    tb += f"""
wrdata phase1a_results.txt v(env_fast) v(env_avg) v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

echo "=== Done ==="
quit
.endc

.end
"""

    out_path = os.path.join(OUT_DIR, 'tb_phase1a.spice')
    with open(out_path, 'w') as f:
        f.write(tb)
    print(f"Written: {out_path}")

    # Save reference data
    np.savez(os.path.join(OUT_DIR, 'phase1a_reference.npz'),
             times=times, env_fast_v=env_fast_v, env_avg_v=env_avg_v,
             detections=np.array(detections))

    print(f"\nExpected Python detection times:")
    for i, d in enumerate(detections[:14]):
        print(f"  det{i+1}: {d/FS:.4f}s")


def generate_phase1b():
    """Phase 1B: Test fast envelope follower with synthetic input."""

    # Create a synthetic input: periodic pulses above baseline
    # 5 beats: pulse at t=0.5, 1.3, 2.1, 2.9, 3.7
    # Each pulse: 50ms ramp up to peak, 50ms ramp down
    duration = 5.0
    dt = 1e-3  # 1ms resolution
    n = int(duration / dt)
    t = np.arange(n) * dt

    # Baseline at vbias, pulses add 100mV peak
    signal = np.full(n, VBASE)
    beat_times = [0.5, 1.3, 2.1, 2.9, 3.7]
    pulse_width = 0.040  # 40ms total pulse width
    peak_amp = 0.100     # 100mV above baseline

    for bt in beat_times:
        # Triangular pulse
        for i in range(n):
            dt_pulse = t[i] - bt
            if -pulse_width/2 < dt_pulse < 0:
                # Rising edge
                signal[i] = VBASE + peak_amp * (1 + dt_pulse/(pulse_width/2))
            elif 0 <= dt_pulse < pulse_width/2:
                # Falling edge
                signal[i] = VBASE + peak_amp * (1 - dt_pulse/(pulse_width/2))

    pwl_str = format_pwl(t, signal)

    tb = f"""* ============================================================
* Phase 1B: Fast Envelope Follower Test
* ============================================================
* Synthetic pulsed input: 5 pulses at 0.5, 1.3, 2.1, 2.9, 3.7s
* Each pulse: 100mV peak above 0.300V baseline, 40ms wide
*
* Expected: env should peak near 0.400V, decay toward ~0.300V
*           with tau ~ 100ms between beats
* ============================================================

.lib {SKY130_LIB} tt

.include {CIRCUIT_DIR}/rpeak_detector_v2.spice

.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 0.300

* Synthetic pulsed input
Vin vin 0 PWL(
{pwl_str}
+ )

* DUT: fast envelope follower only
Xenv vin env_out vdd vss vbias fast_envelope

* Initial condition: start at baseline
.ic v(env_out)=0.300

.tran 100u 5 0 100u

* Measurements
.meas tran env_pk1 max v(env_out) from=0.45 to=0.6
.meas tran env_pk2 max v(env_out) from=1.25 to=1.4
.meas tran env_at_0_7 find v(env_out) at=0.7
.meas tran env_at_0_9 find v(env_out) at=0.9
.meas tran env_at_1_1 find v(env_out) at=1.1

.save v(vin) v(env_out)

.control
set filetype=ascii
run

echo ""
echo "=== PHASE 1B: ENVELOPE FOLLOWER TEST ==="
print env_pk1 env_pk2
echo "--- Decay profile after first pulse ---"
print env_at_0_7 env_at_0_9 env_at_1_1
echo ""

wrdata phase1b_results.txt v(vin) v(env_out)

echo "=== Done ==="
quit
.endc

.end
"""

    out_path = os.path.join(OUT_DIR, 'tb_phase1b.spice')
    with open(out_path, 'w') as f:
        f.write(tb)
    print(f"\nWritten: {out_path}")


def generate_phase1c():
    """Phase 1C: Full envelope+detection path with Python rectified signal as input."""
    record = wfdb.rdrecord(os.path.join(DB_PATH, RECORD))
    ecg = record.p_signal[:, 0]

    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    sigs = get_intermediate_signals(ecg, FS)
    detections = detect_final(ecg, FS)
    detections = [d for d in detections if d < n_samples]

    times = np.arange(n_samples) / FS
    rect = sigs['rectified']
    scale = 0.15 / (rect.max() + 1e-15)

    # Scale rectified signal to voltage: baseline 0.300V, peaks up to ~0.450V
    rect_v = VBASE + rect * scale

    print(f"Phase 1C: Record {RECORD}, {TEST_DURATION_S}s, {len(detections)} Python detections")
    print(f"Rect range: {rect_v.min():.4f}V - {rect_v.max():.4f}V")

    # Downsample 2x to reduce file size (rect changes slower than raw ECG)
    step = 2
    t_ds = times[::step]
    r_ds = rect_v[::step]

    rect_pwl = format_pwl(t_ds, r_ds)

    det_comments = "\n".join([f"* Python det {i+1}: t={d/FS:.4f}s"
                               for i, d in enumerate(detections)])

    # Threshold: use a moderate fixed value
    # Python env_fast peaks at ~0.438V, avg is ~0.317V
    # threshold = 0.300 + 0.25*(0.438-0.300) = 0.3345
    threshold_v = 0.320

    tb = f"""* ============================================================
* Phase 1C: Full Envelope + Detection Path Test
* ============================================================
* Tests: rect_in -> fast_envelope -> smoother -> noise_gate +
*        rect_in -> slow_average -> noise_gate
*        + threshold_comp + AND + rpeak_hold
*
* Input: Python-computed rectified ECG as PWL
* Record {RECORD}, {TEST_DURATION_S}s, {len(detections)} Python detections
{det_comments}
* ============================================================

.lib {SKY130_LIB} tt

.include {CIRCUIT_DIR}/inverter.spice
.include {CIRCUIT_DIR}/rpeak_hold.spice
.include {CIRCUIT_DIR}/rpeak_detector_v2.spice

.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 0.300

* ---- Python-computed rectified signal as PWL ----
Vrect rect_in 0 PWL(
{rect_pwl}
+ )

* ---- Threshold ----
Vthresh threshold 0 {threshold_v}

* ---- DUT: full rpeak_detector_v2 ----
Xdet rect_in threshold rpeak_out vdd vss vbias
+ env_fast_raw env_fast env_avg noise_gate_out thr_gate_out
+ rpeak_detector_v2

* Initial conditions: set envelope caps near baseline
.ic v(env_fast_raw)=0.300 v(env_fast)=0.300 v(env_avg)=0.300

* ---- Simulation ----
.tran 50u {TEST_DURATION_S} 0 50u

* ---- Measurements ----
"""

    # Detection time measurements
    for i in range(1, min(len(detections)+3, 20)):
        tb += f".meas tran det{i} when v(rpeak_out)=0.45 rise={i}\n"

    # Envelope peak measurements
    tb += "\n* Envelope characteristics\n"
    tb += ".meas tran efr_max max v(env_fast_raw)\n"
    tb += ".meas tran ef_max max v(env_fast)\n"
    tb += ".meas tran ea_max max v(env_avg)\n"
    tb += ".meas tran efr_min min v(env_fast_raw)\n"
    tb += ".meas tran ef_min min v(env_fast)\n"
    tb += ".meas tran ea_min min v(env_avg)\n"

    tb += f"""
.save v(rect_in) v(env_fast_raw) v(env_fast) v(env_avg)
+ v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

.control
set filetype=ascii
run

echo ""
echo "=== PHASE 1C: FULL ENVELOPE + DETECTION PATH ==="
echo "--- Envelope ranges ---"
echo "--- Detection times ---"
"""
    for i in range(1, min(len(detections)+3, 20)):
        tb += f"* det{i}\n"

    tb += f"""
wrdata phase1c_results.txt v(rect_in) v(env_fast_raw) v(env_fast) v(env_avg) v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

echo "=== Done ==="
quit
.endc

.end
"""

    out_path = os.path.join(OUT_DIR, 'tb_phase1c.spice')
    with open(out_path, 'w') as f:
        f.write(tb)
    print(f"Written: {out_path}")

    print(f"\nExpected Python detection times:")
    for i, d in enumerate(detections[:18]):
        print(f"  det{i+1}: {d/FS:.4f}s")


def generate_phase2a():
    """Phase 2A: Full chain test with raw ECG input."""
    record = wfdb.rdrecord(os.path.join(DB_PATH, RECORD))
    ecg = record.p_signal[:, 0]

    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    detections = detect_final(ecg, FS)
    detections = [d for d in detections if d < n_samples]

    times = np.arange(n_samples) / FS

    # Scale ECG to voltage: baseline ~0.300V, typical R-peak ~0.400V
    # Record 100 ECG range: about -0.3 to 1.1 mV (physical).
    # Map to 0.250V–0.450V range (200mV span, centered at 0.350V)
    ecg_min, ecg_max = ecg.min(), ecg.max()
    ecg_v = 0.250 + (ecg - ecg_min) / (ecg_max - ecg_min) * 0.200

    print(f"Phase 2A: Record {RECORD}, {TEST_DURATION_S}s, {len(detections)} Python detections")
    print(f"ECG voltage range: {ecg_v.min():.4f}V - {ecg_v.max():.4f}V")
    print(f"ECG baseline (median): {np.median(ecg_v):.4f}V")

    # Downsample 1x (keep full resolution for ECG fidelity)
    step = 1
    t_ds = times[::step]
    e_ds = ecg_v[::step]

    ecg_pwl = format_pwl(t_ds, e_ds)

    det_comments = "\n".join([f"* Python det {i+1}: t={d/FS:.4f}s"
                               for i, d in enumerate(detections)])

    # For direct ECG input (no AC coupling):
    # ECG baseline is at ~0.287V, QRS peaks reach ~0.450V
    # env_fast will track QRS peaks, env_avg tracks baseline
    # Threshold between baseline and peak: 0.287 + 0.25*(0.45-0.287) = 0.328V
    threshold_v = 0.320

    tb = f"""* ============================================================
* Phase 2A: Direct ECG -> Envelope Detection Path
* ============================================================
* Tests: ECG -> rpeak_detector_v2 (no AC coupling stage)
* The ECG signal feeds directly into the dual-envelope detector.
* QRS peaks are the dominant feature without differentiation.
*
* Record {RECORD}, {TEST_DURATION_S}s, {len(detections)} Python detections
{det_comments}
* ============================================================

.lib {SKY130_LIB} tt

.include {CIRCUIT_DIR}/inverter.spice
.include {CIRCUIT_DIR}/rpeak_hold.spice
.include {CIRCUIT_DIR}/rpeak_detector_v2.spice

.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 0.300

* ---- Raw ECG signal as PWL ----
Vecg ecg_in 0 PWL(
{ecg_pwl}
+ )

* ---- Threshold ----
Vthresh threshold 0 {threshold_v}

* ---- DUT: rpeak_detector_v2 (direct ECG input) ----
Xdet ecg_in threshold rpeak_out vdd vss vbias
+ env_fast_raw env_fast env_avg noise_gate_out thr_gate_out
+ rpeak_detector_v2

* Initial conditions: set near ECG baseline
.ic v(env_fast_raw)={np.median(ecg_v):.4f} v(env_fast)={np.median(ecg_v):.4f} v(env_avg)={np.median(ecg_v):.4f}

* ---- Simulation ----
.tran 50u {TEST_DURATION_S} 0 50u

* ---- Measurements ----
"""

    for i in range(1, min(len(detections)+3, 20)):
        tb += f".meas tran det{i} when v(rpeak_out)=0.45 rise={i}\n"

    tb += "\n* Signal ranges\n"
    tb += ".meas tran efr_max max v(env_fast_raw)\n"
    tb += ".meas tran ef_max max v(env_fast)\n"
    tb += ".meas tran ea_max max v(env_avg)\n"
    tb += ".meas tran efr_min min v(env_fast_raw)\n"
    tb += ".meas tran ea_min min v(env_avg)\n"

    tb += f"""
.save v(ecg_in) v(env_fast_raw) v(env_fast) v(env_avg)
+ v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

.control
set filetype=ascii
run

echo ""
echo "=== PHASE 2A: FULL CHAIN TEST ==="
echo "--- Signal ranges ---"
echo "--- Detection times ---"

wrdata phase2a_results.txt v(ecg_in) v(env_fast_raw) v(env_fast) v(env_avg) v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

echo "=== Done ==="
quit
.endc

.end
"""

    out_path = os.path.join(OUT_DIR, 'tb_phase2a.spice')
    with open(out_path, 'w') as f:
        f.write(tb)
    print(f"Written: {out_path}")

    print(f"\nExpected Python detection times:")
    for i, d in enumerate(detections[:18]):
        print(f"  det{i+1}: {d/FS:.4f}s")


if __name__ == '__main__':
    generate_phase1a()
    generate_phase1b()
    generate_phase1c()
    generate_phase2a()
