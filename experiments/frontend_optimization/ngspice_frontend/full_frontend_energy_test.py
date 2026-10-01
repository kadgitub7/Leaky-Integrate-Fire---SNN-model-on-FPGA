"""Full frontend energy measurement using actual MIT-BIH ECG data.

Extracts multi-beat ECG segments from a real MIT-BIH record,
generates ngspice PWL files, runs the complete frontend circuit,
and measures actual total supply current for true energy-per-beat.
"""

import os
import sys
import json
import re
import subprocess
import numpy as np

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MITDB_DIR = os.path.join(REPO_ROOT, 'mitdb_data')
NGSPICE_CON = "C:/Spice64/bin/ngspice_con.exe"

VDD = 0.9
VREF = VDD / 2.0


def load_mitbih_segment(record_id='100', n_beats=6):
    """Load a multi-beat ECG segment from MIT-BIH using wfdb.

    Returns: (time_s, lead0_volts, lead1_volts, rpeak_times_s)
    ECG is scaled to fit within [0.1*VDD, 0.8*VDD] for analog circuit headroom.
    """
    try:
        import wfdb
    except ImportError:
        print("Installing wfdb...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wfdb', '-q'])
        import wfdb

    record_path = os.path.join(MITDB_DIR, record_id)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    fs = record.fs  # 360 Hz

    # Find beat annotations (exclude non-beat markers)
    beat_syms = set('NLRejAaJSVEFfQ/')
    beat_samples = [s for s, sym in zip(ann.sample, ann.symbol) if sym in beat_syms]

    # Take n_beats+1 starting from beat 10 (skip early transients)
    start_beat = 10
    if start_beat + n_beats + 1 > len(beat_samples):
        start_beat = 0
    selected = beat_samples[start_beat:start_beat + n_beats + 1]

    # Segment: 0.3s before first R-peak to 0.3s after last R-peak
    margin_samples = int(0.3 * fs)
    seg_start = max(0, selected[0] - margin_samples)
    seg_end = min(len(record.p_signal), selected[-1] + margin_samples)

    signals = record.p_signal[seg_start:seg_end, :]
    n_samples = signals.shape[0]
    n_leads = min(signals.shape[1], 2)

    time_s = np.arange(n_samples) / fs

    # R-peak times relative to segment start
    rpeak_times = [(s - seg_start) / fs for s in selected]

    # Scale ECG to analog range [0.1*VDD, 0.8*VDD]
    # Raw MIT-BIH is in mV, typical range [-1, +2] mV
    v_min = 0.1 * VDD
    v_max = 0.8 * VDD
    scaled = []
    for lead in range(n_leads):
        sig = signals[:, lead]
        sig_min, sig_max = np.percentile(sig, [1, 99])
        sig_norm = (sig - sig_min) / (sig_max - sig_min + 1e-12)
        sig_scaled = v_min + sig_norm * (v_max - v_min)
        sig_scaled = np.clip(sig_scaled, 0.05, VDD - 0.05)
        scaled.append(sig_scaled)

    # Pad to 2 leads if only 1
    if n_leads == 1:
        scaled.append(scaled[0].copy())

    print(f"  Record {record_id}: {n_samples} samples at {fs} Hz = {time_s[-1]:.2f}s")
    print(f"  {len(rpeak_times)} R-peaks at: {[f'{t:.3f}' for t in rpeak_times]}")
    print(f"  ECG scaled to [{v_min:.2f}, {v_max:.2f}]V")

    return time_s, scaled[0], scaled[1], rpeak_times, fs


def write_pwl_file(filepath, time_s, voltage):
    """Write ngspice PWL-compatible text file."""
    with open(filepath, 'w') as f:
        step = 4
        for i in range(0, len(time_s), step):
            f.write(f"{time_s[i]:.7e} {voltage[i]:.7e}\n")
    print(f"  Wrote {filepath} ({os.path.getsize(filepath)} bytes)")


def format_pwl_inline(time_s, voltage, step=4):
    """Format PWL data as inline string for ngspice (no file= needed)."""
    pairs = []
    for i in range(0, len(time_s), step):
        pairs.append(f"{time_s[i]:.6e} {voltage[i]:.6e}")
    return "\n+ ".join(pairs)


def generate_timing_pwl(rpeak_times, total_time, fs):
    """Generate timing signal PWL files from R-peak locations.

    Returns dict of {signal_name: (times, values)} for:
    - rpeak_trigger: 5ms pulse at each R-peak
    - qrs_gate: high for 150ms centered on R-peak
    - pwave_gate: high for 100ms ending 20ms before R-peak
    - sample_clk: 2ms pulse 200ms after R-peak
    """
    dt = 0.001  # 1ms resolution for timing edges
    pulse_rpeak = 0.005    # 5ms rpeak trigger pulse
    qrs_window = 0.150     # 150ms QRS gate
    qrs_pre = 0.050        # QRS gate starts 50ms before R-peak
    pw_window = 0.100      # 100ms P-wave window
    pw_end_before = 0.020  # P-wave window ends 20ms before R-peak
    sh_delay = 0.200       # S&H samples 200ms after R-peak
    sh_pulse = 0.002       # 2ms sample pulse

    signals = {}
    for name in ['rpeak_trigger', 'qrs_gate', 'pwave_gate', 'sample_clk', 'sample_clk_bar']:
        times = [0.0]
        values = [0.0]

        for rp in rpeak_times:
            if name == 'rpeak_trigger':
                t_on = rp
                t_off = rp + pulse_rpeak
            elif name == 'qrs_gate':
                t_on = rp - qrs_pre
                t_off = rp - qrs_pre + qrs_window
            elif name == 'pwave_gate':
                t_on = rp - pw_end_before - pw_window
                t_off = rp - pw_end_before
            elif name == 'sample_clk':
                t_on = rp + sh_delay
                t_off = rp + sh_delay + sh_pulse
            elif name == 'sample_clk_bar':
                # Inverted sample clock — handled below
                t_on = rp + sh_delay
                t_off = rp + sh_delay + sh_pulse

            if t_on < 0:
                t_on = 0
            if t_off > total_time:
                continue

            rise_edge = dt * 0.1  # fast edges
            if name == 'sample_clk_bar':
                times.extend([t_on - rise_edge, t_on, t_off, t_off + rise_edge])
                values.extend([VDD, 0.0, 0.0, VDD])
            else:
                times.extend([t_on - rise_edge, t_on, t_off, t_off + rise_edge])
                values.extend([0.0, VDD, VDD, 0.0])

        # Ensure ends at total_time
        times.append(total_time)
        values.append(values[-1] if name != 'sample_clk_bar' else VDD)

        # Fix sample_clk_bar initial value
        if name == 'sample_clk_bar':
            values[0] = VDD

        signals[name] = (np.array(times), np.array(values))

    return signals


def write_timing_files(signals, output_dir):
    """Write all timing PWL files."""
    for name, (times, values) in signals.items():
        filepath = os.path.join(output_dir, f"{name}.txt")
        with open(filepath, 'w') as f:
            for t, v in zip(times, values):
                f.write(f"{t:.7e} {v:.7e}\n")
        print(f"  Wrote {name}.txt ({len(times)} points)")


def write_full_testbench(output_dir, total_time, corner, temp_C,
                         time_s, lead0, lead1, timing_signals):
    """Generate the complete frontend energy testbench.

    Uses all fixed circuit files and preprocessed PDK.
    Embeds all PWL data inline (ngspice-43 file= syntax is broken).
    Measures total VDD current for actual energy calculation.
    """
    meas_start = 1.0
    meas_end = total_time - 0.5

    # Format inline PWL data
    ecg0_pwl = format_pwl_inline(time_s, lead0)
    ecg1_pwl = format_pwl_inline(time_s, lead1)

    def fmt_timing(name):
        t, v = timing_signals[name]
        pairs = [f"{ti:.7e} {vi:.6e}" for ti, vi in zip(t, v)]
        return "\n+ ".join(pairs)

    rpeak_pwl = fmt_timing('rpeak_trigger')
    qrs_gate_pwl = fmt_timing('qrs_gate')
    pwave_gate_pwl = fmt_timing('pwave_gate')
    sample_clk_pwl = fmt_timing('sample_clk')
    sample_clkb_pwl = fmt_timing('sample_clk_bar')

    spice = f"""* ============================================================
* FULL FRONTEND ENERGY TEST - All Components, Real MIT-BIH ECG
* SKY130 transistor-level, {corner} corner, T={temp_C}C
* ============================================================

.title full_frontend_energy_{corner}_{temp_C}C

.lib "sky130_minimal.lib.spice" {corner}
.option scale=1.0u
.option method=gear
.option gmin=1e-15
.option abstol=1e-15
.option reltol=0.005
.temp {temp_C}

.include "ota_5t_subthreshold.spice"
* bias_gen.spice not included - using external Vbias source
.include "inverter.spice"
.include "resistor_div.spice"
.include "otac_lpf.spice"
.include "otac_hpf.spice"
.include "otac_bpf.spice"
.include "rectifier.spice"
.include "comparator.spice"
.include "rr_timing.spice"
.include "beat_instability.spice"
.include "pwave_energy.spice"
.include "qrs_width.spice"
.include "slope_ratio.spice"
.include "qrs_symmetry.spice"
.include "sh_channel.spice"
.include "sh_bank.spice"

* === SUPPLIES ===
Vdd vdd 0 {VDD}
Vss vss 0 0

Xvref_gen vdd vss vref resistor_div

* External bias at 0.35V (self-biased generator needs redesign for VDD=0.9V
* due to HVT PMOS Vth~0.7V leaving insufficient mirror headroom;
* in a real chip: dedicated low-VDD bias circuit or off-chip reference)
Vbias_src vbias 0 0.35

* === ECG INPUTS - MIT-BIH record (inline PWL) ===
Vecg0 ecg_raw_L0 0 PWL(
+ {ecg0_pwl})
Vecg1 ecg_raw_L1 0 PWL(
+ {ecg1_pwl})

* === STAGE 1: Input Conditioning (ALWAYS-ON, 8 OTAs) ===
Xbuf0 ecg_raw_L0 ecg_buf_L0 ecg_buf_L0 vdd vss vbias ota_5t
Xhpf0 ecg_buf_L0 ecg_hp_L0 vdd vss vbias otac_hpf cap_w=100 cap_l=100
Xlpf0 ecg_hp_L0 ecg_cond_L0 vdd vss vbias otac_lpf cap_w=17 cap_l=17

Xbuf1 ecg_raw_L1 ecg_buf_L1 ecg_buf_L1 vdd vss vbias ota_5t
Xhpf1 ecg_buf_L1 ecg_hp_L1 vdd vss vbias otac_hpf cap_w=100 cap_l=100
Xlpf1 ecg_hp_L1 ecg_cond_L1 vdd vss vbias otac_lpf cap_w=17 cap_l=17

* === STAGE 2: R-Peak Detection (ALWAYS-ON, 9 OTAs) ===
Xbpf ecg_cond_L0 bpf_out vdd vss vbias otac_bpf cl_hpf=39 cw_hpf=39 cl_lpf=24 cw_lpf=24
Xrect bpf_out rect_out vref vdd vss vbias rectifier
Xenv rect_out env_out vdd vss vbias otac_lpf cap_w=77 cap_l=77
Xthresh env_out thresh_out vdd vss vbias otac_lpf cap_w=50 cap_l=50
Xcmp env_out thresh_out rpeak_internal vdd vss vbias comparator

* === TIMING - External PWL (hardware: monostable from rpeak) ===
Vrpeak rpeak_trigger 0 PWL(
+ {rpeak_pwl})
Vqrs_gate qrs_gate 0 PWL(
+ {qrs_gate_pwl})
Vpwave_gate pwave_gate 0 PWL(
+ {pwave_gate_pwl})
Vsample_clk sample_clk 0 PWL(
+ {sample_clk_pwl})
Vsample_clkb sample_clk_bar 0 PWL(
+ {sample_clkb_pwl})

* === STAGE 3: Feature Extraction ===
Xrr rpeak_trigger f_rr_ratio f_rr_asym vdd vss vbias rr_timing
Xinstab0 ecg_cond_L0 qrs_gate rpeak_trigger f_instab_L0 vdd vss vbias vref beat_instability
Xinstab1 ecg_cond_L1 qrs_gate rpeak_trigger f_instab_L1 vdd vss vbias vref beat_instability
Xpwave ecg_cond_L0 pwave_gate rpeak_trigger f_pwave vdd vss vbias vref pwave_energy
Xwidth vref ecg_cond_L0 qrs_gate rpeak_trigger f_width vdd vss vbias qrs_width
Xslope ecg_cond_L0 qrs_gate rpeak_trigger f_slope vdd vss vbias vref slope_ratio
Xsym qrs_gate rpeak_trigger f_width f_sym rpeak_trigger vdd vss vbias qrs_symmetry

* === STAGE 4: Sample-and-Hold Bank (8 channels) ===
Xsh f_rr_ratio f_rr_asym f_instab_L0 f_instab_L1 f_pwave f_width f_slope f_sym
+ o_rr_ratio o_rr_asym o_instab_L0 o_instab_L1 o_pwave o_width o_slope o_sym
+ sample_clk sample_clk_bar vdd vss sh_bank

* === SIMULATION ===
.tran 200u {total_time} uic

.meas tran I_total_avg AVG i(Vdd) FROM={meas_start} TO={meas_end}
.meas tran I_total_rms RMS i(Vdd) FROM={meas_start} TO={meas_end}

.control
  run
  echo ""
  echo "============================================================"
  echo " FULL FRONTEND ENERGY - SKY130 {corner} {temp_C}C"
  echo "============================================================"
  echo ""
  echo "--- Supply current measurements ---"
  print I_total_avg I_total_rms
  echo ""
  echo "--- Full current waveform (decimated) ---"
  print -i(Vdd)
  quit
.endc

.end
"""
    filepath = os.path.join(output_dir, f"_full_energy_test_{corner}_{temp_C}.spice")
    with open(filepath, 'w') as f:
        f.write(spice)
    print(f"  Wrote testbench: {filepath}")
    return filepath


def run_ngspice(spice_file):
    """Run ngspice in batch mode."""
    print(f"  Running ngspice on {os.path.basename(spice_file)}...")
    result = subprocess.run(
        [NGSPICE_CON, "-b", spice_file],
        capture_output=True, text=True,
        cwd=SCRIPT_DIR, timeout=3600  # 60 min timeout for full transient
    )
    return result.stdout, result.stderr


def parse_energy_results(stdout, stderr, total_time, rpeak_times, corner, temp_C):
    """Parse ngspice output for supply current and compute energy."""
    results = {
        'corner': corner,
        'temp_C': temp_C,
        'vdd': VDD,
    }

    # Parse .meas results
    for line in stdout.split('\n'):
        line_lower = line.lower().strip()
        if 'i_total_avg' in line_lower and '=' in line:
            try:
                val = float(line.split('=')[1].strip())
                results['I_avg_A'] = abs(val)
            except (ValueError, IndexError):
                pass
        elif 'i_total_rms' in line_lower and '=' in line:
            try:
                val = float(line.split('=')[1].strip())
                results['I_rms_A'] = abs(val)
            except (ValueError, IndexError):
                pass

    # Also parse from print output
    if 'I_avg_A' not in results:
        for line in stdout.split('\n'):
            if 'i_total_avg' in line.lower():
                nums = re.findall(r'[-+]?\d+\.?\d*e[+-]?\d+', line)
                if nums:
                    results['I_avg_A'] = abs(float(nums[0]))

    if 'I_avg_A' in results:
        i_avg = results['I_avg_A']
        p_avg = i_avg * VDD
        n_beats = len(rpeak_times) - 1
        rr_avg = (rpeak_times[-1] - rpeak_times[0]) / n_beats if n_beats > 0 else 0.833
        e_per_beat = p_avg * rr_avg

        results.update({
            'I_avg_nA': i_avg * 1e9,
            'P_avg_nW': p_avg * 1e9,
            'rr_avg_s': rr_avg,
            'n_beats': n_beats,
            'E_per_beat_nJ': e_per_beat * 1e9,
            'E_per_beat_pJ': e_per_beat * 1e12,
        })

    # Check for errors
    if 'error' in stderr.lower() or 'Error' in stderr:
        error_lines = [l for l in stderr.split('\n') if 'error' in l.lower() or 'Error' in l]
        results['errors'] = error_lines[:10]

    return results


def main():
    print("=" * 60)
    print("  Full Frontend Energy Test — MIT-BIH ECG + SKY130")
    print("=" * 60)

    # Check ngspice exists
    if not os.path.exists(NGSPICE_CON):
        print(f"ERROR: ngspice not found at {NGSPICE_CON}")
        return

    # Check MIT-BIH data
    if not os.path.exists(MITDB_DIR):
        print(f"ERROR: MIT-BIH data not found at {MITDB_DIR}")
        return

    # Step 1: Load real ECG data
    print("\n--- Step 1: Loading MIT-BIH record 100 ---")
    record_id = '100'
    n_beats = 4
    time_s, lead0, lead1, rpeak_times, fs = load_mitbih_segment(record_id, n_beats)
    total_time = time_s[-1]

    # Step 2: Write ECG PWL files
    print("\n--- Step 2: Writing ECG PWL files ---")
    write_pwl_file(os.path.join(SCRIPT_DIR, "ecg_L0.txt"), time_s, lead0)
    write_pwl_file(os.path.join(SCRIPT_DIR, "ecg_L1.txt"), time_s, lead1)

    # Step 3: Generate timing signals from known R-peak locations
    print("\n--- Step 3: Generating timing signals ---")
    timing = generate_timing_pwl(rpeak_times, total_time, fs)
    write_timing_files(timing, SCRIPT_DIR)

    # Step 4: Run simulations
    corners_temps = [
        ('tt', 27),
    ]

    all_results = []

    for corner, temp in corners_temps:
        print(f"\n{'=' * 60}")
        print(f"  Running {corner} corner at {temp}C")
        print(f"{'=' * 60}")

        # Write testbench
        spice_file = write_full_testbench(SCRIPT_DIR, total_time, corner, temp,
                                          time_s, lead0, lead1, timing)

        # Run ngspice
        stdout, stderr = run_ngspice(spice_file)

        # Save raw output for debugging
        out_file = os.path.join(SCRIPT_DIR, f"_full_energy_output_{corner}_{temp}.txt")
        with open(out_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)
        print(f"  Raw output saved to: {out_file}")

        # Parse results
        results = parse_energy_results(stdout, stderr, total_time, rpeak_times, corner, temp)
        all_results.append(results)

        if 'E_per_beat_nJ' in results:
            print(f"\n  RESULTS ({corner}/{temp}C):")
            print(f"    Average supply current: {results['I_avg_nA']:.3f} nA")
            print(f"    Average power:          {results['P_avg_nW']:.3f} nW")
            print(f"    Energy per beat:        {results['E_per_beat_nJ']:.3f} nJ")
            print(f"    RR interval (avg):      {results['rr_avg_s']:.3f} s")
            print(f"    Number of beats:        {results['n_beats']}")
        else:
            print(f"\n  WARNING: Could not extract energy results for {corner}/{temp}C")
            if 'errors' in results:
                print(f"  Errors: {results['errors'][:5]}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  ENERGY SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        if 'E_per_beat_nJ' in r:
            print(f"  {r['corner']:>6} {r['temp_C']:>4}C: "
                  f"I_avg={r['I_avg_nA']:.3f} nA, "
                  f"E={r['E_per_beat_nJ']:.3f} nJ/beat, "
                  f"P={r['P_avg_nW']:.3f} nW")

    # Save results
    output_path = os.path.join(SCRIPT_DIR, "full_frontend_energy_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'description': 'Full frontend energy from ngspice transient simulation with real MIT-BIH ECG',
            'record': record_id,
            'n_beats': n_beats,
            'vdd': VDD,
            'method': 'Full transient simulation — actual total VDD current, not OTA-count estimate',
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
