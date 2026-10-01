"""Validate analog frontend against digital feature extraction.

Runs the same MIT-BIH beats through:
  1. Digital Python extraction (same as the RF model training)
  2. Full analog ngspice frontend simulation

Compares feature values using Spearman rank correlation.
If correlations are high, the analog frontend preserves the discriminative
information and the classifier accuracy should transfer.
"""

import os, sys, re, json, subprocess
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MITDB_DIR = os.path.join(REPO_ROOT, 'mitdb_data')
NGSPICE_CON = "C:/Spice64/bin/ngspice_con.exe"

VDD = 0.9
VREF = VDD / 2.0
FS = 360

# AAMI beat classification
aami_map = {}
for sym in ['N','L','R','e','j']: aami_map[sym] = 0
for sym in ['A','a','J','S']: aami_map[sym] = 1
for sym in ['V','E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/','f','Q']: aami_map[sym] = 4

# Beat window
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130
PW_S, PW_E = 16, QRS_S

# The 8 selected features and their indices in the digital feature vector
SELECTED_FEATURES = [
    ('rr_ratio', 2),
    ('rr_asymmetry', 3),
    ('beat_instability_L0', 37),
    ('beat_instability_L1', 38),
    ('pwave_energy_L0', 30),    # testbench uses L0, not L1
    ('qrs_width_L0', 7),
    ('slope_ratio_L0', 17),
    ('qrs_symmetry_L0', 34),
]

# Analog feature node names (raw, before S&H)
ANALOG_NODES = [
    'f_rr_ratio',
    'f_rr_asym',
    'f_instab_l0',
    'f_instab_l1',
    'f_pwave',
    'f_width',
    'f_slope',
    'f_sym',
]


def load_record(record_id):
    """Load a full MIT-BIH record."""
    try:
        import wfdb
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wfdb', '-q'])
        import wfdb

    record_path = os.path.join(MITDB_DIR, record_id)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    return record, ann


def extract_digital_features(record, ann, beat_start, n_beats):
    """Extract the 8 selected features digitally for n_beats starting at beat_start.

    Returns: list of dicts with feature values, R-peak sample indices, labels.
    """
    from scipy.signal import butter, filtfilt

    signals = record.p_signal
    fs = record.fs
    num_leads = min(signals.shape[1], 2)

    # P-wave bandpass filter
    nyq = fs / 2
    b, a = butter(2, [3.0/nyq, 8.0/nyq], btype='band')
    pwave_filtered = np.zeros_like(signals)
    for lead in range(num_leads):
        pwave_filtered[:, lead] = filtfilt(b, a, signals[:, lead])

    # Get valid beats
    valid = []
    for idx, sym in zip(ann.sample, ann.symbol):
        if idx - WIN_L >= 0 and idx + WIN_R < len(signals) and sym in aami_map:
            valid.append((idx, sym))

    if beat_start + n_beats + 1 > len(valid):
        n_beats = len(valid) - beat_start - 1

    results = []
    prev_peak = [None, None]
    local_rrs = []

    # Process some earlier beats for EWMA/state warm-up
    warmup_start = max(0, beat_start - 10)

    for i in range(warmup_start, beat_start + n_beats):
        if i < 0 or i >= len(valid):
            continue

        idx, sym = valid[i]
        beat = signals[idx - WIN_L : idx + WIN_R, :]
        beat_pw = pwave_filtered[idx - WIN_L : idx + WIN_R, :]

        # RR intervals
        pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
        post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

        local_rrs.append(pre_rr)
        if len(local_rrs) > 20:
            local_rrs.pop(0)
        local_rr = np.mean(local_rrs) if local_rrs else 0.833

        rr_ratio = pre_rr / (local_rr + 1e-8)
        rr_asymmetry = pre_rr / (post_rr + 1e-8)

        # Per-lead features
        feat = {}
        for lead in range(num_leads):
            qrs = beat[QRS_S:QRS_E, lead]
            abs_qrs = np.abs(qrs)
            peak = np.max(abs_qrs) if len(abs_qrs) > 0 else 0.0

            # QRS width
            width = 0.0
            if peak > 1e-6:
                threshold = 0.3 * peak
                above = abs_qrs > threshold
                if np.any(above):
                    first = np.argmax(above)
                    last = len(above) - 1 - np.argmax(above[::-1])
                    width = (last - first) * 1000.0 / fs

            # Slope ratio
            dqrs = np.diff(qrs) * fs
            if len(dqrs) > 0:
                up = max(np.max(dqrs), 1e-8)
                down = max(abs(np.min(dqrs)), 1e-8)
                slope_ratio = up / (down + 1e-8)
            else:
                slope_ratio = 1.0

            # QRS symmetry
            if peak > 1e-6:
                peak_idx = np.argmax(abs_qrs)
                threshold = 0.3 * peak
                above = abs_qrs > threshold
                if np.any(above):
                    onset = np.argmax(above)
                    qrs_w = max(width, 1e-6) * fs / 1000.0
                    time_to_peak = peak_idx - onset
                    symmetry = np.clip(time_to_peak / (qrs_w + 1e-8), 0.0, 1.0)
                else:
                    symmetry = 0.5
            else:
                symmetry = 0.5

            # P-wave energy
            pw_seg = beat_pw[PW_S:PW_E, lead]
            pwave_energy = np.sum(pw_seg**2) / fs

            # Beat instability
            if prev_peak[lead] is not None and prev_peak[lead] > 1e-6:
                instability = abs(peak - prev_peak[lead]) / (prev_peak[lead] + 1e-8)
            else:
                instability = 0.0
            prev_peak[lead] = peak

            if lead == 0:
                feat['qrs_width_L0'] = width
                feat['slope_ratio_L0'] = slope_ratio
                feat['qrs_symmetry_L0'] = symmetry
                feat['pwave_energy_L0'] = pwave_energy
                feat['beat_instability_L0'] = instability
            else:
                feat['beat_instability_L1'] = instability

        feat['rr_ratio'] = rr_ratio
        feat['rr_asymmetry'] = rr_asymmetry
        feat['label'] = aami_map[sym]
        feat['rpeak_sample'] = idx
        feat['rpeak_time_s'] = idx / fs

        if i >= beat_start:
            results.append(feat)

    return results


def generate_validation_testbench(record, ann, beat_start, n_beats, batch_id):
    """Generate an ngspice testbench for a batch of beats.

    Includes .meas statements to capture S&H output voltages.
    Returns: (spice_filepath, sample_times, rpeak_times)
    """
    signals = record.p_signal
    fs = record.fs
    num_leads = min(signals.shape[1], 2)

    valid = []
    for idx, sym in zip(ann.sample, ann.symbol):
        if idx - WIN_L >= 0 and idx + WIN_R < len(signals) and sym in aami_map:
            valid.append((idx, sym))

    if beat_start + n_beats + 1 > len(valid):
        n_beats = len(valid) - beat_start - 1

    selected = valid[beat_start:beat_start + n_beats + 1]
    margin_samples = int(0.3 * fs)
    seg_start = max(0, selected[0][0] - margin_samples)
    seg_end = min(len(signals), selected[-1][0] + margin_samples)

    seg_signals = signals[seg_start:seg_end, :]
    n_samples = seg_signals.shape[0]
    time_s = np.arange(n_samples) / fs
    total_time = time_s[-1]

    rpeak_times = [(s[0] - seg_start) / fs for s in selected]

    # Scale ECG to analog range
    v_min, v_max = 0.1 * VDD, 0.8 * VDD
    scaled = []
    for lead in range(min(num_leads, 2)):
        sig = seg_signals[:, lead]
        sig_min, sig_max = np.percentile(sig, [1, 99])
        sig_norm = (sig - sig_min) / (sig_max - sig_min + 1e-12)
        sig_scaled = v_min + sig_norm * (v_max - v_min)
        sig_scaled = np.clip(sig_scaled, 0.05, VDD - 0.05)
        scaled.append(sig_scaled)
    if num_leads == 1:
        scaled.append(scaled[0].copy())

    # Format inline PWL
    def fmt_pwl(t, v, step=4):
        pairs = []
        for i in range(0, len(t), step):
            pairs.append(f"{t[i]:.6e} {v[i]:.6e}")
        return "\n+ ".join(pairs)

    ecg0_pwl = fmt_pwl(time_s, scaled[0])
    ecg1_pwl = fmt_pwl(time_s, scaled[1])

    # Generate timing signals
    dt = 0.001
    pulse_rpeak = 0.005
    qrs_window = 0.150
    qrs_pre = 0.050
    pw_window = 0.100
    pw_end_before = 0.020
    sh_delay = 0.200
    sh_pulse = 0.002

    def gen_timing(name, rpeak_times_list):
        times = [0.0]
        values = [0.0 if name != 'sample_clk_bar' else VDD]
        for rp in rpeak_times_list:
            if name == 'rpeak_trigger':
                t_on, t_off = rp, rp + pulse_rpeak
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
                t_on = rp + sh_delay
                t_off = rp + sh_delay + sh_pulse
            if t_on < 0:
                t_on = 0
            if t_off > total_time:
                continue
            rise = dt * 0.1
            if name == 'sample_clk_bar':
                times.extend([t_on - rise, t_on, t_off, t_off + rise])
                values.extend([VDD, 0.0, 0.0, VDD])
            else:
                times.extend([t_on - rise, t_on, t_off, t_off + rise])
                values.extend([0.0, VDD, VDD, 0.0])
        times.append(total_time)
        values.append(values[-1] if name != 'sample_clk_bar' else VDD)
        if name == 'sample_clk_bar':
            values[0] = VDD
        return times, values

    def gen_timing_custom(target_rpeaks, all_rpeaks, offset, dur, t_total, inverted):
        """Generate timing signal for specific rpeaks with custom offset."""
        times = [0.0]
        values = [VDD if inverted else 0.0]
        for rp in target_rpeaks:
            t_on = rp + offset
            t_off = t_on + dur
            if t_on < 0:
                t_on = 0
            if t_off > t_total:
                continue
            rise = 0.0001
            if inverted:
                times.extend([t_on - rise, t_on, t_off, t_off + rise])
                values.extend([VDD, 0.0, 0.0, VDD])
            else:
                times.extend([t_on - rise, t_on, t_off, t_off + rise])
                values.extend([0.0, VDD, VDD, 0.0])
        times.append(t_total)
        values.append(VDD if inverted else 0.0)
        return times, values

    timing = {}
    for name in ['rpeak_trigger', 'qrs_gate', 'pwave_gate', 'sample_clk', 'sample_clk_bar']:
        timing[name] = gen_timing(name, rpeak_times)

    def fmt_timing(t_list, v_list):
        pairs = [f"{t:.7e} {v:.6e}" for t, v in zip(t_list, v_list)]
        return "\n+ ".join(pairs)

    rpeak_pwl = fmt_timing(*timing['rpeak_trigger'])
    qrs_gate_pwl = fmt_timing(*timing['qrs_gate'])
    pwave_gate_pwl = fmt_timing(*timing['pwave_gate'])
    sample_clk_pwl = fmt_timing(*timing['sample_clk'])
    sample_clkb_pwl = fmt_timing(*timing['sample_clk_bar'])

    # Measure raw feature nodes at rp - 5ms (before rpeak resets integrators)
    # This gives the final computed value for integrator-based features
    # and the most recent value for rr/instab (which update at rpeak)
    # Skip first 2 R-peaks for warm-up
    meas_rpeaks = rpeak_times[2:-1]

    meas_lines = []
    sample_info = []
    for bi, rp in enumerate(meas_rpeaks):
        t_meas = rp - 0.005
        if t_meas < 0.02 or t_meas > total_time - 0.01:
            continue
        sample_info.append((bi, rp))
        for node in ANALOG_NODES:
            meas_lines.append(
                f".meas tran {node}_b{bi} FIND v({node}) AT={t_meas:.6e}"
            )
    meas_block = "\n".join(meas_lines)
    n_meas_beats = len(sample_info)

    spice = f"""* ANALOG vs DIGITAL VALIDATION - Batch {batch_id}
* MIT-BIH record, full frontend, SKY130 tt 27C

.title validate_batch_{batch_id}

.lib "sky130_minimal.lib.spice" tt
.option scale=1.0u
.option method=gear
.option gmin=1e-15
.option abstol=1e-15
.option reltol=0.005
.temp 27

.include "ota_5t_subthreshold.spice"
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

Vdd vdd 0 {VDD}
Vss vss 0 0
Xvref_gen vdd vss vref resistor_div
Vbias_src vbias 0 0.35

Vecg0 ecg_raw_L0 0 PWL(
+ {ecg0_pwl})
Vecg1 ecg_raw_L1 0 PWL(
+ {ecg1_pwl})

* Stage 1: Input Conditioning
Xbuf0 ecg_raw_L0 ecg_buf_L0 ecg_buf_L0 vdd vss vbias ota_5t
Xhpf0 ecg_buf_L0 ecg_hp_L0 vdd vss vbias otac_hpf cap_w=100 cap_l=100
Xlpf0 ecg_hp_L0 ecg_cond_L0 vdd vss vbias otac_lpf cap_w=17 cap_l=17

Xbuf1 ecg_raw_L1 ecg_buf_L1 ecg_buf_L1 vdd vss vbias ota_5t
Xhpf1 ecg_buf_L1 ecg_hp_L1 vdd vss vbias otac_hpf cap_w=100 cap_l=100
Xlpf1 ecg_hp_L1 ecg_cond_L1 vdd vss vbias otac_lpf cap_w=17 cap_l=17

* Stage 2: R-Peak Detection
Xbpf ecg_cond_L0 bpf_out vdd vss vbias otac_bpf cl_hpf=39 cw_hpf=39 cl_lpf=24 cw_lpf=24
Xrect bpf_out rect_out vref vdd vss vbias rectifier
Xenv rect_out env_out vdd vss vbias otac_lpf cap_w=77 cap_l=77
Xthresh env_out thresh_out vdd vss vbias otac_lpf cap_w=50 cap_l=50
Xcmp env_out thresh_out rpeak_internal vdd vss vbias comparator

* Timing signals (from annotations, not from R-peak detector)
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

* Stage 3: Feature Extraction
Xrr rpeak_trigger f_rr_ratio f_rr_asym vdd vss vbias rr_timing
Xinstab0 ecg_cond_L0 qrs_gate rpeak_trigger f_instab_L0 vdd vss vbias vref beat_instability
Xinstab1 ecg_cond_L1 qrs_gate rpeak_trigger f_instab_L1 vdd vss vbias vref beat_instability
Xpwave ecg_cond_L0 pwave_gate rpeak_trigger f_pwave vdd vss vbias vref pwave_energy
Xwidth vref ecg_cond_L0 qrs_gate rpeak_trigger f_width vdd vss vbias qrs_width
Xslope ecg_cond_L0 qrs_gate rpeak_trigger f_slope vdd vss vbias vref slope_ratio
Xsym qrs_gate rpeak_trigger f_width f_sym rpeak_trigger vdd vss vbias qrs_symmetry

* Stage 4: Sample-and-Hold
Xsh f_rr_ratio f_rr_asym f_instab_L0 f_instab_L1 f_pwave f_width f_slope f_sym
+ o_rr_ratio o_rr_asym o_instab_L0 o_instab_L1 o_pwave o_width o_slope o_sym
+ sample_clk sample_clk_bar vdd vss sh_bank

.tran 200u {total_time} uic

* Measure S&H outputs at each sample time
{meas_block}

.control
  run
  echo "=== VALIDATION COMPLETE ==="
  quit
.endc

.end
"""
    filepath = os.path.join(SCRIPT_DIR, f"_validate_batch_{batch_id}.spice")
    with open(filepath, 'w') as f:
        f.write(spice)

    return filepath, n_meas_beats, rpeak_times


def parse_meas_results(stdout, stderr, n_beats_expected):
    """Parse .meas tran results from ngspice output."""
    results = {}
    combined = stdout + '\n' + stderr
    for line in combined.split('\n'):
        line_stripped = line.strip().lower()
        for node in ANALOG_NODES:
            safe = node.replace('.', '_')
            pattern = f"{safe}_b(\\d+)\\s*=\\s*([-+]?\\d+\\.?\\d*e[+-]?\\d+)"
            m = re.search(pattern, line_stripped)
            if m:
                beat_idx = int(m.group(1))
                value = float(m.group(2))
                if node not in results:
                    results[node] = {}
                results[node][beat_idx] = value

    return results


def run_batch(record, ann, record_id, beat_start, n_beats, batch_id):
    """Run one batch: digital extraction + analog simulation."""
    print(f"\n--- Batch {batch_id}: record {record_id}, beats {beat_start}-{beat_start+n_beats-1} ---")

    # Digital extraction
    print("  Extracting digital features...")
    digital = extract_digital_features(record, ann, beat_start, n_beats)
    print(f"  Got {len(digital)} digital feature vectors")

    # Generate and run analog simulation
    print("  Generating testbench...")
    spice_file, n_meas_beats, rpeak_times = generate_validation_testbench(
        record, ann, beat_start, n_beats, batch_id)
    print(f"  Testbench has {n_meas_beats} measurement points")

    print(f"  Running ngspice (this may take 5-15 minutes)...")
    try:
        result = subprocess.run(
            [NGSPICE_CON, "-b", spice_file],
            capture_output=True, text=True,
            cwd=SCRIPT_DIR, timeout=1800
        )
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("  ERROR: ngspice timed out (30 min)")
        return None, None

    # Save raw output
    out_file = os.path.join(SCRIPT_DIR, f"_validate_output_{batch_id}.txt")
    with open(out_file, 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n=== STDERR ===\n")
        f.write(stderr)

    # Check for errors
    if 'error' in stderr.lower():
        err_lines = [l for l in stderr.split('\n') if 'error' in l.lower()]
        if any('singular' in l.lower() for l in err_lines):
            print(f"  WARNING: Convergence issues: {err_lines[0]}")

    # Parse analog results
    analog = parse_meas_results(stdout, stderr, n_meas_beats)
    n_parsed = sum(len(v) for v in analog.values())
    print(f"  Parsed {n_parsed} analog measurements across {len(analog)} nodes")

    if not analog:
        print("  ERROR: No analog measurements parsed!")
        return digital, None

    return digital, analog


def main():
    print("=" * 70)
    print("  ANALOG vs DIGITAL FEATURE VALIDATION")
    print("  MIT-BIH ECG + SKY130 Full Frontend")
    print("=" * 70)

    if not os.path.exists(NGSPICE_CON):
        print(f"ERROR: ngspice not found at {NGSPICE_CON}")
        return
    if not os.path.exists(MITDB_DIR):
        print(f"ERROR: MIT-BIH data not found at {MITDB_DIR}")
        return

    # Mix of DS1 and DS2 records for inter-patient diversity
    # ~8 usable beats per batch (9 in testbench, skip first/last)
    # beat_start=5 gives RR history warm-up
    ALL_BATCHES = [
        ('100', 5, 9),
        ('103', 5, 9),
        ('105', 5, 9),
        ('111', 5, 9),
        ('113', 5, 9),
        ('200', 5, 9),
        ('202', 5, 9),
        ('210', 5, 9),
        ('212', 5, 9),
        ('219', 5, 9),
        ('221', 5, 9),
        ('231', 5, 9),
        ('100', 20, 9),
        ('200', 20, 9),
    ]

    # --quick: 2 batches, --medium: 6 batches, default: all
    if '--quick' in sys.argv:
        batches = ALL_BATCHES[:2]
    elif '--medium' in sys.argv:
        batches = ALL_BATCHES[:6]
    else:
        batches = ALL_BATCHES

    all_digital = []
    all_analog = []

    for batch_id, (rec_id, beat_start, n_beats) in enumerate(batches):
        print(f"\nLoading record {rec_id}...")
        try:
            record, ann = load_record(rec_id)
        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")
            continue

        digital, analog = run_batch(record, ann, rec_id, beat_start, n_beats, batch_id)

        if digital is None or analog is None:
            print(f"  Skipping batch {batch_id} (simulation failed)")
            continue

        # Align digital and analog beats
        # Analog measurements skip first 2 rpeaks (indices 0,1) and last 1
        # So analog beat bi=0 corresponds to rpeak index 2 in the testbench
        # Digital features: index 2 corresponds to the same rpeak
        n_analog = max(len(v) for v in analog.values()) if analog else 0
        n_matched = 0

        for bi in range(n_analog):
            dig_idx = bi + 2  # digital beat index (skip first 2)
            if dig_idx >= len(digital):
                break

            dig = digital[dig_idx]
            ana = {}
            valid = True
            for node in ANALOG_NODES:
                if node in analog and bi in analog[node]:
                    ana[node] = analog[node][bi]
                else:
                    valid = False
                    break

            if valid:
                all_digital.append(dig)
                all_analog.append(ana)
                n_matched += 1

        print(f"  Matched {n_matched} beats for this batch")
        print(f"  Running total: {len(all_digital)} matched beats")

    # Analysis
    if len(all_digital) < 5:
        print("\nERROR: Too few matched beats for correlation analysis")
        return

    print(f"\n{'=' * 70}")
    print(f"  CORRELATION ANALYSIS ({len(all_digital)} beats)")
    print(f"{'=' * 70}")

    feature_pairs = [
        ('rr_ratio', 'f_rr_ratio'),
        ('rr_asymmetry', 'f_rr_asym'),
        ('beat_instability_L0', 'f_instab_l0'),
        ('beat_instability_L1', 'f_instab_l1'),
        ('pwave_energy_L0', 'f_pwave'),
        ('qrs_width_L0', 'f_width'),
        ('slope_ratio_L0', 'f_slope'),
        ('qrs_symmetry_L0', 'f_sym'),
    ]

    results_table = []
    for dig_name, ana_node in feature_pairs:
        dig_vals = np.array([d[dig_name] for d in all_digital])
        ana_vals = np.array([a[ana_node] for a in all_analog])

        # Remove NaN/inf
        mask = np.isfinite(dig_vals) & np.isfinite(ana_vals)
        dig_clean = dig_vals[mask]
        ana_clean = ana_vals[mask]

        if len(dig_clean) < 5:
            print(f"  {dig_name:25s}  SKIP (too few valid points)")
            continue

        rho, pval = spearmanr(dig_clean, ana_clean)
        pearson = np.corrcoef(dig_clean, ana_clean)[0, 1] if np.std(dig_clean) > 1e-10 and np.std(ana_clean) > 1e-10 else 0.0

        status = "OK" if abs(rho) > 0.7 else "WARN" if abs(rho) > 0.4 else "FAIL"

        print(f"  {dig_name:25s}  Spearman={rho:+.3f}  Pearson={pearson:+.3f}"
              f"  dig=[{np.min(dig_clean):.4f}, {np.max(dig_clean):.4f}]"
              f"  ana=[{np.min(ana_clean):.4f}, {np.max(ana_clean):.4f}]"
              f"  [{status}]")

        results_table.append({
            'feature': dig_name,
            'analog_node': ana_node,
            'spearman_rho': float(rho),
            'spearman_pval': float(pval),
            'pearson_r': float(pearson),
            'n_valid': int(np.sum(mask)),
            'dig_range': [float(np.min(dig_clean)), float(np.max(dig_clean))],
            'ana_range': [float(np.min(ana_clean)), float(np.max(ana_clean))],
            'dig_mean': float(np.mean(dig_clean)),
            'ana_mean': float(np.mean(ana_clean)),
            'status': status,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    n_ok = sum(1 for r in results_table if r['status'] == 'OK')
    n_warn = sum(1 for r in results_table if r['status'] == 'WARN')
    n_fail = sum(1 for r in results_table if r['status'] == 'FAIL')
    print(f"  Total beats compared: {len(all_digital)}")
    print(f"  Features OK (rho>0.7):   {n_ok}/{len(results_table)}")
    print(f"  Features WARN (0.4-0.7): {n_warn}/{len(results_table)}")
    print(f"  Features FAIL (rho<0.4): {n_fail}/{len(results_table)}")

    if n_ok == len(results_table):
        print("\n  PASS: All features show strong monotonic correlation.")
        print("  The analog frontend preserves discriminative information.")
        print("  Classifier accuracy should transfer from digital to analog.")
    elif n_fail == 0:
        print("\n  PARTIAL: Most features correlate, some weakly.")
        print("  Check WARN features for circuit issues or scaling problems.")
    else:
        print("\n  ISSUES: Some features show weak/no correlation.")
        print("  These circuits may need debugging before classifier can work.")

    # Per-class distribution check
    print(f"\n{'=' * 70}")
    print(f"  PER-CLASS ANALOG VOLTAGE RANGES")
    print(f"{'=' * 70}")
    labels = [d['label'] for d in all_digital]
    class_names = ['N', 'S', 'V', 'F', 'Q']
    for _, ana_node in feature_pairs:
        ana_vals = np.array([a[ana_node] for a in all_analog])
        print(f"  {ana_node:15s}", end="")
        for cls in range(5):
            mask = np.array(labels) == cls
            if np.any(mask):
                vals = ana_vals[mask]
                print(f"  {class_names[cls]}:{np.mean(vals):.3f}", end="")
        print()

    # Save results
    output = {
        'description': 'Analog vs digital feature validation',
        'n_beats': len(all_digital),
        'records': [b[0] for b in batches],
        'correlations': results_table,
    }
    out_path = os.path.join(SCRIPT_DIR, "validation_analog_vs_digital.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
