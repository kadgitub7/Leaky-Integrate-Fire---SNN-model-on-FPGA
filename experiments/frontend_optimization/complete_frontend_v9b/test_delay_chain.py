"""
Modular test: delay_timer + inverter buffers + pulse_gen chain
=============================================================
Tests the supplementary circuit chain in ISOLATION.
Feeds a clean rpeak_raw level (simulating rpeak_detector output),
measures when each trigger pulse fires, and compares to target timing.

Target trigger timing (from PWL test that gave rho 0.97-0.99):
  trig_s14:     R+14ms  (5 samples at 360Hz)
  trig_wid:     R+28ms  (10 samples)
  trig_s69:     R+69ms  (25 samples)
  trig_bi_copy: R+300ms
  trig_rr_copy: R+10ms  (fast timer)

Pipeline: rpeak_raw → delay_timer(comparator LEVELS) → inverter buffers → pulse_gen(PULSES)
"""

import subprocess, os, re, time
import numpy as np

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")

VDD = 0.9

def interp_sf(sf_data, target_ms):
    """Interpolate SF curve to get voltage at target time."""
    for i in range(len(sf_data) - 1):
        t0, v0 = sf_data[i]
        t1, v1 = sf_data[i+1]
        if t0 <= target_ms <= t1:
            frac = (target_ms - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    if target_ms < sf_data[0][0]:
        rate = (sf_data[0][1] - sf_data[1][1]) / (sf_data[1][0] - sf_data[0][0])
        return sf_data[0][1] + rate * (sf_data[0][0] - target_ms)
    return sf_data[-1][1]


def run_test():
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    # rpeak_raw: goes HIGH at t=100ms (R-peak), drops LOW at t=110ms
    # This simulates rpeak_detector: HIGH during QRS, drops ~10ms after R-peak
    # Timer starts discharging at rpeak_raw drop (t=110ms)
    rpk_rise = 0.100   # R-peak time
    rpk_drop = 0.110   # rpeak_raw drops (QRS end)
    rpk_drop_from_r = 10  # ms from R-peak to rpk_drop
    t_end = 0.600      # enough time for 300ms bi_copy trigger

    # Measured SF discharge curve from first run (main timer, 100pF)
    # (ms_after_rpk_drop, SF_voltage)
    sf_main = [(0, 0.48325), (1, 0.48293), (2, 0.48241), (5, 0.48084),
               (10, 0.47824), (15, 0.47563), (20, 0.47299), (30, 0.46777),
               (50, 0.45736), (70, 0.44698), (100, 0.43142), (150, 0.40563),
               (200, 0.38001), (300, 0.32922), (400, 0.27907)]

    sf_fast = [(0, 0.48325), (1, 0.46623), (2, 0.44104), (3, 0.41600),
               (5, 0.36623), (8, 0.29291), (10, 0.24477), (15, 0.12730)]

    # Compute thresholds for each target trigger time
    # Target timing is relative to R-peak. Timer starts at rpk_drop (R+10ms).
    # So discharge_time = target_from_R - rpk_drop_from_r
    targets = {
        's14': 14,    # R+14ms → discharge 4ms
        'wid': 28,    # R+28ms → discharge 18ms
        's69': 69,    # R+69ms → discharge 59ms
        'bi':  300,   # R+300ms → discharge 290ms
        'rrc': 10,    # R+10ms → discharge 0ms (fast timer)
    }

    # Pipeline delay: ~2.3ms for main timer triggers, ~1ms for fast timer
    # Compensate by making SF cross the threshold earlier
    PIPELINE_MAIN = 2.5  # ms (slightly overcompensate to hit target)
    PIPELINE_FAST = 1.0  # ms

    thresholds = {}
    print("THRESHOLD COMPUTATION (pipeline-compensated):")
    print(f"  rpk_drop is at R+{rpk_drop_from_r}ms")
    print(f"  Pipeline delay: main={PIPELINE_MAIN}ms, fast={PIPELINE_FAST}ms")
    print(f"  {'Trigger':>8s} {'Target':>8s} {'Discharge':>10s} {'Compensated':>12s} {'SF at t':>10s} {'Timer':>6s}")
    for name, target_ms in targets.items():
        discharge_ms = target_ms - rpk_drop_from_r
        if name == 'rrc':
            comp_ms = max(0.3, discharge_ms - PIPELINE_FAST)
            vth = interp_sf(sf_fast, comp_ms)
            timer = 'fast'
        else:
            comp_ms = max(0.5, discharge_ms - PIPELINE_MAIN)
            vth = interp_sf(sf_main, comp_ms)
            timer = 'main'
        thresholds[name] = vth
        print(f"  {name:>8s} R+{target_ms:3d}ms  {discharge_ms:6.1f}ms  {comp_ms:8.1f}ms  {vth:.5f}V  {timer:>6s}")

    sp = f""".title delay_chain_test
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 0.300

* rpeak_raw: simulates rpeak_detector output
* HIGH during QRS (100-110ms), LOW otherwise
Vrpk rpeak_raw 0 PWL(0 0 0.098 0 0.100 {VDD} 0.110 {VDD} 0.1105 0 {t_end} 0)

.include "{fp}/inverter.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/pulse_gen.spice"

* === DELAY TIMER ===
* Thresholds calibrated from measured SF curves
Vth1 vth_14 0 {thresholds['s14']:.6f}
Vth2 vth_69 0 {thresholds['s69']:.6f}
Vth3 vth_28 0 {thresholds['wid']:.6f}
Vth4 vth_300 0 {thresholds['bi']:.6f}
Vth5 vth_10 0 {thresholds['rrc']:.6f}

Xdelay rpeak_raw dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_copy_lev vdd vss vbias vth_14 vth_69 vth_28 vth_300 vth_10 delay_timer

* === INVERTER BUFFERS (sharpen slow comparator edges) ===
Xinv_s14a dt_s14_lev dt_s14_buf1 vdd vss inverter_sub
Xinv_s14b dt_s14_buf1 dt_s14_buf vdd vss inverter_sub
Xinv_s69a dt_s69_lev dt_s69_buf1 vdd vss inverter_sub
Xinv_s69b dt_s69_buf1 dt_s69_buf vdd vss inverter_sub
Xinv_wida dt_wid_lev dt_wid_buf1 vdd vss inverter_sub
Xinv_widb dt_wid_buf1 dt_wid_buf vdd vss inverter_sub
Xinv_bia  dt_bi_lev  dt_bi_buf1 vdd vss inverter_sub
Xinv_bib  dt_bi_buf1 dt_bi_buf vdd vss inverter_sub
Xinv_rrca dt_rr_copy_lev dt_rrc_buf1 vdd vss inverter_sub
Xinv_rrcb dt_rrc_buf1 dt_rrc_buf vdd vss inverter_sub

* === PULSE GENERATORS (convert level transitions to 0.5ms pulses) ===
Xpg_s14 dt_s14_buf trig_s14 vdd vss vbias pulse_gen
Xpg_s69 dt_s69_buf trig_s69 vdd vss vbias pulse_gen
Xpg_wid dt_wid_buf trig_wid vdd vss vbias pulse_gen
Xpg_bi  dt_bi_buf  trig_bi_copy vdd vss vbias pulse_gen
Xpg_rrc dt_rrc_buf trig_rr_copy vdd vss vbias pulse_gen

.tran 0.05m {t_end}

* === MEASUREMENTS ===

* 1. SF timer discharge curve (relative to rpk_drop at 110ms)
"""
    # Measure SF at multiple times after rpk_drop
    for dt_ms in [0, 1, 2, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400]:
        t = rpk_drop + dt_ms / 1000.0
        if t < t_end - 0.001:
            sp += f".meas tran sf_main_{dt_ms:03d}ms FIND V(xdelay.sf_timer) AT={t:.6f}\n"

    # Fast timer SF
    for dt_ms in [0, 1, 2, 3, 5, 8, 10, 15, 20]:
        t = rpk_drop + dt_ms / 1000.0
        if t < t_end - 0.001:
            sp += f".meas tran sf_fast_{dt_ms:03d}ms FIND V(xdelay.sf_timer2) AT={t:.6f}\n"

    # Fine-resolution trigger pulse measurements (0.5ms steps near expected times)
    # Each trigger should fire as a ~0.5ms pulse at the target time
    sp += "\n* 2. TRIGGER PULSE TIMING (0.5ms resolution)\n"

    # For each trigger, measure in a window around expected time
    # trig_rrc: expect at rpk+0ms (fast timer)
    for dt_10 in range(0, 40):  # 0 to 4ms in 0.1ms steps
        dt_ms = dt_10 / 10.0
        t = rpk_drop + dt_ms / 1000.0
        sp += f".meas tran trig_rrc_{dt_10:03d} FIND V(trig_rr_copy) AT={t:.7f}\n"

    # trig_s14: expect at rpk+4ms
    for dt_10 in range(0, 120):  # 0 to 12ms in 0.1ms steps
        dt_ms = dt_10 / 10.0
        t = rpk_drop + dt_ms / 1000.0
        sp += f".meas tran trig_s14_{dt_10:03d} FIND V(trig_s14) AT={t:.7f}\n"

    # trig_wid: expect at rpk+18ms
    for dt_10 in range(100, 300):  # 10 to 30ms in 0.1ms steps
        dt_ms = dt_10 / 10.0
        t = rpk_drop + dt_ms / 1000.0
        sp += f".meas tran trig_wid_{dt_10:03d} FIND V(trig_wid) AT={t:.7f}\n"

    # trig_s69: expect at rpk+59ms
    for dt_10 in range(500, 700):  # 50 to 70ms in 0.1ms steps
        dt_ms = dt_10 / 10.0
        t = rpk_drop + dt_ms / 1000.0
        sp += f".meas tran trig_s69_{dt_10:03d} FIND V(trig_s69) AT={t:.7f}\n"

    # trig_bi: expect at rpk+290ms
    for dt_10 in range(2800, 3000):  # 280 to 300ms in 0.1ms steps
        dt_ms = dt_10 / 10.0
        t = rpk_drop + dt_ms / 1000.0
        sp += f".meas tran trig_bi_{dt_10:03d} FIND V(trig_bi_copy) AT={t:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_test_delay_chain.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)

    print("Running delay chain testbench...")
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=300, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    with open(log) as f:
        text = f.read()

    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass

    # Check for errors
    errors = [l.strip() for l in text.split('\n')
              if 'error' in l.lower() and 'reltol' not in l.lower() and l.strip()]
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:5]: print(f"  {e}")

    print(f"\nExtracted {len(results)} measurements")

    # === ANALYSIS ===

    # 1. SF discharge curve (still useful as reference)
    print("\n" + "=" * 60)
    print("1. SOURCE FOLLOWER DISCHARGE CURVES")
    print("=" * 60)
    print(f"\n   Main timer (100pF):")
    print(f"   {'dt_ms':>6s} {'SF (V)':>8s}")
    for dt_ms in [0, 1, 2, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 400]:
        v = results.get(f'sf_main_{dt_ms:03d}ms')
        if v is not None:
            print(f"   {dt_ms:6d} {v:8.5f}")

    print(f"\n   Fast timer (2pF):")
    for dt_ms in [0, 1, 2, 3, 5, 8, 10, 15, 20]:
        v = results.get(f'sf_fast_{dt_ms:03d}ms')
        if v is not None:
            print(f"   {dt_ms:6d} {v:8.5f}")

    # 2. Find trigger pulse timing from fine-resolution data
    print("\n" + "=" * 60)
    print("2. TRIGGER PULSE TIMING (0.1ms resolution)")
    print("=" * 60)

    def find_pulse(prefix, start_idx, end_idx):
        """Find pulse start/end in 0.1ms steps."""
        pulse_start = None
        pulse_end = None
        pulse_peak = 0
        for idx in range(start_idx, end_idx):
            v = results.get(f'{prefix}_{idx:03d}')
            if v is not None:
                if v > pulse_peak:
                    pulse_peak = v
                if pulse_start is None and v > 0.45:
                    pulse_start = idx / 10.0  # convert to ms
                elif pulse_start is not None and pulse_end is None and v < 0.45:
                    pulse_end = idx / 10.0
        return pulse_start, pulse_end, pulse_peak

    trig_specs = [
        ('trig_rrc', 'RR copy', 10, 0, 40),
        ('trig_s14', 'S14',     14, 0, 120),
        ('trig_wid', 'Width',   28, 100, 300),
        ('trig_s69', 'S69',     69, 500, 700),
        ('trig_bi',  'BI copy', 300, 2800, 3000),
    ]

    print(f"\n   rpk_drop is at R+{rpk_drop_from_r}ms")
    print(f"\n   {'Trigger':>10s} {'Target':>8s} {'Pulse_ms':>10s} {'Width_ms':>10s} {'Actual':>10s} {'Error':>8s} {'Peak_V':>8s}")
    print("   " + "-" * 72)

    for prefix, label, target_ms, si, ei in trig_specs:
        ps, pe, pk = find_pulse(prefix, si, ei)
        if ps is not None:
            actual_ms = ps + rpk_drop_from_r
            error_ms = actual_ms - target_ms
            width_ms = pe - ps if pe else '?'
            width_s = f"{width_ms:.1f}" if isinstance(width_ms, float) else width_ms
            print(f"   {label:>10s} R+{target_ms:3d}ms  rpk+{ps:5.1f}ms  {width_s:>6s}ms  R+{actual_ms:5.1f}ms  {error_ms:+6.1f}ms  {pk:.3f}V")
        else:
            print(f"   {label:>10s} R+{target_ms:3d}ms  NO PULSE DETECTED (peak={pk:.3f}V)")

    # Print raw pulse data for debugging
    print("\n   Raw pulse traces (showing only HIGH values > 0.1V):")
    for prefix, label, target_ms, si, ei in trig_specs:
        highs = []
        for idx in range(si, ei):
            v = results.get(f'{prefix}_{idx:03d}')
            if v is not None and v > 0.1:
                highs.append((idx/10.0, v))
        if highs:
            print(f"\n   {label}:")
            for t, v in highs:
                bar = "#" * int(v * 40)
                print(f"     rpk+{t:6.1f}ms: {v:.4f}V {bar}")
        else:
            print(f"\n   {label}: all values < 0.1V")


if __name__ == '__main__':
    run_test()
