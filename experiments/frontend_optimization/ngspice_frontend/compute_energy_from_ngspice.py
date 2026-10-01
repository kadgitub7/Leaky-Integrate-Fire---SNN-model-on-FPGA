"""Compute frontend energy from ngspice OTA current measurements.

Uses actual BSIM4 simulation data from ngspice + SKY130 PDK.
"""

import json
import os
import subprocess
import re
import sys

NGSPICE_CON = "C:/Spice64/bin/ngspice_con.exe"
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
VDD = 0.9
RR_INTERVAL = 0.833  # seconds at 72 bpm
GATED_DURATION = 0.150  # 150ms per beat for gated circuits


def run_ngspice(spice_file, working_dir=None):
    """Run ngspice in batch mode, return stdout."""
    if working_dir is None:
        working_dir = FRONTEND_DIR
    result = subprocess.run(
        [NGSPICE_CON, "-b", spice_file],
        capture_output=True, text=True,
        cwd=working_dir, timeout=120
    )
    return result.stdout, result.stderr


def parse_sweep_data(stdout):
    """Parse ngspice DC sweep output into (vbias, isupply) pairs."""
    pairs = []
    in_data = False
    for line in stdout.split('\n'):
        if '---' in line and 'Index' not in line and 'v-sweep' not in line:
            in_data = True
            continue
        if in_data and line.strip():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vbias = float(parts[1])
                    isupply = float(parts[2])
                    pairs.append((vbias, isupply))
                except ValueError:
                    continue
    return pairs


def find_bias_for_current(pairs, target_i):
    """Interpolate to find Vbias that gives target current."""
    for i in range(len(pairs) - 1):
        v1, i1 = pairs[i]
        v2, i2 = pairs[i+1]
        if i1 <= target_i <= i2 or i2 <= target_i <= i1:
            frac = (target_i - i1) / (i2 - i1)
            return v1 + frac * (v2 - v1)
    return None


def compute_frontend_energy(ota_current_nA, temp_C=27, corner="tt"):
    """Compute full frontend energy given per-OTA supply current.

    Architecture from the 8-feature optimized set:
    - Always-on: 20 OTAs (2-lead input conditioning + R-peak detection)
      - Per lead: 1 buffer OTA + 1 HPF OTA + 1 LPF OTA = 3 OTAs x 2 leads = 6
      - R-peak: 3 BPF OTAs + 1 rectifier (3 OTAs) + 1 envelope (1 OTA) + 1 comparator (1 OTA)
      -       = 6 per lead, but shared between leads -> ~9 OTAs
      - Timing: 2 OTAs for RR interval measurement
      - EWMA: 1 OTA
      - Misc: 2 OTAs (mid-rail ref gen uses resistor divider, not OTA)
      - Total: ~20 OTAs always on

    - Gated (active during beat window, ~150ms):
      - beat_instability: 4 OTAs + switches per lead x 2 = 8 OTAs
      - pwave_energy: 3 BPF OTAs + 1 integrator = 4 OTAs per lead (1 lead) = 4
      - qrs_width: 1 comparator OTA = 1
      - slope_ratio: 4 OTAs per lead (1 lead) = 4
      - qrs_symmetry: 0 OTAs (just current source + cap + divider)
      - S&H bank: negligible static current
      - Gilbert cell dividers: ~8 transistors each, negligible current
      - Total: ~17 gated OTAs (not 24 OTA-equiv as originally estimated)
    """
    i_ota = ota_current_nA * 1e-9  # Convert to Amps

    # Always-on power
    n_always_on = 20
    p_always_on = n_always_on * i_ota * VDD  # Watts
    e_always_on = p_always_on * RR_INTERVAL   # Joules per beat

    # Gated power
    n_gated = 17
    p_gated = n_gated * i_ota * VDD
    e_gated = p_gated * GATED_DURATION

    # S&H bank leakage (8 channels, ~0.1 pA per channel from switch leakage)
    e_sh = 8 * 0.1e-12 * VDD * RR_INTERVAL

    # Gilbert cell dividers (4 total, ~2 transistors conducting ~0.5 nA each during gated)
    e_gilbert = 4 * 2 * 0.5e-9 * VDD * GATED_DURATION

    e_total = e_always_on + e_gated + e_sh + e_gilbert

    return {
        'corner': corner,
        'temp_C': temp_C,
        'ota_current_nA': ota_current_nA,
        'ota_power_pW': i_ota * VDD * 1e12,
        'n_always_on': n_always_on,
        'n_gated': n_gated,
        'e_always_on_nJ': e_always_on * 1e9,
        'e_gated_nJ': e_gated * 1e9,
        'e_sh_nJ': e_sh * 1e9,
        'e_gilbert_nJ': e_gilbert * 1e9,
        'e_total_nJ': e_total * 1e9,
        'always_on_pct': e_always_on / e_total * 100,
        'gated_pct': e_gated / e_total * 100,
    }


def run_ota_sweep(corner="tt", temp_C=27):
    """Run ngspice OTA sweep at given corner and temperature."""

    spice_content = f"""* OTA current sweep - {corner} corner, T={temp_C}C
.title ota_sweep_{corner}_{temp_C}C

.lib "sky130_minimal.lib.spice" {corner}
.option scale=1.0u
.temp {temp_C}

.include "ota_5t_subthreshold.spice"

Vdd vdd 0 {VDD}
Vss vss 0 0
Vcm inp 0 {VDD/2}
Vdiff inn inp 0
XOTA inn inp vout vdd vss vbias ota_5t
Cload vout 0 10p
Vbias vbias 0 0.35

.dc Vbias 0.30 0.50 0.005

.control
  run
  print -i(Vdd)
  quit
.endc

.end
"""

    spice_file = os.path.join(FRONTEND_DIR, f"_temp_sweep_{corner}_{temp_C}.spice")
    with open(spice_file, 'w') as f:
        f.write(spice_content)

    try:
        stdout, stderr = run_ngspice(spice_file)
        pairs = parse_sweep_data(stdout)
        return pairs
    finally:
        try:
            os.remove(spice_file)
        except OSError:
            pass


def main():
    print("=" * 60)
    print("  Frontend Energy Validation - ngspice + SKY130 PDK")
    print("=" * 60)
    print()

    # Run nominal sweep
    print("Running nominal sweep (tt, 27C)...")
    pairs = run_ota_sweep("tt", 27)

    if not pairs:
        print("ERROR: No sweep data returned!")
        return

    print(f"  Got {len(pairs)} data points")
    print()

    # Print current table
    print("  Vbias(V)  |  I_OTA(nA)  |  P_OTA(pW)  |  gm_est(pS)")
    print("  ----------|-------------|-------------|-------------")
    for vb, isup in pairs:
        i_nA = isup * 1e9
        p_pW = isup * VDD * 1e12
        # Estimate gm from subthreshold: gm = Id/(2*n*Vt), n~1.35
        gm_pS = (isup/2) / (1.35 * 0.02585) * 1e12
        if vb in [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.42, 0.45, 0.50]:
            print(f"  {vb:.2f}      |  {i_nA:9.3f}  |  {p_pW:9.1f}  |  {gm_pS:9.2f}")

    # Find bias for target currents
    print()
    print("--- Target Operating Points ---")
    for target_nA in [0.25, 0.5, 1.0, 2.0, 3.0]:
        vb = find_bias_for_current(pairs, target_nA * 1e-9)
        if vb:
            print(f"  {target_nA:.1f} nA/OTA -> Vbias = {vb:.3f}V")

    # Compute energy at the nominal bias (0.35V -> 0.257 nA)
    print()
    print("=" * 60)
    print("  ENERGY BREAKDOWN (per heartbeat)")
    print("=" * 60)

    # Energy at Vbias = 0.35V (from sweep data)
    i_at_0p35 = None
    for vb, isup in pairs:
        if abs(vb - 0.35) < 0.001:
            i_at_0p35 = isup * 1e9
            break

    if i_at_0p35:
        result = compute_frontend_energy(i_at_0p35, 27, "tt")
        print(f"\n  At Vbias = 0.35V ({result['ota_current_nA']:.3f} nA/OTA):")
        print(f"    Always-on ({result['n_always_on']} OTAs): {result['e_always_on_nJ']:.3f} nJ ({result['always_on_pct']:.1f}%)")
        print(f"    Gated ({result['n_gated']} OTAs):         {result['e_gated_nJ']:.3f} nJ ({result['gated_pct']:.1f}%)")
        print(f"    S&H bank:                 {result['e_sh_nJ']:.4f} nJ")
        print(f"    Gilbert cells:            {result['e_gilbert_nJ']:.4f} nJ")
        print(f"    TOTAL:                    {result['e_total_nJ']:.3f} nJ/beat")
        print(f"    Average power:            {result['e_total_nJ'] / RR_INTERVAL:.3f} nW")

    # Also compute at 1 nA target (may need higher bias)
    vb_1nA = find_bias_for_current(pairs, 1e-9)
    if vb_1nA:
        i_at_1nA = 1.0
        result_1nA = compute_frontend_energy(i_at_1nA, 27, "tt")
        print(f"\n  At Vbias = {vb_1nA:.3f}V (1.0 nA/OTA target):")
        print(f"    Always-on ({result_1nA['n_always_on']} OTAs): {result_1nA['e_always_on_nJ']:.3f} nJ ({result_1nA['always_on_pct']:.1f}%)")
        print(f"    Gated ({result_1nA['n_gated']} OTAs):         {result_1nA['e_gated_nJ']:.3f} nJ ({result_1nA['gated_pct']:.1f}%)")
        print(f"    TOTAL:                    {result_1nA['e_total_nJ']:.3f} nJ/beat")
        print(f"    Average power:            {result_1nA['e_total_nJ'] / RR_INTERVAL:.3f} nW")

    # Corner and temperature sweep
    print()
    print("=" * 60)
    print("  CORNER / TEMPERATURE SWEEP")
    print("=" * 60)

    results_all = []
    corners_temps = [
        ("tt", 27), ("tt", 37), ("tt", 50),
        ("ss", 27), ("ss", 37),
        ("ff", 27), ("ff", 50),
    ]

    for corner, temp in corners_temps:
        print(f"\n  Running {corner}/{temp}C...", end=" ", flush=True)
        try:
            sweep = run_ota_sweep(corner, temp)
            if sweep:
                # Find current at Vbias = 0.35V
                i_035 = None
                for vb, isup in sweep:
                    if abs(vb - 0.35) < 0.001:
                        i_035 = isup * 1e9
                        break

                if i_035:
                    energy = compute_frontend_energy(i_035, temp, corner)
                    results_all.append(energy)
                    print(f"I_OTA = {i_035:.3f} nA, E = {energy['e_total_nJ']:.3f} nJ/beat")
                else:
                    print("No data at Vbias=0.35V")
            else:
                print("No sweep data")
        except Exception as e:
            print(f"Error: {e}")

    # Summary table
    print()
    print("=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Corner':>6} {'Temp':>5} | {'I_OTA(nA)':>10} | {'E_total(nJ)':>12} | {'P_avg(nW)':>10}")
    print(f"  {'------':>6} {'-----':>5} | {'----------':>10} | {'------------':>12} | {'----------':>10}")
    for r in results_all:
        print(f"  {r['corner']:>6} {r['temp_C']:>4}C | {r['ota_current_nA']:>10.3f} | {r['e_total_nJ']:>12.3f} | {r['e_total_nJ']/RR_INTERVAL:>10.3f}")

    if results_all:
        e_min = min(r['e_total_nJ'] for r in results_all)
        e_max = max(r['e_total_nJ'] for r in results_all)
        print(f"\n  Energy range: {e_min:.3f} - {e_max:.3f} nJ/beat")

    # Save results
    output_path = os.path.join(FRONTEND_DIR, "ngspice_energy_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'description': 'Frontend energy from ngspice + SKY130 PDK BSIM4 simulation',
            'vdd': VDD,
            'rr_interval': RR_INTERVAL,
            'gated_duration': GATED_DURATION,
            'nominal': results_all[0] if results_all else None,
            'all_corners': results_all,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
