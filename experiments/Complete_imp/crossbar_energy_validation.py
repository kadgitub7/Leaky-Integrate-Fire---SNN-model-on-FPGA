"""
CROSSBAR ENERGY VALIDATION using SKY130B ReRAM PDK Model
=========================================================
Uses the EXACT I-V equation from sky130_fd_pr_reram__reram_cell.va
(Verilog-A compact model) to compute validated energy per cell read,
per MAC, and total classifier energy.

This is equivalent to an ngspice transient simulation but computed
analytically from the calibrated PDK parameters. For a short read pulse
at low voltage (0.1-0.2V), the filament does not change (non-destructive
read), so the steady-state I-V equation is exact.

PDK source: skywater-pdk-libs-sky130_fd_pr_reram
Model file: cells/reram_cell/sky130_fd_pr_reram__reram_cell.va

Run: python experiments/Complete_imp/crossbar_energy_validation.py
"""

import numpy as np
import json
import os

# ================================================================
# SKY130B ReRAM PDK PARAMETERS (from .va.params.json)
# ================================================================
# These are the foundry-calibrated values, NOT estimates

I_K1           = 6.140e-5     # current calibration [A]
TOX            = 5.0e-9       # oxide thickness [m]
TFILAMENT_MAX  = 4.9e-9       # max filament = min resistance (LRS) [m]
TFILAMENT_MIN  = 3.3e-9       # min filament = max resistance (HRS) [m]
TFILAMENT_REF  = 4.7249e-9    # filament reference for calibration [m]
V_REF          = 0.430        # voltage calibration [V]
AREA_OX        = 0.1024e-12   # TE/BE overlap area [m^2]

# Filament dynamics (for read-disturb check)
EACT_GEN       = 1.501        # activation energy generation [eV]
EACT_REC       = 1.500        # activation energy recombination [eV]
VELOCITY_K1    = 150           # velocity calibration [m/s]
GAMMA_K0       = 16.5          # enhancement factor calibration
GAMMA_K1       = -1.25         # enhancement factor calibration
A0             = 0.25e-9       # atomic distance [m]
TEMP_0         = 300           # temperature [K]

# Physical constants
K_BOLTZMANN    = 1.380649e-23  # [J/K]
Q_ELECTRON     = 1.602176634e-19  # [C]
KT_OVER_Q      = (K_BOLTZMANN * TEMP_0) / Q_ELECTRON  # ~0.02585 eV at 300K

# ================================================================
# NMOS ACCESS TRANSISTOR MODEL (simplified for linear region)
# sky130_fd_pr__nfet_01v8: W=0.42u, L=0.15u
# ================================================================
# At VGS = 1.8V (wordline high), the NMOS is strongly ON
# Using simplified linear region: R_on = 1 / (mu_n * Cox * (W/L) * (VGS - Vth))
# For sky130 nfet_01v8: Vth ~ 0.49V, mu_n*Cox ~ 270 uA/V^2
# R_on ~ 1 / (270e-6 * (0.42/0.15) * (1.8 - 0.49)) = 1 / (270e-6 * 2.8 * 1.31)
#       ~ 1 / (990.4e-6) ~ 1010 ohms

NMOS_VTH       = 0.49         # threshold voltage [V]
NMOS_MU_COX    = 270e-6       # mu_n * Cox [A/V^2]
NMOS_W         = 0.42e-6      # width [m]
NMOS_L         = 0.15e-6      # length [m]
VGS_READ       = 1.8          # wordline voltage during read [V]

def nmos_ron(vgs=VGS_READ):
    """On-resistance of the access transistor in linear region"""
    return 1.0 / (NMOS_MU_COX * (NMOS_W / NMOS_L) * (vgs - NMOS_VTH))

# ================================================================
# ReRAM I-V MODEL (from Verilog-A line 98)
# ================================================================

def reram_current(v_reram, tfilament):
    """
    Exact I-V from SKY130B PDK Verilog-A model:
    I = I_k1 * exp(-(Tox - Tfilament)/(Tox - Tfilament_ref)) * sinh(V/V_ref)
    """
    exponent = -(TOX - tfilament) / (TOX - TFILAMENT_REF)
    return I_K1 * np.exp(exponent) * np.sinh(v_reram / V_REF)

def reram_conductance(tfilament):
    """Small-signal conductance dI/dV at V=0"""
    exponent = -(TOX - tfilament) / (TOX - TFILAMENT_REF)
    return I_K1 * np.exp(exponent) / V_REF

def tfilament_for_conductance(g_target):
    """Find Tfilament that gives a target conductance [S]"""
    g_ratio = g_target * V_REF / I_K1
    log_ratio = np.log(g_ratio)
    tfilament = TOX + log_ratio * (TOX - TFILAMENT_REF)
    return np.clip(tfilament, TFILAMENT_MIN, TFILAMENT_MAX)

def cell_current_with_nmos(v_bl, tfilament):
    """
    Actual cell current including NMOS access transistor.
    The NMOS and ReRAM form a series path: V_bl = V_nmos + V_reram.
    Solve iteratively for the voltage split.
    """
    ron = nmos_ron()
    # Simple iterative solver
    v_reram = v_bl * 0.9  # initial guess: most voltage across ReRAM
    for _ in range(50):
        i_reram = reram_current(v_reram, tfilament)
        v_nmos = i_reram * ron
        v_reram_new = v_bl - v_nmos
        if abs(v_reram_new - v_reram) < 1e-6:
            break
        v_reram = 0.5 * (v_reram + v_reram_new)  # damped update
    return reram_current(v_reram, tfilament), v_reram

# ================================================================
# FILAMENT GROWTH RATE (read-disturb check)
# ================================================================

def filament_growth_rate(v_reram, tfilament, temperature=TEMP_0):
    """
    dTfilament/dt from Verilog-A model lines 92-95.
    Positive = filament growing (toward LRS), negative = shrinking.
    For a read pulse, this should be near zero.
    """
    kT_q = (K_BOLTZMANN * temperature) / Q_ELECTRON
    gamma = GAMMA_K0 + GAMMA_K1 * ((TOX - tfilament) / 1e-9) ** 3
    rate = VELOCITY_K1 * (
        np.exp(-EACT_GEN / kT_q) * np.exp(gamma * A0 / TOX * v_reram / kT_q) -
        np.exp(-EACT_REC / kT_q) * np.exp(-gamma * A0 / TOX * v_reram / kT_q)
    )
    return rate

# ================================================================
# PARAMETER SWEEP
# ================================================================

# Sweep ranges
V_READ_SWEEP   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
T_READ_SWEEP   = [50e-9, 100e-9, 200e-9, 500e-9, 1e-6]
G_LRS_SWEEP_US = [10, 20, 30, 50, 75, 100, 120]  # target conductance in uS

# Classifier architecture MACs
ARCH_CONFIGS = {
    'MLP_20_10':  {'macs': 16*20 + 20*10 + 10*5,  'layers': 3, 'name': 'MLP 20-10'},
    'MLP_40_20':  {'macs': 16*40 + 40*20 + 20*5,  'layers': 3, 'name': 'MLP 40-20'},
    'MLP_48_24':  {'macs': 16*48 + 48*24 + 24*5,  'layers': 3, 'name': 'MLP 48-24'},
    'MLP_80_40':  {'macs': 16*80 + 80*40 + 40*5,  'layers': 3, 'name': 'MLP 80-40'},
    'SNN_20_10_s12': {'macs_fc1': 16*20, 'macs_per_step': 20*20 + 20*10 + 10*10 + 10*5,
                      'steps': 12, 'fire_rate': 0.165, 'name': 'SNN 20-10 s12'},
    'SNN_48_24_s12': {'macs_fc1': 16*48, 'macs_per_step': 48*48 + 48*24 + 24*24 + 24*5,
                      'steps': 12, 'fire_rate': 0.165, 'name': 'SNN 48-24 s12'},
}

# Energy model parameters (periphery)
E_WAKE_NJ  = {'opt': 0.5, 'nom': 1.5, 'pess': 3.0}
P_PERIPH_NW = {'opt': 20, 'nom': 50, 'pess': 100}

# Frontend energy
TIMING_POWER_NW  = 5.0
TIMING_DUR_S     = 0.833
MORPH_POWER_NW   = 95.0
MORPH_DUR_S      = 0.100
AFE_POWER_NW     = 80.0
AFE_DUR_S        = 0.833

def main():
    print("=" * 80)
    print("SKY130B ReRAM CROSSBAR ENERGY VALIDATION")
    print("Using foundry-calibrated Verilog-A model parameters")
    print("=" * 80)

    ron = nmos_ron()
    print(f"\nNMOS access transistor R_on = {ron:.1f} ohms (at VGS={VGS_READ}V)")

    # ================================================================
    # 1. CONDUCTANCE MAPPING: Tfilament -> G
    # ================================================================
    print("\n" + "=" * 80)
    print("1. CONDUCTANCE MAPPING (Tfilament -> Conductance)")
    print("=" * 80)
    print(f"{'Tfilament (nm)':>15} {'G_ss (uS)':>12} {'State':>8}")
    print("-" * 40)

    tfilaments = np.linspace(TFILAMENT_MIN, TFILAMENT_MAX, 20)
    for tf in tfilaments:
        g = reram_conductance(tf) * 1e6
        state = "HRS" if tf < 3.5e-9 else ("LRS" if tf > 4.5e-9 else "mid")
        print(f"{tf*1e9:15.3f} {g:12.3f} {state:>8}")

    # Full range
    g_min = reram_conductance(TFILAMENT_MIN) * 1e6
    g_max = reram_conductance(TFILAMENT_MAX) * 1e6
    print(f"\nPDK conductance range: {g_min:.3f} uS (HRS) to {g_max:.3f} uS (LRS)")
    print(f"Ratio (on/off): {g_max/g_min:.1f}")

    # ================================================================
    # 2. I-V CHARACTERISTICS at target conductance levels
    # ================================================================
    print("\n" + "=" * 80)
    print("2. CELL I-V AT TARGET CONDUCTANCE LEVELS")
    print("=" * 80)

    results = {}

    for g_target_us in G_LRS_SWEEP_US:
        g_target = g_target_us * 1e-6
        tf = tfilament_for_conductance(g_target)
        g_actual = reram_conductance(tf) * 1e6

        # Check if achievable
        if tf <= TFILAMENT_MIN or tf >= TFILAMENT_MAX:
            achievable = f"CLAMPED (actual={g_actual:.1f} uS)"
        else:
            achievable = f"OK (actual={g_actual:.1f} uS)"

        print(f"\n--- Target G = {g_target_us} uS | Tfilament = {tf*1e9:.3f} nm | {achievable} ---")
        print(f"  {'V_read':>8} {'I_ideal(uA)':>12} {'I_w/NMOS(uA)':>14} {'V_reram(V)':>10} {'P_cell(nW)':>10}")

        for v_read in V_READ_SWEEP:
            i_ideal = reram_current(v_read, tf)
            i_actual, v_reram = cell_current_with_nmos(v_read, tf)
            p_cell = v_read * i_actual  # power from supply perspective

            results[(g_target_us, v_read)] = {
                'i_actual': i_actual,
                'v_reram': v_reram,
                'p_cell': p_cell,
                'tfilament': tf,
                'g_actual_us': g_actual if tf > TFILAMENT_MIN and tf < TFILAMENT_MAX else g_actual,
            }

            print(f"  {v_read:8.3f} {i_ideal*1e6:12.4f} {i_actual*1e6:14.4f} {v_reram:10.4f} {p_cell*1e9:10.4f}")

    # ================================================================
    # 3. ENERGY PER CELL READ SWEEP
    # ================================================================
    print("\n" + "=" * 80)
    print("3. ENERGY PER CELL READ (V_read x t_read x G_LRS)")
    print("=" * 80)

    energy_table = {}

    print(f"\n{'G(uS)':>6} {'V_read':>7} {'t_read':>8} {'E_cell(fJ)':>11} {'E_cell(pJ)':>11}")
    print("-" * 50)

    for g_us in G_LRS_SWEEP_US:
        for v_read in V_READ_SWEEP:
            for t_read in T_READ_SWEEP:
                key = (g_us, v_read)
                if key not in results:
                    continue
                r = results[key]
                e_cell = r['p_cell'] * t_read  # energy = power * time
                energy_table[(g_us, v_read, t_read)] = e_cell

    # Print a focused subset (the most relevant combinations)
    print("\nFocused sweep (V_read = 0.10, 0.15, 0.20V):")
    print(f"{'G(uS)':>6} {'V_read':>7} {'t_read(ns)':>10} {'E_cell(fJ)':>11} {'I_cell(uA)':>11}")
    print("-" * 55)
    for g_us in [30, 50, 75, 100, 120]:
        for v_read in [0.10, 0.15, 0.20]:
            for t_read in [100e-9, 200e-9, 500e-9]:
                key = (g_us, v_read, t_read)
                if key in energy_table:
                    r = results[(g_us, v_read)]
                    print(f"{g_us:6d} {v_read:7.2f} {t_read*1e9:10.0f} "
                          f"{energy_table[key]*1e15:11.2f} {r['i_actual']*1e6:11.4f}")

    # ================================================================
    # 4. ENERGY PER MAC (4-row column sum)
    # ================================================================
    print("\n" + "=" * 80)
    print("4. ENERGY PER MAC (4-cell column, mixed conductances)")
    print("=" * 80)

    # A MAC = apply V_read to all 4 wordlines, sum currents on one bitline
    # Cells in a column have different weights = different conductances
    # For opt/nom/pess: use uniform conductance at min/median/max LRS
    mac_configs = {
        'optimistic':  {'g_list_us': [30, 30, 30, 30],  'label': 'All min LRS (30 uS)'},
        'nominal':     {'g_list_us': [50, 75, 75, 100], 'label': 'Mixed (50,75,75,100 uS)'},
        'pessimistic': {'g_list_us': [120, 120, 120, 120], 'label': 'All max LRS (120 uS)'},
        'typical_mixed': {'g_list_us': [30, 75, 100, 10], 'label': 'Typical (30,75,100,10-HRS)'},
    }

    print(f"\n{'Config':>16} {'V_read':>7} {'t_read(ns)':>10} {'E_MAC(fJ)':>10} {'E_MAC(pJ)':>10} {'I_total(uA)':>12}")
    print("-" * 72)

    mac_energies = {}

    for config_name, config in mac_configs.items():
        for v_read in V_READ_SWEEP:
            for t_read in T_READ_SWEEP:
                e_mac = 0
                i_total = 0
                for g_us in config['g_list_us']:
                    g = g_us * 1e-6
                    tf = tfilament_for_conductance(g)
                    i_cell, _ = cell_current_with_nmos(v_read, tf)
                    e_mac += v_read * i_cell * t_read
                    i_total += i_cell

                mac_energies[(config_name, v_read, t_read)] = e_mac

                if t_read == 200e-9:  # print at recommended t_read
                    print(f"{config_name:>16} {v_read:7.2f} {t_read*1e9:10.0f} "
                          f"{e_mac*1e15:10.2f} {e_mac*1e12:10.4f} {i_total*1e6:12.4f}")

    # ================================================================
    # 5. READ-DISTURB ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("5. READ-DISTURB ANALYSIS")
    print("=" * 80)
    print("Filament growth rate during read (should be near zero for non-destructive read)")
    print(f"\n{'G(uS)':>6} {'V_read':>7} {'Tfilament(nm)':>14} {'dTf/dt(m/s)':>14} "
          f"{'drift/1000reads':>18} {'drift_%':>10}")
    print("-" * 78)

    for g_us in [30, 75, 120]:
        tf = tfilament_for_conductance(g_us * 1e-6)
        for v_read in [0.10, 0.15, 0.20, 0.25, 0.30]:
            _, v_reram = cell_current_with_nmos(v_read, tf)
            rate = filament_growth_rate(v_reram, tf)
            # Drift over 1000 reads at 500ns each
            total_time = 1000 * 500e-9  # 0.5 ms
            drift = rate * total_time
            drift_pct = abs(drift) / (TFILAMENT_MAX - TFILAMENT_MIN) * 100

            safe = "OK" if drift_pct < 1.0 else ("WARN" if drift_pct < 5.0 else "FAIL")
            print(f"{g_us:6d} {v_read:7.2f} {tf*1e9:14.3f} {rate:14.4e} "
                  f"{drift*1e9:14.4f} nm {drift_pct:9.4f}% {safe}")

    # ================================================================
    # 6. RECOMMENDED OPERATING POINT
    # ================================================================
    print("\n" + "=" * 80)
    print("6. RECOMMENDED OPERATING POINT")
    print("=" * 80)

    # Find safe V_read (read-disturb < 1% after 10k reads)
    safe_points = []
    for v_read in V_READ_SWEEP:
        max_drift_pct = 0
        for g_us in [30, 75, 120]:
            tf = tfilament_for_conductance(g_us * 1e-6)
            _, v_reram = cell_current_with_nmos(v_read, tf)
            rate = filament_growth_rate(v_reram, tf)
            total_time = 10000 * 500e-9
            drift = rate * total_time
            drift_pct = abs(drift) / (TFILAMENT_MAX - TFILAMENT_MIN) * 100
            max_drift_pct = max(max_drift_pct, drift_pct)
        safe_points.append((v_read, max_drift_pct))
        print(f"  V_read={v_read:.2f}V: max drift after 10k reads = {max_drift_pct:.4f}%"
              f"  {'SAFE' if max_drift_pct < 1.0 else 'RISKY'}")

    # Select best safe V_read: use 0.15V for energy efficiency + adequate SNR
    # All voltages are safe (read-disturb negligible), so choose based on energy
    # 0.15V gives ~10 uA read current at 75 uS, sufficient for sense amp
    recommended_v = 0.15
    recommended_t = 200e-9  # good balance of speed and energy

    print(f"\n  RECOMMENDED: V_read = {recommended_v:.2f}V, t_read = {recommended_t*1e9:.0f}ns")

    # Compute E_per_MAC at recommended point
    for config_name in ['optimistic', 'nominal', 'pessimistic']:
        e = mac_energies.get((config_name, recommended_v, recommended_t))
        if e:
            print(f"    E_MAC ({config_name:>12}): {e*1e15:.2f} fJ = {e*1e12:.4f} pJ")

    # ================================================================
    # 7. TOTAL CLASSIFIER ENERGY
    # ================================================================
    print("\n" + "=" * 80)
    print("7. TOTAL CLASSIFIER ENERGY (per classification)")
    print("=" * 80)

    v_op = recommended_v
    t_op = recommended_t

    for arch_key, arch in ARCH_CONFIGS.items():
        print(f"\n--- {arch['name']} ---")

        if 'macs' in arch:
            # MLP: simple MAC count
            total_macs = arch['macs']
            n_layers = arch['layers']
            t_infer = n_layers * t_op  # one crossbar activation per layer

            for scenario in ['optimistic', 'nominal', 'pessimistic']:
                e_mac = mac_energies.get((scenario, v_op, t_op), 0)
                e_cells = total_macs * e_mac
                e_wake = E_WAKE_NJ[scenario[:3] if scenario != 'pessimistic' else 'pess'] * 1e-9
                p_periph = P_PERIPH_NW[scenario[:3] if scenario != 'pessimistic' else 'pess'] * 1e-9
                e_periph = p_periph * t_infer
                e_total_cls = e_cells + e_wake + e_periph

                print(f"  {scenario:>12}: E_cells={e_cells*1e9:.4f} nJ, "
                      f"E_wake={e_wake*1e9:.2f} nJ, E_periph={e_periph*1e15:.1f} fJ, "
                      f"E_total_cls={e_total_cls*1e9:.4f} nJ")
                print(f"{'':>16} MACs={total_macs}, t_infer={t_infer*1e9:.0f} ns, "
                      f"E/MAC={e_mac*1e15:.2f} fJ")

        else:
            # SNN: account for sparsity and timesteps
            macs_fc1 = arch['macs_fc1']
            macs_per_step = arch['macs_per_step']
            steps = arch['steps']
            fire_rate = arch['fire_rate']
            total_macs = macs_fc1 + macs_per_step * steps * fire_rate
            t_infer = (1 + steps * 2) * t_op  # fc1 + (fwd+rec per step)

            for scenario in ['optimistic', 'nominal', 'pessimistic']:
                e_mac = mac_energies.get((scenario, v_op, t_op), 0)
                e_cells = total_macs * e_mac
                e_wake = E_WAKE_NJ[scenario[:3] if scenario != 'pessimistic' else 'pess'] * 1e-9
                p_periph = P_PERIPH_NW[scenario[:3] if scenario != 'pessimistic' else 'pess'] * 1e-9
                e_periph = p_periph * t_infer
                e_total_cls = e_cells + e_wake + e_periph

                print(f"  {scenario:>12}: E_cells={e_cells*1e9:.4f} nJ, "
                      f"E_wake={e_wake*1e9:.2f} nJ, E_periph={e_periph*1e15:.1f} fJ, "
                      f"E_total_cls={e_total_cls*1e9:.4f} nJ")
                print(f"{'':>16} effective_MACs={total_macs:.0f} (fc1={macs_fc1} + "
                      f"{macs_per_step}*{steps}*{fire_rate:.3f}), "
                      f"t_infer={t_infer*1e9:.0f} ns")

    # ================================================================
    # 8. FULL-CHAIN ENERGY SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("8. FULL-CHAIN ENERGY SUMMARY (per heartbeat)")
    print("=" * 80)

    e_afe = AFE_POWER_NW * 1e-9 * AFE_DUR_S
    e_timing = TIMING_POWER_NW * 1e-9 * TIMING_DUR_S
    e_morph = MORPH_POWER_NW * 1e-9 * MORPH_DUR_S
    e_frontend = e_timing + e_morph

    print(f"  AFE (IA+filter+Rpeak):   {e_afe*1e9:.2f} nJ ({AFE_POWER_NW:.0f} nW * {AFE_DUR_S:.3f}s)")
    print(f"  Frontend timing:         {e_timing*1e9:.2f} nJ ({TIMING_POWER_NW:.0f} nW * {TIMING_DUR_S:.3f}s)")
    print(f"  Frontend morphology:     {e_morph*1e9:.2f} nJ ({MORPH_POWER_NW:.0f} nW * {MORPH_DUR_S:.3f}s)")
    print(f"  Frontend total:          {e_frontend*1e9:.2f} nJ")

    # Best MLP config
    best_mlp = ARCH_CONFIGS['MLP_48_24']
    total_macs = best_mlp['macs']
    for scenario, label in [('optimistic', 'opt'), ('nominal', 'nom'), ('pessimistic', 'pess')]:
        e_mac = mac_energies.get((scenario, v_op, t_op), 0)
        e_cells = total_macs * e_mac
        e_wake = E_WAKE_NJ[label] * 1e-9
        p_periph = P_PERIPH_NW[label] * 1e-9
        e_periph = p_periph * (best_mlp['layers'] * t_op)
        e_cls = e_cells + e_wake + e_periph
        e_total = e_afe + e_frontend + e_cls
        print(f"\n  MLP 48-24 ({scenario}):")
        print(f"    Classifier: {e_cls*1e9:.4f} nJ  (cells={e_cells*1e9:.4f}, wake={e_wake*1e9:.2f}, periph={e_periph*1e15:.0f} fJ)")
        print(f"    FULL CHAIN: {e_total*1e9:.2f} nJ")

    # ================================================================
    # 9. COMPARISON WITH OUR PREVIOUS ESTIMATES
    # ================================================================
    print("\n" + "=" * 80)
    print("9. VALIDATION vs PREVIOUS ESTIMATES")
    print("=" * 80)

    prev_e_mac_opt  = 0.3375  # pJ (V^2*G*t formula)
    prev_e_mac_nom  = 0.8438
    prev_e_mac_pess = 1.35

    for scenario, prev in [('optimistic', prev_e_mac_opt),
                            ('nominal', prev_e_mac_nom),
                            ('pessimistic', prev_e_mac_pess)]:
        e_mac_new = mac_energies.get((scenario, v_op, t_op), 0) * 1e12
        if e_mac_new > 0:
            ratio = e_mac_new / prev
            print(f"  E_MAC {scenario:>12}: prev={prev:.4f} pJ, PDK={e_mac_new:.4f} pJ, "
                  f"ratio={ratio:.3f}x {'(higher)' if ratio > 1 else '(lower)'}")

    print("\nNote: Previous estimates used V^2*G*t (ohmic model).")
    print("PDK model uses sinh(V/V_ref) nonlinearity + NMOS voltage drop.")
    print("The difference shows the importance of using calibrated device models.")

    # ================================================================
    # 10. EXPORT RESULTS
    # ================================================================
    export = {
        'pdk_source': 'sky130_fd_pr_reram__reram_cell.va',
        'recommended_v_read': recommended_v,
        'recommended_t_read': recommended_t,
        'nmos_ron_ohms': ron,
        'conductance_range_us': {
            'hrs_min': round(g_min, 4),
            'lrs_max': round(g_max, 4),
            'on_off_ratio': round(g_max / g_min, 1),
        },
        'e_per_mac_pj': {},
        'classifier_energy_nj': {},
    }

    for scenario in ['optimistic', 'nominal', 'pessimistic']:
        e = mac_energies.get((scenario, v_op, t_op), 0)
        export['e_per_mac_pj'][scenario] = round(e * 1e12, 6)

    for arch_key, arch in ARCH_CONFIGS.items():
        if 'macs' in arch:
            total_macs = arch['macs']
            for scenario in ['optimistic', 'nominal', 'pessimistic']:
                label = scenario[:3] if scenario != 'pessimistic' else 'pess'
                e_mac = mac_energies.get((scenario, v_op, t_op), 0)
                e_cells = total_macs * e_mac
                e_wake = E_WAKE_NJ[label] * 1e-9
                e_cls = e_cells + e_wake
                export['classifier_energy_nj'][f"{arch['name']}_{scenario}"] = round(e_cls * 1e9, 6)

    out_path = os.path.join(os.path.dirname(__file__), 'crossbar_energy_results.json')
    with open(out_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\nResults exported to: {out_path}")

    # ================================================================
    # 11. VALUES TO UPDATE IN BAKEOFF SCRIPTS
    # ================================================================
    print("\n" + "=" * 80)
    print("11. UPDATE THESE VALUES IN bakeoff_snn.py AND bakeoff_mlp.py")
    print("=" * 80)

    for scenario in ['optimistic', 'nominal', 'pessimistic']:
        e = mac_energies.get((scenario, v_op, t_op), 0)
        label = scenario.upper()[:3]
        if scenario == 'pessimistic':
            label = 'PESS'
        print(f"E_PER_MAC_PJ_{label} = {e*1e12:.6f}  # pJ, from PDK model at V_read={v_op}V, t_read={t_op*1e9:.0f}ns")

    print(f"\nV_READ = {v_op}")
    print(f"T_READ = {t_op}")

    # Also compute what G values map to these scenarios
    for scenario in ['optimistic', 'nominal', 'pessimistic']:
        config = mac_configs[scenario]
        g_avg = np.mean(config['g_list_us'])
        tf = tfilament_for_conductance(g_avg * 1e-6)
        g_actual = reram_conductance(tf) * 1e6
        if tf > TFILAMENT_MIN and tf < TFILAMENT_MAX:
            print(f"G_LRS_{scenario.upper()[:3] if scenario != 'pessimistic' else 'PESS'} = "
                  f"{g_actual:.1f}e-6  # uS, Tfilament={tf*1e9:.3f} nm")
        else:
            print(f"G_LRS_{scenario.upper()[:3] if scenario != 'pessimistic' else 'PESS'} = "
                  f"{g_actual:.1f}e-6  # uS, CLAMPED at PDK limit")


if __name__ == '__main__':
    main()
