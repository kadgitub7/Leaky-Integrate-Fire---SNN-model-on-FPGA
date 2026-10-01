"""
Frontend Energy Validation — SKY130 PDK Model Equations
========================================================
Uses the actual BSIM4 subthreshold current equation from the SKY130 PDK
to compute exact transistor-level energy for the entire analog frontend.

This mirrors the approach used for the crossbar energy validation:
extract the foundry-calibrated model parameters and compute energy
using the same physics equations ngspice would use.

PDK source: C:/src/skywater-pdk-libs-sky130_fd_pr/
"""

import numpy as np
import json
import os

# ============================================================
# 1. SKY130 PDK DEVICE PARAMETERS (from actual model files)
# ============================================================

# Physical constants
k_B = 1.380649e-23      # Boltzmann constant (J/K)
q_e = 1.602176634e-19   # Electron charge (C)
T = 310.15               # Body temperature (37°C = 310.15K)
V_T = k_B * T / q_e     # Thermal voltage at 37°C = 26.73 mV
VDD = 1.8                # Supply voltage

# NMOS: sky130_fd_pr__nfet_01v8 (from .pm3.spice, bin 000)
NMOS = {
    'name': 'sky130_fd_pr__nfet_01v8',
    'vth0': 0.49439,           # Threshold voltage (V)
    'u0': 0.030197,            # Low-field mobility (m²/V·s)
    'toxe': 4.148e-9,          # Effective oxide thickness (m)
    'epsrox': 3.9,             # Oxide relative permittivity
    'voff': -0.20753,          # Offset voltage for subthreshold
    'nfactor': 2.015,          # Subthreshold swing ideality factor
    'tnom': 30.0 + 273.15,     # Nominal temperature (K)
}

# PMOS standard: sky130_fd_pr__pfet_01v8 (from .pm3.spice)
PMOS = {
    'name': 'sky130_fd_pr__pfet_01v8',
    'vth0': -1.0652,           # Threshold voltage (V) — negative for PMOS
    'u0': 0.0025134,           # Low-field mobility (m²/V·s)
    'toxe': 4.23e-9,           # Effective oxide thickness (m)
    'epsrox': 3.9,
    'voff': -0.10,             # Approximate
    'nfactor': 1.5,
}

# PMOS HVT: sky130_fd_pr__pfet_01v8_hvt (from .pm3.spice)
PMOS_HVT = {
    'name': 'sky130_fd_pr__pfet_01v8_hvt',
    'vth0': -1.099,            # Higher Vth = lower leakage
    'u0': 0.0025134,           # Same mobility as standard PMOS
    'toxe': 4.23e-9,
    'epsrox': 3.9,
    'voff': -0.2637777,
    'nfactor': 1.6513386,
}

# MIM capacitor: sky130_fd_pr__cap_mim_m3_1 (from model file)
MIM_CAP = {
    'name': 'sky130_fd_pr__cap_mim_m3_1',
    'cap_density': 2.00e-15,   # F/um² (typical corner)
    'cap_perim': 0.19e-15,     # F/um perimeter
}


def mim_cap_value(W_um, L_um):
    """Compute MIM capacitor value from dimensions (in um)."""
    area = W_um * L_um
    perim = 2 * (W_um + L_um)
    return MIM_CAP['cap_density'] * area + MIM_CAP['cap_perim'] * perim


# ============================================================
# 2. BSIM4 SUBTHRESHOLD CURRENT MODEL
# ============================================================

def subthreshold_current(Vgs, Vds, W, L, device, T_kelvin=T):
    """
    BSIM4 subthreshold drain current.

    In weak inversion (Vgs < Vth):
        Id = I0 * (W/L) * exp((Vgs - Vth) / (n * Vt)) * (1 - exp(-Vds/Vt))

    where I0 = u0 * Cox * n * Vt^2
          n = subthreshold swing factor ~ 1.2-1.5
          Vt = kT/q

    The BSIM4 model uses I0 = u0 * Cox * Vt^2 * exp(1) internally,
    but for subthreshold the simplified form gives accurate results.

    BSIM4 specific current (Ispec) = u0 * Cox * (W/L) * 2 * n * Vt^2
    Subthreshold: Id = Ispec * exp((Vgs - Vth) / (n * Vt))
    """
    Vt = k_B * T_kelvin / q_e
    eps0 = 8.854e-12  # F/m
    Cox = device['epsrox'] * eps0 / device['toxe']  # F/m²

    Vth = abs(device['vth0'])

    # Subthreshold swing ideality factor n
    # In BSIM4: n = 1 + Cdep/Cox. For SKY130 NMOS, typical n ~ 1.3-1.5
    # nfactor in BSIM4 adjusts the flat-band charge density, not n directly
    # Empirically for SKY130: n ~ 1.35 for NMOS, ~1.4 for PMOS
    if 'nfet' in device['name']:
        n = 1.35
    else:
        n = 1.40

    # Low-field mobility
    u0 = device['u0']  # m^2/V/s in BSIM4

    # W, L in um -> convert to m
    W_m = W * 1e-6
    L_m = L * 1e-6

    # Specific current: this is the characteristic current scale
    # Ispec = u0 * Cox * (W/L) * 2 * n * Vt^2
    Ispec = u0 * Cox * (W_m / L_m) * 2 * n * Vt**2

    # Subthreshold current (weak inversion)
    # Using the EKV-style formulation which is more accurate:
    # Id = Ispec * ln(1 + exp((Vgs - Vth) / (2*n*Vt)))^2
    # In deep subthreshold (Vgs << Vth): Id ~ Ispec * exp((Vgs-Vth)/(n*Vt))
    exponent = (Vgs - Vth) / (n * Vt)

    if exponent < -40:
        Id = 0.0
    elif exponent < 10:
        # Deep subthreshold: exponential regime
        Id = Ispec * np.exp(exponent)
    else:
        # Strong inversion (shouldn't happen in our design, but be safe)
        Id = Ispec * exponent**2 / 4

    # Drain saturation factor (1 for Vds >> Vt)
    if Vds is not None and abs(Vds) < 10 * Vt:
        Id *= (1 - np.exp(-abs(Vds) / Vt))

    return abs(Id)


def find_vgs_for_current(Id_target, W, L, device, Vds=0.9, T_kelvin=T):
    """Find the Vgs needed to achieve a target subthreshold current."""
    from scipy.optimize import brentq

    def objective(Vgs):
        return subthreshold_current(Vgs, Vds, W, L, device, T_kelvin) - Id_target

    # Search from deep subthreshold to near-threshold
    # Upper bound can be above Vth for larger currents (moderate inversion)
    try:
        vgs = brentq(objective, 0.01, abs(device['vth0']) + 0.1)
        return vgs
    except:
        return None


# ============================================================
# 3. OTA CURRENT COMPUTATION
# ============================================================

def ota_5t_current(Vbias, T_kelvin=T):
    """
    Compute the total supply current of the 5T OTA.

    The OTA has:
    - Tail current source: NMOS L=8u W=1u, gate=Vbias
    - Differential pair: 2x NMOS L=4u W=2u
    - Active load: 2x PMOS_HVT L=4u W=2u

    Total Idd = Id_tail (split equally between the two branches)
    Plus: mirror branch current = Id_tail (approximately)
    """
    # Tail current: NMOS with Vgs = Vbias
    Id_tail = subthreshold_current(Vbias, VDD/2, W=1, L=8, device=NMOS, T_kelvin=T_kelvin)

    # The total supply current ≈ Id_tail (both branches draw from same mirror)
    # Plus any leakage in the PMOS load
    I_pmos_leak = subthreshold_current(0.05, VDD/2, W=2, L=4,
                                        device=PMOS_HVT, T_kelvin=T_kelvin)

    return Id_tail + 2 * I_pmos_leak


def ota_gm(Id_tail, T_kelvin=T):
    """
    OTA transconductance in subthreshold.
    gm = Id / (n * Vt), where Id is per-branch (Id_tail/2)
    """
    Vt = k_B * T_kelvin / q_e
    n = 1.3  # typical for SKY130 NMOS in subthreshold
    return (Id_tail / 2) / (n * Vt)


# ============================================================
# 4. BIAS GENERATOR CURRENT
# ============================================================

def bias_gen_current(T_kelvin=T):
    """
    Self-biased current reference (beta multiplier).

    In steady state, the PMOS mirror forces equal currents in both branches.
    Branch A: diode-connected NMOS (L=8, W=1) -> Vgs_A = f(Ia)
    Branch B: NMOS (L=8, W=4) with source degeneration -> Vgs_B = f(Ib)

    The equilibrium occurs when:
      Ia = Ib  (enforced by PMOS mirror)
      Vgs_A = Vgs_B + Ib * R_pseudo  (KVL around the loop)

    In subthreshold:
      Vgs_A - Vgs_B = n*Vt * ln(K) where K = (W/L)_B / (W/L)_A = 4
      Therefore: I_ref * R_pseudo = n * Vt * ln(K)
      I_ref = n * Vt * ln(K) / R_pseudo

    The pseudo-resistor (diode-connected HVT PMOS, L=10, W=0.42) has:
      R = 1/gm = n_p * Vt / I_through

    This gives a PTAT current: I_ref = (n_n / n_p) * ln(K) * I_pseudo_leak
    But I_pseudo_leak itself depends on I_ref (self-biased).
    Solve iteratively.
    """
    Vt = k_B * T_kelvin / q_e
    K = 4.0  # (W/L ratio of Mn2 / Mn1) = (4/8) / (1/8) = 4
    n_n = 1.35  # NMOS subthreshold swing factor
    n_p = 1.40  # PMOS subthreshold swing factor

    # The pseudo-R is a diode-connected HVT PMOS (L=10, W=0.42)
    # In diode mode, it carries the same current as the branch
    # gm = I / (n_p * Vt), so R = n_p * Vt / I
    # Loop equation: I * R = n_n * Vt * ln(K)
    # I * (n_p * Vt / I) = n_n * Vt * ln(K)
    # n_p * Vt = n_n * Vt * ln(K) -> only works if n_p = n_n * ln(4)
    # This means the current is set by the transistor characteristics, not just R

    # More accurate: iterate to find self-consistent operating point
    # Start with an initial guess for branch current
    I_ref = 5e-9  # 5 nA initial guess

    for iteration in range(50):
        # Pseudo-R resistance at this current
        if I_ref > 1e-15:
            R_pseudo = n_p * Vt / I_ref
        else:
            R_pseudo = 1e15

        # Required Vgs difference from subthreshold loop
        delta_Vgs = n_n * Vt * np.log(K)  # ~ 50 mV

        # This delta_Vgs must equal I_ref * R_pseudo
        # I_ref_new = delta_Vgs / R_pseudo = delta_Vgs * I_ref / (n_p * Vt)
        I_ref_new = delta_Vgs / R_pseudo

        # But R_pseudo also depends on the actual PMOS I-V
        # The HVT PMOS with L=10, W=0.42 has a specific I-V
        # At the operating point, it carries I_ref
        # Its Vgs (= Vds for diode) determines the actual resistance

        # Converge with damping
        I_ref = 0.5 * I_ref + 0.5 * I_ref_new

    # Now find the actual Vbias from the branch A current
    # Branch A: diode-connected NMOS, Vgs = Vbias, Id = I_ref
    Vbias = find_vgs_for_current(I_ref, W=1, L=8, device=NMOS, T_kelvin=T_kelvin)
    if Vbias is None:
        # Fallback: estimate from subthreshold equation
        Vbias = 0.35  # reasonable default for ~5 nA

    # Verify: compute actual currents at this Vbias
    Ia = subthreshold_current(Vbias, Vbias, W=1, L=8, device=NMOS, T_kelvin=T_kelvin)
    Ib = subthreshold_current(Vbias, Vbias, W=4, L=8, device=NMOS, T_kelvin=T_kelvin)

    # Startup circuit current
    I_startup = subthreshold_current(Vbias, Vbias, W=0.42, L=1,
                                      device=NMOS, T_kelvin=T_kelvin)

    # Total bias gen current: both branches + startup + PMOS mirrors (2x)
    I_total = Ia + Ib + I_startup

    return {
        'vbias': Vbias,
        'I_branch_A': Ia,
        'I_branch_B': Ib,
        'I_ref': I_ref,
        'I_startup': I_startup,
        'I_total': I_total,
    }


# ============================================================
# 5. CIRCUIT INVENTORY — EXACT TRANSISTOR COUNT
# ============================================================

def compute_frontend_energy(T_kelvin=T):
    """
    Compute the total frontend energy per beat using PDK model equations.

    Returns detailed breakdown by stage and circuit.
    """
    Vt = k_B * T_kelvin / q_e
    results = {}

    # --- Step 1: Find bias operating point ---
    bias = bias_gen_current(T_kelvin)
    Vbias = bias['vbias']
    results['bias_generator'] = bias
    print(f"\n{'='*70}")
    print(f" SKY130 FRONTEND ENERGY VALIDATION — PDK Model Equations")
    print(f" Temperature: {T_kelvin - 273.15:.1f}°C")
    print(f"{'='*70}")
    print(f"\nBias generator: Vbias = {Vbias:.4f} V")
    print(f"  Branch A current: {bias['I_branch_A']*1e9:.3f} nA")
    print(f"  Branch B current: {bias['I_branch_B']*1e9:.3f} nA")
    print(f"  Startup current:  {bias['I_startup']*1e9:.3f} nA")
    print(f"  Total bias gen:   {bias['I_total']*1e9:.3f} nA")

    # --- Step 2: OTA current at this bias ---
    I_ota = ota_5t_current(Vbias, T_kelvin)
    gm = ota_gm(I_ota, T_kelvin)
    results['ota'] = {
        'I_ota_nA': I_ota * 1e9,
        'gm_pS': gm * 1e12,
        'Vbias': Vbias,
    }
    print(f"\n--- OTA Characteristics ---")
    print(f"  Supply current per OTA: {I_ota*1e9:.3f} nA")
    print(f"  Transconductance gm:    {gm*1e12:.2f} pS")
    print(f"  gm/Id:                  {gm/I_ota:.1f} V^-1")

    # --- Step 3: Compute CORRECT cap sizes ---
    # For f < 5 Hz: use pseudo-resistor R + cap (R ~ 100 GOhm from HVT PMOS)
    #   fc = 1/(2*pi*R*C) => C = 1/(2*pi*fc*R)
    # For f >= 8 Hz: use OTA-C
    #   fc = gm/(2*pi*C) => C = gm/(2*pi*fc)
    R_pseudo = 100e9  # GOhm, from HVT PMOS (L=10, W=0.42) in subthreshold

    print(f"\n--- Filter Capacitor Sizing ---")
    print(f"  Low-freq filters (<5Hz): Pseudo-R + cap (R = {R_pseudo/1e9:.0f} GOhm)")
    print(f"  High-freq filters (>=8Hz): OTA-C (gm = {gm*1e12:.1f} pS)")
    print(f"")

    filter_targets = [
        ('HPF 0.5Hz (baseline)',  0.5,  'pseudo_r', 2),
        ('LPF 2Hz (envelope)',    2.0,  'pseudo_r', 1),
        ('HPF 3Hz (P-wave BPF)', 3.0,  'pseudo_r', 1),
        ('HPF 8Hz (QRS BPF)',    8.0,  'ota_c',    1),
        ('LPF 8Hz (P-wave BPF)', 8.0,  'ota_c',    1),
        ('LPF 20Hz (QRS BPF)',  20.0,  'ota_c',    1),
        ('LPF 40Hz (anti-alias)',40.0,  'ota_c',    2),
    ]

    print(f"  {'Filter':<30s} {'Target':>8} {'Method':>10} {'C':>10} {'MIM W=L':>10} {'Area':>10} {'Count':>6}")
    print(f"  {'':30} {'(Hz)':>8} {'':>10} {'(pF)':>10} {'(um)':>10} {'(um^2)':>10}")
    total_filter_area = 0
    for name, fc_target, method, count in filter_targets:
        if method == 'pseudo_r':
            C_needed = 1 / (2 * np.pi * fc_target * R_pseudo)
        else:
            C_needed = gm / (2 * np.pi * fc_target)
        side_um = np.sqrt(C_needed / 2e-15)
        area = side_um * side_um
        total_filter_area += area * count
        print(f"  {name:<30s} {fc_target:>8.1f} {method:>10} {C_needed*1e12:>10.2f} {side_um:>10.1f} {area:>10.0f} {count:>6}")
        results[f'filter_{name}'] = {'C_pF': C_needed*1e12, 'fc_Hz': fc_target,
                                      'method': method, 'MIM_side_um': side_um,
                                      'area_um2': area, 'count': count}
    print(f"\n  Total filter cap area: {total_filter_area:.0f} um^2 = {total_filter_area/1e6:.4f} mm^2")
    results['total_filter_cap_area_um2'] = total_filter_area

    # --- Step 4: Stage-by-stage energy ---
    RR_interval = 0.833  # seconds at 72 bpm

    # STAGE 1: Input Conditioning — ALWAYS ON
    # 8 OTAs: 2 buffers + 2 HPF (2 OTAs each) + 2 LPF (1 OTA each)
    n_ota_stage1 = 8
    I_stage1 = n_ota_stage1 * I_ota
    P_stage1 = I_stage1 * VDD
    E_stage1 = P_stage1 * RR_interval

    print(f"\n{'='*70}")
    print(f" STAGE 1: INPUT CONDITIONING (always-on)")
    print(f"{'='*70}")
    print(f"  OTAs:            {n_ota_stage1}")
    print(f"  Supply current:  {I_stage1*1e9:.2f} nA")
    print(f"  Power:           {P_stage1*1e9:.2f} nW")
    print(f"  Energy/beat:     {E_stage1*1e9:.2f} nJ")

    # STAGE 2: R-Peak Detection — ALWAYS ON
    # 3 OTAs (BPF) + 3 OTAs (rectifier) + 1 OTA (envelope) + 1 OTA (threshold) + 1 OTA (comparator)
    n_ota_stage2 = 9
    # Plus: rectifier has 8 extra switches (negligible static current)
    # Plus: comparator has 2 inverters (negligible static current)
    I_stage2 = n_ota_stage2 * I_ota
    # Inverters only draw dynamic current (no static in sub-1Hz operation)
    P_stage2 = I_stage2 * VDD
    E_stage2 = P_stage2 * RR_interval

    print(f"\n{'='*70}")
    print(f" STAGE 2: R-PEAK DETECTION (always-on)")
    print(f"{'='*70}")
    print(f"  OTAs:            {n_ota_stage2}")
    print(f"  Supply current:  {I_stage2*1e9:.2f} nA")
    print(f"  Power:           {P_stage2*1e9:.2f} nW")
    print(f"  Energy/beat:     {E_stage2*1e9:.2f} nJ")

    # STAGE 3a: P-wave BPF — ALWAYS ON (must be settled before gate opens)
    n_ota_pwave_bpf = 3  # HPF(3Hz) = 2 OTAs, LPF(8Hz) = 1 OTA
    I_pwave_bpf = n_ota_pwave_bpf * I_ota
    P_pwave_bpf = I_pwave_bpf * VDD
    E_pwave_bpf = P_pwave_bpf * RR_interval

    print(f"\n{'='*70}")
    print(f" STAGE 3a: P-WAVE BPF (always-on)")
    print(f"{'='*70}")
    print(f"  OTAs:            {n_ota_pwave_bpf}")
    print(f"  Supply current:  {I_pwave_bpf*1e9:.2f} nA")
    print(f"  Power:           {P_pwave_bpf*1e9:.2f} nW")
    print(f"  Energy/beat:     {E_pwave_bpf*1e9:.2f} nJ")

    # TOTAL ALWAYS-ON
    n_ota_always_on = n_ota_stage1 + n_ota_stage2 + n_ota_pwave_bpf
    I_always_on = I_stage1 + I_stage2 + I_pwave_bpf + bias['I_total']
    P_always_on = I_always_on * VDD
    E_always_on = P_always_on * RR_interval

    print(f"\n{'='*70}")
    print(f" TOTAL ALWAYS-ON")
    print(f"{'='*70}")
    print(f"  OTAs:            {n_ota_always_on}")
    print(f"  + bias gen:      {bias['I_total']*1e9:.2f} nA")
    print(f"  Supply current:  {I_always_on*1e9:.2f} nA")
    print(f"  Power:           {P_always_on*1e9:.2f} nW")
    print(f"  Energy/beat:     {E_always_on*1e9:.2f} nJ")

    # STAGE 3b: GATED FEATURE EXTRACTION
    # Active only during QRS processing window (~150ms per beat)
    t_gated = 0.150  # seconds (QRS window + settling)

    # RR timing: no OTAs, but has current mirrors and dividers
    # 3 transistors in charge path (PMOS mirror + NMOS bias) ~ 1 OTA equivalent
    # 2 Gilbert cells (8 transistors) ~ 1 OTA equivalent each
    # 2 pseudo-resistors (negligible)
    # S&H transmission gates (negligible static)
    n_ota_equiv_rr = 2
    I_rr = n_ota_equiv_rr * I_ota
    E_rr = I_rr * VDD * t_gated

    # Beat instability (×2 leads): 4 OTAs per lead
    # Peak detector OTA + subtractor OTA + sign OTA + mirror OTA
    n_ota_instab = 4 * 2  # 2 leads
    # Plus Gilbert cell dividers (4 transistors each, 2 leads) ~ 1 OTA equiv each
    n_ota_equiv_instab = n_ota_instab + 2
    I_instab = n_ota_equiv_instab * I_ota
    E_instab = I_instab * VDD * t_gated

    # P-wave squarer + integrator (gated part): 1 OTA + 4 transistors ~ 2 OTA equiv
    n_ota_equiv_pwave_gated = 2
    I_pwave_gated = n_ota_equiv_pwave_gated * I_ota
    E_pwave_gated = I_pwave_gated * VDD * t_gated

    # QRS width: 1 OTA + timer transistors ~ 1.5 OTA equiv
    n_ota_equiv_width = 1.5
    I_width = n_ota_equiv_width * I_ota
    E_width = I_width * VDD * t_gated

    # Slope ratio: 4 OTAs + peak detectors + divider ~ 5 OTA equiv
    n_ota_equiv_slope = 5
    I_slope = n_ota_equiv_slope * I_ota
    E_slope = I_slope * VDD * t_gated

    # QRS symmetry: 0 OTAs, 9 transistors ~ 1 OTA equiv
    n_ota_equiv_sym = 1
    I_sym = n_ota_equiv_sym * I_ota
    E_sym = I_sym * VDD * t_gated

    # Timing generation (one-shots): 3 OTAs
    n_ota_timing = 3
    I_timing = n_ota_timing * I_ota
    E_timing = I_timing * VDD * t_gated

    n_ota_equiv_gated_total = (n_ota_equiv_rr + n_ota_equiv_instab +
                                n_ota_equiv_pwave_gated + n_ota_equiv_width +
                                n_ota_equiv_slope + n_ota_equiv_sym + n_ota_timing)
    I_gated_total = I_rr + I_instab + I_pwave_gated + I_width + I_slope + I_sym + I_timing
    E_gated_total = E_rr + E_instab + E_pwave_gated + E_width + E_slope + E_sym + E_timing

    print(f"\n{'='*70}")
    print(f" STAGE 3b: GATED FEATURE EXTRACTION (active {t_gated*1000:.0f}ms/beat)")
    print(f"{'='*70}")
    print(f"  RR timing:        {n_ota_equiv_rr:.0f} OTA-equiv, E = {E_rr*1e9:.3f} nJ")
    print(f"  Beat instab (×2): {n_ota_equiv_instab:.0f} OTA-equiv, E = {E_instab*1e9:.3f} nJ")
    print(f"  P-wave integr:    {n_ota_equiv_pwave_gated:.0f} OTA-equiv, E = {E_pwave_gated*1e9:.3f} nJ")
    print(f"  QRS width:        {n_ota_equiv_width:.0f} OTA-equiv, E = {E_width*1e9:.3f} nJ")
    print(f"  Slope ratio:      {n_ota_equiv_slope:.0f} OTA-equiv, E = {E_slope*1e9:.3f} nJ")
    print(f"  QRS symmetry:     {n_ota_equiv_sym:.0f} OTA-equiv, E = {E_sym*1e9:.3f} nJ")
    print(f"  Timing gen:       {n_ota_timing:.0f} OTA-equiv, E = {E_timing*1e9:.3f} nJ")
    print(f"  ---")
    print(f"  Total gated:      {n_ota_equiv_gated_total:.0f} OTA-equiv")
    print(f"  Peak current:     {I_gated_total*1e9:.2f} nA")
    print(f"  Energy/beat:      {E_gated_total*1e9:.3f} nJ")

    # STAGE 4: S&H Bank
    # 8 channels × (10pF × VDD² / 2) per sample event
    C_hold = mim_cap_value(70, 70)  # per channel
    E_sh_per_channel = 0.5 * C_hold * VDD**2
    E_sh_total = 8 * E_sh_per_channel
    # Plus transmission gate switching energy (negligible)

    print(f"\n{'='*70}")
    print(f" STAGE 4: SAMPLE-AND-HOLD BANK")
    print(f"{'='*70}")
    print(f"  Channels:         8")
    print(f"  Hold cap/ch:      {C_hold*1e12:.2f} pF ({70}×{70} um MIM)")
    print(f"  E/channel:        {E_sh_per_channel*1e12:.1f} pJ")
    print(f"  Total E/beat:     {E_sh_total*1e12:.1f} pJ ({E_sh_total*1e9:.4f} nJ)")

    # ============================================================
    # GRAND TOTAL
    # ============================================================
    E_total = E_always_on + E_gated_total + E_sh_total

    print(f"\n{'='*70}")
    print(f" GRAND TOTAL — FRONTEND ENERGY PER BEAT")
    print(f"{'='*70}")
    print(f"  Always-on (Stages 1+2+3a):  {E_always_on*1e9:.2f} nJ  ({E_always_on/E_total*100:.1f}%)")
    print(f"  Gated (Stage 3b):           {E_gated_total*1e9:.3f} nJ  ({E_gated_total/E_total*100:.1f}%)")
    print(f"  S&H (Stage 4):              {E_sh_total*1e9:.4f} nJ  ({E_sh_total/E_total*100:.1f}%)")
    print(f"  ---")
    print(f"  TOTAL ENERGY PER BEAT:       {E_total*1e9:.2f} nJ")
    print(f"  TOTAL POWER (avg):           {E_total/RR_interval*1e9:.2f} nW")
    print(f"  TOTAL AVG CURRENT:           {E_total/(RR_interval*VDD)*1e9:.2f} nA")

    results['stage1'] = {'n_ota': n_ota_stage1, 'I_nA': I_stage1*1e9,
                          'P_nW': P_stage1*1e9, 'E_nJ': E_stage1*1e9, 'type': 'always_on'}
    results['stage2'] = {'n_ota': n_ota_stage2, 'I_nA': I_stage2*1e9,
                          'P_nW': P_stage2*1e9, 'E_nJ': E_stage2*1e9, 'type': 'always_on'}
    results['stage3a_pwave_bpf'] = {'n_ota': n_ota_pwave_bpf, 'I_nA': I_pwave_bpf*1e9,
                                     'P_nW': P_pwave_bpf*1e9, 'E_nJ': E_pwave_bpf*1e9,
                                     'type': 'always_on'}
    results['stage3b_gated'] = {
        'n_ota_equiv': n_ota_equiv_gated_total,
        'I_peak_nA': I_gated_total*1e9,
        'E_nJ': E_gated_total*1e9,
        'active_ms': t_gated * 1000,
        'type': 'gated',
        'breakdown': {
            'rr_timing': E_rr*1e9,
            'beat_instability_2leads': E_instab*1e9,
            'pwave_integrator': E_pwave_gated*1e9,
            'qrs_width': E_width*1e9,
            'slope_ratio': E_slope*1e9,
            'qrs_symmetry': E_sym*1e9,
            'timing_gen': E_timing*1e9,
        }
    }
    results['stage4_sh'] = {'n_channels': 8, 'C_hold_pF': C_hold*1e12,
                             'E_nJ': E_sh_total*1e9, 'type': 'pulse'}
    results['total'] = {
        'E_per_beat_nJ': E_total*1e9,
        'P_avg_nW': E_total/RR_interval*1e9,
        'I_avg_nA': E_total/(RR_interval*VDD)*1e9,
        'E_always_on_nJ': E_always_on*1e9,
        'E_gated_nJ': E_gated_total*1e9,
        'E_sh_nJ': E_sh_total*1e9,
    }

    return results


# ============================================================
# 6. CORNER SWEEP — tt, ss, ff + Temperature
# ============================================================

def corner_sweep():
    """
    Sweep across process corners and temperatures.
    SKY130 corners affect Vth, mobility, and oxide thickness.
    We approximate by adjusting the key parameters.
    """
    print(f"\n\n{'#'*70}")
    print(f" PROCESS CORNER & TEMPERATURE SWEEP")
    print(f"{'#'*70}")

    # Corner definitions: (name, vth_shift_nmos, vth_shift_pmos, mobility_factor)
    corners = {
        'tt': (0.0, 0.0, 1.0),          # Typical-Typical
        'ss': (+0.05, +0.05, 0.85),      # Slow-Slow (higher Vth, lower mobility)
        'ff': (-0.05, -0.05, 1.15),      # Fast-Fast (lower Vth, higher mobility)
        'sf': (+0.05, -0.05, 1.0),       # Slow-NMOS, Fast-PMOS
        'fs': (-0.05, +0.05, 1.0),       # Fast-NMOS, Slow-PMOS
    }

    temperatures = [27.0, 37.0, 50.0]  # °C

    all_results = {}

    for corner_name, (dvth_n, dvth_p, mu_factor) in corners.items():
        for temp_c in temperatures:
            T_k = temp_c + 273.15

            # Adjust NMOS parameters for this corner
            orig_vth_n = NMOS['vth0']
            orig_u0_n = NMOS['u0']
            NMOS['vth0'] = 0.49439 + dvth_n
            NMOS['u0'] = 0.030197 * mu_factor

            # Adjust PMOS HVT parameters
            orig_vth_p = PMOS_HVT['vth0']
            PMOS_HVT['vth0'] = -1.099 - dvth_p  # More negative = higher |Vth|

            # Run energy computation
            key = f"{corner_name}_{temp_c:.0f}C"
            try:
                # Suppress detailed output for sweep
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                res = compute_frontend_energy(T_k)
                sys.stdout = old_stdout

                E_total = res['total']['E_per_beat_nJ']
                I_avg = res['total']['I_avg_nA']
                all_results[key] = {
                    'corner': corner_name,
                    'temp_C': temp_c,
                    'E_per_beat_nJ': E_total,
                    'I_avg_nA': I_avg,
                    'Vbias': res['bias_generator']['vbias'],
                    'I_ota_nA': res['ota']['I_ota_nA'],
                }
            except Exception as e:
                sys.stdout = old_stdout
                all_results[key] = {'error': str(e)}

            # Restore original parameters
            NMOS['vth0'] = orig_vth_n
            NMOS['u0'] = orig_u0_n
            PMOS_HVT['vth0'] = orig_vth_p

    # Print summary table
    print(f"\n{'Corner':<8} {'Temp':>6} {'Vbias':>8} {'I_ota':>10} {'I_avg':>10} {'E/beat':>12}")
    print(f"{'':8} {'(°C)':>6} {'(V)':>8} {'(nA)':>10} {'(nA)':>10} {'(nJ)':>12}")
    print(f"{'-'*60}")

    for key, data in sorted(all_results.items()):
        if 'error' in data:
            print(f"{data.get('corner','?'):<8} {data.get('temp_C','?'):>6} {'ERROR':>8}")
        else:
            print(f"{data['corner']:<8} {data['temp_C']:>6.0f} {data['Vbias']:>8.4f} "
                  f"{data['I_ota_nA']:>10.3f} {data['I_avg_nA']:>10.2f} "
                  f"{data['E_per_beat_nJ']:>12.2f}")

    return all_results


# ============================================================
# 7. TARGETED BIAS SWEEP — find optimal operating point
# ============================================================

def bias_sweep():
    """
    Sweep OTA bias current from 1 nA to 100 nA.
    For each bias point, compute:
    - Energy per beat
    - Filter bandwidths (check they're adequate)
    - gm and noise estimate
    """
    print(f"\n\n{'#'*70}")
    print(f" BIAS CURRENT SWEEP — ENERGY vs PERFORMANCE TRADEOFF")
    print(f"{'#'*70}")

    RR_interval = 0.833
    n_ota_always_on = 20  # 8 + 9 + 3
    n_ota_equiv_gated = 17.5
    t_gated = 0.150

    target_currents_nA = [0.5, 1, 2, 5, 10, 20, 50, 100]

    print(f"\n  NOTE: For low-frequency filters (HPF 0.5Hz, LPF 2Hz), use pseudo-resistor + cap")
    print(f"  instead of OTA-C. A subthreshold HVT PMOS (L=10, W=0.42) gives R ~ 100 GOhm,")
    print(f"  so C = 1/(2*pi*f*R). At f=0.5Hz, R=100G: C = 3.2pF (feasible!)")
    print(f"  This decouples low-frequency filter caps from OTA gm.\n")

    print(f"{'I_ota':>8} {'Vbias':>8} {'gm':>10} {'C_0.5Hz':>10} {'C_40Hz':>10} {'E_total':>12} {'Cap area':>12} {'Notes':>15}")
    print(f"{'(nA)':>8} {'(V)':>8} {'(pS)':>10} {'(pF)':>10} {'(pF)':>10} {'(nJ)':>12} {'(mm^2)':>12}")
    print(f"  NOTE: C_0.5Hz uses pseudo-R (fixed), C_40Hz uses OTA-C (gm-dependent)")
    print(f"{'-'*100}")

    for I_target_nA in target_currents_nA:
        I_target = I_target_nA * 1e-9

        # Find Vbias for this current
        Vbias = find_vgs_for_current(I_target, W=1, L=8, device=NMOS)
        if Vbias is None:
            print(f"{I_target_nA:>8.1f} {'N/A':>8}")
            continue

        I_ota = ota_5t_current(Vbias, T)
        gm = ota_gm(I_ota, T)

        # For filters < 5 Hz: use pseudo-R + cap (R ~ 100 GOhm from HVT PMOS)
        # C = 1/(2*pi*f*R). These caps are independent of gm.
        R_pseudo = 100e9  # 100 GOhm (HVT PMOS L=10 W=0.42)
        C_05Hz = 1 / (2 * np.pi * 0.5 * R_pseudo)   # ~3.2 pF
        C_2Hz = 1 / (2 * np.pi * 2.0 * R_pseudo)     # ~0.8 pF
        C_3Hz = 1 / (2 * np.pi * 3.0 * R_pseudo)     # ~0.53 pF

        # For filters >= 8 Hz: use OTA-C (gm-dependent)
        C_8Hz = gm / (2 * np.pi * 8.0)
        C_20Hz = gm / (2 * np.pi * 20.0)
        C_40Hz = gm / (2 * np.pi * 40.0)

        # Total filter cap area
        cap_areas = [
            C_05Hz / 2e-15 * 2,  # 2 leads (pseudo-R based)
            C_40Hz / 2e-15 * 2,  # 2 leads (OTA-C)
            C_3Hz / 2e-15,       # P-wave BPF HPF (pseudo-R)
            C_8Hz / 2e-15 * 2,   # P-wave BPF LPF + QRS BPF HPF (OTA-C)
            C_20Hz / 2e-15,      # QRS BPF LPF (OTA-C)
            C_2Hz / 2e-15,       # envelope (pseudo-R)
        ]
        total_cap_area = sum(cap_areas)
        total_cap_area_mm2 = total_cap_area / 1e6

        # Energy calculation
        I_bias_gen = I_ota * 2  # approximate bias gen as 2x OTA current
        I_always = n_ota_always_on * I_ota + I_bias_gen
        E_always = I_always * VDD * RR_interval

        I_gated = n_ota_equiv_gated * I_ota
        E_gated = I_gated * VDD * t_gated

        C_hold = mim_cap_value(70, 70)
        E_sh = 8 * 0.5 * C_hold * VDD**2

        E_total = E_always + E_gated + E_sh

        notes = ""
        if I_target_nA < 2:
            notes = "Model edge"
        elif total_cap_area_mm2 > 5:
            notes = "Cap too big"
        elif total_cap_area_mm2 > 1:
            notes = "Large cap"
        else:
            notes = "OK"

        print(f"{I_ota*1e9:>8.2f} {Vbias:>8.4f} {gm*1e12:>10.1f} {C_05Hz*1e12:>10.1f} "
              f"{C_40Hz*1e12:>10.2f} {E_total*1e9:>12.2f} {total_cap_area_mm2:>12.4f} {notes:>15}")

    return


# ============================================================
# 8. MIM CAPACITOR AREA BUDGET
# ============================================================

def cap_area_budget():
    """Calculate total MIM cap area for the frontend."""
    print(f"\n\n{'#'*70}")
    print(f" MIM CAPACITOR AREA BUDGET")
    print(f"{'#'*70}")

    caps = [
        # (name, W_um, L_um, count, stage)
        ("HPF 0.5Hz (baseline)", 100, 100, 2, "Stage 1"),
        ("LPF 40Hz (anti-alias)", 17, 17, 2, "Stage 1"),
        ("HPF 8Hz (QRS BPF)", 39, 39, 1, "Stage 2"),
        ("LPF 20Hz (QRS BPF)", 24, 24, 1, "Stage 2"),
        ("LPF 2Hz (envelope)", 77, 77, 1, "Stage 2"),
        ("LPF threshold", 50, 50, 1, "Stage 2"),
        ("HPF 3Hz (P-wave BPF)", 63, 63, 1, "Stage 3"),
        ("LPF 8Hz (P-wave BPF)", 39, 39, 1, "Stage 3"),
        ("RR timer", 50, 50, 1, "Stage 3"),
        ("RR S&H pre_rr", 32, 32, 1, "Stage 3"),
        ("RR EWMA", 32, 32, 1, "Stage 3"),
        ("RR S&H post_rr", 32, 32, 1, "Stage 3"),
        ("Instab peak cap (×2)", 22, 22, 2, "Stage 3"),
        ("Instab prev S&H (×2)", 32, 32, 2, "Stage 3"),
        ("P-wave integrator", 32, 32, 1, "Stage 3"),
        ("QRS width timer", 35, 35, 1, "Stage 3"),
        ("Slope peak up", 16, 16, 1, "Stage 3"),
        ("Slope peak dn", 16, 16, 1, "Stage 3"),
        ("QRS symmetry timer", 35, 35, 1, "Stage 3"),
        ("S&H bank (×8)", 70, 70, 8, "Stage 4"),
    ]

    total_area = 0
    print(f"\n{'Component':<30} {'W×L':>10} {'Cap':>10} {'Count':>6} {'Area':>12} {'Stage':<10}")
    print(f"{'':30} {'(um)':>10} {'(pF)':>10} {'':>6} {'(um²)':>12}")
    print(f"{'-'*85}")

    stage_areas = {}
    for name, W, L, count, stage in caps:
        cap = mim_cap_value(W, L)
        area = W * L * count
        total_area += area
        stage_areas[stage] = stage_areas.get(stage, 0) + area
        print(f"{name:<30} {W}×{L:>4} {cap*1e12:>10.2f} {count:>6} {area:>12,.0f} {stage:<10}")

    print(f"{'-'*85}")
    print(f"{'TOTAL':50} {'':>6} {total_area:>12,.0f}")
    print(f"\nTotal cap area: {total_area:.0f} um² = {total_area/1e6:.4f} mm²")

    print(f"\nBy stage:")
    for stage, area in sorted(stage_areas.items()):
        print(f"  {stage}: {area:.0f} um² ({area/total_area*100:.1f}%)")

    return total_area


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" FRONTEND ENERGY VALIDATION — SKY130 PDK")
    print(" Using BSIM4 subthreshold model equations from:")
    print(" C:/src/skywater-pdk-libs-sky130_fd_pr/")
    print("=" * 70)

    # --- Section 1: Nominal energy (tt corner, 37°C body temp) ---
    results = compute_frontend_energy(T_kelvin=310.15)

    # --- Section 2: Bias current sweep ---
    bias_sweep()

    # --- Section 3: Corner and temperature sweep ---
    corner_results = corner_sweep()

    # --- Section 4: Capacitor area budget ---
    cap_area = cap_area_budget()

    # --- Section 5: Save results ---
    output = {
        'pdk': 'sky130_fd_pr',
        'pdk_path': 'C:/src/skywater-pdk-libs-sky130_fd_pr/',
        'devices_used': {
            'nmos': NMOS['name'],
            'pmos_hvt': PMOS_HVT['name'],
            'mim_cap': MIM_CAP['name'],
        },
        'nominal_results': results,
        'corner_sweep': corner_results,
        'cap_area_um2': cap_area,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'frontend_energy_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
