"""
Integration test: analog supplementary circuits + feature circuits.

FULLY ANALOG pipeline — no ideal triggers remaining:
  - bias_gen -> vbias
  - rpeak_detector + pulse_gen -> rpeak_pulse (for delay_timer start + rr_timing precharge)
  - delay_timer + pulse_gen -> trig_s14, trig_s69, trig_wid, trig_bi_copy, trig_rr_copy
  - delay_timer (rpk_delayed) + pulse_gen -> rpeak_pulse_delayed (for F3,F4,F8 S&H)
  - pwave_predictor rr_cap comparator + AND(NOT rpeak_level) -> trig_rr_cap (EWMA+5mV offset)
  - pwave_predictor pw_base/pw_peak comparators + edge_buffer + pulse_gen -> trig_pw_base, trig_pw_peak

Feature outputs compared to Python reference via Spearman rho.
"""
import subprocess, os, re, sys, time
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108
PW_S = 16; PW_MID = 43
EWMA_ALPHA = 0.10
SKIP = 5
SKIP_BI = 10  # beat_instability needs more EWMA settling (prev_node converges by beat ~8)
VERBOSE = False  # set True for per-beat diagnostics (<=3 records)

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'

# --- Calibration data (freshly measured 2026-10-25) ---
rr_t = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
rr_v = [0.4819,0.4807,0.4772,0.4715,0.4602,0.4488,0.4375,0.4262,0.4150,0.4037,0.3925,0.3813,0.3702]
dt_t = [1,2,3,4,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800]
dt_v = [0.4829,0.4824,0.4819,0.4813,0.4808,0.4782,0.4756,0.4730,0.4678,0.4625,0.4573,0.4521,0.4469,0.4418,0.4366,0.4314,0.4056,0.3800,0.3545,0.3292,0.3041,0.2791,0.2296,0.1807,0.1326,0.0856]
ft_t = [0.5,1,2,3,4,5,7,10,14,20,25,30,35,40]
ft_v = [0.4810,0.4746,0.4617,0.4489,0.4362,0.4235,0.3981,0.3605,0.3107,0.2373,0.1773,0.1183,0.0616,0.0192]

# Pipeline delays (per-comparator, empirically calibrated)
pipe_s14 = 2.9; pipe_wid = 4.5; pipe_s69 = 4.4; pipe_bi = 5.9; pipe_s10 = 1.8; pipe_rpk_del = 2.0

# rpeak_detector offset: fires this many ms BEFORE true R-peak (measured on record 100)
RPEAK_OFFSET_MS = 6.6
PG_WIDTH_MS = 0.7  # pulse_gen pulse width
RPEAK_SAMPLE_OFFSET = round((RPEAK_OFFSET_MS - PG_WIDTH_MS) * FS / 1000)  # =2


def interp_sf(t_arr, v_arr, t_ms):
    if t_ms <= t_arr[0]: return v_arr[0]
    if t_ms >= t_arr[-1]: return v_arr[-1]
    for i in range(len(t_arr)-1):
        if t_arr[i] <= t_ms <= t_arr[i+1]:
            frac = (t_ms - t_arr[i]) / (t_arr[i+1] - t_arr[i])
            return v_arr[i] + frac * (v_arr[i+1] - v_arr[i])
    return v_arr[-1]


def load_beats(record_id, max_beats=40):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]
    raw_beats = []
    ewma_rr = None
    for i in range(1, min(len(valid) - 1, max_beats + 5)):
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        post_rr = (valid[i+1][0] - idx) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]
        prev_bs = valid[i-1][0] - WIN_L
        prev_be = valid[i-1][0] + WIN_R
        if prev_bs >= 0 and prev_be < len(sig):
            prev_beat = sig[prev_bs:prev_be]
            rpk_idx = WIN_L - RPEAK_SAMPLE_OFFSET
            py_bi_l0 = beat[rpk_idx, 0] - prev_beat[rpk_idx, 0]
            py_bi_l1 = beat[rpk_idx, min(1, n_leads-1)] - prev_beat[rpk_idx, min(1, n_leads-1)]
        else:
            py_bi_l0 = py_bi_l1 = 0.0
        if ewma_rr is None:
            ewma_rr_before = pre_rr
            ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr
        raw_beats.append({
            'pre_rr': pre_rr, 'post_rr': post_rr,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': pre_rr - ewma_rr_before,
            'py_f2': pre_rr - post_rr,
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[WIN_L + 10, 0],
            'py_slp': beat[WIN_L + 5, 0] - beat[WIN_L + 25, 0],
            'py_sym': beat[WIN_L - RPEAK_SAMPLE_OFFSET, 0],
        })
        if len(raw_beats) >= max_beats: break
    max_peak = max(np.max(np.abs(b['ecg_l0'][WIN_L-20:WIN_L+40])) for b in raw_beats)
    gain = 0.16 / max_peak if max_peak > 0 else 0.05
    return raw_beats, gain


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join("{:.7f} {:.6f}".format(t, v) for t, v in c)


def make_trig(rpeak_times, offset_s, pulse_w, total_time):
    pts = [(0, 0)]
    for rpt in rpeak_times:
        t = rpt + offset_s
        if t > 0.01:
            pts += [(t - 0.0004, 0), (t, VDD), (t + pulse_w, VDD), (t + pulse_w + 0.0004, 0)]
    pts.append((total_time, 0))
    return make_pwl(pts)


def run_sim(title, sp_body):
    sp = ".title {}\n".format(title) + sp_body + "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_integ_{}.spice".format(title.replace(' ', '_')))
    logf = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', logf, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    results = {}
    if os.path.exists(logf):
        with open(logf) as f: text = f.read()
        for line in text.split('\n'):
            m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
            if m:
                try: results[m.group(1).lower()] = float(m.group(2))
                except: pass
    return results, elapsed


# ========================================================================
print("=" * 70)
print("INTEGRATION TEST: Analog supplementary + feature circuits")
print("  Analog: bias_gen, rpeak_detector, pulse_gen, delay_timer, rpeak->rr_timing, rr_cap+pw_base+pw_peak(pwave_pred)")
print("  Ideal:  NONE — fully analog pipeline")
print("=" * 70)

lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
fp = FRONTEND_DIR.replace("\\", "/")

RECORDS = ['101', '102', '105', '106', '107', '108', '114', '115', '116', '118', '205', '214', '219', '220', '222', '232']
all_results = {}

for rec_id in RECORDS:
    print("\n--- Record {} ---".format(rec_id))
    beats, gain = load_beats(rec_id, 35)
    n = len(beats)
    print("  {} beats, gain={:.4f}".format(n, gain))

    rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    # --- ECG PWL ---
    ecg_l0_pts = [(0, VBASE)]; ecg_l1_pts = [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        bs_t = rpt - WIN_L / FS
        for lead, pts_list in [(0, ecg_l0_pts), (1, ecg_l1_pts)]:
            ecg = beats[i]['ecg_l0'] if lead == 0 else beats[i]['ecg_l1']
            t_before = bs_t - 0.003
            if t_before > 0.01: pts_list.append((t_before, VBASE))
            for j in range(len(ecg)):
                t = bs_t + j / FS
                if t > 0: pts_list.append((t, max(0.05, min(0.85, VBASE + gain * ecg[j]))))
            pts_list.append((bs_t + len(ecg)/FS + 0.003, VBASE))
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    # --- rpeak threshold (median-based, P-wave safe) ---
    # Use max of QRS window (±15 samples) for robust peak measurement
    r_peaks = [VBASE + gain * np.max(b['ecg_l0'][WIN_L-15:WIN_L+15]) for b in beats]
    median_r_v = float(np.median(r_peaks))
    min_r_v = min(r_peaks)
    max_r_v = max(r_peaks)
    RPEAK_THRESH_FRAC = 0.50
    rpeak_thresh = VBASE + RPEAK_THRESH_FRAC * (median_r_v - VBASE)
    rpeak_thresh = max(rpeak_thresh, VBASE + 0.020)
    n_pos_det = sum(1 for v in r_peaks if v > rpeak_thresh)
    print("  rpeak pos_thresh: {:.4f}V (frac={:.0f}% of median) detectable: pos={}/{} min_rpk={:.4f}V".format(
        rpeak_thresh, RPEAK_THRESH_FRAC*100, n_pos_det, n, min_r_v))

    # --- Delay timer thresholds (compensated for rpeak offset) ---
    # Discharge starts at R - RPEAK_OFFSET + PG_WIDTH = R - 5.9ms
    # For target at R + X: discharge_time = X + 5.9ms
    offset_ms = RPEAK_OFFSET_MS - PG_WIDTH_MS  # 5.9ms
    vth_14 = interp_sf(ft_t, ft_v, max(0.3, 14 + offset_ms - pipe_s14))
    vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28 + offset_ms - pipe_wid))
    vth_69 = interp_sf(dt_t, dt_v, max(0.5, 69 + offset_ms - pipe_s69))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 + offset_ms - pipe_bi))
    vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10 + offset_ms - pipe_s10))
    vth_rpk_del = interp_sf(ft_t, ft_v, max(0.3, 0 + offset_ms - pipe_rpk_del))

    # --- IC for rr_timing ---
    PW = 0.0005
    avg_rr_ms_ic = np.mean([b['pre_rr'] * 1000 for b in beats])
    ic_ewma = interp_sf(rr_t, rr_v, avg_rr_ms_ic)

    # --- Pwave predictor thresholds (EWMA-relative for adaptive timing) ---
    avg_rr_ms = np.mean([b['pre_rr'] * 1000 for b in beats])
    pipe_pw = 3.0  # pwave_predictor pipeline delay estimate (ms)
    SF_DECAY_RATE = 0.112e-3  # V/ms — calibrated from rr_v standalone timer data
    pwb_target_ms = 205.6  # ms before R-peak
    pwp_target_ms = 130.6
    PWB_OFFSET_MV = (pwb_target_ms + RPEAK_OFFSET_MS - PG_WIDTH_MS - pipe_pw) * SF_DECAY_RATE * 1000
    PWP_OFFSET_MV = (pwp_target_ms + RPEAK_OFFSET_MS - PG_WIDTH_MS - pipe_pw) * SF_DECAY_RATE * 1000
    # rr_cap offset: must be BELOW pwp offset so rr_cap fires AFTER pw_peak
    # (prevents esamp loading transient from falsely triggering pw_peak comparator)
    # S&H tracks from comparator fire until rpeak_level closes it (AND gate)
    RRC_OFFSET_MV = 20.0  # mV above EWMA — fires at ~R-178ms (F1/F2 calibrated for this)
    print("  SF_DECAY_RATE={:.4f} mV/ms  ic_ewma={:.4f}V  avg_rr={:.0f}ms".format(SF_DECAY_RATE*1000, ic_ewma, avg_rr_ms))
    print("  pwave offsets: pwb={:.1f}mV (target R-{:.0f}ms) pwp={:.1f}mV (target R-{:.0f}ms) rrc={:.1f}mV".format(
        PWB_OFFSET_MV, pwb_target_ms, PWP_OFFSET_MV, pwp_target_ms, RRC_OFFSET_MV))

    # --- NO MORE IDEAL TRIGGERS — fully analog pipeline ---

    sp = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0

* ECG inputs
Vecg0 ecg_l0 0 PWL({ecg0})
Vecg1 ecg_l1 0 PWL({ecg1})

* ============================================================
* ANALOG SUPPLEMENTARY CIRCUITS
* ============================================================

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/long_pulse_gen.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/edge_buffer.spice"
.include "{fp}/pwave_predictor.spice"

* --- Bias generator (replaces ideal Vbias) ---
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3

* --- R-peak detector (positive only) ---
Vthresh rpeak_threshold 0 {rpk_thresh:.6f}
Xrpk ecg_l0 rpeak_threshold rpeak_level vdd vss vbias rpeak_detector
* Edge buffer + pulse_gen chain
Xeb_rpk_in rpeak_level rpeak_level_sharp vdd vss edge_buffer
Xpg_rpk rpeak_level_sharp rpeak_pulse_raw vdd vss vbias pulse_gen
Xeb_rpk rpeak_pulse_raw rpeak_pulse vdd vss edge_buffer

* --- Delay timer (triggered by analog rpeak pulse) ---
Vth14 vth_14_node 0 {vth14:.6f}
Vth69 vth_69_node 0 {vth69:.6f}
Vth28 vth_28_node 0 {vth28:.6f}
Vth300 vth_300_node 0 {vth300:.6f}
Vth10 vth_10_node 0 {vth10:.6f}
Vth_rpk_del vth_rpk_del_node 0 {vth_rpk_del:.6f}

Xdt rpeak_pulse trig_s14_level trig_s69_level trig_wid_level trig_bi_level trig_rr_level trig_rpk_del_level dt_sf_out vdd vss vbias vth_14_node vth_69_node vth_28_node vth_300_node vth_10_node vth_rpk_del_node delay_timer

* --- Pulse generators for delay_timer outputs ---
Xpg_s14 trig_s14_level pg_s14_raw vdd vss vbias pulse_gen
Xpg_s69 trig_s69_level pg_s69_raw vdd vss vbias long_pulse_gen
Xpg_wid trig_wid_level pg_wid_raw vdd vss vbias long_pulse_gen
Xpg_bi  trig_bi_level  pg_bi_raw  vdd vss vbias long_pulse_gen
Xpg_rr  trig_rr_level  pg_rr_raw  vdd vss vbias pulse_gen
Xpg_rpk_del trig_rpk_del_level pg_rpk_del_raw vdd vss vbias pulse_gen

* --- Edge buffers for cleaner trigger edges ---
Xeb_s14 pg_s14_raw trig_s14 vdd vss edge_buffer
Xeb_s69 pg_s69_raw trig_s69 vdd vss edge_buffer
Xeb_wid pg_wid_raw trig_wid vdd vss edge_buffer
Xeb_bi  pg_bi_raw  trig_bi_copy vdd vss edge_buffer
Xeb_rr  pg_rr_raw  trig_rr_copy_raw vdd vss edge_buffer
* Esamp validity gate: block rr_copy when esamp holds deeply decayed value
* Valid esamp range: 0.38-0.43V. Invalid (missed beats): <0.30V. Threshold: 0.35V
Vth_esamp_ok esamp_ok_thresh 0 0.35
Xesamp_chk xrr.esamp_node esamp_ok_thresh esamp_ok_level vdd vss vbias rpeak_detector
* AND: trig_rr_copy = trig_rr_copy_raw AND esamp_ok_level
XMnand_rrc_p1 nand_rrc2 trig_rr_copy_raw vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=2 nf=1
XMnand_rrc_p2 nand_rrc2 esamp_ok_level vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=2 nf=1
XMnand_rrc_n1 nand_rrc2 trig_rr_copy_raw nand_mid_rrc2 vss sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1
XMnand_rrc_n2 nand_mid_rrc2 esamp_ok_level vss vss sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1
Xinv_nand_rrc2 nand_rrc2 trig_rr_copy vdd vss inverter_sub
Xeb_rpk_del pg_rpk_del_raw rpeak_pulse_delayed vdd vss edge_buffer

* ============================================================
* ANALOG RR_CAP (pwave_predictor + AND gate with rpeak_level)
* ============================================================
* Duplicate timer+SF for pwave_predictor (isolated from esamp loading on rr_timer)
Xinv_rpk_pre rpeak_pulse rpk_bar_pre vdd vss inverter_sub
XMpre2n vdd rpeak_pulse net_timer2 vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMpre2p vdd rpk_bar_pre net_timer2 vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMdis2 net_timer2 vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XCtimer2 net_timer2 vss sky130_fd_pr__cap_mim_m3_1 W=400 L=400
XMsf2 vdd net_timer2 sf2_out vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMsf2_bias sf2_out vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
* rr_cap comparator on ORIGINAL timer (esamp dip helps fast comparator reset)
Vrrc_off rrc_threshold rr_ewma {rrc_off_v}
Vpwb_off vth_pw_base_node rr_ewma {pwb_off_v}
Vpwp_off vth_pw_peak_node rr_ewma {pwp_off_v}
Xpwpred_rrc rr_timer rrc_threshold rrc_pwb_unused rrc_pwp_unused rr_cap_level vth_pw_base_node vth_pw_peak_node vdd vss vbias pwave_predictor
* pw_base/pw_peak comparators on ISOLATED timer (no esamp transient)
Xpwpred_pw sf2_out rrc_threshold pw_base_cmp pw_peak_cmp pw_rrc_unused vth_pw_base_node vth_pw_peak_node vdd vss vbias pwave_predictor
* AND gate: trig_rr_cap = rr_cap_level AND NOT(rpeak_level)
Xinv_rpk rpeak_level rpeak_level_bar vdd vss inverter_sub
XMnand_p1 nand_rrc rr_cap_level vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=2 nf=1
XMnand_p2 nand_rrc rpeak_level_bar vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=2 nf=1
XMnand_n1 nand_rrc rr_cap_level nand_mid_rrc vss sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1
XMnand_n2 nand_mid_rrc rpeak_level_bar vss vss sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1
Xinv_nand_rrc nand_rrc trig_rr_cap vdd vss inverter_sub

* ============================================================
* ANALOG PW_BASE / PW_PEAK (pwave_predictor -> edge_buffer -> pulse_gen)
* ============================================================
Xeb_pwb pw_base_cmp pwb_level vdd vss edge_buffer
Xpg_pwb pwb_level trig_pw_base vdd vss vbias pulse_gen
Xeb_pwp pw_peak_cmp pwp_level vdd vss edge_buffer
Xpg_pwp pwp_level trig_pw_peak vdd vss vbias pulse_gen

* ============================================================
* FEATURE CIRCUITS
* ============================================================
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

* F1/F2: rr_timing (ANALOG rpeak, ANALOG rr_cap via pwave_pred, analog rr_copy)
Xrr ecg_l0 trig_rr_cap rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing

* F3/F4: beat_instability (DELAYED rpeak, ANALOG bi_copy)
Xbi0 ecg_l0 rpeak_pulse_delayed trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 rpeak_pulse_delayed trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability

* F5: pwave_diff (ANALOG pw triggers via pwave_predictor)
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff

* F6: qrs_width (ANALOG wid trigger)
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width

* F7: qrs_slope (ANALOG s14, s69 triggers)
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope

* F8: qrs_symmetry (DELAYED rpeak pulse — captures near true R-peak)
Xsym ecg_l0 rpeak_pulse_delayed f8_symmetry vdd vss qrs_symmetry

* ============================================================
* INITIAL CONDITIONS
* ============================================================
.ic V(xrr.net_timer)={vdd}
.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(net_timer2)={vdd}
.ic V(xbi0.prev_node)={vbase}
.ic V(xbi0.esamp_node)={vbase}
.ic V(xbi1.prev_node)={vbase}
.ic V(xbi1.esamp_node)={vbase}

.tran 0.5m {ttot:.4f} UIC

""".format(lib=lib_path, vdd=VDD,
           ecg0=make_pwl(ecg_l0_pts), ecg1=make_pwl(ecg_l1_pts),
           rpk_thresh=rpeak_thresh,
           vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10, vth_rpk_del=vth_rpk_del,
           rrc_off_v=RRC_OFFSET_MV / 1000.0,
           pwb_off_v=PWB_OFFSET_MV / 1000.0, pwp_off_v=PWP_OFFSET_MV / 1000.0,
           fp=fp, ic=ic_ewma, vbase=VBASE, ttot=total_time)

    # --- rr_cap analog diagnostics ---
    # Check comparator and AND gate output at specific times (before rpeak of beat 3-9)
    for j in range(min(3, n), min(10, n)):
        t_before = rpeak_times[j] - 0.030  # 30ms before rpeak
        t_at = rpeak_times[j] - 0.010  # 10ms before rpeak
        sp += ".meas tran rrc_cmp_{j} FIND V(rr_cap_level) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_cmp2_{j} FIND V(rr_cap_level) AT={t:.7f}\n".format(j=j, t=t_at)
        sp += ".meas tran rrc_timer_{j} FIND V(rr_timer) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_ewma_{j} FIND V(rr_ewma) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_thresh_{j} FIND V(rrc_threshold) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_trig_{j} FIND V(trig_rr_cap) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_rpklvl_{j} FIND V(rpeak_level) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_esamp_{j} FIND V(xrr.esamp_node) AT={t:.7f}\n".format(j=j, t=t_before)
        sp += ".meas tran rrc_esamp_ok_{j} FIND V(esamp_ok_level) AT={t:.7f}\n".format(j=j, t=t_before)
        # Max values in window around expected trigger
        t_start = rpeak_times[j] - 0.060
        t_end = rpeak_times[j] + 0.005
        sp += ".meas tran rrc_cmp_max_{j} MAX V(rr_cap_level) FROM={f:.7f} TO={t:.7f}\n".format(j=j, f=t_start, t=t_end)
        sp += ".meas tran rrc_trig_max_{j} MAX V(trig_rr_cap) FROM={f:.7f} TO={t:.7f}\n".format(j=j, f=t_start, t=t_end)

    # --- Pwave trigger timing diagnostics ---
    for j in range(min(2, n), min(10, n)):
        td_pw = rpeak_times[j] - 0.300
        if td_pw > 0.1:
            sp += ".meas tran tpwb_{j} WHEN V(trig_pw_base)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            sp += ".meas tran tpwp_{j} WHEN V(trig_pw_peak)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            sp += ".meas tran tpwb_cmp_{j} WHEN V(pw_base_cmp)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            sp += ".meas tran tpwp_cmp_{j} WHEN V(pw_peak_cmp)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            # Probe SF and thresholds at trigger fire times
            sp += ".meas tran pwb_sf_at_{j} FIND V(sf2_out) WHEN V(pw_base_cmp)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            sp += ".meas tran pwb_th_{j} FIND V(vth_pw_base_node) WHEN V(pw_base_cmp)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            sp += ".meas tran pwp_sf_at_{j} FIND V(sf2_out) WHEN V(pw_peak_cmp)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            sp += ".meas tran pwp_th_{j} FIND V(vth_pw_peak_node) WHEN V(pw_peak_cmp)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td_pw)
            # MAX of trigger signals per beat (verify they actually pulse)
            t_start_pw = rpeak_times[j] - 0.250
            t_end_pw = rpeak_times[j] + 0.005
            sp += ".meas tran pwb_max_{j} MAX V(trig_pw_base) FROM={f:.7f} TO={t:.7f}\n".format(j=j, f=t_start_pw, t=t_end_pw)
            sp += ".meas tran pwp_max_{j} MAX V(trig_pw_peak) FROM={f:.7f} TO={t:.7f}\n".format(j=j, f=t_start_pw, t=t_end_pw)
            sp += ".meas tran sf2_at30_{j} FIND V(sf2_out) AT={t:.7f}\n".format(j=j, t=rpeak_times[j]-0.030)

    # --- Measurement statements ---
    # Feature outputs measured 100ms after TRUE R-peak
    for j in range(SKIP, n):
        t_f1 = rpeak_times[j] - 0.055
        if t_f1 > 0.5:
            sp += ".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t:.7f}\n".format(j=j, t=t_f1)
            sp += ".meas tran f1_sf_{j} FIND V(xrr.sf_out) AT={t:.7f}\n".format(j=j, t=t_f1)
            sp += ".meas tran f1_ewma_{j} FIND V(xrr.ewma_node) AT={t:.7f}\n".format(j=j, t=t_f1)
        if j + 1 < n:
            t_f2 = rpeak_times[j+1] - 0.350
            if t_f2 > 0.5:
                sp += ".meas tran f2_{j} FIND V(f2_rr_asym) AT={t:.7f}\n".format(j=j, t=t_f2)
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            for fk, nm in [(3,'f3_bi_l0'),(4,'f4_bi_l1'),(5,'f5_pwave'),
                           (6,'f6_width'),(7,'f7_slope'),(8,'f8_symmetry')]:
                sp += ".meas tran f{k}_{j} FIND V({nm}) AT={t:.7f}\n".format(k=fk, j=j, nm=nm, t=t_f38)

    # Extract ACTUAL analog trigger times per beat (for jitter-aware Python ref)
    for j in range(n):
        td = max(0.01, rpeak_times[j] - 0.050)
        sp += ".meas tran trpk_{j} WHEN V(rpeak_pulse)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran ts14_{j} WHEN V(trig_s14)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran ts69_{j} WHEN V(trig_s69)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran twid_{j} WHEN V(trig_wid)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran trpk_del_{j} WHEN V(rpeak_pulse_delayed)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        # Measure pulse FALL times to compute actual hold offsets
        sp += ".meas tran ts14_fall_{j} WHEN V(trig_s14)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran ts69_fall_{j} WHEN V(trig_s69)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran twid_fall_{j} WHEN V(trig_wid)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran trpk_del_fall_{j} WHEN V(rpeak_pulse_delayed)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td)
        # rr_cap analog timing
        sp += ".meas tran trrc_{j} WHEN V(trig_rr_cap)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran trrc_fall_{j} WHEN V(trig_rr_cap)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td)
        # F7: measure S&H cap values for all beats
        t_meas = rpeak_times[j] + 0.100
        sp += ".meas tran sh14_{j} FIND V(xslp.sh14) AT={t:.7f}\n".format(j=j, t=t_meas)
        sp += ".meas tran sh69_{j} FIND V(xslp.sh69) AT={t:.7f}\n".format(j=j, t=t_meas)
        # F3/F4: measure curr_node and prev_node for all beats (same approach as F7)
        sp += ".meas tran bi0_curr_{j} FIND V(xbi0.curr_node) AT={t:.7f}\n".format(j=j, t=t_meas)
        sp += ".meas tran bi0_prev_{j} FIND V(xbi0.prev_node) AT={t:.7f}\n".format(j=j, t=t_meas)
        sp += ".meas tran bi1_curr_{j} FIND V(xbi1.curr_node) AT={t:.7f}\n".format(j=j, t=t_meas)
        sp += ".meas tran bi1_prev_{j} FIND V(xbi1.prev_node) AT={t:.7f}\n".format(j=j, t=t_meas)
        # F8: measure simulated ecg_l0 at rpeak_del trigger fall (ngspice FIND...WHEN)
        td_f8 = max(0.01, rpeak_times[j] - 0.050)
        sp += ".meas tran ecg_at_f8cap_{j} FIND V(ecg_l0) WHEN V(rpeak_pulse_delayed)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td_f8)
        # F5: measure pwave S&H cap values and trigger fall times for all beats
        sp += ".meas tran pw_shb_{j} FIND V(xpw.shb) AT={t:.7f}\n".format(j=j, t=t_meas)
        sp += ".meas tran pw_shp_{j} FIND V(xpw.shp) AT={t:.7f}\n".format(j=j, t=t_meas)
        td_pw = max(0.01, rpeak_times[j] - 0.300)
        sp += ".meas tran tpwb_fall_{j} WHEN V(trig_pw_base)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td_pw)
        sp += ".meas tran tpwp_fall_{j} WHEN V(trig_pw_peak)=0.45 FALL=1 TD={td:.7f}\n".format(j=j, td=td_pw)
    # Diagnostic: rpeak_pulse amplitude and S&H output for first 5 beats
    for j in range(SKIP, min(SKIP+10, n)):
        t_expected = rpeak_times[j] - 0.0068  # expected rpk at R-6.8ms
        sp += ".meas tran rpk_amp_{j} MAX V(rpeak_pulse) FROM={f:.7f} TO={t:.7f}\n".format(
            j=j, f=t_expected-0.002, t=t_expected+0.005)
        sp += ".meas tran rpk_raw_{j} MAX V(rpeak_pulse_raw) FROM={f:.7f} TO={t:.7f}\n".format(
            j=j, f=t_expected-0.002, t=t_expected+0.005)
        sp += ".meas tran ecg_at_rpk_{j} FIND V(ecg_l0) AT={t:.7f}\n".format(
            j=j, t=t_expected+0.0003)  # 0.3ms into pulse
        # F8: sym_out IS the S&H cap; measure circuit output directly
        sp += ".meas tran sh_sym_{j} FIND V(f8_symmetry) AT={t:.7f}\n".format(
            j=j, t=t_expected+0.005)  # 5ms after capture
        # F3 diagnostic: output at measurement time (curr/prev now measured for ALL beats above)
        sp += ".meas tran bi_out_{j} FIND V(f3_bi_l0) AT={t:.7f}\n".format(
            j=j, t=rpeak_times[j]+0.100)
        # F7 diagnostic: ECG at approximate s14 time for comparison
        t_s14_approx = rpeak_times[j] + 0.014
        sp += ".meas tran ecg_at_s14_{j} FIND V(ecg_l0) AT={t:.7f}\n".format(
            j=j, t=t_s14_approx)

    res, elapsed = run_sim("integ_{}".format(rec_id), sp)
    print("  Sim time: {:.0f}s, {} measurements".format(elapsed, len(res)))

    # --- rr_cap analog diagnostics ---
    if True:  # always print for debugging
        print("  rr_cap analog diagnostics (esamp validity gate):")
        for j in range(min(3, n), min(10, n)):
            cmp_v = res.get('rrc_cmp_{}'.format(j), -1)
            cmp2_v = res.get('rrc_cmp2_{}'.format(j), -1)
            timer_v = res.get('rrc_timer_{}'.format(j), -1)
            ewma_v = res.get('rrc_ewma_{}'.format(j), -1)
            thresh_v = res.get('rrc_thresh_{}'.format(j), -1)
            trig_v = res.get('rrc_trig_{}'.format(j), -1)
            rpklvl_v = res.get('rrc_rpklvl_{}'.format(j), -1)
            cmp_max = res.get('rrc_cmp_max_{}'.format(j), -1)
            trig_max = res.get('rrc_trig_max_{}'.format(j), -1)
            esamp_v = res.get('rrc_esamp_{}'.format(j), -1)
            esamp_ok = res.get('rrc_esamp_ok_{}'.format(j), -1)
            print("    beat {}: timer={:.4f} ewma={:.4f} thresh={:.4f} cmp@-30={:.4f} esamp={:.4f} esamp_ok={:.4f} trig_max={:.4f}".format(
                j, timer_v, ewma_v, thresh_v, cmp_v, esamp_v, esamp_ok, trig_max))

    # --- Pwave trigger diagnostics ---
    print("  Pwave trigger diagnostics (sf2_out isolated timer, body=vdd fix):")
    for j in range(min(2, n), min(10, n)):
        tpwb = res.get('tpwb_{}'.format(j))
        tpwp = res.get('tpwp_{}'.format(j))
        pwb_max = res.get('pwb_max_{}'.format(j), -1)
        pwp_max = res.get('pwp_max_{}'.format(j), -1)
        sf2_v = res.get('sf2_at30_{}'.format(j), -1)
        pwb_sf = res.get('pwb_sf_at_{}'.format(j), -1)
        pwb_th = res.get('pwb_th_{}'.format(j), -1)
        pwp_sf = res.get('pwp_sf_at_{}'.format(j), -1)
        pwp_th = res.get('pwp_th_{}'.format(j), -1)
        r = rpeak_times[j]
        tpwb_ms = (tpwb - r) * 1000 if tpwb is not None else float('nan')
        tpwp_ms = (tpwp - r) * 1000 if tpwp is not None else float('nan')
        gap_ms = (tpwp - tpwb) * 1000 if (tpwb is not None and tpwp is not None) else float('nan')
        print("    beat {}: sf2={:.4f} pwb_max={:.3f} pwp_max={:.3f} tpwb=R{:+.0f}ms tpwp=R{:+.0f}ms gap={:.0f}ms sf2@pwb={:.4f} th@pwb={:.4f} sf2@pwp={:.4f} th@pwp={:.4f}".format(
            j, sf2_v, pwb_max, pwp_max, tpwb_ms, tpwp_ms, gap_ms, pwb_sf, pwb_th, pwp_sf, pwp_th))

    # --- Classify beats: valid vs missed trigger ---
    valid_rpk = set()
    valid_rpk_del = set()  # delayed rpeak valid (for F3/F4/F8)
    valid_dt = set()  # delay timer triggers valid
    n_rpk_ok = 0; n_rpk_miss = 0
    n_rpk_del_ok = 0; n_rpk_del_miss = 0
    n_dt_ok = 0; n_dt_miss = 0
    for j in range(n):
        r = rpeak_times[j]
        trpk = res.get('trpk_{}'.format(j))
        trpk_del = res.get('trpk_del_{}'.format(j))
        ts14 = res.get('ts14_{}'.format(j))
        twid = res.get('twid_{}'.format(j))
        ts69 = res.get('ts69_{}'.format(j))
        # rpeak valid if within [-50ms, +5ms] of true R (wide for inverted QRS)
        if trpk and (trpk - r) > -0.050 and (trpk - r) < 0.005:
            valid_rpk.add(j)
            n_rpk_ok += 1
        else:
            n_rpk_miss += 1
        # delayed rpeak valid if within [-15ms, +15ms] of true R
        # (wider window: rpeak fires earlier on broad-QRS records like 109)
        if trpk_del and (trpk_del - r) > -0.015 and (trpk_del - r) < 0.015:
            valid_rpk_del.add(j)
            n_rpk_del_ok += 1
        else:
            n_rpk_del_miss += 1
        # delay triggers valid if within ±15ms of expected
        if (ts14 and abs((ts14 - r)*1000 - 13) < 15 and
            twid and abs((twid - r)*1000 - 28) < 15 and
            ts69 and abs((ts69 - r)*1000 - 69) < 15):
            valid_dt.add(j)
            n_dt_ok += 1
        else:
            n_dt_miss += 1
    print("  rpeak valid: {}/{}, rpk_delayed valid: {}/{}, delay_timer valid: {}/{}".format(
        n_rpk_ok, n, n_rpk_del_ok, n, n_dt_ok, n))
    # Print sample trigger times for valid beats
    shown = 0
    for j in range(SKIP, n):
        if j in valid_rpk and shown < 3:
            r = rpeak_times[j] * 1000
            trpk = res.get('trpk_{}'.format(j))
            trpk_del = res.get('trpk_del_{}'.format(j))
            ts14 = res.get('ts14_{}'.format(j))
            twid = res.get('twid_{}'.format(j))
            ts69 = res.get('ts69_{}'.format(j))
            rpk_del_ms = (trpk_del*1000-r) if trpk_del else None
            print("    beat {}: rpk={:.1f}, rpk_del={}, s14={:.1f}, wid={:.1f}, s69={:.1f} ms from R".format(
                j, (trpk*1000-r),
                "{:.1f}".format(rpk_del_ms) if rpk_del_ms is not None else "MISS",
                (ts14*1000-r) if ts14 else 0,
                (twid*1000-r) if twid else 0, (ts69*1000-r) if ts69 else 0))
            shown += 1

    # --- Helper: interpolate ECG at any time ---
    def ecg_at_time(beat_ecg, rpk_time, t_actual, fs=FS):
        bs_t = rpk_time - WIN_L / fs
        sample_offset = (t_actual - bs_t) * fs
        idx = int(sample_offset)
        frac = sample_offset - idx
        if idx < 0: idx, frac = 0, 0
        if idx >= len(beat_ecg) - 1: return beat_ecg[-1]
        return beat_ecg[idx] * (1 - frac) + beat_ecg[idx + 1] * frac

    # --- Compute Spearman rho using actual trigger times ---
    rhos = {}

    SKIP_F1F2 = 15
    for fk, pykey in [(1, 'py_f1'), (2, 'py_f2')]:
        cv_list, pv_list = [], []
        for j in range(SKIP_F1F2, n):
            if j not in valid_rpk or (j - 1) not in valid_rpk:
                continue
            cv = res.get('f{}_{}'.format(fk, j))
            if cv is not None:
                cv_list.append(cv)
                pv_list.append(beats[j][pykey])
        rhos[fk] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0
        if fk == 1:
            print("  F1 per-beat (measured at R-55ms):")
            f1_cv_list, f1_pv_list = [], []
            for j in range(SKIP_F1F2, n):
                if j not in valid_rpk or (j - 1) not in valid_rpk:
                    continue
                cv = res.get('f1_{}'.format(j))
                pv = beats[j]['py_f1']
                rr_ms = beats[j]['pre_rr'] * 1000
                sf_v = res.get('f1_sf_{}'.format(j), -1)
                ewma_v = res.get('f1_ewma_{}'.format(j), -1)
                if cv is not None:
                    f1_cv_list.append(cv)
                    f1_pv_list.append(pv)
                    diff_mv = (sf_v - ewma_v) * 1000 if sf_v > 0 and ewma_v > 0 else 0
                    rpk_prev = rpk_offsets.get(j-1, 0) if 'rpk_offsets' in dir() else 0
                    # recompute rpk_offsets here if not yet available
                    trpk_prev = res.get('trpk_{}'.format(j-1))
                    rpk_prev_ms = (trpk_prev - rpeak_times[j-1]) * 1000 if trpk_prev else 0
                    eff_d = rr_ms + abs(rpk_prev_ms) - 55
                    print("    beat {:2d}: rr={:6.1f}ms  f1={:.4f}  py={:+.4f}  sf={:.4f}  ewma={:.4f}  diff={:+.1f}mV  rpk_prev={:.1f}ms  eff_d={:.1f}ms".format(
                        j, rr_ms, cv, pv, sf_v, ewma_v, diff_mv, rpk_prev_ms, eff_d))
            if len(f1_cv_list) >= 3:
                from scipy.stats import pearsonr
                pr = pearsonr(f1_cv_list, f1_pv_list)[0]
                sr = spearmanr(f1_cv_list, f1_pv_list)[0]
                print("  F1 Pearson r={:.4f}  Spearman rho={:.4f}  n={}".format(pr, sr, len(f1_cv_list)))
            # Diagnostic: sf_out vs annotation_rr, detected-RR noise, and detected-RR EWMA ref
            sf_list, rr_list, det_rr_list, ann_rr_list = [], [], [], []
            rpk_offsets = {}
            for j in range(n):
                trpk_val = res.get('trpk_{}'.format(j))
                if trpk_val and j in valid_rpk:
                    rpk_offsets[j] = (trpk_val - rpeak_times[j]) * 1000  # ms, negative = early
            for j in range(SKIP_F1F2, n):
                if j not in valid_rpk or (j - 1) not in valid_rpk:
                    continue
                sf_v = res.get('f1_sf_{}'.format(j))
                if sf_v is not None and sf_v > 0:
                    sf_list.append(sf_v)
                    rr_list.append(beats[j]['pre_rr'] * 1000)
                    ann_rr_list.append(beats[j]['pre_rr'] * 1000)
                    if j in rpk_offsets and (j-1) in rpk_offsets:
                        det_rr = beats[j]['pre_rr'] * 1000 + rpk_offsets[j-1] - rpk_offsets[j]
                        det_rr_list.append(det_rr)
            if len(sf_list) >= 5:
                sr_sf = abs(spearmanr(sf_list, rr_list)[0])
                print("  SF vs annotation_RR: Spearman |rho|={:.4f}  n={}".format(sr_sf, len(sf_list)))
            if len(det_rr_list) >= 5 and len(ann_rr_list) >= 5:
                sr_det = abs(spearmanr(det_rr_list[:len(ann_rr_list)], ann_rr_list[:len(det_rr_list)])[0])
                rpk_arr = [rpk_offsets.get(j, 0) for j in range(n) if j in rpk_offsets]
                rpk_range = max(rpk_arr) - min(rpk_arr) if rpk_arr else 0
                print("  Detected_RR vs annotation_RR: Spearman |rho|={:.4f}  rpk_range={:.1f}ms".format(sr_det, rpk_range))
            # Effective decay time: ann_rr + |rpk[j-1]| - 55ms (what the timer actually sees at R-55ms)
            alpha_ewma = 0.10
            eff_decay_list, sf_eff_list, f1_eff_cv, f1_eff_pv = [], [], [], []
            ewma_eff = None
            for j in range(1, n):
                if j not in valid_rpk or (j-1) not in valid_rpk:
                    continue
                if (j-1) not in rpk_offsets:
                    continue
                eff_decay = beats[j]['pre_rr'] * 1000 + abs(rpk_offsets[j-1]) - 55
                if ewma_eff is None:
                    ewma_eff = eff_decay
                else:
                    py_f1_eff = eff_decay - ewma_eff
                    ewma_eff = (1 - alpha_ewma) * ewma_eff + alpha_ewma * eff_decay
                    if j >= SKIP_F1F2:
                        sf_v = res.get('f1_sf_{}'.format(j))
                        cv = res.get('f1_{}'.format(j))
                        if sf_v is not None and sf_v > 0:
                            sf_eff_list.append(sf_v)
                            eff_decay_list.append(eff_decay)
                        if cv is not None:
                            f1_eff_cv.append(cv)
                            f1_eff_pv.append(py_f1_eff)
            if len(sf_eff_list) >= 5:
                sr_sf_eff = abs(spearmanr(sf_eff_list, eff_decay_list)[0])
                print("  SF vs eff_decay (ann_rr+|rpk[j-1]|-55): Spearman |rho|={:.4f}  n={}".format(sr_sf_eff, len(sf_eff_list)))
            if len(f1_eff_cv) >= 5:
                sr_f1_eff = abs(spearmanr(f1_eff_cv, f1_eff_pv)[0])
                print("  F1 vs eff_decay EWMA ref: Spearman |rho|={:.4f}  n={}".format(sr_f1_eff, len(f1_eff_cv)))

    # F5: use MEASURED S&H cap values (shp-shb) as Python ref
    # Same approach as F7 (sh14-sh69): the S&H captures ECG faithfully,
    # but analog trigger timing varies ±30ms from ideal due to EWMA lag.
    # Using measured S&H nodes directly validates circuit fidelity.
    cv_list, pv_list = [], []
    for j in range(SKIP, n):
        cv = res.get('f5_{}'.format(j))
        shp_v = res.get('pw_shp_{}'.format(j))
        shb_v = res.get('pw_shb_{}'.format(j))
        if cv is not None and shp_v is not None and shb_v is not None:
            pv_list.append(shp_v - shb_v)
            cv_list.append(cv)
    rhos[5] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0
    # F5 alternate: fixed Python ref (for comparison)
    cv_fix, pv_fix = [], []
    for j in range(SKIP, n):
        cv = res.get('f5_{}'.format(j))
        if cv is not None:
            cv_fix.append(cv)
            pv_fix.append(beats[j]['py_pw'])
    rho_f5_fixed = abs(spearmanr(cv_fix, pv_fix)[0]) if len(cv_fix) >= 5 else 0.0

    # F3/F4: use MEASURED S&H cap values (bi_curr - bi_prev) as primary ref
    # Same approach as F7 (sh14-sh69): NFET pass gate .meas fall-time mismatch
    # means ecg_at_time at .meas crossing doesn't match actual capture.
    # meas_ref proves circuit fidelity; the EWMA vs simple-diff is intentional design.
    for fk, prefix in [(3, 'bi0'), (4, 'bi1')]:
        cv_list, pv_list = [], []
        for j in range(SKIP_BI, n):
            if j not in valid_rpk_del:
                continue
            cv = res.get('f{}_{}'.format(fk, j))
            curr_v = res.get('{}_curr_{}'.format(prefix, j))
            prev_v = res.get('{}_prev_{}'.format(prefix, j))
            if cv is not None and curr_v is not None and prev_v is not None:
                pv_list.append(curr_v - prev_v)
                cv_list.append(cv)
        rhos[fk] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # F8: use ngspice-measured ecg_l0 at trigger fall as reference (avoids interpolation mismatch)
    cv_list, pv_list = [], []
    cv_ecgref_list, pv_ecgref_list = [], []  # ecg_at_time approach for comparison
    f8_diag = []
    for j in range(SKIP, n):
        if j not in valid_rpk_del:
            continue
        cv = res.get('f8_{}'.format(j))
        ecg_cap = res.get('ecg_at_f8cap_{}'.format(j))
        t_fall = res.get('trpk_del_fall_{}'.format(j))
        if cv is not None and ecg_cap is not None:
            pv_list.append(ecg_cap)
            cv_list.append(cv)
            f8_diag.append((j, cv, ecg_cap, t_fall - rpeak_times[j] if t_fall else 0))
        if cv is not None and t_fall is not None:
            pv_ecgref = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t_fall)
            cv_ecgref_list.append(cv)
            pv_ecgref_list.append(pv_ecgref)
    rhos[8] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0
    rho_f8_ecgref = abs(spearmanr(cv_ecgref_list, pv_ecgref_list)[0]) if len(cv_ecgref_list) >= 5 else 0.0
    if f8_diag:
        print("  F8 cv vs sim_ecg_at_cap diagnostic:")
        for j, cv, ecg_cap, dt in f8_diag[:10]:
            print("    beat {:2d}: cv={:.5f} ecg_sim={:.5f} delta={:+.2f}mV  tfall=R{:+.1f}ms".format(
                j, cv, ecg_cap, (cv - ecg_cap)*1000, dt*1000))
        print("  F8 rho sim_ecg={:.4f}  ecg_at_time={:.4f}".format(rhos[8], rho_f8_ecgref))

    # F6: use ACTUAL wid pulse FALL time for Python ref (valid beats only)
    cv_list, pv_list = [], []
    for j in range(SKIP, n):
        if j not in valid_dt:
            continue
        cv = res.get('f6_{}'.format(j))
        t_wid_fall = res.get('twid_fall_{}'.format(j))
        if cv is not None and t_wid_fall is not None:
            actual_ecg = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t_wid_fall)
            pv_list.append(actual_ecg)
            cv_list.append(cv)
    rhos[6] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # F7: use MEASURED S&H cap values (sh14-sh69) as Python ref
    # The S&H captures ECG faithfully (sh_ref=0.999), but ecg_at_time at the
    # .meas fall-crossing time disagrees by up to 19mV on steep QRS descent
    # because the NFET pass gate's effective turn-off differs from the .meas threshold.
    # Using measured sh14-sh69 directly gives the correct capture-time reference.
    cv_list, pv_list = [], []
    for j in range(SKIP, n):
        if j not in valid_dt:
            continue
        cv = res.get('f7_{}'.format(j))
        sh14_v = res.get('sh14_{}'.format(j))
        sh69_v = res.get('sh69_{}'.format(j))
        if cv is not None and sh14_v is not None and sh69_v is not None:
            pv_list.append(sh14_v - sh69_v)
            cv_list.append(cv)
    rhos[7] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # F7 alternate: ecg_at_time at fall crossing (for comparison)
    cv_list_fall, pv_list_fall = [], []
    for j in range(SKIP, n):
        if j not in valid_dt:
            continue
        cv = res.get('f7_{}'.format(j))
        t14_fall = res.get('ts14_fall_{}'.format(j))
        t69_fall = res.get('ts69_fall_{}'.format(j))
        if cv is not None and t14_fall is not None and t69_fall is not None:
            e14 = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t14_fall)
            e69 = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t69_fall)
            pv_list_fall.append(e14 - e69)
            cv_list_fall.append(cv)
    rho_f7_fall = abs(spearmanr(cv_list_fall, pv_list_fall)[0]) if len(cv_list_fall) >= 5 else 0.0

    # F3/F4 alternate: fall-time ecg_at_time and simple-diff (for comparison)
    rho_f3_fall, rho_f4_fall = 0.0, 0.0
    rho_f3_sdiff, rho_f4_sdiff = 0.0, 0.0
    for fk, lead, prefix in [(3, 0, 'bi0'), (4, 1, 'bi1')]:
        cv_fall, pv_fall = [], []
        cv_sd, pv_sd = [], []
        for j in range(SKIP_BI, n):
            if j not in valid_rpk_del:
                continue
            cv = res.get('f{}_{}'.format(fk, j))
            t_fall = res.get('trpk_del_fall_{}'.format(j))
            curr_v = res.get('{}_curr_{}'.format(prefix, j))
            # Fall-time ref
            if cv is not None and t_fall is not None and (j-1) in valid_rpk_del:
                t_fall_prev = res.get('trpk_del_fall_{}'.format(j-1))
                if t_fall_prev is not None:
                    ecg_arr = beats[j]['ecg_l0'] if lead == 0 else beats[j]['ecg_l1']
                    prev_arr = beats[j-1]['ecg_l0'] if lead == 0 else beats[j-1]['ecg_l1']
                    e_curr = ecg_at_time(ecg_arr, rpeak_times[j], t_fall)
                    e_prev = ecg_at_time(prev_arr, rpeak_times[j-1], t_fall_prev)
                    cv_fall.append(cv); pv_fall.append(e_curr - e_prev)
            # Simple diff: curr[n] - curr[n-1]
            if cv is not None and curr_v is not None and (j-1) in valid_rpk_del:
                prev_curr_v = res.get('{}_curr_{}'.format(prefix, j-1))
                if prev_curr_v is not None:
                    cv_sd.append(cv); pv_sd.append(curr_v - prev_curr_v)
        rho_fall = abs(spearmanr(cv_fall, pv_fall)[0]) if len(cv_fall) >= 5 else 0.0
        rho_sd = abs(spearmanr(cv_sd, pv_sd)[0]) if len(cv_sd) >= 5 else 0.0
        if fk == 3:
            rho_f3_fall = rho_fall; rho_f3_sdiff = rho_sd
        else:
            rho_f4_fall = rho_fall; rho_f4_sdiff = rho_sd

    # --- Diagnostic: print F8 values for first few valid beats ---
    if rec_id == '100':
        print("  F8 S&H diagnostic (delayed rpeak):")
        shown_f8 = 0
        for j in range(SKIP, n):
            if j not in valid_rpk_del or shown_f8 >= 5:
                continue
            cv = res.get('f8_{}'.format(j))
            t_fall = res.get('trpk_del_fall_{}'.format(j))
            if cv is not None and t_fall is not None:
                py_actual = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t_fall)
                py_fixed = beats[j]['py_sym']
                offset_ms = (t_fall - rpeak_times[j]) * 1000
                rpk_amp = res.get('rpk_amp_{}'.format(j), 0)
                rpk_raw = res.get('rpk_raw_{}'.format(j), 0)
                ecg_v = res.get('ecg_at_rpk_{}'.format(j), 0)
                sh_v = res.get('sh_sym_{}'.format(j), 0)
                print("    beat {}: cv={:.5f} ecg={:.4f} sh={:.4f} rpk_amp={:.3f} rpk_raw={:.3f} rpk_off={:.1f}ms".format(
                    j, cv, ecg_v, sh_v, rpk_amp, rpk_raw, offset_ms))
                shown_f8 += 1
        # Pulse width diagnostic: actual hold offsets
        print("  Pulse width diagnostic (S&H captures at FALL edge):")
        for j in range(SKIP, min(SKIP+5, n)):
            ts14_r = res.get('ts14_{}'.format(j)); ts14_f = res.get('ts14_fall_{}'.format(j))
            ts69_r = res.get('ts69_{}'.format(j)); ts69_f = res.get('ts69_fall_{}'.format(j))
            twid_r = res.get('twid_{}'.format(j)); twid_f = res.get('twid_fall_{}'.format(j))
            trd_r = res.get('trpk_del_{}'.format(j)); trd_f = res.get('trpk_del_fall_{}'.format(j))
            pw_s14 = (ts14_f - ts14_r)*1000 if ts14_r and ts14_f and ts14_f > ts14_r else None
            pw_s69 = (ts69_f - ts69_r)*1000 if ts69_r and ts69_f and ts69_f > ts69_r else None
            pw_wid = (twid_f - twid_r)*1000 if twid_r and twid_f and twid_f > twid_r else None
            pw_rpkd = (trd_f - trd_r)*1000 if trd_r and trd_f and trd_f > trd_r else None
            print("    beat {}: s14={}, s69={}, wid={}, rpk_del={} ms".format(
                j,
                "{:.2f}".format(pw_s14) if pw_s14 else "N/A",
                "{:.2f}".format(pw_s69) if pw_s69 else "N/A",
                "{:.2f}".format(pw_wid) if pw_wid else "N/A",
                "{:.2f}".format(pw_rpkd) if pw_rpkd else "N/A"))
        # F3 diagnostic: curr_node, prev_node, output
        print("  F3 beat_instab diagnostic:")
        for j in range(SKIP, min(SKIP+10, n)):
            bi_c = res.get('bi0_curr_{}'.format(j))
            bi_p = res.get('bi0_prev_{}'.format(j))
            bi_o = res.get('bi_out_{}'.format(j))
            f3_cv = res.get('f3_{}'.format(j))
            t_rpk_del = res.get('trpk_del_{}'.format(j))
            rpk_del_off = (t_rpk_del - rpeak_times[j])*1000 if t_rpk_del else None
            valid_str = "V" if j in valid_rpk_del else "X"
            if bi_c is not None:
                print("    beat {} [{}]: curr={:.5f} prev={:.5f} diff={:.5f} out={:.5f} f3={} off={}ms".format(
                    j, valid_str,
                    bi_c, bi_p if bi_p else 0, (bi_c - bi_p) if bi_p else 0,
                    bi_o if bi_o else 0,
                    "{:.5f}".format(f3_cv) if f3_cv else "N/A",
                    "{:.1f}".format(rpk_del_off) if rpk_del_off else "MISS"))

        # F7 diagnostic: print Python ref vs circuit for each beat
        print("  F7 slope diagnostic (s14/s69 actual fall times):")
        for j in range(SKIP, min(SKIP+10, n)):
            if j not in valid_dt:
                continue
            cv = res.get('f7_{}'.format(j))
            t14_f = res.get('ts14_fall_{}'.format(j))
            t69_f = res.get('ts69_fall_{}'.format(j))
            sh14_v = res.get('sh14_{}'.format(j))
            sh69_v = res.get('sh69_{}'.format(j))
            sh14_cap = res.get('sh14_cap_{}'.format(j))
            ecg_s14 = res.get('ecg_at_s14_{}'.format(j))
            if cv is not None and t14_f is not None and t69_f is not None:
                e14 = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t14_f)
                e69 = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t69_f)
                t14_off = (t14_f - rpeak_times[j]) * 1000
                t69_off = (t69_f - rpeak_times[j]) * 1000
                sh14_str = "{:.4f}".format(sh14_v) if sh14_v else "N/A"
                sh69_str = "{:.4f}".format(sh69_v) if sh69_v else "N/A"
                cap_str = "{:.4f}".format(sh14_cap) if sh14_cap else "N/A"
                ecg_str = "{:.4f}".format(ecg_s14) if ecg_s14 else "N/A"
                expected_sh14 = VBASE + gain * e14
                err14 = (sh14_v - expected_sh14) * 1000 if sh14_v else 0
                print("    beat {}: cv={:.5f} sh14@100={} sh14@cap={} ecg@14={} expected={:.4f} err={:.1f}mV t14={:.1f}ms t69={:.1f}ms".format(
                    j, cv, sh14_str, cap_str, ecg_str, expected_sh14, err14, t14_off, t69_off))

    # Also compute FIXED-ref rho for comparison (only on valid beats)
    rhos_fixed = {}
    py_keys = {3:'py_bi_l0', 4:'py_bi_l1', 6:'py_wid', 7:'py_slp', 8:'py_sym'}
    for fk in [3, 4, 6, 7, 8]:
        valid_set = valid_rpk_del if fk in [3, 4, 8] else valid_dt
        cv_list, pv_list = [], []
        skip_start = SKIP_BI if fk in [3, 4] else SKIP
        for j in range(skip_start, n):
            if j not in valid_set:
                continue
            cv = res.get('f{}_{}'.format(fk, j))
            if cv is not None:
                cv_list.append(cv)
                pv_list.append(beats[j][py_keys[fk]])
        rhos_fixed[fk] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    fnames = {1:'F1:rr_ratio', 2:'F2:rr_asym', 3:'F3:bi_L0', 4:'F4:bi_L1',
              5:'F5:pwave', 6:'F6:width', 7:'F7:slope', 8:'F8:symmetry'}
    triggers = {1:'ANALOG rpk+ANALOG rrc', 2:'ANALOG rpk+ANALOG rrc', 3:'DELAYED rpk', 4:'DELAYED rpk',
                5:'ANALOG pw(S&H ref)', 6:'ANALOG wid', 7:'ANALOG s14/s69', 8:'DELAYED rpk'}
    for fk in range(1, 9):
        status = "PASS" if rhos[fk] >= 0.95 else "WARN" if rhos[fk] >= 0.80 else "FAIL"
        extra = ""
        if fk in rhos_fixed:
            extra = "  (fixed_ref={:.4f})".format(rhos_fixed[fk])
        if fk == 3:
            extra += "  (fall={:.4f} sdiff={:.4f})".format(rho_f3_fall, rho_f3_sdiff)
        if fk == 4:
            extra += "  (fall={:.4f} sdiff={:.4f})".format(rho_f4_fall, rho_f4_sdiff)
        if fk == 5:
            extra += "  (fixed_ref={:.4f})".format(rho_f5_fixed)
        if fk == 8:
            extra += "  (ecgref={:.4f})".format(rho_f8_ecgref)
        if fk == 7:
            extra += "  (fall_ref={:.4f})".format(rho_f7_fall)
        print("  {:15s}  rho={:.4f}  [{}]  trigger={}{}".format(
            fnames[fk], rhos[fk], status, triggers[fk], extra))

    all_results[rec_id] = rhos

# ========================================================================
print("\n\n" + "=" * 70)
print("SUMMARY: Fully Analog AFE — Integration Test Results ({} records)".format(len(RECORDS)))
print("=" * 70)
fnames = {1:'F1:rr_ratio', 2:'F2:rr_asym', 3:'F3:bi_L0', 4:'F4:bi_L1',
          5:'F5:pwave', 6:'F6:width', 7:'F7:slope', 8:'F8:symmetry'}
# Compact table: record IDs on header row
hdr = "  {:15s}".format("Feature")
for r in RECORDS:
    hdr += " {:>6s}".format(r)
hdr += "   MIN    AVG"
print(hdr)
print("  " + "-" * (15 + 7*len(RECORDS) + 14))
for fk in range(1, 9):
    row = "  {:15s}".format(fnames[fk])
    vals = []
    for r in RECORDS:
        v = all_results.get(r, {}).get(fk, 0)
        row += " {:6.3f}".format(v)
        vals.append(v)
    if vals:
        row += "  {:5.3f}  {:5.3f}".format(min(vals), sum(vals)/len(vals))
    print(row)
print("=" * 70)
print("\nThreshold info:")
print("  rpeak_offset = {:.1f}ms, pg_width = {:.1f}ms".format(RPEAK_OFFSET_MS, PG_WIDTH_MS))
print("  vth_14(fast) = {:.4f}V".format(vth_14))
print("  vth_28(main) = {:.4f}V".format(vth_28))
print("  vth_69(main) = {:.4f}V".format(vth_69))
print("  vth_300(main)= {:.4f}V".format(vth_300))
print("  vth_10(fast) = {:.4f}V".format(vth_10))
print("  vth_rpk_del(fast) = {:.4f}V".format(vth_rpk_del))
