"""
Full inter-patient RF accuracy test — ALL 8 features from analog circuits.
NO Python feature computation — everything from SKY130 SPICE circuits.

Feature circuits (all SKY130 PDK analog):
  F1/F2: rr_timing.spice    — EWMA alpha=0.50, real RR intervals
  F3/F4: beat_instability.spice — TG S&H, diode diff pair (per lead)
  F5:    pwave_diff.spice    — diode diff pair
  F6:    qrs_width.spice     — NFET S&H
  F7:    qrs_slope.spice     — 2x S&H + diode diff pair
  F8:    qrs_symmetry.spice  — TG S&H

Stimulus signals (representing support circuit outputs):
  ECG:      PWL waveform (amplified ECG after CCIA, excluded per user)
  Triggers: PWL pulses (delay_timer / pwave_predictor output)
  Vbias:    Fixed 0.300V (bias_gen output)

Trains RF on DS1, tests on DS2 — full MIT-BIH 44-record dataset.
"""
import subprocess, os, re, sys, time, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")
CACHE_DIR = os.path.join(FRONTEND_DIR, "_cache")

VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130; RPEAK_IDX = 20
PW_S = 16; PW_MID = 43
BATCH_SIZE = 40
EWMA_ALPHA = 0.50

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


def load_all_beats(record_id):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    all_peaks = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(np.abs(sig[bs+QRS_S:bs+QRS_E, 0])))
    gain = 0.16 / max(all_peaks) if all_peaks else 0.05

    beats = []
    ewma_rr = None
    for i in range(1, len(valid) - 1):
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        post_rr = (valid[i+1][0] - idx) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]

        if ewma_rr is None:
            ewma_rr_before = pre_rr
            ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr

        py_f1 = pre_rr / ewma_rr_before if ewma_rr_before > 0 else 1.0
        py_f2 = pre_rr / post_rr if post_rr > 0 else 1.0

        prev_bs = valid[i-1][0] - WIN_L
        prev_be = valid[i-1][0] + WIN_R
        if prev_bs >= 0 and prev_be < len(sig):
            prev_beat = sig[prev_bs:prev_be]
            py_bi_l0 = beat[RPEAK_IDX+QRS_S, 0] - prev_beat[RPEAK_IDX+QRS_S, 0]
            py_bi_l1 = beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)] - prev_beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)]
        else:
            py_bi_l0 = py_bi_l1 = 0.0

        beats.append({
            'label': aami_map[sym],
            'pre_rr': pre_rr, 'post_rr': post_rr,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': py_f1, 'py_f2': py_f2,
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[RPEAK_IDX+QRS_S+10, 0],
            'py_slp': beat[RPEAK_IDX+QRS_S+5, 0] - beat[RPEAK_IDX+QRS_S+25, 0],
            'py_sym': beat[RPEAK_IDX+QRS_S, 0],
        })
    return beats, gain


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join(f"{t:.7f} {v:.6f}" for t, v in c)


def make_trig(rpeak_times, offset_s, pulse_w, total_time):
    pts = [(0, 0)]
    for rpt in rpeak_times:
        t = rpt + offset_s
        if t > 0.01:
            pts += [(t - 0.0004, 0), (t, VDD), (t + pulse_w, VDD), (t + pulse_w + 0.0004, 0)]
    pts.append((total_time, 0))
    return make_pwl(pts)


def generate_batch_testbench(beats_batch, gain, batch_id, record_id):
    n = len(beats_batch)
    dt = 1.0 / FS
    PW = 0.0005

    rpeak_times = [2.0]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats_batch[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    ecg_l0_pts, ecg_l1_pts = [(0, VBASE)], [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt
        ecg0 = beats_batch[i]['ecg_l0']
        ecg1 = beats_batch[i]['ecg_l1']
        t_before = beat_start - 0.003
        if t_before > 0.01:
            ecg_l0_pts.append((t_before, VBASE))
            ecg_l1_pts.append((t_before, VBASE))
        for j in range(len(ecg0)):
            t = beat_start + j * dt
            if t > 0:
                ecg_l0_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg0[j]))))
                ecg_l1_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg1[j]))))
        t_after = beat_start + len(ecg0) * dt + 0.003
        ecg_l0_pts.append((t_after, VBASE))
        ecg_l1_pts.append((t_after, VBASE))
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    trig_rpeak = make_trig(rpeak_times, 0, PW, total_time)
    trig_rr_cap = make_trig(rpeak_times, -0.005, PW, total_time)
    trig_rr_copy = make_trig(rpeak_times, 0.010, 0.002, total_time)
    trig_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)
    trig_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)
    trig_rpeak_sh = make_trig(rpeak_times, 0, PW, total_time)
    trig_s14 = make_trig(rpeak_times, 5 / FS, PW, total_time)
    trig_s69 = make_trig(rpeak_times, 25 / FS, PW, total_time)
    trig_wid = make_trig(rpeak_times, 10 / FS, PW, total_time)
    trig_bi_copy = make_trig(rpeak_times, 0.300, 0.002, total_time)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title acc_{record_id}_b{batch_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 0.300

Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

Vrpk trig_rpeak 0 PWL({trig_rpeak})
Vrrc trig_rr_cap 0 PWL({trig_rr_cap})
Vrry trig_rr_copy 0 PWL({trig_rr_copy})
Vpwb trig_pw_base 0 PWL({trig_pw_base})
Vpwp trig_pw_peak 0 PWL({trig_pw_peak})
Vrsh trig_rpeak_sh 0 PWL({trig_rpeak_sh})
Vs14 trig_s14 0 PWL({trig_s14})
Vs69 trig_s69 0 PWL({trig_s69})
Vwid trig_wid 0 PWL({trig_wid})
Vbco trig_bi_copy 0 PWL({trig_bi_copy})

.include "{fp}/inverter.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

Xrr ecg_l0 trig_rr_cap trig_rpeak trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.ewma_node)=0.40 V(xrr.esamp_node)=0.40 V(xrr.prev_node)=0.40
.ic V(xbi0.prev_node)=0.30 V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30 V(xbi1.esamp_node)=0.30

.tran 0.5m {total_time:.4f}

"""
    f12_skip = 5
    f38_skip = 2
    skip = max(f12_skip, f38_skip)

    for j in range(skip, n):
        # F1 for beat j: measure at rpeak[j]-10ms
        t_f1 = rpeak_times[j] - 0.010
        if t_f1 > 0.5:
            sp += f".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t_f1:.7f}\n"

        # F2 for beat j: measure at rpeak[j+1]-10ms (circuit needs next rpeak)
        if j + 1 < n:
            t_f2 = rpeak_times[j + 1] - 0.010
            sp += f".meas tran f2_{j} FIND V(f2_rr_asym) AT={t_f2:.7f}\n"

        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            sp += f".meas tran f3_{j} FIND V(f3_bi_l0) AT={t_f38:.7f}\n"
            sp += f".meas tran f4_{j} FIND V(f4_bi_l1) AT={t_f38:.7f}\n"
            sp += f".meas tran f5_{j} FIND V(f5_pwave) AT={t_f38:.7f}\n"
            sp += f".meas tran f6_{j} FIND V(f6_width) AT={t_f38:.7f}\n"
            sp += f".meas tran f7_{j} FIND V(f7_slope) AT={t_f38:.7f}\n"
            sp += f".meas tran f8_{j} FIND V(f8_symmetry) AT={t_f38:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, f"_acc_{record_id}_b{batch_id}.spice")
    with open(tb, 'w') as f:
        f.write(sp)
    return tb, skip


def run_ngspice(tb):
    log = tb.replace('.spice', '.log')
    try:
        subprocess.run([NGSPICE, '-b', '-o', log, tb],
                       capture_output=True, text=True, timeout=1800, cwd=NGSPICE_DIR)
    except subprocess.TimeoutExpired:
        return None
    if not os.path.exists(log): return None
    with open(log, 'r') as f:
        text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    return results


def process_record(record_id):
    cache_file = os.path.join(CACHE_DIR, f"{record_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        py_feats = data.get('py_features')
        if py_feats is not None:
            return (np.array(data['features']), np.array(data['labels']),
                    np.array(py_feats))
        return (np.array(data['features']), np.array(data['labels']),
                None)

    beats, gain = load_all_beats(record_id)
    if not beats:
        return np.array([]).reshape(0, 8), np.array([]), None

    all_feats, all_labels = [], []
    all_py_feats, all_circuit_feats = [], []
    n_batches = (len(beats) + BATCH_SIZE - 1) // BATCH_SIZE
    n_circuit = 0
    n_fallback = 0

    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end = min((b_idx + 1) * BATCH_SIZE, len(beats))
        batch = beats[start:end]
        if len(batch) < 6:
            continue

        tb, skip = generate_batch_testbench(batch, gain, b_idx, record_id)
        results = run_ngspice(tb)

        if results is None:
            n_fallback += len(batch) - skip
            for j in range(skip, len(batch)):
                orig_idx = start + j
                if orig_idx >= len(beats): break
                bt = beats[orig_idx]
                all_feats.append([bt['py_f1'], bt['py_f2'],
                                  bt['py_bi_l0'], bt['py_bi_l1'],
                                  bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']])
                all_labels.append(bt['label'])
            continue

        for j in range(skip, len(batch)):
            orig_idx = start + j
            if orig_idx >= len(beats): break
            bt = beats[orig_idx]

            f1 = results.get(f'f1_{j}')
            f2 = results.get(f'f2_{j}')
            f3 = results.get(f'f3_{j}')
            f4 = results.get(f'f4_{j}')
            f5 = results.get(f'f5_{j}')
            f6 = results.get(f'f6_{j}')
            f7 = results.get(f'f7_{j}')
            f8 = results.get(f'f8_{j}')

            vals = [f1, f2, f3, f4, f5, f6, f7, f8]
            py_vals = [bt['py_f1'], bt['py_f2'],
                       bt['py_bi_l0'], bt['py_bi_l1'],
                       bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']]
            if any(v is None for v in vals):
                n_fallback += 1
                all_feats.append(py_vals)
            else:
                n_circuit += 1
                all_feats.append(vals)
                all_circuit_feats.append(vals)
                all_py_feats.append(py_vals)
            all_labels.append(bt['label'])

        try:
            os.remove(tb)
            os.remove(tb.replace('.spice', '.log'))
        except: pass

    feats = np.array(all_feats) if all_feats else np.array([]).reshape(0, 8)
    labels = np.array(all_labels) if all_labels else np.array([])
    py_feats = np.array(all_py_feats) if all_py_feats else None
    circuit_feats = np.array(all_circuit_feats) if all_circuit_feats else None

    os.makedirs(CACHE_DIR, exist_ok=True)
    save_data = {
        'features': feats.tolist(), 'labels': labels.tolist(),
        'n_circuit': n_circuit, 'n_fallback': n_fallback
    }
    if py_feats is not None:
        save_data['py_features'] = py_feats.tolist()
        save_data['circuit_features'] = circuit_feats.tolist()
    with open(cache_file, 'w') as f:
        json.dump(save_data, f)

    return feats, labels, py_feats


def run_python_baseline():
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for rec in DS1:
        beats, _ = load_all_beats(rec)
        for bt in beats:
            X_tr.append([bt['py_f1'], bt['py_f2'],
                         bt['py_bi_l0'], bt['py_bi_l1'],
                         bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']])
            y_tr.append(bt['label'])
    for rec in DS2:
        beats, _ = load_all_beats(rec)
        for bt in beats:
            X_te.append([bt['py_f1'], bt['py_f2'],
                         bt['py_bi_l0'], bt['py_bi_l1'],
                         bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']])
            y_te.append(bt['label'])

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_te, y_te = np.array(X_te), np.array(y_te)
    clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)), len(y_tr), len(y_te)


# === MAIN ===
print("=" * 70)
print("FULL INTER-PATIENT ACCURACY TEST — ALL ANALOG FRONTEND")
print("=" * 70)
print("  ALL 8 features from SKY130 analog circuits:")
print("    F1: RR ratio (EWMA, alpha=0.50)     — rr_timing.spice")
print("    F2: RR asymmetry (curr/prev)         — rr_timing.spice")
print("    F3: Beat instability L0              — beat_instability.spice")
print("    F4: Beat instability L1              — beat_instability.spice")
print("    F5: P-wave diff (diode loads)        — pwave_diff.spice")
print("    F6: QRS width tail (R+28ms)          — qrs_width.spice")
print("    F7: QRS slope (R+14ms vs R+69ms)     — qrs_slope.spice")
print("    F8: QRS symmetry (R-peak)            — qrs_symmetry.spice")
print(f"  Classifier: RandomForest(n=300, depth=15, balanced)")
print(f"  Dataset:    MIT-BIH, DS1 train ({len(DS1)} rec) / DS2 test ({len(DS2)} rec)")
print()

print("Running Python-proxy baseline (EWMA F1, alpha=0.50)...")
t0 = time.time()
py_acc, n_tr, n_te = run_python_baseline()
print(f"  Python baseline: {py_acc*100:.2f}% (train={n_tr}, test={n_te}, {time.time()-t0:.1f}s)")
print()

print("Running analog frontend simulations (ALL 8 features from circuits)...")
os.makedirs(CACHE_DIR, exist_ok=True)

from scipy.stats import spearmanr

X_train, y_train, X_test, y_test = [], [], [], []
all_rho_py, all_rho_circuit = [], []
total_records = len(DS1) + len(DS2)
total_circuit = 0
total_fallback = 0

for idx, (split, records) in enumerate([("DS1-train", DS1), ("DS2-test", DS2)]):
    for i, rec in enumerate(records):
        t0 = time.time()
        feats, labels, py_feats = process_record(rec)
        elapsed = time.time() - t0
        rec_num = idx * len(DS1) + i + 1

        cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
        nc = nf = 0
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cdata = json.load(f)
            nc = cdata.get('n_circuit', len(labels))
            nf = cdata.get('n_fallback', 0)
            if py_feats is None and 'py_features' in cdata:
                py_feats = np.array(cdata['py_features'])
            circuit_feats = cdata.get('circuit_features')
            if circuit_feats is not None and py_feats is not None:
                all_rho_circuit.append(np.array(circuit_feats))
                all_rho_py.append(np.array(py_feats) if not isinstance(py_feats, np.ndarray) else py_feats)
        total_circuit += nc
        total_fallback += nf

        cached = " [cached]" if elapsed < 2 else ""
        print(f"  [{rec_num:2d}/{total_records}] Record {rec}: {len(labels):4d} beats"
              f" ({nc} circuit, {nf} fallback), {elapsed:.1f}s{cached}")

        if len(feats) == 0: continue
        if split == "DS1-train":
            X_train.append(feats); y_train.append(labels)
        else:
            X_test.append(feats); y_test.append(labels)

X_train = np.vstack(X_train); y_train = np.concatenate(y_train)
X_test = np.vstack(X_test); y_test = np.concatenate(y_test)

print(f"\n  Train: {len(y_train)} beats, Test: {len(y_test)} beats")
print(f"  Circuit features: {total_circuit} beats ({total_circuit*100/(total_circuit+total_fallback):.1f}%)")
print(f"  Python fallback:  {total_fallback} beats ({total_fallback*100/(total_circuit+total_fallback):.1f}%)")

clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
analog_acc = accuracy_score(y_test, y_pred)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Python baseline accuracy:   {py_acc*100:.2f}%")
print(f"  Analog frontend accuracy:   {analog_acc*100:.2f}%")
print(f"  Delta:                      {(analog_acc - py_acc)*100:+.2f}%")
print()
print("  Per-class breakdown (analog frontend):")
print(classification_report(y_test, y_pred, digits=3))

print("Feature importances:")
names = ['F1:rr_ratio', 'F2:rr_asym', 'F3:bi_L0', 'F4:bi_L1',
         'F5:pwave', 'F6:width', 'F7:slope', 'F8:symmetry']
for name, imp in sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<15s} {imp:.4f}")

print()
print("=" * 70)
print("SPEARMAN RHO: Analog Circuit vs Python Reference")
print("=" * 70)

if all_rho_py and all_rho_circuit:
    rho_py = np.vstack(all_rho_py)
    rho_circuit = np.vstack(all_rho_circuit)
    print(f"  Matched circuit beats: {len(rho_py)}")
    print()
    feature_files = [
        ('F1:rr_ratio',  'rr_timing.spice'),
        ('F2:rr_asym',   'rr_timing.spice'),
        ('F3:bi_L0',     'beat_instability.spice'),
        ('F4:bi_L1',     'beat_instability.spice'),
        ('F5:pwave',     'pwave_diff.spice'),
        ('F6:width',     'qrs_width.spice'),
        ('F7:slope',     'qrs_slope.spice'),
        ('F8:symmetry',  'qrs_symmetry.spice'),
    ]
    for i, (fname, spice_file) in enumerate(feature_files):
        py_col = rho_py[:, i]
        ci_col = rho_circuit[:, i]
        mask = np.isfinite(py_col) & np.isfinite(ci_col)
        if np.sum(mask) < 5:
            print(f"  {fname:<15s}  SKIP (too few valid points)")
            continue
        rho, pval = spearmanr(py_col[mask], ci_col[mask])
        status = "IDEAL" if abs(rho) > 0.95 else "OK" if abs(rho) > 0.7 else "WARN"
        print(f"  {fname:<15s}  rho={rho:+.4f}  p={pval:.2e}  n={np.sum(mask):5d}  [{status}]  {spice_file}")
else:
    print("  No cached rho data — delete _cache/ and re-run to generate.")
    print("  (Existing cache was created before rho tracking was added.)")

print("=" * 70)
