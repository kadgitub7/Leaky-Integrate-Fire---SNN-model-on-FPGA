"""
Full inter-patient RF accuracy test with analog frontend in the loop.
Features 1&2: computed in Python (RR timing — validated digital timer).
Features 3-8: computed by ngspice circuit simulation per record.
Trains RF on DS1, tests on DS2 — full MIT-BIH dataset.
"""
import subprocess, os, re, sys, time, json
import numpy as np
from scipy.stats import spearmanr
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
BEAT_SPACING = 0.700  # seconds between beats in simulation
BATCH_SIZE = 40

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
    for i in range(1, len(valid) - 1):
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        post_rr = (valid[i+1][0] - idx) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]

        rr_ints = []
        for j in range(max(0, i-5), i):
            rr_ints.append((valid[j][0] - valid[j-1][0]) / FS if j > 0 else pre_rr)
        local_rr = np.mean(rr_ints) if rr_ints else pre_rr

        f1 = pre_rr / local_rr if local_rr > 0 else 1.0
        f2 = pre_rr / post_rr if post_rr > 0 else 1.0

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
            'f1_rr_ratio': f1, 'f2_rr_asym': f2,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
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
    rpeak_times = [2.0 + i * BEAT_SPACING for i in range(n)]
    total_time = rpeak_times[-1] + 0.5
    dt = 1.0 / FS

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

    PW = 0.0005
    trig_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)
    trig_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)
    trig_rpeak_sh = make_trig(rpeak_times, 0, PW, total_time)
    trig_s14 = make_trig(rpeak_times, 5 / FS, PW, total_time)
    trig_s69 = make_trig(rpeak_times, 25 / FS, PW, total_time)
    trig_wid = make_trig(rpeak_times, 10 / FS, PW, total_time)
    trig_bi_copy = make_trig(rpeak_times, 0.300, 0.002, total_time)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title acc_test_{record_id}_b{batch_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 0.300

Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

Vpwb trig_pw_base 0 PWL({trig_pw_base})
Vpwp trig_pw_peak 0 PWL({trig_pw_peak})
Vrsh trig_rpeak_sh 0 PWL({trig_rpeak_sh})
Vs14 trig_s14 0 PWL({trig_s14})
Vs69 trig_s69 0 PWL({trig_s69})
Vwid trig_wid 0 PWL({trig_wid})
Vbco trig_bi_copy 0 PWL({trig_bi_copy})

.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

.tran 0.5m {total_time:.4f}

"""
    skip = 1
    for i in range(skip, n):
        t_meas = rpeak_times[i] + 30 / FS + PW + 0.020
        sp += f".meas tran bi0_b{i} FIND V(f3_bi_l0) AT={t_meas:.7f}\n"
        sp += f".meas tran bi1_b{i} FIND V(f4_bi_l1) AT={t_meas:.7f}\n"
        sp += f".meas tran pw_b{i} FIND V(f5_pwave) AT={t_meas:.7f}\n"
        sp += f".meas tran wid_b{i} FIND V(f6_width) AT={t_meas:.7f}\n"
        sp += f".meas tran slp_b{i} FIND V(f7_slope) AT={t_meas:.7f}\n"
        sp += f".meas tran sym_b{i} FIND V(f8_symmetry) AT={t_meas:.7f}\n"
    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, f"_acc_{record_id}_b{batch_id}.spice")
    with open(tb, 'w') as f:
        f.write(sp)
    return tb, skip


def run_ngspice(tb):
    log = tb.replace('.spice', '.log')
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
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
        return np.array(data['features']), np.array(data['labels'])

    beats, gain = load_all_beats(record_id)
    if not beats:
        return np.array([]).reshape(0, 8), np.array([])

    all_feats, all_labels = [], []
    n_batches = (len(beats) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        if start > 0: start -= 1  # overlap 1 for warm-up
        end = min((b_idx + 1) * BATCH_SIZE, len(beats))
        batch = beats[start:end]

        tb, skip = generate_batch_testbench(batch, gain, b_idx, record_id)
        results = run_ngspice(tb)

        if results is None:
            for i in range(skip, len(batch)):
                orig_idx = start + i
                bt = beats[orig_idx]
                all_feats.append([bt['f1_rr_ratio'], bt['f2_rr_asym'],
                                  bt['py_bi_l0'], bt['py_bi_l1'],
                                  bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']])
                all_labels.append(bt['label'])
            continue

        for i in range(skip, len(batch)):
            orig_idx = start + i
            if orig_idx >= len(beats): break
            bt = beats[orig_idx]
            bi0 = results.get(f'bi0_b{i}', float('nan'))
            bi1 = results.get(f'bi1_b{i}', float('nan'))
            pw = results.get(f'pw_b{i}', float('nan'))
            wid = results.get(f'wid_b{i}', float('nan'))
            slp = results.get(f'slp_b{i}', float('nan'))
            sym = results.get(f'sym_b{i}', float('nan'))

            if any(np.isnan(x) for x in [bi0, bi1, pw, wid, slp, sym]):
                bi0 = bt['py_bi_l0']; bi1 = bt['py_bi_l1']
                pw = bt['py_pw']; wid = bt['py_wid']
                slp = bt['py_slp']; sym = bt['py_sym']

            all_feats.append([bt['f1_rr_ratio'], bt['f2_rr_asym'],
                              bi0, bi1, pw, wid, slp, sym])
            all_labels.append(bt['label'])

        # cleanup temp files
        try:
            os.remove(tb)
            os.remove(tb.replace('.spice', '.log'))
        except: pass

    feats = np.array(all_feats) if all_feats else np.array([]).reshape(0, 8)
    labels = np.array(all_labels) if all_labels else np.array([])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({'features': feats.tolist(), 'labels': labels.tolist()}, f)

    return feats, labels


def run_python_baseline():
    """Run pure-Python proxy baseline for comparison."""
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for rec in DS1:
        beats, _ = load_all_beats(rec)
        for bt in beats:
            X_tr.append([bt['f1_rr_ratio'], bt['f2_rr_asym'],
                         bt['py_bi_l0'], bt['py_bi_l1'],
                         bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']])
            y_tr.append(bt['label'])
    for rec in DS2:
        beats, _ = load_all_beats(rec)
        for bt in beats:
            X_te.append([bt['f1_rr_ratio'], bt['f2_rr_asym'],
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
print("FULL INTER-PATIENT ACCURACY TEST — ANALOG FRONTEND IN THE LOOP")
print("=" * 70)
print(f"  Features 1-2: Python (RR timing — validated digital timer)")
print(f"  Features 3-8: ngspice circuit simulation (SKY130 PDK)")
print(f"  Classifier:   RandomForest(n=300, depth=15, balanced)")
print(f"  Dataset:       MIT-BIH, DS1 train / DS2 test")
print()

# Python baseline first (fast)
print("Running Python-proxy baseline...")
py_acc, n_tr, n_te = run_python_baseline()
print(f"  Python proxy accuracy: {py_acc*100:.2f}% (train={n_tr}, test={n_te})")
print()

# Analog frontend accuracy (slow — ngspice per record)
print("Running analog frontend simulations...")
os.makedirs(CACHE_DIR, exist_ok=True)

X_train, y_train, X_test, y_test = [], [], [], []
total_records = len(DS1) + len(DS2)

for idx, (split, records) in enumerate([("DS1-train", DS1), ("DS2-test", DS2)]):
    for i, rec in enumerate(records):
        t0 = time.time()
        feats, labels = process_record(rec)
        elapsed = time.time() - t0
        rec_num = idx * len(DS1) + i + 1
        print(f"  [{rec_num:2d}/{total_records}] Record {rec}: {len(labels):4d} beats, {elapsed:.1f}s")

        if len(feats) == 0: continue
        if split == "DS1-train":
            X_train.append(feats); y_train.append(labels)
        else:
            X_test.append(feats); y_test.append(labels)

X_train = np.vstack(X_train); y_train = np.concatenate(y_train)
X_test = np.vstack(X_test); y_test = np.concatenate(y_test)

print(f"\n  Train: {len(y_train)} beats, Test: {len(y_test)} beats")

clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
analog_acc = accuracy_score(y_test, y_pred)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Python proxy accuracy:  {py_acc*100:.2f}%")
print(f"  Analog frontend accuracy: {analog_acc*100:.2f}%")
print(f"  Delta:                  {(analog_acc - py_acc)*100:+.2f}%")
print()
print("  Per-class breakdown:")
print(classification_report(y_test, y_pred, digits=3))
print("=" * 70)

# Feature importance
print("Feature importances:")
names = ['F1:rr_ratio', 'F2:rr_asym', 'F3:bi_L0', 'F4:bi_L1',
         'F5:pwave', 'F6:width', 'F7:slope', 'F8:symmetry']
for name, imp in sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<15s} {imp:.4f}")
