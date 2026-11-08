"""
Analog Feature Set Sweep: Test what accuracy different analog-realizable
feature sets can achieve for arrhythmia classification.

Compares: digital gold-standard (ratios), current analog (differences),
log-domain (subthreshold MOSFET realizable), raw node voltages, etc.
"""

import os, sys
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score
from collections import Counter

MITDB = os.path.join(os.path.dirname(__file__), '..', '..', 'mitdb_data')

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']

FS = 360; WIN_L = 90; WIN_R = 108; QRS_S = 70; QRS_E = 130
PW_S = 16; PW_E = QRS_S; N_CLASSES = 5
CLASS_NAMES = ['N','S','V','F','Q']

AAMI_MAP = {}
for s in ['N','L','R','e','j']: AAMI_MAP[s] = 0
for s in ['A','a','J','S']:     AAMI_MAP[s] = 1
for s in ['V','E']:             AAMI_MAP[s] = 2
for s in ['F']:                 AAMI_MAP[s] = 3
for s in ['/','f','Q']:         AAMI_MAP[s] = 4

b_pw, a_pw = butter(2, [3.0/(FS/2), 8.0/(FS/2)], btype='band')


def extract_raw_beats(rec_list):
    """Extract raw beat data needed by all feature sets."""
    import wfdb
    all_beats = []
    for rec_id in rec_list:
        path = os.path.join(MITDB, rec_id)
        try:
            rec = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  Skip {rec_id}: {e}")
            continue

        sig = rec.p_signal
        nl = min(sig.shape[1], 2)

        # Bandpass filter for P-wave energy
        pwf = np.zeros_like(sig)
        for ld in range(nl):
            try:
                pwf[:, ld] = filtfilt(b_pw, a_pw, sig[:, ld])
            except:
                pass

        valid = [(i, s) for i, s in zip(ann.sample, ann.symbol)
                 if i - WIN_L >= 0 and i + WIN_R < len(sig) and s in AAMI_MAP]

        for vi, (idx, sym) in enumerate(valid):
            bt = sig[idx - WIN_L:idx + WIN_R, :]
            bpw = pwf[idx - WIN_L:idx + WIN_R, :]

            pre_rr = (idx - valid[vi-1][0]) / FS if vi > 0 else 0.833
            post_rr = (valid[vi+1][0] - idx) / FS if vi < len(valid)-1 else 0.833

            # QRS morphology per lead
            morph = {}
            for ld in range(nl):
                q = bt[QRS_S:QRS_E, ld]
                aq = np.abs(q)
                pk = np.max(aq) if len(aq) > 0 else 0.0
                ar = np.sum(aq) / FS

                # Width at 30% threshold
                w = 0.0
                on_idx = 0
                if pk > 1e-6:
                    th = 0.3 * pk
                    ab = aq > th
                    if np.any(ab):
                        on_idx = np.argmax(ab)
                        off_idx = len(ab) - 1 - np.argmax(ab[::-1])
                        w = (off_idx - on_idx) * 1000.0 / FS

                # Slope
                dq = np.diff(q) * FS
                max_rise = np.max(dq) if len(dq) > 0 else 1e-8
                max_fall = np.abs(np.min(dq)) if len(dq) > 0 else 1e-8
                sr = max(max_rise, 1e-8) / (max(max_fall, 1e-8) + 1e-8)

                # Symmetry
                pi = np.argmax(aq) if pk > 1e-6 else 0
                qw_samp = max(w, 1e-6) * FS / 1000.0
                sy = np.clip((pi - on_idx) / (qw_samp + 1e-8), 0.0, 1.0) if pk > 1e-6 else 0.5

                # P-wave
                pw_seg = bpw[PW_S:PW_E, ld]
                pw_energy = np.sum(pw_seg**2) / FS
                pw_raw = bt[PW_S:PW_E, ld]
                pw_mid = (PW_S + PW_E) // 2
                pw_diff = bt[pw_mid, ld] - bt[PW_S, ld] if bt.shape[0] > pw_mid else 0.0

                # Slope samples (for analog: early - late voltage difference)
                s5 = bt[QRS_S + 5, ld] if QRS_S + 5 < bt.shape[0] else 0.0
                s25 = bt[QRS_S + 25, ld] if QRS_S + 25 < bt.shape[0] else 0.0

                # Analog-equivalent features (raw S&H voltages)
                rpeak_idx = WIN_L  # R-peak is at center of beat window
                analog_wid = bt[rpeak_idx + 10, ld] if rpeak_idx + 10 < bt.shape[0] else 0.0
                analog_sym = bt[rpeak_idx, ld]

                morph[ld] = {
                    'pk': pk, 'ar': ar, 'w': w, 'sr': sr, 'sy': sy,
                    'pw_energy': pw_energy, 'pw_diff': pw_diff,
                    's5': s5, 's25': s25,
                    'slope_diff': s5 - s25,
                    'peak_idx': pi, 'onset_idx': on_idx, 'width_samp': qw_samp,
                    'analog_wid': analog_wid, 'analog_sym': analog_sym,
                }

            all_beats.append({
                'rec_id': rec_id, 'idx': idx, 'vi': vi,
                'label': AAMI_MAP[sym],
                'pre_rr': pre_rr, 'post_rr': post_rr,
                'morph': morph, 'nl': nl,
            })

    return all_beats


def compute_features(beats, set_name, ewma_alpha=0.10):
    """Compute feature vectors for a given feature set configuration."""
    feats = []
    labels = []

    # Per-record state
    rec_state = {}

    for b in beats:
        rid = b['rec_id']
        if rid not in rec_state:
            rec_state[rid] = {
                'rrh': [], 'ewma_rr': None, 'prev_pre_rr': None,
                'prev_pk': [None, None], 'beat_count': 0,
                'ewma_init_rrs': [],
            }
        st = rec_state[rid]
        st['beat_count'] += 1

        pre_rr = b['pre_rr']
        post_rr = b['post_rr']
        m0 = b['morph'].get(0, {})
        m1 = b['morph'].get(1, b['morph'].get(0, {}))

        # Update RR history (for mean-of-last-20)
        st['rrh'].append(pre_rr)
        if len(st['rrh']) > 20:
            st['rrh'].pop(0)
        lrr = np.mean(st['rrh']) if st['rrh'] else 0.833

        # EWMA update
        if st['ewma_rr'] is None:
            st['ewma_init_rrs'].append(pre_rr)
            if len(st['ewma_init_rrs']) >= 10:
                st['ewma_rr'] = np.median(st['ewma_init_rrs'])
            else:
                st['ewma_rr'] = pre_rr
            ewma_before = st['ewma_rr']
        else:
            ewma_before = st['ewma_rr']
            st['ewma_rr'] = ewma_alpha * pre_rr + (1 - ewma_alpha) * st['ewma_rr']

        prev_pk0 = st['prev_pk'][0]
        prev_pk1 = st['prev_pk'][1]
        prev_pre_rr = st['prev_pre_rr']

        pk0 = m0.get('pk', 0.0)
        pk1 = m1.get('pk', 0.0)

        f = None

        if set_name == 'A':
            # Digital gold standard: ratios
            f = np.array([
                pre_rr / (lrr + 1e-8),                                          # rr_ratio
                abs(pk1 - prev_pk1) / (prev_pk1 + 1e-8) if prev_pk1 is not None and prev_pk1 > 1e-6 else 0.0,  # bi_L1
                pre_rr / (post_rr + 1e-8),                                      # rr_asym
                m1.get('pw_energy', 0.0),                                        # pw_energy_L1
                m0.get('w', 0.0),                                                # qrs_width_L0
                abs(pk0 - prev_pk0) / (prev_pk0 + 1e-8) if prev_pk0 is not None and prev_pk0 > 1e-6 else 0.0,  # bi_L0
                m0.get('sr', 1.0),                                               # slope_ratio_L0
                m0.get('sy', 0.5),                                               # qrs_sym_L0
            ])

        elif set_name in ('B', 'C', 'D'):
            # Raw RR + differences
            f = np.array([
                pre_rr,                                                          # raw pre_rr
                ewma_before,                                                     # ewma_rr
                prev_pre_rr if prev_pre_rr is not None else pre_rr,              # prev_pre_rr
                (pk1 - prev_pk1) if prev_pk1 is not None else 0.0,              # unnorm bi_L1
                m1.get('pw_diff', 0.0),                                          # pw voltage diff L1
                m0.get('w', 0.0),                                                # qrs_width_L0
                m0.get('slope_diff', 0.0),                                       # slope voltage diff L0
                m0.get('sy', 0.5),                                               # qrs_sym_L0
            ])

        elif set_name == 'E':
            # 10 features: raw RR + separate peaks
            f = np.array([
                pre_rr,
                ewma_before,
                prev_pre_rr if prev_pre_rr is not None else pre_rr,
                pk0,
                prev_pk0 if prev_pk0 is not None else pk0,
                m1.get('pw_diff', 0.0),
                m0.get('w', 0.0),
                m0.get('slope_diff', 0.0),
                m0.get('sy', 0.5),
                pk1,
            ])

        elif set_name == 'F':
            # Analog differences + slow EWMA + gain
            diff_rr = pre_rr - ewma_before
            diff_asym = pre_rr - post_rr
            f = np.array([
                diff_rr * 20,                                                    # amplified rr diff
                diff_asym * 20,                                                  # amplified asym diff
                (pk0 - prev_pk0) if prev_pk0 is not None else 0.0,              # bi_L0 (diff)
                (pk1 - prev_pk1) if prev_pk1 is not None else 0.0,              # bi_L1 (diff)
                m1.get('pw_diff', 0.0),                                          # pw diff
                m0.get('w', 0.0),                                                # width
                m0.get('slope_diff', 0.0),                                       # slope diff
                m0.get('sy', 0.5),                                               # symmetry
            ])

        elif set_name == 'G':
            # Log-domain ratios
            eps = 1e-4
            f = np.array([
                np.log(max(pre_rr, eps)) - np.log(max(ewma_before, eps)),        # log(rr_ratio)
                np.log(max(pre_rr, eps)) - np.log(max(post_rr, eps)),            # log(rr_asym)
                np.log(max(pk1, eps)) - np.log(max(prev_pk1, eps)) if prev_pk1 is not None and prev_pk1 > eps else 0.0,  # log(bi_L1)
                m1.get('pw_diff', 0.0),                                          # pw diff
                m0.get('w', 0.0),                                                # width
                np.log(max(pk0, eps)) - np.log(max(prev_pk0, eps)) if prev_pk0 is not None and prev_pk0 > eps else 0.0,  # log(bi_L0)
                m0.get('slope_diff', 0.0),                                       # slope diff
                m0.get('sy', 0.5),                                               # symmetry
            ])

        elif set_name == 'H':
            # Set F but with ANALOG-REALISTIC width and symmetry
            diff_rr = pre_rr - ewma_before
            diff_asym = pre_rr - post_rr
            f = np.array([
                diff_rr * 20,                                                    # amplified rr diff
                diff_asym * 20,                                                  # amplified asym diff
                (pk0 - prev_pk0) if prev_pk0 is not None else 0.0,              # bi_L0 (diff)
                (pk1 - prev_pk1) if prev_pk1 is not None else 0.0,              # bi_L1 (diff)
                m1.get('pw_diff', 0.0),                                          # pw diff
                m0.get('analog_wid', 0.0),                                       # ANALOG: raw ECG@R+28ms
                m0.get('slope_diff', 0.0),                                       # slope diff
                m0.get('analog_sym', 0.0),                                       # ANALOG: raw ECG@R-peak
            ])

        elif set_name == 'I':
            # Set H but with BOTH lead peaks for beat instability (raw voltages)
            diff_rr = pre_rr - ewma_before
            diff_asym = pre_rr - post_rr
            f = np.array([
                diff_rr * 20,                                                    # amplified rr diff
                diff_asym * 20,                                                  # amplified asym diff
                pk0,                                                             # raw current peak L0
                prev_pk0 if prev_pk0 is not None else pk0,                       # raw prev peak L0
                m1.get('pw_diff', 0.0),                                          # pw diff
                m0.get('analog_wid', 0.0),                                       # ANALOG width
                m0.get('slope_diff', 0.0),                                       # slope diff
                m0.get('analog_sym', 0.0),                                       # ANALOG symmetry
            ])

        # Update state
        st['prev_pk'] = [pk0, pk1]
        st['prev_pre_rr'] = pre_rr

        if f is not None:
            feats.append(f)
            labels.append(b['label'])

    return np.array(feats), np.array(labels)


def train_and_eval(X_tr, y_tr, X_te, y_te, hidden=(64,)):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500,
                        random_state=42, early_stopping=True,
                        validation_fraction=0.1)
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)

    acc = np.mean(y_pred == y_te)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(N_CLASSES)))
    recalls = {}
    for i, cn in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        recalls[cn] = cm[i, i] / total if total > 0 else 0.0
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)

    return acc, recalls, macro_f1, cm


def main():
    print("=" * 70)
    print("ANALOG FEATURE SET SWEEP")
    print("=" * 70)

    print("\nLoading DS1 (train)...")
    beats_tr = extract_raw_beats(DS1)
    print(f"  {len(beats_tr)} beats")
    print(f"  Class dist: {dict(sorted(Counter(b['label'] for b in beats_tr).items()))}")

    print("\nLoading DS2 (test)...")
    beats_te = extract_raw_beats(DS2)
    print(f"  {len(beats_te)} beats")
    print(f"  Class dist: {dict(sorted(Counter(b['label'] for b in beats_te).items()))}")

    configs = [
        ('A', 'Digital gold (ratios)', 0.10),
        ('F', 'Diffs + slow EWMA + 20x gain', 0.015),
        ('H', 'Set F + analog width/sym', 0.015),
        ('I', 'Set H + raw peaks (no bi diff)', 0.015),
        ('G', 'Log-domain ratios, a=0.05', 0.05),
    ]

    results = []

    for set_name, desc, alpha in configs:
        print(f"\n{'='*70}")
        print(f"Feature Set {set_name}: {desc}")
        print(f"{'='*70}")

        X_tr, y_tr = compute_features(beats_tr, set_name, ewma_alpha=alpha)
        X_te, y_te = compute_features(beats_te, set_name, ewma_alpha=alpha)
        print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

        for hid_name, hid in [('64', (64,)), ('128-64', (128, 64))]:
            acc, recalls, macro_f1, cm = train_and_eval(X_tr, y_tr, X_te, y_te, hidden=hid)
            print(f"\n  MLP {hid_name}: acc={acc:.4f}, macro_F1={macro_f1:.4f}")
            for cn in CLASS_NAMES:
                print(f"    {cn}: recall={recalls[cn]:.4f}")

            results.append({
                'set': set_name, 'desc': desc, 'hidden': hid_name,
                'acc': acc, 'macro_f1': macro_f1, 'recalls': recalls,
            })

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Set':<5} {'Hidden':<8} {'Acc':>6} {'mF1':>6} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}  {'Description'}"
    print(header)
    print("-" * 100)
    for r in results:
        rc = r['recalls']
        line = (f"{r['set']:<5} {r['hidden']:<8} "
                f"{r['acc']:>6.3f} {r['macro_f1']:>6.3f} "
                f"{rc['N']:>6.3f} {rc['S']:>6.3f} {rc['V']:>6.3f} "
                f"{rc['F']:>6.3f} {rc['Q']:>6.3f}  {r['desc']}")
        print(line)


if __name__ == '__main__':
    main()
