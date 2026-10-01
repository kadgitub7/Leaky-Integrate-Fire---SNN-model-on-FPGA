"""
Test a single new feature: prematurity z-score
z = (median_rr - pre_rr) / max(rr_std, floor)

Hypothesis: this patient-normalizes prematurity and should have
lower domain shift than raw pre_rr (1.26 sigma shift) or rr_ratio.

Also tests: incomplete compensatory pause score
icp = 2*median_rr - (pre_rr + post_rr)  (positive = incomplete = S-like)
"""
import sys, os, time
import numpy as np
import wfdb
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from data_proven import DS1, DS2, aami_map, QRS_START, QRS_END, WIN_LEFT, WIN_RIGHT, N_TEMPLATE, EWMA_ALPHA_MORPH, EWMA_GATE_THRESH, EWMA_INIT_BEATS

t0 = time.time()

FEAT_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1', 'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1', 'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
    # New features
    'z_premature',       # (median_rr - pre_rr) / max(rr_std, 0.02)
    'icp_score',         # 2*median_rr - (pre_rr + post_rr), normalized
    'rr_local_rank',     # percentile rank of pre_rr in last 20 beats (0=shortest)
    'rr_delta',          # (prev_rr - pre_rr) / prev_rr  — sudden shortening
    'premature_normal',  # z_premature * templ_corr_L0 — premature AND normal QRS
]


def extract_with_zscore(rec_list):
    all_labels, all_features, all_records = [], [], []
    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            fs = record.fs
            num_leads = min(signals.shape[1], 2)
            valid = []
            for idx, sym in zip(annotation.sample, annotation.symbol):
                if idx - WIN_LEFT >= 0 and idx + WIN_RIGHT < len(signals) and sym in aami_map:
                    valid.append((idx, sym))

            ewma_area = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []
            rr_window = []  # last 20 RR intervals for rank

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - WIN_LEFT : idx + WIN_RIGHT, :]
                all_labels.append(aami_map[sym])
                all_records.append(rec_id)

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - 10), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833

                rr_history.append(pre_rr)
                if len(rr_history) > 10: rr_history.pop(0)

                rr_window.append(pre_rr)
                if len(rr_window) > 20: rr_window.pop(0)

                # Median of recent RR (more robust than mean to outliers)
                median_rr = np.median(rr_history) if len(rr_history) >= 2 else local_rr
                rr_std = np.std(rr_history) if len(rr_history) >= 2 else 0.05

                feat = np.zeros(len(FEAT_NAMES), dtype=np.float32)

                # Standard 16 features
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                for lead in range(num_leads):
                    qrs = beat[QRS_START:QRS_END, lead]
                    abs_qrs = np.abs(qrs)
                    peak = np.max(abs_qrs) if len(abs_qrs) > 0 else 0.0
                    width = 0.0
                    if peak > 1e-6:
                        threshold = 0.3 * peak
                        above = abs_qrs > threshold
                        if np.any(above):
                            first = np.argmax(above)
                            last = len(above) - 1 - np.argmax(above[::-1])
                            width = (last - first) * 1000.0 / fs
                    area = np.sum(abs_qrs) / fs
                    dqrs = np.diff(qrs) * fs
                    max_slope = np.max(np.abs(dqrs)) if len(dqrs) > 0 else 0.0
                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]
                    feat[6 + lead] = width
                    feat[8 + lead] = area
                    feat[10 + lead] = max_slope
                    if i < EWMA_INIT_BEATS:
                        init_areas[lead].append(max(area, 1e-6))
                        init_templates[lead].append(qrs_ds.copy())
                    if ewma_area[lead] is None:
                        if i >= EWMA_INIT_BEATS - 1 and len(init_areas[lead]) >= EWMA_INIT_BEATS:
                            ewma_area[lead] = np.median(init_areas[lead])
                            ewma_template[lead] = np.median(init_templates[lead], axis=0)
                        else:
                            ewma_area[lead] = max(area, 1e-6)
                            ewma_template[lead] = qrs_ds.copy()
                    feat[12 + lead] = area / (ewma_area[lead] + 1e-8)
                    norm_c = np.linalg.norm(qrs_ds)
                    norm_t = np.linalg.norm(ewma_template[lead])
                    tc = np.dot(qrs_ds, ewma_template[lead]) / (norm_c * norm_t + 1e-8) if norm_c > 1e-8 and norm_t > 1e-8 else 1.0
                    feat[14 + lead] = tc
                    if tc > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

                # === NEW FEATURES ===

                # Z-score: how many patient-SDs early is this beat?
                floor = 0.020  # 20ms floor prevents divide-by-zero for very regular rhythms
                z_premature = (median_rr - pre_rr) / max(rr_std, floor)
                feat[16] = z_premature

                # Incomplete compensatory pause score
                # For PACs: pre_rr + post_rr < 2*median_rr (incomplete)
                # For PVCs: pre_rr + post_rr ~ 2*median_rr (complete)
                # Normalize by median_rr to make patient-independent
                icp = (2 * median_rr - (pre_rr + post_rr)) / (median_rr + 1e-8)
                feat[17] = icp

                # Local rank: what percentile is this RR in the last 20?
                if len(rr_window) >= 3:
                    rank = sum(1 for r in rr_window if r < pre_rr) / len(rr_window)
                else:
                    rank = 0.5
                feat[18] = rank

                # RR delta: sudden shortening from previous beat
                if i >= 2:
                    prev_rr = (valid[i-1][0] - valid[i-2][0]) / fs
                    rr_delta = (prev_rr - pre_rr) / (prev_rr + 1e-8)
                else:
                    rr_delta = 0.0
                feat[19] = rr_delta

                # Premature AND normal QRS (the S-beat signature)
                feat[20] = z_premature * feat[14]  # z_premature * templ_corr_L0

                all_features.append(feat)
        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")
    return np.array(all_labels), np.array(all_features), all_records


if __name__ == '__main__':
    print(f"\nZ-Score Feature Diagnostic")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ds1_labels, ds1_feats, ds1_recs = extract_with_zscore(DS1)
    ds2_labels, ds2_feats, ds2_recs = extract_with_zscore(DS2)
    print(f"DS1: {len(ds1_labels)} beats | DS2: {len(ds2_labels)} beats")

    new_feat_idx = list(range(16, len(FEAT_NAMES)))
    new_names = FEAT_NAMES[16:]

    # ============================================================
    # 1. Per-feature S vs N analysis for new features
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  NEW FEATURE ANALYSIS: S vs N separation")
    print(f"{'='*70}")

    for ds_name, labels, feats in [("DS1", ds1_labels, ds1_feats), ("DS2", ds2_labels, ds2_feats)]:
        s_mask = labels == 1
        n_mask = labels == 0
        print(f"\n  {ds_name} (S={sum(s_mask)}, N={sum(n_mask)}):")
        print(f"  {'Feature':>25} {'S mean':>8} {'N mean':>8} {'S med':>8} {'N med':>8} {'Cohen d':>8} {'Overlap':>8}")
        for fi in new_feat_idx:
            s_vals = feats[s_mask, fi]
            n_vals = feats[n_mask, fi]
            s_mu, s_sd = s_vals.mean(), s_vals.std() + 1e-8
            n_mu, n_sd = n_vals.mean(), n_vals.std() + 1e-8
            pooled = np.sqrt((s_sd**2 + n_sd**2) / 2)
            d = abs(s_mu - n_mu) / pooled
            overlap = max(0, 1.0 - d / 3.0)
            print(f"  {FEAT_NAMES[fi]:>25} {s_mu:>8.3f} {n_mu:>8.3f} "
                  f"{np.median(s_vals):>8.3f} {np.median(n_vals):>8.3f} "
                  f"{d:>8.3f} {overlap:>7.1%}")

    # ============================================================
    # 2. Domain shift for new features
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  DOMAIN SHIFT: new features, S-class DS1 vs DS2")
    print(f"{'='*70}")
    s1 = ds1_feats[ds1_labels == 1]
    s2 = ds2_feats[ds2_labels == 1]
    print(f"\n  {'Feature':>25} {'DS1 S':>10} {'DS2 S':>10} {'Shift':>8}  vs old")
    for fi in new_feat_idx:
        s1_mu = s1[:, fi].mean()
        s2_mu = s2[:, fi].mean()
        s1_sd = s1[:, fi].std() + 1e-8
        shift = abs(s2_mu - s1_mu) / s1_sd
        print(f"  {FEAT_NAMES[fi]:>25} {s1_mu:>10.4f} {s2_mu:>10.4f} {shift:>7.3f}sigma")

    # Compare to old features
    print(f"\n  Reference - old feature shifts:")
    old_shifts = {'pre_rr': 0, 'rr_ratio': 2, 'rr_asymmetry': 3, 'rr_std_10': 5}
    for name, fi in old_shifts.items():
        s1_mu = s1[:, fi].mean()
        s2_mu = s2[:, fi].mean()
        s1_sd = s1[:, fi].std() + 1e-8
        shift = abs(s2_mu - s1_mu) / s1_sd
        print(f"  {name:>25} {s1_mu:>10.4f} {s2_mu:>10.4f} {shift:>7.3f}sigma")

    # ============================================================
    # 3. Per-patient z_premature distribution
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PER-PATIENT z_premature: S beats vs N beats")
    print(f"{'='*70}")
    for ds_name, labels, feats, recs in [("DS1", ds1_labels, ds1_feats, ds1_recs),
                                          ("DS2", ds2_labels, ds2_feats, ds2_recs)]:
        print(f"\n  {ds_name}:")
        patients = sorted(set(recs))
        for rec_id in patients:
            rec_mask = np.array([r == rec_id for r in recs])
            s_in_rec = (labels == 1) & rec_mask
            n_in_rec = (labels == 0) & rec_mask
            n_s = sum(s_in_rec)
            if n_s == 0:
                continue
            s_z = feats[s_in_rec, 16]  # z_premature
            n_z = feats[n_in_rec, 16]
            sep = s_z.mean() - n_z.mean()
            print(f"    {rec_id}: S z={s_z.mean():>6.2f}+-{s_z.std():.2f} (n={n_s:>4}) | "
                  f"N z={n_z.mean():>6.2f}+-{n_z.std():.2f} | "
                  f"sep={sep:>+6.2f}")

    # ============================================================
    # 4. Binary S-vs-N classifier with z_premature alone
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  BINARY S vs N: feature set comparison")
    print(f"{'='*70}")

    sn_train = (ds1_labels == 0) | (ds1_labels == 1)
    sn_test = (ds2_labels == 0) | (ds2_labels == 1)
    X_tr = ds1_feats[sn_train]
    y_tr = (ds1_labels[sn_train] == 1).astype(int)
    X_te = ds2_feats[sn_test]
    y_te = (ds2_labels[sn_test] == 1).astype(int)

    configs = {
        '16 standard':          list(range(16)),
        '16 + z_premature':     list(range(16)) + [16],
        '16 + all_new (21)':    list(range(21)),
        'timing_6 + z_pre':     list(range(6)) + [16],
        'z_pre + icp + rank':   [16, 17, 18],
        'z_pre + rr_delta + icp': [16, 17, 19],
        'z_pre + premature_normal': [16, 20],
        'rr_asym + z_pre + icp': [3, 16, 17],
    }

    # GradientBoosting with oversampling (best from prior diagnostic)
    s_idx = np.where(y_tr == 1)[0]
    n_idx = np.where(y_tr == 0)[0]
    n_target = len(n_idx) // 10
    if len(s_idx) < n_target:
        extra = np.random.choice(s_idx, n_target - len(s_idx), replace=True)
        bal_idx = np.concatenate([n_idx, s_idx, extra])
    else:
        bal_idx = np.concatenate([n_idx, s_idx])
    np.random.shuffle(bal_idx)

    print(f"\n  GradientBoosting (10% S oversample):")
    print(f"  {'Config':>30} {'S Prec':>8} {'S Rec':>8} {'S F1':>8} {'N lost':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, idx in configs.items():
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
        gb.fit(X_tr[bal_idx][:, idx], y_tr[bal_idx])
        pred = gb.predict(X_te[:, idx])
        cm = confusion_matrix(y_te, pred)
        s_prec = cm[1,1] / (cm[0,1] + cm[1,1] + 1e-8)
        s_rec = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-8)
        s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec + 1e-8)
        n_lost = cm[0,1]  # false S predictions on actual N
        print(f"  {name:>30} {s_prec:>7.1%} {s_rec:>7.1%} {s_f1:>7.3f} {n_lost:>8}")

    # ============================================================
    # 5. Threshold sweep: what S recall can we get at 1% N false positive?
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  THRESHOLD SWEEP: z_premature alone")
    print(f"{'='*70}")
    z_s = ds2_feats[ds2_labels == 1, 16]
    z_n = ds2_feats[ds2_labels == 0, 16]
    print(f"\n  DS2 z_premature: S mean={z_s.mean():.3f} std={z_s.std():.3f} | "
          f"N mean={z_n.mean():.3f} std={z_n.std():.3f}")
    print(f"\n  {'Threshold':>10} {'S recall':>10} {'N FP rate':>10} {'S caught':>10} {'N false':>10}")
    for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        s_caught = sum(z_s > thresh)
        n_false = sum(z_n > thresh)
        s_recall = s_caught / len(z_s)
        n_fp = n_false / len(z_n)
        print(f"  {thresh:>10.1f} {s_recall:>9.1%} {n_fp:>9.2%} {s_caught:>10} {n_false:>10}")

    # ============================================================
    # 6. 5-class with best new feature set
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  5-CLASS RF: 16 vs 16+z_premature vs 21 features")
    print(f"{'='*70}")
    for name, idx in [('16 standard', list(range(16))),
                       ('16 + z_premature', list(range(16)) + [16]),
                       ('all 21', list(range(21)))]:
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(ds1_feats[:, idx], ds1_labels)
        pred = rf.predict(ds2_feats[:, idx])
        acc = np.mean(pred == ds2_labels) * 100
        cm = confusion_matrix(ds2_labels, pred)
        s_rec = cm[1,1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
        v_rec = cm[2,2] / cm[2].sum() * 100 if cm[2].sum() > 0 else 0
        print(f"\n  {name}: {acc:.2f}% overall | S recall: {s_rec:.1f}% | V recall: {v_rec:.1f}%")
        print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
        for i, lbl in enumerate(['N','S','V','F','Q']):
            print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
