"""
S-class Deep Diagnostic: Why S beats are nearly undetectable
=============================================================
S-class (supraventricular ectopic: A, a, J, S symbols) is the #1 bottleneck.
Current: 2.2% recall (1,777/1,837 misclassified as N).
Even RF only gets F1 = 0.05-0.09.

This script:
  1. Feature-level analysis: which features separate S from N at all?
  2. Per-patient S distribution: is S concentrated in a few patients?
  3. Feature interaction search: are there 2-feature combos that help?
  4. P-wave / pre-QRS morphology: can we extract a new S-discriminative feature?
  5. Prematurity analysis: how well does timing alone separate S from N?
  6. Train a binary S-vs-N classifier to find the accuracy ceiling

Usage:
  python experiments/diag_s_class.py
"""

import sys, os, time
import numpy as np
import wfdb
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from data_proven import (extract_features, DS1, DS2, aami_map,
                         EWMA_ALPHA_MORPH, N_TEMPLATE, EWMA_GATE_THRESH,
                         EWMA_INIT_BEATS, QRS_START, QRS_END, WIN_LEFT, WIN_RIGHT)

t0 = time.time()

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1', 'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1', 'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1'
]


def extract_features_with_extra(rec_list):
    """Extract 16 standard features + extra S-discriminative candidates."""
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

                # Standard 16 features
                feat = np.zeros(16, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                qrs_data = {}
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
                    qrs_data[lead] = qrs

                # === EXTRA S-DISCRIMINATIVE FEATURES ===
                extra = []

                # E1: Pre-QRS morphology (P-wave region) — samples 40-70 (before QRS)
                for lead in range(num_leads):
                    pre_qrs = beat[40:QRS_START, lead]
                    pre_qrs_energy = np.sum(pre_qrs**2)
                    pre_qrs_max = np.max(np.abs(pre_qrs)) if len(pre_qrs) > 0 else 0.0
                    pre_qrs_slope = np.max(np.abs(np.diff(pre_qrs) * fs)) if len(pre_qrs) > 1 else 0.0
                    extra.extend([pre_qrs_energy, pre_qrs_max, pre_qrs_slope])

                # E2: Post-QRS morphology (T-wave region) — samples 130-180
                for lead in range(num_leads):
                    post_qrs = beat[QRS_END:min(QRS_END+50, len(beat)), lead]
                    post_qrs_energy = np.sum(post_qrs**2)
                    post_qrs_max = np.max(np.abs(post_qrs)) if len(post_qrs) > 0 else 0.0
                    extra.extend([post_qrs_energy, post_qrs_max])

                # E3: Prematurity index (key S discriminator)
                prematurity = 1.0 - feat[2]  # 1 - rr_ratio: how much earlier than expected
                extra.append(prematurity)

                # E4: Compensatory pause indicator
                # S beats typically have incomplete compensatory pause
                # V beats typically have full compensatory pause
                # post_rr / local_rr: >1 means post-beat interval longer than average
                post_rr_ratio = post_rr / (local_rr + 1e-8)
                extra.append(post_rr_ratio)

                # E5: Pre-RR change rate (how sudden the premature beat is)
                if i >= 2:
                    prev_rr = (valid[i-1][0] - valid[i-2][0]) / fs
                    rr_change = (pre_rr - prev_rr) / (prev_rr + 1e-8)
                else:
                    rr_change = 0.0
                extra.append(rr_change)

                # E6: QRS morphology difference from template (L2 distance)
                for lead in range(num_leads):
                    qrs = beat[QRS_START:QRS_END, lead]
                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]
                    if ewma_template[lead] is not None:
                        l2_diff = np.linalg.norm(qrs_ds - ewma_template[lead])
                    else:
                        l2_diff = 0.0
                    extra.append(l2_diff)

                # E7: Heart rate (beats per minute from local RR)
                hr = 60.0 / (local_rr + 1e-8)
                extra.append(hr)

                # E8: RR ratio squared (emphasizes prematurity)
                extra.append(feat[2] ** 2)

                # E9: Interaction: prematurity * template correlation
                # S beats should be premature AND have similar morphology to N
                extra.append(prematurity * feat[14])  # L0 templ_corr

                # E10: Interaction: prematurity * compensatory pause
                extra.append(prematurity * post_rr_ratio)

                combined = np.concatenate([feat, np.array(extra, dtype=np.float32)])
                all_features.append(combined)
        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features), all_records


EXTRA_NAMES = [
    'pre_qrs_energy_L0', 'pre_qrs_max_L0', 'pre_qrs_slope_L0',
    'pre_qrs_energy_L1', 'pre_qrs_max_L1', 'pre_qrs_slope_L1',
    'post_qrs_energy_L0', 'post_qrs_max_L0',
    'post_qrs_energy_L1', 'post_qrs_max_L1',
    'prematurity', 'post_rr_ratio', 'rr_change_rate',
    'qrs_l2_diff_L0', 'qrs_l2_diff_L1',
    'heart_rate', 'rr_ratio_sq',
    'prematurity_x_templ_corr', 'prematurity_x_comp_pause'
]
ALL_NAMES = FEATURE_NAMES + EXTRA_NAMES


if __name__ == '__main__':
    print(f"\nS-Class Deep Diagnostic")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract features with extras
    print(f"\n{'='*65}")
    print(f"  Extracting features from DS1 and DS2...")
    print(f"{'='*65}")
    ds1_labels, ds1_features, ds1_records = extract_features_with_extra(DS1)
    ds2_labels, ds2_features, ds2_records = extract_features_with_extra(DS2)
    print(f"  DS1: {len(ds1_labels)} beats ({sum(ds1_labels==1)} S-class)")
    print(f"  DS2: {len(ds2_labels)} beats ({sum(ds2_labels==1)} S-class)")

    # ================================================================
    # 1. Per-feature S vs N separation (DS1 train)
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  1. FEATURE-LEVEL S vs N SEPARATION (DS1)")
    print(f"{'='*65}")
    s_mask = ds1_labels == 1
    n_mask = ds1_labels == 0
    s_feats = ds1_features[s_mask]
    n_feats = ds1_features[n_mask]

    print(f"\n  {'Feature':>30} {'S mean':>8} {'N mean':>8} {'Diff':>8} {'S std':>8} {'N std':>8} {'Overlap':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for fi in range(len(ALL_NAMES)):
        s_vals = s_feats[:, fi]
        n_vals = n_feats[:, fi]
        s_mu, s_sd = s_vals.mean(), s_vals.std() + 1e-8
        n_mu, n_sd = n_vals.mean(), n_vals.std() + 1e-8
        # Cohen's d: standardized mean difference
        pooled_sd = np.sqrt((s_sd**2 + n_sd**2) / 2)
        cohens_d = abs(s_mu - n_mu) / (pooled_sd + 1e-8)
        # Overlap coefficient (approximate)
        overlap = 1.0 - min(cohens_d / 3.0, 1.0)
        print(f"  {ALL_NAMES[fi]:>30} {s_mu:>8.3f} {n_mu:>8.3f} {s_mu-n_mu:>+8.3f} "
              f"{s_sd:>8.3f} {n_sd:>8.3f} {overlap:>7.1%}")

    # ================================================================
    # 2. Per-patient S distribution
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  2. PER-PATIENT S DISTRIBUTION")
    print(f"{'='*65}")
    for ds_name, labels, records in [("DS1", ds1_labels, ds1_records), ("DS2", ds2_labels, ds2_records)]:
        print(f"\n  {ds_name}:")
        patient_s = defaultdict(int)
        patient_total = defaultdict(int)
        for lbl, rec in zip(labels, records):
            patient_total[rec] += 1
            if lbl == 1:
                patient_s[rec] += 1
        patients_with_s = {k: v for k, v in patient_s.items() if v > 0}
        print(f"    Patients with S beats: {len(patients_with_s)}/{len(set(records))}")
        for rec_id in sorted(patients_with_s, key=lambda x: -patients_with_s[x]):
            pct = patients_with_s[rec_id] / patient_total[rec_id] * 100
            print(f"      {rec_id}: {patients_with_s[rec_id]:>5} S beats ({pct:.1f}% of total)")

    # ================================================================
    # 3. Domain shift analysis for S-class features
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  3. DOMAIN SHIFT: S-class features DS1 vs DS2")
    print(f"{'='*65}")
    s1_feats = ds1_features[ds1_labels == 1]
    s2_feats = ds2_features[ds2_labels == 1]
    if len(s1_feats) > 0 and len(s2_feats) > 0:
        print(f"\n  {'Feature':>30} {'DS1 S mean':>10} {'DS2 S mean':>10} {'Shift(sd)':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for fi in range(len(ALL_NAMES)):
            s1_mu = s1_feats[:, fi].mean()
            s2_mu = s2_feats[:, fi].mean()
            s1_sd = s1_feats[:, fi].std() + 1e-8
            shift = abs(s2_mu - s1_mu) / s1_sd
            marker = " ***" if shift > 0.5 else ""
            print(f"  {ALL_NAMES[fi]:>30} {s1_mu:>10.4f} {s2_mu:>10.4f} {shift:>9.3f}{marker}")

    # ================================================================
    # 4. Binary S-vs-N classifier (RF) — accuracy ceiling
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  4. BINARY S vs N CLASSIFIER — ACCURACY CEILING")
    print(f"{'='*65}")

    # Inter-patient: train on DS1, test on DS2
    sn_mask_train = (ds1_labels == 0) | (ds1_labels == 1)
    sn_mask_test = (ds2_labels == 0) | (ds2_labels == 1)
    X_train = ds1_features[sn_mask_train]
    y_train = (ds1_labels[sn_mask_train] == 1).astype(int)
    X_test = ds2_features[sn_mask_test]
    y_test = (ds2_labels[sn_mask_test] == 1).astype(int)

    print(f"\n  Train: {sum(y_train==0)} N, {sum(y_train==1)} S")
    print(f"  Test:  {sum(y_test==0)} N, {sum(y_test==1)} S")

    # 4a. RF with standard 16 features
    rf_16 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_16.fit(X_train[:, :16], y_train)
    pred_16 = rf_16.predict(X_test[:, :16])
    print(f"\n  RF (16 standard features):")
    print(f"  {classification_report(y_test, pred_16, target_names=['N', 'S'])}")
    cm16 = confusion_matrix(y_test, pred_16)
    print(f"    N: {cm16[0,0]} correct, {cm16[0,1]} false S")
    print(f"    S: {cm16[1,1]} correct, {cm16[1,0]} missed S")

    # 4b. RF with all features (16 + extras)
    rf_all = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_all.fit(X_train, y_train)
    pred_all = rf_all.predict(X_test)
    print(f"\n  RF (16 + {len(EXTRA_NAMES)} extra features = {len(ALL_NAMES)} total):")
    print(f"  {classification_report(y_test, pred_all, target_names=['N', 'S'])}")
    cm_all = confusion_matrix(y_test, pred_all)
    print(f"    N: {cm_all[0,0]} correct, {cm_all[0,1]} false S")
    print(f"    S: {cm_all[1,1]} correct, {cm_all[1,0]} missed S")

    # Feature importance for the extended RF
    importances = rf_all.feature_importances_
    ranked = sorted(zip(ALL_NAMES, importances), key=lambda x: -x[1])
    print(f"\n  Feature importance (S vs N, all features):")
    for name, imp in ranked[:20]:
        bar = '#' * int(imp * 200)
        print(f"    {name:>30}: {imp:.4f} {bar}")

    # 4c. GradientBoosting (stronger learner)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    # Use SMOTE-like oversampling for S
    s_idx = np.where(y_train == 1)[0]
    n_idx = np.where(y_train == 0)[0]
    # Oversample S to 10% of N
    n_s_target = len(n_idx) // 10
    if len(s_idx) < n_s_target:
        extra_idx = np.random.choice(s_idx, n_s_target - len(s_idx), replace=True)
        balanced_idx = np.concatenate([n_idx, s_idx, extra_idx])
    else:
        balanced_idx = np.concatenate([n_idx, s_idx])
    np.random.shuffle(balanced_idx)
    gb.fit(X_train[balanced_idx], y_train[balanced_idx])
    pred_gb = gb.predict(X_test)
    print(f"\n  GradientBoosting (all features, 10% S oversample):")
    print(f"  {classification_report(y_test, pred_gb, target_names=['N', 'S'])}")
    cm_gb = confusion_matrix(y_test, pred_gb)
    print(f"    N: {cm_gb[0,0]} correct, {cm_gb[0,1]} false S")
    print(f"    S: {cm_gb[1,1]} correct, {cm_gb[1,0]} missed S")

    # ================================================================
    # 5. What makes S beats that ARE correctly classified different?
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  5. CORRECTLY vs INCORRECTLY CLASSIFIED S BEATS")
    print(f"{'='*65}")

    # Use RF all-feature predictions on DS2
    s_test_mask = y_test == 1
    s_test_feats = X_test[s_test_mask]
    s_test_preds = pred_all[s_test_mask]
    correct_s = s_test_feats[s_test_preds == 1]
    wrong_s = s_test_feats[s_test_preds == 0]

    if len(correct_s) > 0 and len(wrong_s) > 0:
        print(f"\n  Correctly detected S: {len(correct_s)} | Missed S: {len(wrong_s)}")
        print(f"\n  {'Feature':>30} {'Detected S':>12} {'Missed S':>12} {'Diff':>8}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
        for fi in range(min(len(ALL_NAMES), s_test_feats.shape[1])):
            c_mu = correct_s[:, fi].mean()
            w_mu = wrong_s[:, fi].mean()
            diff = c_mu - w_mu
            if abs(diff) > 0.01:
                print(f"  {ALL_NAMES[fi]:>30} {c_mu:>12.4f} {w_mu:>12.4f} {diff:>+8.4f}")

    # ================================================================
    # 6. 5-class RF with extra features (inter-patient)
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  6. 5-CLASS RF WITH EXTRA FEATURES (inter-patient)")
    print(f"{'='*65}")

    rf5_16 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf5_16.fit(ds1_features[:, :16], ds1_labels)
    pred5_16 = rf5_16.predict(ds2_features[:, :16])
    acc_16 = np.mean(pred5_16 == ds2_labels) * 100
    print(f"\n  RF 5-class (16 features): {acc_16:.2f}%")
    print(classification_report(ds2_labels, pred5_16,
          target_names=['N','S','V','F','Q'], zero_division=0))

    rf5_all = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf5_all.fit(ds1_features, ds1_labels)
    pred5_all = rf5_all.predict(ds2_features)
    acc_all = np.mean(pred5_all == ds2_labels) * 100
    print(f"\n  RF 5-class ({len(ALL_NAMES)} features): {acc_all:.2f}%")
    print(classification_report(ds2_labels, pred5_all,
          target_names=['N','S','V','F','Q'], zero_division=0))
    cm5 = confusion_matrix(ds2_labels, pred5_all)
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N','S','V','F','Q']):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm5[i])}")

    # Feature importance for 5-class extended
    imp5 = rf5_all.feature_importances_
    ranked5 = sorted(zip(ALL_NAMES, imp5), key=lambda x: -x[1])
    print(f"\n  Feature importance (5-class, all features):")
    for name, imp in ranked5[:20]:
        bar = '#' * int(imp * 200)
        print(f"    {name:>30}: {imp:.4f} {bar}")

    # Compare: which extra features actually improved 5-class accuracy?
    delta = acc_all - acc_16
    print(f"\n  Accuracy delta from extra features: {delta:+.2f}%")
    print(f"  16-feature RF: {acc_16:.2f}% | {len(ALL_NAMES)}-feature RF: {acc_all:.2f}%")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
