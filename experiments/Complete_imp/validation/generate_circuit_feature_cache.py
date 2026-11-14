"""
Generate circuit_features_cache.npz — the pre-extracted circuit-matching features
used to validate the 94.44% accuracy claim.

This uses the CORRECTED F2 (backward-looking asymmetry) that matches the actual
rr_timing.spice circuit computation.

Run: python -I generate_circuit_feature_cache.py
Output: circuit_features_cache.npz
"""
import sys, os, time
import numpy as np
import wfdb
from sklearn.preprocessing import StandardScaler

FS = 360
WIN_L, WIN_R = 90, 108
RPEAK_IDX = WIN_L
EWMA_ALPHA = 0.015
PW_S, PW_MID = 16, 43
SLOPE_EARLY, SLOPE_LATE = 5, 25
WIDTH_OFF = 30

aami_map = {}
for sym in ['N','L','R','e','j']: aami_map[sym] = 0
for sym in ['A','a','J','S']: aami_map[sym] = 1
for sym in ['V','E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/','f','Q']: aami_map[sym] = 4

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', '..', 'mitdb_data')

FEATURE_NAMES = [
    'F1_rr_deviation',    # pre_rr - ewma_rr
    'F4_bi_L1',           # peak_L1 - prev_peak_L1
    'F2_rr_asymmetry',    # pre_rr - prev_pre_rr (BACKWARD-looking, corrected)
    'F5_pwave_diff',      # ECG@R-131ms - ECG@R-206ms
    'F6_width_proxy',     # ECG@R+83ms (raw tail voltage)
    'F3_bi_L0',           # peak_L0 - prev_peak_L0
    'F7_slope',           # ECG@R+14ms - ECG@R+69ms
    'F8_rpeak_voltage',   # ECG@R-peak
]


def extract_corrected_circuit_features(rec_list, data_dir):
    """Extract features matching EXACTLY what the analog circuits compute.
    F2 uses backward-looking asymmetry (current_rr - previous_rr)."""
    all_labels, all_features = [], []
    for rec_id in rec_list:
        path = os.path.join(data_dir, rec_id)
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')
        sig = record.p_signal
        n_leads = min(sig.shape[1], 2)
        valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
                 if idx - WIN_L >= 0 and idx + WIN_R < len(sig) and sym in aami_map]

        ewma_rr = None
        prev_pre_rr = None

        for i in range(1, len(valid) - 1):
            idx, sym = valid[i]
            pre_rr = (idx - valid[i-1][0]) / FS
            bs, be = idx - WIN_L, idx + WIN_R
            if bs < 0 or be >= len(sig):
                continue
            beat = sig[bs:be]

            if ewma_rr is None:
                ewma_before = pre_rr
                ewma_rr = pre_rr
            else:
                ewma_before = ewma_rr
                ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr

            prev_bs = valid[i-1][0] - WIN_L
            prev_be = valid[i-1][0] + WIN_R
            if prev_bs >= 0 and prev_be < len(sig):
                prev_beat = sig[prev_bs:prev_be]
                bi_l0 = beat[RPEAK_IDX, 0] - prev_beat[RPEAK_IDX, 0]
                l1_idx = min(1, n_leads - 1)
                bi_l1 = beat[RPEAK_IDX, l1_idx] - prev_beat[RPEAK_IDX, l1_idx]
            else:
                bi_l0 = bi_l1 = 0.0

            f2_asym = (pre_rr - prev_pre_rr) if prev_pre_rr is not None else 0.0

            feat = np.zeros(8, dtype=np.float32)
            feat[0] = pre_rr - ewma_before
            feat[1] = bi_l1
            feat[2] = f2_asym
            feat[3] = beat[PW_MID, 0] - beat[PW_S, 0]
            feat[4] = beat[RPEAK_IDX + WIDTH_OFF, 0]
            feat[5] = bi_l0
            feat[6] = beat[RPEAK_IDX + SLOPE_EARLY, 0] - beat[RPEAK_IDX + SLOPE_LATE, 0]
            feat[7] = beat[RPEAK_IDX, 0]

            prev_pre_rr = pre_rr
            all_labels.append(aami_map[sym])
            all_features.append(feat)

    return np.array(all_labels), np.array(all_features)


if __name__ == '__main__':
    print(f"Data dir: {os.path.abspath(DATA)}")
    print(f"DS1: {len(DS1)} records, DS2: {len(DS2)} records")

    print("Extracting DS1 (train)...")
    t0 = time.time()
    y_tr, X_tr = extract_corrected_circuit_features(DS1, DATA)
    print(f"  Train: {X_tr.shape}, {time.time()-t0:.1f}s")

    print("Extracting DS2 (test)...")
    t0 = time.time()
    y_te, X_te = extract_corrected_circuit_features(DS2, DATA)
    print(f"  Test: {X_te.shape}, {time.time()-t0:.1f}s")

    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'circuit_features_cache.npz')
    np.savez(out,
             X_tr=X_tr, y_tr=y_tr,
             X_te=X_te, y_te=y_te,
             X_tr_scaled=X_tr_n, X_te_scaled=X_te_n,
             scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_,
             feature_names=FEATURE_NAMES,
             ds1_records=DS1, ds2_records=DS2,
             ewma_alpha=np.float64(EWMA_ALPHA))

    print(f"\nSaved: {out}")
    print(f"  Train: {X_tr.shape[0]} beats, Test: {X_te.shape[0]} beats")
    print(f"  Scaler mean: {scaler.mean_}")
    print(f"  Scaler scale: {scaler.scale_}")
    print(f"  Feature names: {FEATURE_NAMES}")

    # Quick sanity: train one MLP to confirm accuracy
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=500, random_state=44,
        early_stopping=True, validation_fraction=0.1,
        learning_rate='adaptive', learning_rate_init=1e-3,
        alpha=1e-4, batch_size=128
    )
    mlp.fit(X_tr_n, y_tr)
    acc = accuracy_score(y_te, mlp.predict(X_te_n)) * 100
    print(f"\n  Sanity check — seed 44 accuracy: {acc:.2f}%")
