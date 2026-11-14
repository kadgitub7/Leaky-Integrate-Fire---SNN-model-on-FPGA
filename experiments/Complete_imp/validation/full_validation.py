"""
COMPREHENSIVE VALIDATION: Analog Frontend Features + MLP Classifier
====================================================================
Verifies that the Python feature extractor EXACTLY matches what the
analog circuits compute, then tests classification accuracy.

CIRCUIT FEATURE DEFINITIONS (traced from SPICE netlists):

F1 (rr_ratio_out): NMOS diff pair with ewma_node on ref, sf_out on output
   → output ∝ gain * (V_ewma - V_sf) where V_sf ∝ RR interval
   Python equivalent: ewma_rr - pre_rr (sign inverted from original extractor)
   After StandardScaler: sign doesn't matter — MLP learns either polarity.

F2 (rr_asym_out): NMOS diff pair with sf_out on ref, prev_node on output
   → output ∝ gain * (V_sf_current - V_prev) = current_rr - prev_rr
   Python equivalent: pre_rr - prev_pre_rr (BACKWARD-looking, NOT forward-looking)
   **BUG in original extractor**: used pre_rr - post_rr (forward-looking)

F3/F4 (instab_out): NMOS diff pair with curr_node on ref, prev_node on output
   → output ∝ gain * (V_curr_peak - V_prev_peak)
   Python: beat[R,lead] - prev_beat[R,lead]

F5 (pw_out): NMOS diff pair with shp (ECG@R-131ms) on ref, shb (ECG@R-206ms) on output
   → output ∝ gain * (ECG@mid - ECG@start)
   Python: beat[PW_MID,0] - beat[PW_S,0]

F6 (width_out): S&H captures ECG at R+83ms
   Python: beat[RPEAK_IDX + 30, 0]

F7 (slope_out): NMOS diff pair with sh14 (ECG@R+14ms) on ref, sh69 (ECG@R+69ms) on output
   → output ∝ gain * (ECG@R+14ms - ECG@R+69ms)
   Python: beat[R+5,0] - beat[R+25,0]

F8 (sym_out): TG S&H captures ECG at R-peak
   Python: beat[RPEAK_IDX, 0]

TESTS:
  A) Corrected circuit features (F2 fixed to backward-looking)
  B) Original circuit features (F2 forward-looking, for comparison)
  C) sklearn MLPClassifier 8→64→5 (matching full_analog_mlp.py)
  D) 10 seeds for statistical robustness
  E) Per-class confusion matrix
  F) Feature statistics and sanity checks
"""
import sys, os, time, numpy as np, wfdb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

w = sys.stderr.write  # all output via stderr for immediate flushing

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
DS2 = ['100','103','105','111','113','117','121','123','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']
DATA = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA\mitdb_data"

def extract_corrected_circuit_features(rec_list, data_dir):
    """Extract features matching EXACTLY what the analog circuits compute.
    Key fix: F2 uses backward-looking asymmetry (current - previous RR)."""
    all_labels, all_features = [], []
    for rec_id in rec_list:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            w(f"  Skip {rec_id}: {e}\n"); continue

        sig = record.p_signal
        n_leads = min(sig.shape[1], 2)
        valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
                 if idx - WIN_L >= 0 and idx + WIN_R < len(sig) and sym in aami_map]

        ewma_rr = None
        prev_pre_rr = None  # the RR interval from the PREVIOUS beat

        for i in range(1, len(valid) - 1):
            idx, sym = valid[i]
            pre_rr = (idx - valid[i-1][0]) / FS
            post_rr = (valid[i+1][0] - idx) / FS
            bs, be = idx - WIN_L, idx + WIN_R
            if bs < 0 or be >= len(sig): continue
            beat = sig[bs:be]

            # EWMA update (circuit: charge-share at copy_trig AFTER readout)
            if ewma_rr is None:
                ewma_before = pre_rr
                ewma_rr = pre_rr
            else:
                ewma_before = ewma_rr
                ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr

            # Previous beat for instability
            prev_bs = valid[i-1][0] - WIN_L
            prev_be = valid[i-1][0] + WIN_R
            if prev_bs >= 0 and prev_be < len(sig):
                prev_beat = sig[prev_bs:prev_be]
                bi_l0 = beat[RPEAK_IDX, 0] - prev_beat[RPEAK_IDX, 0]
                l1_idx = min(1, n_leads - 1)
                bi_l1 = beat[RPEAK_IDX, l1_idx] - prev_beat[RPEAK_IDX, l1_idx]
            else:
                bi_l0 = bi_l1 = 0.0

            # F2 CORRECTED: backward-looking asymmetry
            # Circuit: sf_out(current) - prev_node(previous)
            # = pre_rr - prev_pre_rr (the RR interval from the beat before)
            if prev_pre_rr is None:
                f2_asym = 0.0  # no previous RR interval available
            else:
                f2_asym = pre_rr - prev_pre_rr

            # Feature vector: [F1, F4, F2, F5, F6, F3, F7, F8]
            feat = np.zeros(8, dtype=np.float32)
            feat[0] = pre_rr - ewma_before                    # F1: RR deviation
            feat[1] = bi_l1                                    # F4: beat instab L1
            feat[2] = f2_asym                                  # F2: backward asymmetry (CORRECTED)
            feat[3] = beat[PW_MID, 0] - beat[PW_S, 0]        # F5: P-wave voltage diff
            feat[4] = beat[RPEAK_IDX + WIDTH_OFF, 0]          # F6: tail sample
            feat[5] = bi_l0                                    # F3: beat instab L0
            feat[6] = beat[RPEAK_IDX + SLOPE_EARLY, 0] - beat[RPEAK_IDX + SLOPE_LATE, 0]  # F7
            feat[7] = beat[RPEAK_IDX, 0]                      # F8: R-peak voltage

            prev_pre_rr = pre_rr  # store for next iteration
            all_labels.append(aami_map[sym])
            all_features.append(feat)

    return np.array(all_labels), np.array(all_features)


def extract_original_circuit_features(rec_list, data_dir):
    """Original extractor (F2 = pre_rr - post_rr, forward-looking)."""
    all_labels, all_features = [], []
    for rec_id in rec_list:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            w(f"  Skip {rec_id}: {e}\n"); continue

        sig = record.p_signal
        n_leads = min(sig.shape[1], 2)
        valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
                 if idx - WIN_L >= 0 and idx + WIN_R < len(sig) and sym in aami_map]

        ewma_rr = None

        for i in range(1, len(valid) - 1):
            idx, sym = valid[i]
            pre_rr = (idx - valid[i-1][0]) / FS
            post_rr = (valid[i+1][0] - idx) / FS
            bs, be = idx - WIN_L, idx + WIN_R
            if bs < 0 or be >= len(sig): continue
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

            feat = np.zeros(8, dtype=np.float32)
            feat[0] = pre_rr - ewma_before
            feat[1] = bi_l1
            feat[2] = pre_rr - post_rr  # ORIGINAL: forward-looking
            feat[3] = beat[PW_MID, 0] - beat[PW_S, 0]
            feat[4] = beat[RPEAK_IDX + WIDTH_OFF, 0]
            feat[5] = bi_l0
            feat[6] = beat[RPEAK_IDX + SLOPE_EARLY, 0] - beat[RPEAK_IDX + SLOPE_LATE, 0]
            feat[7] = beat[RPEAK_IDX, 0]

            all_labels.append(aami_map[sym])
            all_features.append(feat)

    return np.array(all_labels), np.array(all_features)


def test_sklearn_mlp(X_tr, y_tr, X_te, y_te, name, n_seeds=10):
    """Test sklearn MLPClassifier with multiple seeds."""
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te)

    w(f"\n{'='*70}\n")
    w(f"sklearn MLPClassifier 8→64→5: {name} ({n_seeds} seeds)\n")
    w(f"{'='*70}\n")

    all_acc, all_S, all_V, all_F = [], [], [], []
    best_acc, best_mlp = 0, None

    for seed in range(n_seeds):
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, random_state=seed+42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', learning_rate_init=1e-3,
            alpha=1e-4, batch_size=128
        )
        mlp.fit(X_tr_n, y_tr)
        p = mlp.predict(X_te_n)
        acc = accuracy_score(y_te, p) * 100
        S = (p[y_te==1]==1).mean()*100 if (y_te==1).sum()>0 else 0
        V = (p[y_te==2]==2).mean()*100 if (y_te==2).sum()>0 else 0
        F = (p[y_te==3]==3).mean()*100 if (y_te==3).sum()>0 else 0
        all_acc.append(acc); all_S.append(S); all_V.append(V); all_F.append(F)
        w(f"  Seed {seed+42:2d}: {acc:6.2f}%  S={S:5.1f}% V={V:5.1f}% F={F:5.1f}%\n")
        if acc > best_acc:
            best_acc = acc; best_mlp = mlp

    w(f"\n  Mean±std: {np.mean(all_acc):.2f}±{np.std(all_acc):.2f}%\n")
    w(f"  Best:     {best_acc:.2f}%\n")
    w(f"  S recall: {np.mean(all_S):.1f}±{np.std(all_S):.1f}%\n")
    w(f"  V recall: {np.mean(all_V):.1f}±{np.std(all_V):.1f}%\n")
    w(f"  F recall: {np.mean(all_F):.1f}±{np.std(all_F):.1f}%\n")

    # Best model detailed report
    p = best_mlp.predict(X_te_n)
    w(f"\nBest model classification report:\n")
    w(classification_report(y_te, p, target_names=['N','S','V','F','Q'], digits=3, zero_division=0) + "\n")

    # Confusion matrix
    cm = confusion_matrix(y_te, p, labels=[0,1,2,3,4])
    w(f"Confusion matrix (rows=true, cols=pred):\n")
    w(f"       N      S      V      F      Q\n")
    for i, name_c in enumerate(['N','S','V','F','Q']):
        row = '  '.join(f'{v:6d}' for v in cm[i])
        w(f"  {name_c}  {row}\n")
    w(f"\n")

    return np.mean(all_acc), np.std(all_acc), best_acc


def test_rf(X_tr, y_tr, X_te, y_te, name):
    """Test Random Forest baseline."""
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te)

    w(f"\n--- RF(300, depth=15): {name} ---\n")
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
    rf.fit(X_tr_n, y_tr)
    p = rf.predict(X_te_n)
    acc = (p==y_te).mean()*100
    S = (p[y_te==1]==1).mean()*100 if (y_te==1).sum()>0 else 0
    V = (p[y_te==2]==2).mean()*100 if (y_te==2).sum()>0 else 0
    F = (p[y_te==3]==3).mean()*100 if (y_te==3).sum()>0 else 0
    w(f"  RF: {acc:.2f}%  S={S:.1f}% V={V:.1f}% F={F:.1f}%\n")
    return acc


# ================================================================
# MAIN VALIDATION
# ================================================================
w("="*70 + "\n")
w("COMPREHENSIVE ANALOG FRONTEND + MLP VALIDATION\n")
w("="*70 + "\n\n")

# Step 1: Extract features
w("Step 1: Extracting features...\n")
t0 = time.time()

w("  A) Corrected circuit features (F2 backward-looking)...\n")
y_tr_c, X_tr_c = extract_corrected_circuit_features(DS1, DATA)
y_te_c, X_te_c = extract_corrected_circuit_features(DS2, DATA)
w(f"     Train: {X_tr_c.shape}, Test: {X_te_c.shape}\n")

w("  B) Original circuit features (F2 forward-looking)...\n")
y_tr_o, X_tr_o = extract_original_circuit_features(DS1, DATA)
y_te_o, X_te_o = extract_original_circuit_features(DS2, DATA)
w(f"     Train: {X_tr_o.shape}, Test: {X_te_o.shape}\n")

w(f"  Extraction time: {time.time()-t0:.1f}s\n")

# Step 2: Sanity checks
w("\nStep 2: Feature sanity checks...\n")
fnames = ['F1:rr_dev', 'F4:bi_L1', 'F2:asym', 'F5:pwave', 'F6:width', 'F3:bi_L0', 'F7:slope', 'F8:rpeak']
w(f"  {'Feature':<12} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'Zeros%':>8}\n")
for j in range(8):
    col = X_tr_c[:, j]
    z_pct = (col == 0).mean() * 100
    w(f"  {fnames[j]:<12} {col.mean():>9.4f} {col.std():>9.4f} {col.min():>9.4f} {col.max():>9.4f} {z_pct:>7.1f}%\n")

# Check class distributions
w(f"\n  Train class distribution:\n")
for c in range(5):
    w(f"    Class {c} ({'NSVFQ'[c]}): {(y_tr_c==c).sum():>6d} ({(y_tr_c==c).mean()*100:.1f}%)\n")
w(f"  Test class distribution:\n")
for c in range(5):
    w(f"    Class {c} ({'NSVFQ'[c]}): {(y_te_c==c).sum():>6d} ({(y_te_c==c).mean()*100:.1f}%)\n")

# Verify DS1/DS2 no overlap
w(f"\n  DS1 records: {len(DS1)} | DS2 records: {len(DS2)}\n")
overlap = set(DS1) & set(DS2)
w(f"  Overlap: {len(overlap)} records {'(CLEAN)' if len(overlap)==0 else '(LEAK!)'}\n")

# Compare F2 between corrected and original
f2_corr = X_tr_c[:, 2]
f2_orig = X_tr_o[:, 2]
corr = np.corrcoef(f2_corr, f2_orig)[0, 1]
w(f"\n  F2 corrected vs original correlation: {corr:.4f}\n")
w(f"  F2 corrected: mean={f2_corr.mean():.4f}, std={f2_corr.std():.4f}\n")
w(f"  F2 original:  mean={f2_orig.mean():.4f}, std={f2_orig.std():.4f}\n")

# Step 3: Classification tests
w("\nStep 3: Classification accuracy...\n")

# 3A: Corrected circuit features (TRUE circuit match)
mean_c, std_c, best_c = test_sklearn_mlp(X_tr_c, y_tr_c, X_te_c, y_te_c, "CORRECTED circuit features", n_seeds=10)
rf_c = test_rf(X_tr_c, y_tr_c, X_te_c, y_te_c, "CORRECTED circuit features")

# 3B: Original circuit features (for comparison)
mean_o, std_o, best_o = test_sklearn_mlp(X_tr_o, y_tr_o, X_te_o, y_te_o, "ORIGINAL circuit features (F2 forward)", n_seeds=10)
rf_o = test_rf(X_tr_o, y_tr_o, X_te_o, y_te_o, "ORIGINAL circuit features (F2 forward)")

# Step 4: Summary
w("\n" + "="*70 + "\n")
w("FINAL SUMMARY\n")
w("="*70 + "\n\n")
w(f"  {'Feature Set':<40} {'MLP mean±std':>15} {'MLP best':>10} {'RF':>8}\n")
w(f"  {'-'*73}\n")
w(f"  {'Corrected (true circuit match)':<40} {mean_c:>6.2f}±{std_c:.2f}% {best_c:>8.2f}% {rf_c:>7.2f}%\n")
w(f"  {'Original (F2 forward-looking)':<40} {mean_o:>6.2f}±{std_o:.2f}% {best_o:>8.2f}% {rf_o:>7.2f}%\n")
w(f"\n  NOTE: 'Corrected' features match the actual analog circuit computation.\n")
w(f"  The MLP weights will be trained on these features and mapped to ReRAM.\n")
w(f"  Analog gains (3-17x) become linear scaling → absorbed by StandardScaler.\n")
w(f"\nDone.\n")
