"""
Extract the 8 features that MATCH what our analog circuits actually compute.
These are DIFFERENCES and RAW SAMPLES, not ratios/energy/normalized values.

Circuit-matching feature definitions:
  F1: pre_rr - ewma_rr        (RR deviation from EWMA baseline)
  F2: pre_rr - post_rr        (RR asymmetry as difference)
  F3: peak_L0 - prev_peak_L0  (beat instability L0, signed difference)
  F4: peak_L1 - prev_peak_L1  (beat instability L1, signed difference)
  F5: ECG@pw_mid - ECG@pw_start (P-wave voltage difference)
  F6: ECG@R+83ms              (raw tail sample — QRS width proxy)
  F7: ECG@R+14ms - ECG@R+69ms (slope voltage difference)
  F8: ECG@R-peak              (raw R-peak voltage — symmetry proxy)

Classifier order: [F1, F4, F2, F5, F6, F3, F7, F8]
"""
import numpy as np
import wfdb, os

FS = 360
WIN_L, WIN_R = 90, 108
RPEAK_IDX = WIN_L  # R-peak is at index 90 in the beat window
EWMA_ALPHA = 0.015

# QRS region offsets (relative to R-peak index in beat)
QRS_OFF = 0  # QRS center offset from R-peak

# Timing offsets for slope samples (relative to R-peak)
SLOPE_EARLY = 5   # R+14ms ≈ R+5 samples at 360Hz
SLOPE_LATE = 25   # R+69ms ≈ R+25 samples at 360Hz

# Width tail sample offset
WIDTH_OFF = 30    # R+83ms ≈ R+30 samples at 360Hz

# P-wave window (relative to beat start)
PW_S = 16
PW_MID = 43  # midpoint of P-wave window

aami_map = {}
for sym in ['N','L','R','e','j']: aami_map[sym] = 0
for sym in ['A','a','J','S']: aami_map[sym] = 1
for sym in ['V','E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/','f','Q']: aami_map[sym] = 4

CIRCUIT_FEATURE_NAMES = [
    'rr_deviation',      # F1: pre_rr - ewma_rr
    'bi_L1_diff',        # F4→classifier[1]: peak_L1 - prev_peak_L1
    'rr_asym_diff',      # F2: pre_rr - post_rr
    'pwave_vdiff',       # F5: ECG@pw_mid - ECG@pw_start
    'tail_sample_L0',    # F6: ECG@R+83ms
    'bi_L0_diff',        # F3→classifier[5]: peak_L0 - prev_peak_L0
    'slope_vdiff',       # F7: ECG@early - ECG@late
    'rpeak_voltage_L0',  # F8: ECG@R-peak
]

def extract_circuit_features(rec_list, data_dir='./mitdb_data'):
    """Extract features matching what the analog circuits compute."""
    all_labels, all_features = [], []
    for rec_id in rec_list:
        path = os.path.join(data_dir, rec_id)
        try:
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
        except Exception as e:
            print(f"  Skip {rec_id}: {e}")
            continue

        sig = record.p_signal
        n_leads = min(sig.shape[1], 2)

        valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
                 if idx - WIN_L >= 0 and idx + WIN_R < len(sig) and sym in aami_map]

        ewma_rr = None
        prev_peak_l0 = None
        prev_peak_l1 = None

        for i in range(1, len(valid) - 1):
            idx, sym = valid[i]
            pre_rr = (idx - valid[i-1][0]) / FS
            post_rr = (valid[i+1][0] - idx) / FS
            bs = idx - WIN_L
            be = idx + WIN_R
            if bs < 0 or be >= len(sig):
                continue
            beat = sig[bs:be]

            # EWMA update
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

            # Feature vector in CLASSIFIER ORDER: [F1, F4, F2, F5, F6, F3, F7, F8]
            feat = np.zeros(8, dtype=np.float32)
            feat[0] = pre_rr - ewma_before                           # F1: RR deviation
            feat[1] = bi_l1                                           # F4→cls[1]: beat instab L1
            feat[2] = pre_rr - post_rr                                # F2: RR asymmetry diff
            feat[3] = beat[PW_MID, 0] - beat[PW_S, 0]               # F5: P-wave voltage diff
            feat[4] = beat[RPEAK_IDX + WIDTH_OFF, 0]                 # F6: tail sample
            feat[5] = bi_l0                                           # F3→cls[5]: beat instab L0
            feat[6] = beat[RPEAK_IDX + SLOPE_EARLY, 0] - beat[RPEAK_IDX + SLOPE_LATE, 0]  # F7: slope diff
            feat[7] = beat[RPEAK_IDX, 0]                             # F8: R-peak voltage

            prev_peak_l0 = beat[RPEAK_IDX, 0]
            prev_peak_l1 = beat[RPEAK_IDX, min(1, n_leads-1)]

            all_labels.append(aami_map[sym])
            all_features.append(feat)

    return np.array(all_labels), np.array(all_features)
