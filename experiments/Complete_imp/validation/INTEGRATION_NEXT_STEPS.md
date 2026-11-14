# Integration Next Steps

**Status**: Validation complete. Document is conditionally valid.

---

## Phase 1: Build Full ngspice Feature Cache (BLOCKER)

**What**: Run `full_accuracy_test.py` on all 44 MIT-BIH records with the fixed v12 rr_timing circuit.
**Why**: The 94.44% accuracy is from Python simulation only. We need actual circuit outputs to confirm.
**Current state**: 6/44 records cached (112, 113, 118, 122, 200, 231)
**Estimated time**: 10-15 hours total (~15-20 min per record)

Steps:
1. Fix DS2 list — add record `123` to both `full_accuracy_test.py` and `full_validation.py`
2. Fix stale comment in `full_accuracy_test.py` line 6 (says alpha=0.10, code uses 0.015)
3. Run `full_accuracy_test.py` for remaining 38 records (can run in batches)
4. Verify per-record rho values stay >= 0.90 as new records complete

**Success criteria**: All 44 records cached, rho >= 0.90 for all features on all records.

---

## Phase 2: Validate Accuracy on Full Circuit Features

**What**: Once cache is complete, train MLP on circuit features from all DS1 records and test on all DS2.
**Why**: Confirms that actual circuit outputs (not Python approximation) achieve ~94% accuracy.

Steps:
1. Load all DS1 circuit_features from cache, train MLPClassifier 8->64->5
2. Test on all DS2 circuit_features
3. Run 10 seeds, report mean +/- std
4. Compare per-class recall (S, V, F) to the Python projection (69.2%, 86.3%, 57.2%)

**Success criteria**: Overall accuracy >= 93%, S recall >= 50%, V recall >= 80%

---

## Phase 3: Retrain Weights and Map to ReRAM

**What**: Take the MLP trained on circuit features and map its weights to ReRAM conductances.
**Why**: The current `full_analog_weights.npz` was trained on digital features. Must retrain on circuit features.

Steps:
1. Train final MLP on all DS1 circuit features (best seed from Phase 2)
2. Extract W1 (8x64), b1 (64), W2 (64x5), b2 (5)
3. Map to differential G+/G- conductances using `map_weights_to_conductance()`
4. Generate L1 and L2 crossbar netlists
5. Run 72 unit-input ngspice characterizations to extract effective weights
6. Verify SW-vs-analog weight agreement >= 99%

**Success criteria**: Weight fidelity >= 99%, crossbar netlists generated.

---

## Phase 4: Full Analog Chain Test

**What**: End-to-end test — frontend circuits produce features, classifier crossbar produces outputs.
**Why**: Proves the complete analog system works without any digital intermediary.

Steps:
1. Generate combined testbench: frontend_top.spice -> input conditioning -> classifier crossbar -> WTA
2. Run on a representative subset of beats (e.g., 100 beats from each class)
3. Verify WTA selects correct class
4. Measure classification accuracy, per-class recall

**Success criteria**: Accuracy consistent with Phase 2 (within 1-2% of MLP result).

---

## Phase 5: Characterize and Document

Steps:
1. Energy estimation (frontend + classifier + WTA)
2. Area estimation from device counts
3. Timing analysis (latency per beat)
4. Document final performance metrics

---

## Files to Fix Now (non-blocking)

| File | Issue | Fix |
|------|-------|-----|
| `full_accuracy_test.py` line 6 | Comment says "alpha=0.10" | Change to "alpha=0.015" |
| `full_accuracy_test.py` DS2 list | Missing record 123 | Add '123' after '121' |
| `full_validation.py` DS2 list | Missing record 123 | Add '123' after '121' |
| `final_e2e_test.py` | EWMA_ALPHA=0.10, uses RF not MLP, 5 records only | Deprecate or rewrite to match current design |
