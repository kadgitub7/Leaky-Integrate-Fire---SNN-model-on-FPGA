"""Run analog frontend ngspice simulations for missing DS2 records."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'frontend_optimization', 'complete_frontend'))
from full_accuracy_test import process_record, CACHE_DIR

MISSING = ['214', '219', '221', '222', '228', '231', '232', '233', '234']

os.makedirs(CACHE_DIR, exist_ok=True)
for i, rec in enumerate(MISSING):
    cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        print(f"[{i+1}/{len(MISSING)}] Record {rec}: already cached, skipping", flush=True)
        continue
    t0 = time.time()
    print(f"[{i+1}/{len(MISSING)}] Record {rec}: simulating...", flush=True)
    feats, labels = process_record(rec)
    elapsed = time.time() - t0
    print(f"  -> {len(labels)} beats in {elapsed:.0f}s", flush=True)

print("All missing records complete.", flush=True)
