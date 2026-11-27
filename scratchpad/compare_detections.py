"""Compare SPICE detection times against Python reference."""
import numpy as np

# Python detection times (from generate_spice_tb.py output)
python_det_s = [
    0.0222, 1.0278, 1.8417, 2.6306, 3.4194,
    4.2083, 5.0250, 5.6806, 6.6750, 7.5167,
    8.3278, 9.1194, 9.8889
]

# SPICE detection times (from Phase 1A output)
spice_det_s = [
    0.139089, 0.948672, 1.76424, 2.55318, 3.34957,
    4.13553, 4.95090, 5.60636, 6.59714, 7.44267,
    8.25566, 9.04694, 9.81594
]

print("Detection Time Comparison (Phase 1A)")
print("=" * 60)
print(f"{'Beat':>4} {'Python (s)':>12} {'SPICE (s)':>12} {'Diff (ms)':>12}")
print("-" * 60)

diffs = []
for i, (p, s) in enumerate(zip(python_det_s, spice_det_s)):
    diff_ms = (s - p) * 1000
    diffs.append(diff_ms)
    print(f"{i+1:4d} {p:12.4f} {s:12.4f} {diff_ms:12.1f}")

print("-" * 60)
diffs = np.array(diffs)
print(f"Mean difference: {diffs.mean():.1f} ms")
print(f"Std deviation:   {diffs.std():.1f} ms")
print(f"Max |diff|:      {np.abs(diffs).max():.1f} ms")
print(f"\nTotal: {len(spice_det_s)} SPICE, {len(python_det_s)} Python detections")

# The rpeak_hold adds ~0-300ms hold time, and the noise gate
# comparator has some propagation delay. The systematic offset
# is expected and can be calibrated.
offset = diffs[1:].mean()  # skip first beat (startup transient)
print(f"\nSystematic offset (excluding beat 1): {offset:.1f} ms")
print(f"Jitter after offset correction: {diffs[1:].std():.1f} ms")
