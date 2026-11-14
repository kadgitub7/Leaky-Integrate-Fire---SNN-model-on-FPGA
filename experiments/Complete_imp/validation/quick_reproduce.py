"""Quick reproduction: sklearn MLPClassifier on features_8_cache.npz"""
import sys, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

CACHE = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA\experiments\Complete_imp\classifier\features_8_cache.npz"
d = np.load(CACHE)
X_tr, y_tr, X_te, y_te = d['X_tr'], d['y_tr'], d['X_te'], d['y_te']
sys.stderr.write(f"Loaded: Train={X_tr.shape}, Test={X_te.shape}\n")
sys.stderr.write(f"Train mean range: [{X_tr.mean(0).min():.3f}, {X_tr.mean(0).max():.3f}]\n")
sys.stderr.write(f"Train std range:  [{X_tr.std(0).min():.3f}, {X_tr.std(0).max():.3f}]\n")
sys.stderr.write(f"Class dist test: {dict(zip(*np.unique(y_te, return_counts=True)))}\n")

scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_te_n = scaler.transform(X_te)

sys.stderr.write("\n=== sklearn MLPClassifier (exact full_analog_mlp.py config) ===\n")
best_acc, best_mlp = 0, None
for seed in range(5):
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
    sys.stderr.write(f"  Seed {seed+42}: {acc:.2f}%  S={S:.1f}% V={V:.1f}%\n")
    if acc > best_acc:
        best_acc = acc; best_mlp = mlp

p = best_mlp.predict(X_te_n)
sys.stderr.write(f"\nBest: {best_acc:.2f}%\n")
sys.stderr.write(classification_report(y_te, p, target_names=['N','S','V','F','Q'], digits=3, zero_division=0) + "\n")

sys.stderr.write("\n=== RF(300, depth=15) baseline ===\n")
rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
rf.fit(X_tr_n, y_tr)
p = rf.predict(X_te_n)
rf_acc = (p==y_te).mean()*100
S = (p[y_te==1]==1).mean()*100 if (y_te==1).sum()>0 else 0
V = (p[y_te==2]==2).mean()*100 if (y_te==2).sum()>0 else 0
sys.stderr.write(f"RF: {rf_acc:.2f}%  S={S:.1f}% V={V:.1f}%\n")
sys.stderr.write(classification_report(y_te, p, target_names=['N','S','V','F','Q'], digits=3, zero_division=0) + "\n")
