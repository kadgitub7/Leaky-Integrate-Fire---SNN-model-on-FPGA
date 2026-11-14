"""Test: sklearn MLPClassifier on CIRCUIT-MATCHING features (differences, raw voltages).
This answers: can the analog circuits' actual outputs achieve ~96% with the same 8->64->5?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_circuit_features import extract_circuit_features
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']
DATA = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA\mitdb_data"

sys.stderr.write("Extracting circuit-matching features...\n")
t0 = time.time()
y_tr, X_tr = extract_circuit_features(DS1, DATA)
y_te, X_te = extract_circuit_features(DS2, DATA)
sys.stderr.write(f"  Done in {time.time()-t0:.1f}s. Train={X_tr.shape}, Test={X_te.shape}\n")

scaler = StandardScaler()
X_tr_n = scaler.fit_transform(X_tr)
X_te_n = scaler.transform(X_te)

sys.stderr.write("\n=== sklearn MLPClassifier 8->64->5 on CIRCUIT features ===\n")
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
sys.stderr.write(f"\nBest sklearn MLP on circuit features: {best_acc:.2f}%\n")
sys.stderr.write(classification_report(y_te, p, target_names=['N','S','V','F','Q'], digits=3, zero_division=0) + "\n")

sys.stderr.write("\n=== RF(300, depth=15) on CIRCUIT features ===\n")
rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
rf.fit(X_tr_n, y_tr)
p = rf.predict(X_te_n)
rf_acc = (p==y_te).mean()*100
S = (p[y_te==1]==1).mean()*100 if (y_te==1).sum()>0 else 0
V = (p[y_te==2]==2).mean()*100 if (y_te==2).sum()>0 else 0
sys.stderr.write(f"RF on circuit features: {rf_acc:.2f}%  S={S:.1f}% V={V:.1f}%\n")
sys.stderr.write(classification_report(y_te, p, target_names=['N','S','V','F','Q'], digits=3, zero_division=0) + "\n")
