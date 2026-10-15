"""
MLP SIZE SWEEP — Find smallest MLP matching RF 95.70% on the fixed 8 features.

Step 1: extract + cache features (run once, ~5 min)
Step 2: sweep MLPs on cached data (fast, <1 min)

python experiments/Complete_imp/classifier/mlp_size_sweep.py
"""
import os, sys, time, json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, 'features_8_cache.npz')

ALL_FEATURE_NAMES = [
    'pre_rr','post_rr','rr_ratio','rr_asymmetry','compensatory_ratio','rr_std_10','rr_std_20',
    'qrs_width_L0','qrs_width_L1','qrs_area_L0','qrs_area_L1','max_slope_L0','max_slope_L1',
    'peak_amp_L0','peak_amp_L1','qrs_energy_L0','qrs_energy_L1','slope_ratio_L0','slope_ratio_L1',
    'rel_area_L0','rel_area_L1','rel_width_L0','rel_width_L1','rel_peak_L0','rel_peak_L1',
    'templ_corr_L0','templ_corr_L1','area_ratio_L0L1','width_diff_L0L1','corr_L0L1',
    'pwave_energy_L0','pwave_energy_L1','st_discordance_L0','st_discordance_L1',
    'qrs_symmetry_L0','qrs_symmetry_L1','interlead_divergence',
    'beat_instability_L0','beat_instability_L1',
]
FEATURE_NAMES_8 = [
    'rr_ratio','beat_instability_L1','rr_asymmetry','pwave_energy_L1',
    'qrs_width_L0','beat_instability_L0','slope_ratio_L0','qrs_symmetry_L0',
]
FEAT_IDX = [ALL_FEATURE_NAMES.index(f) for f in FEATURE_NAMES_8]
N_CLASSES = 5; CLASS_NAMES = ['N','S','V','F','Q']
E_PER_MAC_PJ_NOM = 1.2679

def build_cache():
    """Extract features from MIT-BIH and cache to disk."""
    import wfdb
    from scipy.signal import butter, filtfilt

    aami_map = {}
    for s in ['N','L','R','e','j']: aami_map[s]=0
    for s in ['A','a','J','S']:     aami_map[s]=1
    for s in ['V','E']:             aami_map[s]=2
    for s in ['F']:                 aami_map[s]=3
    for s in ['/','f','Q']:         aami_map[s]=4

    DS1 = ['101','106','108','109','112','114','115','116','118','119',
           '122','124','201','203','205','207','208','209','215','220','223','230']
    DS2 = ['100','103','105','111','113','117','121','200','202','210',
           '212','213','214','219','221','222','228','231','232','233','234']

    FS=360; WIN_L=90; WIN_R=108; QRS_S=70; QRS_E=130; PW_S=16; PW_E=QRS_S
    ST_S=144; ST_E=min(173,WIN_L+WIN_R); N_TEMPLATE=6
    EWMA_ALPHA=0.015; EWMA_GATE=0.85; EWMA_INIT=10; N_FEAT=39
    b,a = butter(2,[3.0/(FS/2),8.0/(FS/2)],btype='band')

    def extract(rec_list):
        labels, feats = [], []
        for rec_id in rec_list:
            path = os.path.join('./mitdb_data', rec_id)
            try:
                rec=wfdb.rdrecord(path); ann=wfdb.rdann(path,'atr')
                sig=rec.p_signal; fs=rec.fs; nl=min(sig.shape[1],2)
                pwf=np.zeros_like(sig)
                for ld in range(nl):
                    try: pwf[:,ld]=filtfilt(b,a,sig[:,ld])
                    except: pass
                valid=[(i,s) for i,s in zip(ann.sample,ann.symbol) if i-WIN_L>=0 and i+WIN_R<len(sig) and s in aami_map]
                ewa=[None]*nl;eww=[None]*nl;ewp=[None]*nl;ewt=[None]*nl
                ia=[[] for _ in range(nl)];iw=[[] for _ in range(nl)];ip_=[[] for _ in range(nl)];it_=[[] for _ in range(nl)]
                rrh=[];pp=[None]*nl
                for vi,(idx,sym) in enumerate(valid):
                    bt=sig[idx-WIN_L:idx+WIN_R,:];bpw=pwf[idx-WIN_L:idx+WIN_R,:]
                    prr=(idx-valid[vi-1][0])/fs if vi>0 else 0.833
                    porr=(valid[vi+1][0]-idx)/fs if vi<len(valid)-1 else 0.833
                    lrrs=[(valid[j][0]-valid[j-1][0])/fs for j in range(max(1,vi-20),vi+1)]
                    lrr=np.mean(lrrs) if lrrs else 0.833
                    rrh.append(prr)
                    if len(rrh)>20: rrh.pop(0)
                    f=np.zeros(N_FEAT,dtype=np.float32)
                    f[0]=prr;f[1]=porr;f[2]=prr/(lrr+1e-8);f[3]=prr/(porr+1e-8)
                    f[4]=(prr+porr)/(2*lrr+1e-8)
                    f[5]=np.std(rrh[-10:]) if len(rrh)>=2 else 0.0
                    f[6]=np.std(rrh) if len(rrh)>=2 else 0.0
                    qs=[];ws=[];ars=[];pas_=[]
                    for ld in range(nl):
                        q=bt[QRS_S:QRS_E,ld];aq=np.abs(q)
                        pk=np.max(aq) if len(aq)>0 else 0.0;ar=np.sum(aq)/fs
                        w=0.0
                        if pk>1e-6:
                            th=0.3*pk;ab=aq>th
                            if np.any(ab): w=(len(ab)-1-np.argmax(ab[::-1])-np.argmax(ab))*1000.0/fs
                        dq=np.diff(q)*fs;ms=np.max(np.abs(dq)) if len(dq)>0 else 0.0
                        en=np.sum(q**2)/fs
                        if len(dq)>0: sr=max(np.max(dq),1e-8)/(max(abs(np.min(dq)),1e-8)+1e-8)
                        else: sr=1.0
                        ti=np.linspace(0,len(q)-1,N_TEMPLATE).astype(int);qd=q[ti]
                        if vi<EWMA_INIT:
                            ia[ld].append(max(ar,1e-6));iw[ld].append(max(w,1e-6))
                            ip_[ld].append(max(pk,1e-6));it_[ld].append(qd.copy())
                        if ewa[ld] is None:
                            if vi>=EWMA_INIT-1 and len(ia[ld])>=EWMA_INIT:
                                ewa[ld]=np.median(ia[ld]);eww[ld]=np.median(iw[ld])
                                ewp[ld]=np.median(ip_[ld]);ewt[ld]=np.median(it_[ld],axis=0)
                            else:
                                ewa[ld]=max(ar,1e-6);eww[ld]=max(w,1e-6)
                                ewp[ld]=max(pk,1e-6);ewt[ld]=qd.copy()
                        nc=np.linalg.norm(qd);nt=np.linalg.norm(ewt[ld])
                        tc=np.dot(qd,ewt[ld])/(nc*nt+1e-8) if nc>1e-8 and nt>1e-8 else 1.0
                        if tc>EWMA_GATE:
                            al=EWMA_ALPHA
                            ewa[ld]=al*max(ar,1e-6)+(1-al)*ewa[ld];eww[ld]=al*max(w,1e-6)+(1-al)*eww[ld]
                            ewp[ld]=al*max(pk,1e-6)+(1-al)*ewp[ld];ewt[ld]=al*qd+(1-al)*ewt[ld]
                        f[7+ld]=w;f[9+ld]=ar;f[11+ld]=ms;f[13+ld]=pk;f[15+ld]=en;f[17+ld]=sr
                        f[19+ld]=ar/(ewa[ld]+1e-8);f[21+ld]=w/(eww[ld]+1e-8)
                        f[23+ld]=pk/(ewp[ld]+1e-8);f[25+ld]=tc
                        qs.append(q);ws.append(w);ars.append(ar);pas_.append(pk)
                        pw=bpw[PW_S:PW_E,ld];f[30+ld]=np.sum(pw**2)/fs
                        qi=np.sum(q)/fs;se=min(ST_E,bt.shape[0]);ss_=bt[ST_S:se,ld]
                        si=np.sum(ss_)/fs if len(ss_)>0 else 0.0
                        f[32+ld]=(1.0 if qi>=0 else -1.0)*si
                        if pk>1e-6:
                            pi=np.argmax(aq);th=0.3*pk;ab=aq>th
                            if np.any(ab):
                                on=np.argmax(ab);qw=max(w,1e-6)*fs/1000.0
                                sy=np.clip((pi-on)/(qw+1e-8),0.0,1.0)
                            else: sy=0.5
                        else: sy=0.5
                        f[34+ld]=sy
                        if pp[ld] is not None and pp[ld]>1e-6: ins=abs(pk-pp[ld])/(pp[ld]+1e-8)
                        else: ins=0.0
                        f[37+ld]=ins;pp[ld]=pk
                    if nl==2:
                        f[27]=ars[0]/(ars[1]+1e-8);f[28]=abs(ws[0]-ws[1])
                        n0=np.linalg.norm(qs[0]);n1=np.linalg.norm(qs[1])
                        f[29]=np.dot(qs[0],qs[1])/(n0*n1+1e-8) if n0>1e-8 and n1>1e-8 else 0.0
                        f[36]=(ars[0]-ars[1])**2/(ars[0]**2+ars[1]**2+1e-8)
                    labels.append(aami_map[sym]);feats.append(f)
            except Exception as e:
                print(f"  Skip {rec_id}: {e}")
        return np.array(labels), np.array(feats)

    print("Extracting DS1 (train)...")
    y_tr, X_tr_all = extract(DS1)
    print(f"  {len(y_tr)} beats")
    print("Extracting DS2 (test)...")
    y_te, X_te_all = extract(DS2)
    print(f"  {len(y_te)} beats")

    X_tr = X_tr_all[:, FEAT_IDX]; X_te = X_te_all[:, FEAT_IDX]
    mu, sd = X_tr.mean(0), X_tr.std(0)+1e-8
    X_tr = (X_tr-mu)/sd; X_te = (X_te-mu)/sd

    np.savez(CACHE_FILE, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
    print(f"Cached to {CACHE_FILE}")
    return X_tr, y_tr, X_te, y_te

def load_data():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}")
        d = np.load(CACHE_FILE)
        return d['X_tr'], d['y_tr'], d['X_te'], d['y_te']
    else:
        print("No cache found, extracting features...")
        return build_cache()

def main():
    from collections import Counter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score

    X_tr, y_tr, X_te, y_te = load_data()
    print(f"Train: {len(y_tr)}, Test: {len(y_te)}")
    print(f"Train dist: {dict(sorted(Counter(y_tr).items()))}")

    # RF baseline
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42,
                                n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr, y_tr); rf_pred = rf.predict(X_te)
    rf_acc = (rf_pred==y_te).mean()*100
    rf_f1 = f1_score(y_te, rf_pred, average=None, zero_division=0)
    print(f"\nRF BASELINE: {rf_acc:.2f}%  F1: N={rf_f1[0]:.3f} S={rf_f1[1]:.3f} V={rf_f1[2]:.3f} F={rf_f1[3]:.3f} Q={rf_f1[4]:.3f}")

    # MLP sweep
    configs = [
        (8,),(10,),(12,),(16,),(20,),(24,),(32,),(48,),(64,),(96,),(128,),
        (8,4),(10,5),(12,6),(16,8),(20,10),(24,12),(32,16),(48,24),(64,32),
        (32,16,8),(64,32,16),
    ]

    print(f"\n{'='*90}")
    print(f"MLP SWEEP: {len(configs)} configs x 5 seeds  |  RF={rf_acc:.2f}%")
    print(f"{'='*90}")

    results = []
    for hs in configs:
        arch = "8-"+"-".join(str(h) for h in hs)+"-5"
        macs=0;prev=8
        for h in hs: macs+=prev*h;prev=h
        macs+=prev*5
        e_nj = macs*E_PER_MAC_PJ_NOM*1e-3

        accs=[];f1a={c:[] for c in CLASS_NAMES}
        t0=time.time()
        for seed in range(5):
            mlp = MLPClassifier(hidden_layer_sizes=hs, max_iter=500, random_state=seed+42,
                                early_stopping=True, validation_fraction=0.1,
                                learning_rate='adaptive', learning_rate_init=1e-3,
                                alpha=1e-4, batch_size=128)
            mlp.fit(X_tr, y_tr); pred=mlp.predict(X_te)
            acc=(pred==y_te).mean()*100
            f1=f1_score(y_te, pred, average=None, zero_division=0)
            accs.append(acc)
            for ci,c in enumerate(CLASS_NAMES): f1a[c].append(f1[ci])

        el=time.time()-t0;mu=np.mean(accs);sd=np.std(accs);gap=mu-rf_acc
        tag=">>>" if mu>=rf_acc else "   "
        print(f"{tag} {arch:>18s}  MACs={macs:5d}  acc={mu:.2f}+/-{sd:.2f}  max={max(accs):.2f}  "
              f"gap={gap:+.2f}%  S={np.mean(f1a['S']):.3f} V={np.mean(f1a['V']):.3f}  ({el:.1f}s)")

        results.append({'arch':arch,'hidden':list(hs),'macs':macs,'e_nj':round(e_nj,4),
            'acc_mean':round(mu,3),'acc_std':round(sd,3),'acc_max':round(max(accs),3),'gap':round(gap,3),
            'f1':{c:round(np.mean(f1a[c]),4) for c in CLASS_NAMES},'accs':[round(a,3) for a in accs]})

    results.sort(key=lambda x:-x['acc_mean'])
    print(f"\n{'='*90}")
    print(f"RANKED  |  RF = {rf_acc:.2f}%")
    print(f"{'='*90}")
    print(f"{'Arch':>18} {'MACs':>6} {'E(nJ)':>7} {'Mean%':>7} {'Max%':>6} {'Gap':>7} {'S':>5} {'V':>5}")
    print("-"*75)
    for r in results:
        m="*" if r['gap']>=0 else " "
        print(f"{m}{r['arch']:>17} {r['macs']:6d} {r['e_nj']:6.3f} {r['acc_mean']:6.2f}% "
              f"{r['acc_max']:5.2f}% {r['gap']:+6.2f}% {r['f1']['S']:5.3f} {r['f1']['V']:5.3f}")

    matches=[r for r in results if r['gap']>=0]
    if matches:
        best=min(matches,key=lambda x:x['macs'])
        print(f"\n>>> SMALLEST MLP >= RF: {best['arch']}  MACs={best['macs']}  acc={best['acc_mean']:.2f}%  E={best['e_nj']:.3f}nJ")
    else:
        c=results[0]
        print(f"\n>>> No match. Closest: {c['arch']} at {c['acc_mean']:.2f}% (gap={c['gap']:+.2f}%)")

    out=os.path.join(SCRIPT_DIR,'mlp_sweep_results.json')
    with open(out,'w') as f: json.dump({'rf_acc':round(rf_acc,3),'results':results},f,indent=2)
    print(f"Saved to {out}")

if __name__=='__main__':
    main()
