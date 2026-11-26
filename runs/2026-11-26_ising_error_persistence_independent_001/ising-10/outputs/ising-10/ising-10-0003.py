import json, numpy as np
raw=json.load(open('/work/raw.json'))
MEAS=5000
out={}
for T in ['1.9','2.0','2.1','2.2','2.3','2.4']:
    d=raw[T]; S2=np.array(d['S2'])/MEAS; S4=np.array(d['S4'])/MEAS
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    # jackknife over replicas
    tot2=S2.sum(); tot4=S4.sum()
    loo2=(tot2-S2)/(R-1); loo4=(tot4-S4)/(R-1)
    Uj=1-loo4/(3*loo2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # also per-replica-block (20 blocks of 50) jackknife for robustness
    nb=20; idx=np.array_split(np.arange(R),nb)
    Ub=[]
    for b in idx:
        mask=np.ones(R,bool); mask[b]=False
        Ub.append(1-S4[mask].mean()/(3*S2[mask].mean()**2))
    Ub=np.array(Ub); errb=np.sqrt((nb-1)/nb*np.sum((Ub-Ub.mean())**2))
    print(T, "U=%.5f err_jk=%.5f err_blk=%.5f"%(U,err,errb))
    out[T]=(U,err)