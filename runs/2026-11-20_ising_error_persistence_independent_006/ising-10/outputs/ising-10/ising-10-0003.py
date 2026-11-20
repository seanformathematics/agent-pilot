
import json, numpy as np, os
p='/work/res.json' if os.path.exists('/work/res.json') else '/tmp/res.json'
res=json.load(open(p))
def U_of(s2,s4):
    # s2,s4: arrays (R,nblk) of per-replica block averages; pooled over replicas & blocks
    m2=s2.mean(); m4=s4.mean()
    return 1-m4/(3*m2**2)
for T in ['1.9','2.0','2.1','2.2','2.3','2.4']:
    s2=np.array(res[T]['s2']); s4=np.array(res[T]['s4']); s1=np.array(res[T]['s1'])
    R=s2.shape[0]
    U=U_of(s2,s4)
    # replica jackknife
    Uj=np.empty(R)
    tot2=s2.sum(); tot4=s4.sum(); n=s2.size
    for r in range(R):
        m2=(tot2-s2[r].sum())/((R-1)*s2.shape[1]); m4=(tot4-s4[r].sum())/((R-1)*s2.shape[1])
        Uj[r]=1-m4/(3*m2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # block-level check: U per time-block (pooled over replicas), spread
    Ub=[1-s4[:,b].mean()/(3*s2[:,b].mean()**2) for b in range(s2.shape[1])]
    print(T, "U=%.5f err_jk=%.5f  <|m|>=%.4f  blockU std=%.5f"%(U,err,s1.mean(),np.std(Ub,ddof=1)))
