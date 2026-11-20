
import json, numpy as np, os
p='/work/res.json' if os.path.exists('/work/res.json') else '/tmp/res.json'
res=json.load(open(p))
for T in ['1.9','2.0','2.1','2.2','2.3','2.4']:
    s2=np.array(res[T]['s2']); s4=np.array(res[T]['s4']); R=s2.shape[0]; nb=s2.shape[1]
    U=1-s4.mean()/(3*s2.mean()**2)
    tot2=s2.sum(); tot4=s4.sum()
    Uj=np.array([1-((tot4-s4[r].sum())/((R-1)*nb))/(3*((tot2-s2[r].sum())/((R-1)*nb))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, repr(float(U)), repr(float(err)))
