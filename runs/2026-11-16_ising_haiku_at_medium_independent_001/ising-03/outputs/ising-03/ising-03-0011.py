
import json, numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out=[]
for T in Ts:
    d=json.load(open(f'/work/res_{T}.json'))
    a2=np.array(d['m2']); a4=np.array(d['m4']); R=len(a2)
    def U(x2,x4):
        m2=x2.mean(); m4=x4.mean()
        return 1-m4/(3*m2*m2)
    u=U(a2,a4)
    # jackknife over replicas
    tot2=a2.sum(); tot4=a4.sum()
    loo=np.array([1-((tot4-a4[i])/(R-1))/(3*((tot2-a2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    out.append((T,u,err))
    print(f"T={T} U={u:.5f} err={err:.5f} <|m|^2>={a2.mean():.4f}")
