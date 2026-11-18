
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
pilot=np.load('/work/pilot_done.npy')
out=[]
def U_of(m2r,m4r):
    return 1-m4r.mean()/(3*m2r.mean()**2)
for i,T in enumerate(Ts):
    d=np.load(f'/work/prod_{i}.npz')
    M2=d['M2'].astype(np.float64); M4=d['M4'].astype(np.float64)
    nm,R=M2.shape
    m2r=M2.mean(axis=0); m4r=M4.mean(axis=0)
    U=U_of(m2r,m4r)
    tot2=m2r.sum(); tot4=m4r.sum()
    a2=(tot2-m2r)/(R-1); a4=(tot4-m4r)/(R-1)
    jk=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # drift: first half vs second half
    h=nm//2
    U1=U_of(M2[:h].mean(0),M4[:h].mean(0)); U2=U_of(M2[h:].mean(0),M4[h:].mean(0))
    # ensemble-averaged series autocorrelation of m2
    x=M2.mean(axis=1); x=x-x.mean()
    ac=np.correlate(x,x,'full')[len(x)-1:]/ (x.var()*len(x))
    tau=1+2*np.sum(ac[1:200][ac[1:200]>0.05]) if True else 0
    # replica-level sample sigma of U-like per-sweep quantity: pooled std of per-replica-block means
    out.append((T,U,err))
    print(f"T={T:.1f} U={U:.5f} err={err:.5f} | pilot U={pilot[i,1]:.5f} err={pilot[i,2]:.5f} | half1={U1:.5f} half2={U2:.5f} | tau_m2~{tau:.1f} | <m2>={m2r.mean():.4f}")
np.save('/work/final_results.npy', np.array(out))
print("max err", max(o[2] for o in out))
