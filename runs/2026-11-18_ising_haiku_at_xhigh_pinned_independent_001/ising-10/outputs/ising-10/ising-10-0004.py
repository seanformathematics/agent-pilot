
import sys; sys.path.insert(0,'/work')
import numpy as np, time
from sw import SW
N=16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=256; nburn=300; nmeas=1000
rng=np.random.default_rng(2024)
res={}
t0=time.time()
for T in Ts:
    s=SW(N,R,T,rng)
    for _ in range(nburn): s.sweep()
    M2=np.zeros((nmeas,R)); M4=np.zeros((nmeas,R))
    for k in range(nmeas):
        s.sweep()
        m=s.mags()
        M2[k]=m**2; M4[k]=m**4
    # per-replica time averages
    m2r=M2.mean(axis=0); m4r=M4.mean(axis=0)
    U=1-m4r.mean()/(3*m2r.mean()**2)
    # jackknife over replicas
    tot2=m2r.sum(); tot4=m4r.sum()
    jk=[]
    for i in range(R):
        a2=(tot2-m2r[i])/(R-1); a4=(tot4-m4r[i])/(R-1)
        jk.append(1-a4/(3*a2**2))
    jk=np.array(jk)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # per-sweep U fluctuation (sample-level sigma)
    # approximate: sigma of U from pooled per-sweep values across replicas via block
    res[T]=(U,err)
    print(f"T={T} U={U:.5f} jkerr={err:.5f} <m2>={m2r.mean():.4f} <|m|>?")
print("time",time.time()-t0)
np.save('/work/pilot_done.npy', np.array([[T,res[T][0],res[T][1]] for T in Ts]))
