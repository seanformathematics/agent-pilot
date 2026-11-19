
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from sw import SWBatch

N=4; Ns=N*N
states = np.arange(2**Ns, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(Ns, dtype=np.uint32)[None,:]) & 1).astype(np.int8)*2-1
S = bits.reshape(-1,N,N)
E = -(S*np.roll(S,-1,axis=2)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=1)).sum(axis=(1,2))
m = S.sum(axis=(1,2)).astype(np.float64)/Ns
def exact(T):
    w = np.exp(-(E-E.min())/T); w/=w.sum()
    m2=(w*m**2).sum(); m4=(w*m**4).sum()
    return m2,m4,1-m4/(3*m2**2)

def analyze(M,S2,S4,Ns):
    R = M.shape[1]
    out = {}
    for name,(a2,a4) in {"direct":((M/Ns)**2,(M/Ns)**4), "improved":(S2/Ns**2,(3*S2**2-2*S4)/Ns**4)}.items():
        c2 = a2.mean(axis=0); c4 = a4.mean(axis=0)
        U = lambda x2,x4: 1-x4/(3*x2**2)
        Uall = U(c2.mean(), c4.mean())
        s2=c2.sum(); s4=c4.sum()
        Ui = U((s2-c2)/(R-1),(s4-c4)/(R-1))
        err = np.sqrt((R-1)/R*((Ui-Ui.mean())**2).sum())
        out[name]=(c2.mean(), c4.mean(), Uall, err)
    return out

R=500; steps=4000; neq=100
t0=time.time()
for T in (1.5, 2.0, 2.269, 3.0, 5.0):
    sim = SWBatch(N,R,T,seed=777000+int(T*100), start=np.arange(R)%2)
    for _ in range(neq): sim.step()
    Ms=np.empty((steps,R));S2s=np.empty((steps,R));S4s=np.empty((steps,R))
    for t in range(steps):
        M,S2,S4=sim.step(); Ms[t]=M;S2s[t]=S2;S4s[t]=S4
    res = analyze(Ms,S2s,S4s,Ns)
    ex = exact(T)
    print("T=%.3f exact: m2=%.5f m4=%.5f U=%.5f"%(T,ex[0],ex[1],ex[2]), " [%.0fs]"%(time.time()-t0))
    for k,v in res.items():
        print("    %-8s m2=%.5f m4=%.5f U=%.5f +- %.5f  (dev %.2f sigma)"%(k,v[0],v[1],v[2],v[3],(v[2]-ex[2])/v[3]))
