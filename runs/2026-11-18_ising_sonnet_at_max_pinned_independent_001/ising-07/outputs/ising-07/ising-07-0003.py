
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from sw import SWBatch

N=4; Ns=N*N
# exact enumeration
states = np.arange(2**Ns, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(Ns, dtype=np.uint32)[None,:]) & 1).astype(np.int8)*2-1
S = bits.reshape(-1,N,N)
E = -(S*np.roll(S,-1,axis=2)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=1)).sum(axis=(1,2))
Mtot = S.sum(axis=(1,2)).astype(np.float64)
m = Mtot/Ns
def exact(T):
    w = np.exp(-(E-E.min())/T); w/=w.sum()
    m2=(w*m**2).sum(); m4=(w*m**4).sum()
    return m2,m4,1-m4/(3*m2**2), (w*E).sum()/Ns

def jack(M,S2,S4,Ns):
    # M,S2,S4: arrays (steps,R); jackknife over replicas (chains)
    R = M.shape[1]
    out = {}
    for name,(a2,a4) in {"direct":((M/Ns)**2,(M/Ns)**4), "improved":(S2/Ns**2,(3*S2**2-2*S4)/Ns**4)}.items():
        c2 = a2.mean(axis=0); c4 = a4.mean(axis=0)
        U = lambda x2,x4: 1-x4/(3*x2**2)
        Uall = U(c2.mean(), c4.mean())
        Ui = np.array([U((c2.sum()-c2[i])/(R-1),(c4.sum()-c4[i])/(R-1)) for i in range(R)])
        err = np.sqrt((R-1)/R*((Ui-Ui.mean())**2).sum())
        out[name]=(c2.mean(), c4.mean(), Uall, err)
    return out

R=200; steps=1000; neq=100
for T in (1.5, 2.0, 2.269, 3.0, 5.0):
    sim = SWBatch(N,R,T,seed=1000+int(T*100), start=np.arange(R)%2)
    for _ in range(neq): sim.step()
    Ms=[];S2s=[];S4s=[]
    for _ in range(steps):
        M,S2,S4=sim.step(); Ms.append(M);S2s.append(S2);S4s.append(S4)
    res = jack(np.array(Ms,float),np.array(S2s),np.array(S4s),Ns)
    ex = exact(T)
    print("T=%.3f exact: m2=%.5f m4=%.5f U=%.5f"%(T,ex[0],ex[1],ex[2]))
    for k,v in res.items():
        print("    %-8s m2=%.5f m4=%.5f U=%.5f +- %.5f  (dev %.2f sigma)"%(k,v[0],v[1],v[2],v[3],(v[2]-ex[2])/v[3]))
