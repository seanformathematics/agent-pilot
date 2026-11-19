
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from sw import SWBatch

def lnZ_exact(L, T):
    K = 1.0/T
    c2K = np.cosh(2*K); s2K = np.sinh(2*K)
    A = c2K/np.tanh(2*K)
    def gam(l):
        if l == 0:
            return 2*K + np.log(np.tanh(K))      # signed
        return np.arccosh(A - np.cos(np.pi*l/L))
    def lncosh2(x):  # ln(2cosh x)
        x=abs(x); return x + np.log1p(np.exp(-2*x))
    def lnsinh2(x):  # ln|2 sinh x|, sign
        sgn = 1.0 if x>0 else -1.0
        x=abs(x); return x + np.log1p(-np.exp(-2*x)), sgn
    odd = [gam(2*r+1) for r in range(L)]
    even = [gam(2*r) for r in range(L)]
    terms=[]; signs=[]
    terms.append(sum(lncosh2(L*g/2) for g in odd)); signs.append(1.0)
    ls=[lnsinh2(L*g/2) for g in odd]; terms.append(sum(a for a,b in ls)); signs.append(np.prod([b for a,b in ls]))
    terms.append(sum(lncosh2(L*g/2) for g in even)); signs.append(1.0)
    ls=[lnsinh2(L*g/2) for g in even]; terms.append(sum(a for a,b in ls)); signs.append(np.prod([b for a,b in ls]))
    terms=np.array(terms); signs=np.array(signs)
    mx=terms.max()
    tot = (signs*np.exp(terms-mx)).sum()
    return -np.log(2) + 0.5*L*L*np.log(2*s2K) + mx + np.log(tot)

def energy_exact(L,T,h=1e-5):
    # <E> = -d lnZ / d beta
    b=1.0/T
    f=lambda beta: lnZ_exact(L,1.0/beta)
    return -(f(b+h)-f(b-h))/(2*h)

# sanity: exact formula versus brute-force enumeration for L=4
L=4; Ns=16
states = np.arange(2**Ns, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(Ns, dtype=np.uint32)[None,:]) & 1).astype(np.int8)*2-1
S = bits.reshape(-1,L,L)
E = -(S*np.roll(S,-1,axis=2)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=1)).sum(axis=(1,2))
for T in (1.5,2.3,3.0):
    w=np.exp(-(E-E.min())/T); lnZ_bf = np.log(w.sum()) - E.min()/T
    print("L=4 T=%.1f lnZ exact formula %.8f  brute force %.8f ; <E> formula %.6f brute %.6f"%(T,lnZ_exact(4,T),lnZ_bf,energy_exact(4,T),(w*E).sum()/w.sum()))

N=16; Ns=N*N
def energy(s):
    return -(s*np.roll(s,-1,axis=2)).sum(axis=(1,2),dtype=np.int64) - (s*np.roll(s,-1,axis=1)).sum(axis=(1,2),dtype=np.int64)

R=200; steps=1500; neq=200
for T in (1.5, 1.8, 2.1, 2.269, 2.5, 2.8):
    sim = SWBatch(N,R,T,seed=31337+int(T*100), start=np.arange(R)%2)
    for _ in range(neq): sim.step()
    Es=np.empty((steps,R))
    for t in range(steps):
        sim.step(); Es[t]=energy(sim.s)
    ch = Es.mean(axis=0)/Ns
    est=ch.mean(); err=ch.std(ddof=1)/np.sqrt(R)
    ex = energy_exact(N,T)/Ns
    print("N=16 T=%.3f  <E>/N^2: SW %.5f +- %.5f   exact %.5f   dev %.2f sigma"%(T,est,err,ex,(est-ex)/err))
