
import re
src = open('/work/sim.py').read()
src = src.replace("def run(seed, temps, R, equil, block, time_budget, out, hot=True):",
                  "def run(seed, temps, R, equil, block, time_budget, out, hot=True, L=16):")
src = src.replace("    L = 16\n", "")
open('/work/sim.py','w').write(src)
import sys, importlib
sys.path.insert(0,'/work')
import sim; importlib.reload(sim)
import numpy as np, os
# exact enumeration for L=4
L=4; N=L*L
idx = np.arange(2**N)
bits = ((idx[:,None] >> np.arange(N)) & 1)
S = (2*bits-1).reshape(-1,L,L)
E = np.zeros(len(idx))
E -= (S*np.roll(S,1,1)).sum(axis=(1,2)) + (S*np.roll(S,1,2)).sum(axis=(1,2))
m = S.reshape(len(idx),-1).mean(axis=1)
temps = [1.0,1.5,2.0,2.27,2.6,3.0,4.0]
exact = {}
for T in temps:
    w = np.exp(-(E-E.min())/T); w/=w.sum()
    m2=(w*m**2).sum(); m4=(w*m**4).sum()
    exact[T]=1-m4/(3*m2**2)
# MC
nsw, nb, dt = sim.run(seed=7, temps=temps, R=64, equil=500, block=20, time_budget=25, out='/work/val4.npz', L=4)
d=np.load('/work/val4.npz')
m2=d['m2']; m4=d['m4']; R=64; n=nb*20
for k,T in enumerate(temps):
    sl=slice(k*R,(k+1)*R)
    S2=m2[:,sl].sum(axis=0); S4=m4[:,sl].sum(axis=0)
    U=1-(S4.mean()/n)/(3*(S2.mean()/n)**2)
    jk=np.empty(R)
    for i in range(R):
        mk=np.ones(R,bool); mk[i]=False
        jk[i]=1-(S4[mk].mean()/n)/(3*(S2[mk].mean()/n)**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(f"T={T}: MC U={U:.5f} +- {err:.5f}   exact U={exact[T]:.5f}  diff/err={(U-exact[T])/err:.2f}")
print("sweeps", nsw)
