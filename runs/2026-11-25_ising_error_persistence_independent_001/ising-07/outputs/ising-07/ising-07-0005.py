
import sys, numpy as np
sys.path.insert(0,'/work')
from wolff import wolff_run
def U_of(a,b): return 1-b.mean()/(3*a.mean()**2)
for T,seed in [(2.2,2000),(2.3,2001)]:
    m2,m4=wolff_run(16,T,300000,seed,nburn=5000)
    nb=60; L=len(m2)//nb
    us=np.array([U_of(m2[b*L:(b+1)*L],m4[b*L:(b+1)*L]) for b in range(nb)])
    print(T, seed, "U=%.6f err=%.6f"%(U_of(m2,m4), us.std(ddof=1)/np.sqrt(nb)))
