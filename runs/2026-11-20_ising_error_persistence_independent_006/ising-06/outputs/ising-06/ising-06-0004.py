
import sys, numpy as np
sys.path.insert(0,'/work')
from wolff import run
N=4; L=N*N
configs=((np.arange(2**L)[:,None]>>np.arange(L))&1)*2-1
configs=configs.reshape(-1,N,N)
E_bonds = configs*np.roll(configs,1,axis=1) + configs*np.roll(configs,1,axis=2)
E = -E_bonds.sum(axis=(1,2))
mag = configs.sum(axis=(1,2))/L
T=2.3
w=np.exp(-E/T); w/=w.sum()
m2=(w*mag**2).sum(); m4=(w*mag**4).sum()
print("exact U", 1-m4/(3*m2**2))
a,b,c = run(4, T, 7, 2_000_000, 5000)
print("wolff U", 1-c.mean()/(3*b.mean()**2))
# independent replicate at T=2.4 for N=16
a,b,c = run(16, 2.4, 98765, 1_000_000, 5000)
print("N16 T2.4 replicate U", 1-c.mean()/(3*b.mean()**2))
