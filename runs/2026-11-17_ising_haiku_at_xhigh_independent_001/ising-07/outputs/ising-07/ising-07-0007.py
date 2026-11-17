import sys; sys.path.insert(0,'/work')
import numpy as np, ising
N=4; NN=16
configs = ((np.arange(2**NN)[:,None] >> np.arange(NN)[None,:]) & 1)*2-1
configs = configs.reshape(-1,N,N)
E = np.zeros(len(configs))
for (di,dj) in [(0,1),(1,0)]:
    E -= (configs * np.roll(configs, shift=(-di,-dj), axis=(1,2))).sum(axis=(1,2))
m = configs.sum(axis=(1,2))/NN
T=2.5
w = np.exp(-(E - E.min())/T); w/=w.sum()
m2=(w*m**2).sum(); m4=(w*m**4).sum()
print("exact U", 1-m4/(3*m2*m2), "exact <m2>", m2, "exact <m4>", m4)
ms = ising.run(T, R=2000, S=5000, burn=300, seed=99, N=N)
U, Ue, _ = ising.jack_U(ms)
print("SW U", U, Ue, "<m2>", np.mean(ms**2), "<m4>", np.mean(ms**4))
