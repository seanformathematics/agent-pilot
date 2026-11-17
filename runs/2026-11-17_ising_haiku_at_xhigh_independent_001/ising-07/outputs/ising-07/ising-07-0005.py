import sys; sys.path.insert(0,'/work')
import numpy as np, ising, time
# exact enumeration N=4
N=4; NN=16
configs = ((np.arange(2**NN)[:,None] >> np.arange(NN)[None,:]) & 1)*2-1
configs = configs.reshape(-1,N,N)
# bonds
E = np.zeros(len(configs))
for (di,dj) in [(0,1),(1,0)]:
    E -= (configs * np.roll(configs, shift=(-di,-dj), axis=(1,2))).sum(axis=(1,2))
m = configs.sum(axis=(1,2))/NN
for T in [2.0, 2.5]:
    w = np.exp(-(E - E.min())/T); w/=w.sum()
    m2=(w*m**2).sum(); m4=(w*m**4).sum()
    Uex = 1-m4/(3*m2*m2)
    ms = ising.run(T, R=400, S=2000, burn=200, seed=7, N=N)
    U, Ue, _ = ising.jack_U(ms)
    print(f"T={T} exact U={Uex:.5f}  SW U={U:.5f} +- {Ue:.5f}")
