
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=2000; NB=1000; NM=4000
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
def run(T, seed):
    rng = np.random.default_rng(seed)
    beta=1.0/T
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    # precompute acceptance table for dE in {0,4,8}
    acc_tab = np.array([1.0, np.exp(-4*beta), np.exp(-8*beta)])
    def sweep(s):
        for mk in masks:
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nb
            p = acc_tab[np.clip(dE,0,8)//4]
            acc = rng.random(s.shape) < p
            s[acc & mk] *= -1
        return s
    for _ in range(NB): sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); S2h=np.zeros((2,R)); S4h=np.zeros((2,R))
    for k in range(NM):
        sweep(s)
        m = s.sum(axis=(1,2))/(N*N)
        m2=m*m; m4=m2*m2
        S2+=m2; S4+=m4
        h = 0 if k< NM//2 else 1
        S2h[h]+=m2; S4h[h]+=m4
    return S2/NM, S4/NM, S2h/(NM/2), S4h/(NM/2)
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    a2,a4,h2,h4 = run(T,seed)
    np.savez(f'/work/T{T}.npz', a2=a2,a4=a4,h2=h2,h4=h4)
    U = 1-np.mean(a4)/(3*np.mean(a2)**2)
    Uh = [1-np.mean(h4[i])/(3*np.mean(h2[i])**2) for i in range(2)]
    print(T, 'U=',U, 'halves', Uh, 'time',time.time()-t, flush=True)
