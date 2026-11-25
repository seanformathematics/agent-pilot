
import numpy as np, time, os
N=16
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s, tab, rng):
    for c in (0,1):
        nn = (np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2))
        sn = (s*nn).astype(np.int16)+4
        p = tab[sn]
        u = rng.random(s.shape)
        flip = (u < p) & masks[c]
        s = np.where(flip, -s, s).astype(np.int8)
    return s
def run(T, seed, R=1000, nburn=3000, nmeas=30000):
    rng=np.random.default_rng(seed)
    tab=np.zeros(9)
    for sn in (-4,-2,0,2,4):
        dE=2*sn; tab[sn+4]=1.0 if dE<=0 else np.exp(-dE/T)
    s=rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    for k in range(nburn): s=sweep(s,tab,rng)
    S1=np.zeros(R); S2=np.zeros(R); S4=np.zeros(R)
    half=np.zeros((2,R,2))  # per-half sums of m2,m4
    for k in range(nmeas):
        s=sweep(s,tab,rng)
        m=s.reshape(R,-1).sum(axis=1)/(N*N)
        m2=m*m
        S1+=np.abs(m); S2+=m2; S4+=m2*m2
        if k<nmeas//2: half[0,:,0]+=m2; half[0,:,1]+=m2*m2
        else: half[1,:,0]+=m2; half[1,:,1]+=m2*m2
    return dict(S1=S1,S2=S2,S4=S4,n=nmeas,half=half)
def U_of(S2,S4,n):
    m2=S2/n; m4=S4/n
    return 1-m4/(3*m2**2)
for T in [1.9,2.0]:
    t=time.time()
    d=run(T, seed=int(T*1000)+7)
    np.savez(f"/work/T{T}.npz", **d)
    U_r=U_of(d['S2'],d['S4'],d['n'])
    # pooled
    Up=U_of(d['S2'].sum(),d['S4'].sum(),d['n']*len(d['S2']))
    h=d['half']
    Uh=[U_of(h[i,:,0].sum(),h[i,:,1].sum(),d['n']//2*len(d['S2'])) for i in (0,1)]
    print(T, "U pooled",Up, "halves",Uh, "replica-jk err approx", np.std(U_r)/np.sqrt(len(U_r)), "time",time.time()-t, "<|m|>",d['S1'].mean()/d['n'], flush=True)
