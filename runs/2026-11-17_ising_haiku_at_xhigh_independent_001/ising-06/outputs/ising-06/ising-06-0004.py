import numpy as np, time, json
N=16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def make_tables(T):
    xs = np.array([-4,-2,0,2,4])
    p = np.where(xs<=0, 1.0, np.exp(-2.0*xs/T))
    return p.astype(np.float32)
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
def run(R, T, sweeps, burn, seed):
    rng = np.random.default_rng(seed)
    S = np.ones((R,N,N), dtype=np.int8)
    S[R//2:] = -1
    tab = make_tables(T)
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for t in range(sweeps):
        for mk in masks:
            h = np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            x = S*h
            pr = tab[(x+4)>>1]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pr) & mk
            np.negative(S, out=S, where=flip)
        if t>=burn:
            m = S.reshape(R,-1).mean(axis=1).astype(np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2; n+=1
    return S2/n, S4/n
def binder(a2,a4):
    return 1-a4/(3*a2**2)
def jack(a2,a4):
    R=len(a2)
    U=binder(a2.mean(),a4.mean())
    # delete-one jackknife over replicas
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.empty(R)
    for r in range(R):
        m2=(tot2-a2[r])/(R-1); m4=(tot4-a4[r])/(R-1)
        Uj[r]=binder(m2,m4)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
res={}
for T in Ts:
    t0=time.time()
    a2,a4=run(400,T,4000,500,2024+int(T*100))
    U,err=jack(a2,a4)
    res[T]=(U,err)
    print(f"T={T} U={U:.5f} err={err:.5f} time={time.time()-t0:.1f}s", flush=True)
