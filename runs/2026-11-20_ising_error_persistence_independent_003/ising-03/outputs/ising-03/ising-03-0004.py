import time, numpy as np, os
N=16; R=2000; NEQ=2000; NMEAS=8000
mask=np.zeros((N,N),bool); mask[(np.arange(N)[:,None]+np.arange(N)[None,:])%2==0]=True
def sweep(s,T,rng):
    for m in (mask,~mask):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h
        p=np.exp(-dE/T)
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<p)
        s[acc&m]*=-1
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    t0=time.time()
    for _ in range(NEQ): sweep(s,T,rng)
    S2=np.zeros(R); S4=np.zeros(R); SA=np.zeros(R)
    for k in range(NMEAS):
        sweep(s,T,rng)
        m=s.reshape(R,-1).sum(1)/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2; SA+=np.abs(m)
        if k==NMEAS//2-1: S2h=S2.copy(); S4h=S4.copy()
    np.savez(f'/work/T{T}.npz',S2=S2,S4=S4,SA=SA,S2h=S2h,S4h=S4h,NMEAS=NMEAS)
    U=1-(S4/NMEAS).mean()/(3*((S2/NMEAS).mean())**2)
    print(T,"U",U,"time",time.time()-t0,flush=True)
for i,T in enumerate([2.2,2.3,2.4]):
    run(T,200+i)
