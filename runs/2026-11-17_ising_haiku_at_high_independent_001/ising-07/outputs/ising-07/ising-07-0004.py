import numpy as np, time, os
N=16; R=2000
i,j=np.indices((N,N))
ev=((i+j)%2==0); od=~ev
def sweep(S,rng,ptab):
    for mask in (ev,od):
        nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
        dE=2*S*nn
        acc=rng.random(S.shape)<ptab[dE+8]
        S[acc&mask]*=-1
def run(T, seed, burn=1000, meas=6000, blk=100):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    beta=1/T
    ptab=np.array([min(1.0,np.exp(-beta*d)) for d in range(-8,9)])
    for _ in range(burn): sweep(S,rng,ptab)
    s2=np.zeros(R); s4=np.zeros(R); s1=np.zeros(R)
    nb=meas//blk
    b2=np.zeros((R,nb)); b4=np.zeros((R,nb))
    for t in range(meas):
        sweep(S,rng,ptab)
        m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m; m4=m2*m2
        s2+=m2; s4+=m4; s1+=np.abs(m)
        b2[:,t//blk]+=m2; b4[:,t//blk]+=m4
    return dict(T=T,seed=seed,burn=burn,meas=meas,blk=blk,s2=s2,s4=s4,s1=s1,b2=b2,b4=b4)
def U_of(s2,s4,n):
    m2=s2.sum()/(len(s2)*n); m4=s4.sum()/(len(s4)*n)
    return 1-m4/(3*m2*m2)
Ts=[2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    seed=200+k
    d=run(T,seed)
    np.savez(f'/work/T{T:.2f}.npz',**{kk:(np.array(v) if not isinstance(v,(int,float)) else v) for kk,v in d.items()})
    n=d['meas']
    print(T, 'U=',U_of(d['s2'],d['s4'],n), '<|m|>=',d['s1'].sum()/(R*n), 'time',time.time()-t0, flush=True)