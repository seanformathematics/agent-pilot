
import numpy as np, os, time
N=16
nb=[]
for i in range(N):
    for j in range(N):
        nb.append([((i+1)%N)*N+j,((i-1)%N)*N+j,i*N+(j+1)%N,i*N+(j-1)%N])
def run(T, nsteps, seed, nequil=5000):
    rng=np.random.default_rng(seed)
    sl=[1]*(N*N)
    p=1-np.exp(-2.0/T)
    rnd=rng.random(4_000_000).tolist(); ri=0
    M=np.empty(nsteps,dtype=np.int32)
    tot=nequil+nsteps
    starts=rng.integers(N*N,size=tot).tolist()
    for step in range(tot):
        st=starts[step]
        c=sl[st]; sl[st]=-c; stack=[st]
        while stack:
            k=stack.pop()
            for q in nb[k]:
                if sl[q]==c:
                    if rnd[ri]<p:
                        sl[q]=-c; stack.append(q)
                    ri+=1
                    if ri>=4_000_000:
                        rnd=rng.random(4_000_000).tolist(); ri=0
        if step>=nequil: M[step-nequil]=sum(sl)
    return M
temps=[2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
t0=time.time()
for k,T in enumerate(temps):
    M=run(T,1_000_000,2000+k)
    np.save(f'/work/M_{T}.npy',M)
    print(T, time.time()-t0, flush=True)
