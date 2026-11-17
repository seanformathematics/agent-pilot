
import numpy as np, os, time
os.makedirs('/work',exist_ok=True)
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
import sys
temps=[1.5,1.6,1.7,1.8,1.9]
t0=time.time()
for k,T in enumerate(temps):
    M=run(T,1_000_000,1000+k)
    np.save(f'/work/M_{T}.npy',M)
    print(T, time.time()-t0, flush=True)
    if time.time()-t0>500: break
