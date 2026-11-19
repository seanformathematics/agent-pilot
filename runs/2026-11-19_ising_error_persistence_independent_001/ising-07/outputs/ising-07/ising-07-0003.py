import random, math, time, os, json
import numpy as np
N=16; NN=N*N
nbr=[]
for i in range(N):
    for j in range(N):
        nbr.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])

def run(T, nmeas, seed, neq):
    rng=random.Random(seed)
    rnd=rng.random
    s=[1 if rnd()<0.5 else -1 for _ in range(NN)]
    padd=1-math.exp(-2.0/T)
    M=sum(s)
    out=np.empty(nmeas)
    for step in range(nmeas+neq):
        i0=int(rnd()*NN)
        sd=s[i0]
        cl=[i0]; s[i0]=-sd; stack=[i0]
        while stack:
            k=stack.pop()
            for nb in nbr[k]:
                if s[nb]==sd and rnd()<padd:
                    s[nb]=-sd; stack.append(nb); cl.append(nb)
        M-=2*sd*len(cl)
        if step>=neq:
            out[step-neq]=M/NN
    return out

temps=[1.9,2.0,2.1,2.2,2.3,2.4]
os.makedirs('/work/data',exist_ok=True)
t0=time.time()
for idx,T in enumerate(temps):
    m=run(T, 1000000, 1000+idx, 20000)
    np.save(f'/work/data/m_T{T}.npy', m)
    m2=m*m
    U=1-np.mean(m**4)/(3*np.mean(m2)**2)
    print(T, "U=",U, "<|m|>",np.mean(np.abs(m)), "time",time.time()-t0, flush=True)
