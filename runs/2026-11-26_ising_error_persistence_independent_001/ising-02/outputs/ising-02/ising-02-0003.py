import random, math, time, os, json
import numpy as np
os.makedirs('/work',exist_ok=True)
N=16; V=N*N
nb=[[((i//N+1)%N)*N+i%N, ((i//N-1)%N)*N+i%N, (i//N)*N+(i%N+1)%N, (i//N)*N+(i%N-1)%N] for i in range(V)]
def run_chain(T, seed, neq, nmeas, block):
    rng=random.Random(seed)
    s=[1]*V
    M=V
    p=1-math.exp(-2.0/T)
    rnd=rng.random
    def sweep_cluster():
        nonlocal M
        i=int(rnd()*V)
        si=s[i]
        stack=[i]; s[i]=-si; size=1
        while stack:
            j=stack.pop()
            for k in nb[j]:
                if s[k]==si and rnd()<p:
                    s[k]=-si; stack.append(k); size+=1
        M -= 2*si*size
    for _ in range(neq): sweep_cluster()
    nblk=nmeas//block
    S2=np.zeros(nblk); S4=np.zeros(nblk)
    for b in range(nblk):
        a2=0.0; a4=0.0
        for _ in range(block):
            sweep_cluster()
            m2=(M/V)**2
            a2+=m2; a4+=m2*m2
        S2[b]=a2/block; S4[b]=a4/block
    return S2,S4
def binder(m2,m4): return 1-m4/(3*m2**2)
if __name__=="__main__":
    import sys
    Ts=[float(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [1.9,2.0,2.1]
    out={}
    for ti,T in enumerate(Ts):
        t0=time.time()
        res=[]
        for c in range(16):
            seed=100000*int(round(T*100))+c
            S2,S4=run_chain(T,seed,3000,100000,500)
            res.append((S2,S4))
        np.savez(f'/work/T{T:.2f}.npz', S2=np.array([r[0] for r in res]), S4=np.array([r[1] for r in res]))
        print(T, time.time()-t0, flush=True)
