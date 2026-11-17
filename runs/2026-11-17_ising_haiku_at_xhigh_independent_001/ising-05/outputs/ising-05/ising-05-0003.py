
import os, time, math, random
os.makedirs("/work", exist_ok=True)
import numpy as np

def build(N):
    nb=[]
    for x in range(N):
        for y in range(N):
            i=x*N+y
            nb.append((((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N))
    return nb

def run_wolff(N, T, nsweeps, seed, nburn=2000, init=None):
    rng=random.Random(seed)
    nb=build(N)
    V=N*N
    if init is None:
        spins=[1 if rng.random()<0.5 else -1 for _ in range(V)]
    else:
        spins=list(init)
    M=sum(spins)
    p=1.0-math.exp(-2.0/T)
    rnd=rng.random
    rint=rng.randrange
    out_m=np.empty(nsweeps)
    for it in range(nburn+nsweeps):
        i=rint(V)
        s0=spins[i]
        spins[i]=-s0
        stack=[i]
        c=1
        while stack:
            j=stack.pop()
            for k in nb[j]:
                if spins[k]==s0 and rnd()<p:
                    spins[k]=-s0
                    stack.append(k)
                    c+=1
        M-= 2*s0*c
        if it>=nburn:
            out_m[it-nburn]=M/V
    return out_m, spins

def U_from_m(m):
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2*m2)

if __name__=="__main__":
    t=time.time()
    m,_=run_wolff(16, 2.269, 20000, seed=1, nburn=1000)
    dt=time.time()-t
    print("time per 20000 clusters:", dt)
    print("U:", U_from_m(m), "mean|m|", np.mean(np.abs(m)))
    # binned error
    for nb_ in [10,20,50,100,200]:
        bs=len(m)//nb_
        mb=m[:bs*nb_].reshape(nb_,bs)
        Ub=np.array([U_from_m(x) for x in mb])
        print("nbins",nb_,"blocklen",bs,"U std over bins/sqrt(nb)", Ub.std(ddof=1)/np.sqrt(nb_))
