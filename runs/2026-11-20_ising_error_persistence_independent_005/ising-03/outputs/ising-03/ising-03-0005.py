
mod = r'''
import random, math, time, json
import numpy as np
N=16; NN=N*N
nb=[]
for i in range(N):
    for j in range(N):
        nb.append((((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N))
def simulate(T, seed, nclust, nburn, binsize):
    rng=random.Random(seed); rnd=rng.random
    padd=1.0-math.exp(-2.0/T)
    spin=[1 if rnd()<0.5 else -1 for _ in range(NN)]
    M=sum(spin)
    for _ in range(nburn):
        s0=int(rnd()*NN); c=spin[s0]; spin[s0]=-c; stack=[s0]; k=1
        while stack:
            s=stack.pop()
            for t in nb[s]:
                if spin[t]==c and rnd()<padd:
                    spin[t]=-c; stack.append(t); k+=1
        M+=-2*c*k
    nbins=nclust//binsize
    S2=np.zeros(nbins); S4=np.zeros(nbins)
    for b in range(nbins):
        s2=0; s4=0
        for _ in range(binsize):
            s0=int(rnd()*NN); c=spin[s0]; spin[s0]=-c; stack=[s0]; k=1
            while stack:
                s=stack.pop()
                for t in nb[s]:
                    if spin[t]==c and rnd()<padd:
                        spin[t]=-c; stack.append(t); k+=1
            M+=-2*c*k
            m2=M*M
            s2+=m2; s4+=m2*m2
        S2[b]=s2; S4[b]=s4
    return S2,S4
def Uof(a2,a4,m):
    m2=a2/(m*NN*NN); m4=a4/(m*NN**4)
    return 1-m4/(3*m2*m2)
def analyse(S2,S4,n):
    nbins=len(S2)
    full=Uof(S2.sum(),S4.sum(),nbins*n)
    tot2,tot4=S2.sum(),S4.sum()
    jk=np.array([Uof(tot2-S2[b],tot4-S4[b],(nbins-1)*n) for b in range(nbins)])
    err=math.sqrt((nbins-1)/nbins*np.sum((jk-jk.mean())**2))
    return full,err
'''
open('/work/sim.py','w').write(mod)
import sys; sys.path.insert(0,'/work')
import importlib, sim, json, time
import numpy as np
res={}
for T,seed in [(1.9,1901),(2.0,2002)]:
    t0=time.time()
    S2,S4=sim.simulate(T,seed,2_000_000,20000,2000)
    U,err=sim.analyse(S2,S4,2000)
    print('T',T,'U',U,'err',err,'time',time.time()-t0, flush=True)
    json.dump({'T':T,'seed':seed,'S2':S2.tolist(),'S4':S4.tolist()},open(f'/work/raw_T{T}.json','w'))
