
import random, math, time, json, os
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
    return S2, S4

def binder_from_bins(S2,S4,n):
    # per-measurement sums: n measurements per bin, m=M/NN
    nb_=len(S2)
    def U(a2,a4):
        m2=a2/(n*NN*NN); m4=a4/(n*NN**4)
        return 1-m4/(3*m2*m2)
    full=U(S2.sum(),S4.sum())
    jk=np.empty(nb_)
    tot2=S2.sum(); tot4=S4.sum()
    for b in range(nb_):
        jk[b]=U(tot2-S2[b], tot4-S4[b]) if True else 0
    # jackknife with bins: leave-one-out, n_eff = nb_-1 bins
    # adjust: U(leave-one-out) uses (nb_-1)*n measurements
    def Uk(a2,a4,m):
        m2=a2/(m*NN*NN); m4=a4/(m*NN**4)
        return 1-m4/(3*m2*m2)
    jk=np.array([Uk(tot2-S2[b], tot4-S4[b], (nb_-1)*n) for b in range(nb_)])
    err=math.sqrt((nb_-1)/nb_*np.sum((jk-jk.mean())**2))
    return full, err

T=2.2
t0=time.time()
S2,S4=simulate(T, seed=12345, nclust=4_000_000, nburn=20000, binsize=2000)
U,err=binder_from_bins(S2,S4,2000)
print('T',T,'U',U,'err',err,'nbins',len(S2),'time',time.time()-t0)
# check stability across bin-group sizes
for g in [2,5,10,20]:
    nbg=len(S2)//g
    a2=S2[:nbg*g].reshape(nbg,g).sum(1); a4=S4[:nbg*g].reshape(nbg,g).sum(1)
    print('group',g,binder_from_bins(a2,a4,2000*g))
json.dump({'T':T,'S2':S2.tolist(),'S4':S4.tolist()},open(f'/work/raw_T{T}.json','w'))
