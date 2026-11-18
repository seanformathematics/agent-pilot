
import random, math, time, json, os
import numpy as np
N=16
NS=N*N
nb=[]
for x in range(N):
    for y in range(N):
        nb.append([((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N])

def run_wolff(T, nsteps, nburn, seed, start=1):
    rng=random.Random(seed)
    rnd=rng.random; rint=rng.randrange
    s=[start]*NS if start in (1,-1) else [1 if rnd()<0.5 else -1 for _ in range(NS)]
    padd=1.0-math.exp(-2.0/T)
    M=sum(s)
    out=np.empty(nsteps, dtype=np.int32)
    total=nburn+nsteps
    for step in range(total):
        i=rint(NS)
        sg=s[i]
        s[i]=-sg
        stack=[i]
        size=1
        while stack:
            k=stack.pop()
            for j in nb[k]:
                if s[j]==sg and rnd()<padd:
                    s[j]=-sg
                    stack.append(j)
                    size+=1
        M-=2*sg*size
        if step>=nburn:
            out[step-nburn]=M
    return out

def U_of(M):
    m=M/NS
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2**2)

Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    seed=1010+k
    out=run_wolff(T, 1000000, 5000, seed, start=1)
    np.save(f'/work/M_T{T:.2f}_cold.npy', out)
    nbins=1000
    L=len(out)//nbins
    Ub=np.array([U_of(out[j*L:(j+1)*L]) for j in range(nbins)])
    U=U_of(out)
    G=100; gl=len(out)//G
    jkg=np.array([U_of(np.delete(out, slice(g*gl,(g+1)*gl))) for g in range(G)])
    err_jk=np.sqrt((G-1)/G*np.sum((jkg-jkg.mean())**2))
    res[T]=dict(U=float(U), err_binstd=float(np.std(Ub,ddof=1)/np.sqrt(nbins)), err_jk100=float(err_jk), seed=seed, n=len(out), mabs=float(np.mean(np.abs(out))/NS))
    print(T, res[T], "t=",round(time.time()-t0,1), flush=True)
json.dump(res, open('/work/res_part2.json','w'), indent=1)
