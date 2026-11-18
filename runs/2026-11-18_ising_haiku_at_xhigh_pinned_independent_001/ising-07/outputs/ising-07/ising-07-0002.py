
import random, math, time
import numpy as np

N=16
NS=N*N
nb=[]
for x in range(N):
    for y in range(N):
        i=x*N+y
        nb.append([((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N])

def run_wolff(T, nsteps, nburn, seed):
    rng=random.Random(seed)
    rnd=rng.random; rint=rng.randrange
    s=[1]*NS  # ordered start (cold); burn-in will equilibrate
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

t=time.time()
out=run_wolff(2.269, 20000, 2000, 1)
print("time 20000 clusters", time.time()-t)
m=out/NS
print("mean |m|", np.mean(np.abs(m)), "U", 1-np.mean(m**4)/(3*np.mean(m**2)**2))
