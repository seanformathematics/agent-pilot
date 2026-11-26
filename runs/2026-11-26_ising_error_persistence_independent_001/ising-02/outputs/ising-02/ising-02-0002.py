import random, math, time
N=16; V=N*N
nb=[[((i//N+1)%N)*N+i%N, ((i//N-1)%N)*N+i%N, (i//N)*N+(i%N+1)%N, (i//N)*N+(i%N-1)%N] for i in range(V)]
def run(T, nsteps, seed):
    rng=random.Random(seed)
    s=[1]*V
    p=1-math.exp(-2.0/T)
    rnd=rng.random; ch=rng.choice
    tot=0
    for _ in range(nsteps):
        i=int(rnd()*V)
        si=s[i]
        stack=[i]; s[i]=-si; size=1
        while stack:
            j=stack.pop()
            for k in nb[j]:
                if s[k]==si and rnd()<p:
                    s[k]=-si; stack.append(k); size+=1
        tot+=size
    return tot/nsteps
t=time.time(); print(run(2.269,20000,1), time.time()-t)
