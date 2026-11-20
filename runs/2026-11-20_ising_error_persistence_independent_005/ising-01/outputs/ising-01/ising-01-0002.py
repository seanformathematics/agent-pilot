import random, math, time
N=16; L=N*N
nb=[[((i//N+1)%N)*N+(i%N), ((i//N-1)%N)*N+(i%N), (i//N)*N+(i%N+1)%N, (i//N)*N+(i%N-1)%N] for i in range(L)]
def run(T, nsteps, seed):
    rnd=random.Random(seed)
    s=[1]*L
    p=1-math.exp(-2.0/T)
    rr=rnd.random; ri=rnd.randrange
    sizes=0; ms=[]
    t0=time.time()
    for _ in range(nsteps):
        i=ri(L); si=s[i]
        stack=[i]; s[i]=-si; cl=[i]
        while stack:
            j=stack.pop()
            for k in nb[j]:
                if s[k]==si and rr()<p:
                    s[k]=-si; stack.append(k); cl.append(k)
        sizes+=len(cl)
    return time.time()-t0, sizes/nsteps
print(run(2.27,20000,1))
print(run(1.9,20000,2))
print(run(2.4,20000,3))