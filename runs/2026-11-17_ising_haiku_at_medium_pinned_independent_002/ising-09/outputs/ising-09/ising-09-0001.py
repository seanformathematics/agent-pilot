
import time, math, random
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
N=16
def wolff_run(T, nsweeps, seed):
    rng = random.Random(seed)
    p = 1-math.exp(-2.0/T)
    s = [1]*(N*N)
    nb = []
    for i in range(N):
        for j in range(N):
            nb.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])
    out=[]
    for _ in range(nsweeps):
        seed_site = rng.randrange(N*N)
        st=s[seed_site]; s[seed_site]=-st
        stack=[seed_site]; size=1
        while stack:
            k=stack.pop()
            for n in nb[k]:
                if s[n]==st and rng.random()<p:
                    s[n]=-st; stack.append(n); size+=1
        out.append(sum(s))
    return out
t=time.time(); r=wolff_run(2.3,20000,1); print("time per 20000 clusters", time.time()-t)
