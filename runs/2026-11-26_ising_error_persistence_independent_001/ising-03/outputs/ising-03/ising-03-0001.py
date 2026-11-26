
import random, math, time
def run(N, T, nclusters, seed):
    rng = random.Random(seed)
    L = N*N
    nb = []
    for i in range(L):
        x, y = i % N, i // N
        nb.append([((x+1)%N)+y*N, ((x-1)%N)+y*N, x+((y+1)%N)*N, x+((y-1)%N)*N])
    s = [rng.choice((-1,1)) for _ in range(L)]
    M = sum(s)
    p = 1 - math.exp(-2.0/T)
    rnd = rng.random
    t0=time.time()
    tot=0
    for _ in range(nclusters):
        i = int(rnd()*L)
        sg = s[i]
        stack=[i]; s[i]=-sg; size=1
        while stack:
            j = stack.pop()
            for k in nb[j]:
                if s[k]==sg and rnd()<p:
                    s[k]=-sg; stack.append(k); size+=1
        M -= 2*sg*size
        tot+=size
    return time.time()-t0, tot/nclusters
print(run(16,2.269,20000,1))
print(run(16,1.9,20000,2))
