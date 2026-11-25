
import random, math, time, json, sys
def run(N, T, nsteps, seed, nmeas_skip=1):
    rng = random.Random(seed)
    V = N*N
    # neighbor table
    nb = [[0]*4 for _ in range(V)]
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[i][0] = ((x+1)%N)*N+y
            nb[i][1] = ((x-1)%N)*N+y
            nb[i][2] = x*N+(y+1)%N
            nb[i][3] = x*N+(y-1)%N
    s = [1]*V
    M = V
    p = 1-math.exp(-2.0/T)
    rnd = rng.random
    m2s=[]; m4s=[]; ms=[]
    for step in range(nsteps):
        i0 = int(rnd()*V)
        sg = s[i0]
        s[i0] = -sg
        stack=[i0]; size=1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j]==sg and rnd()<p:
                    s[j]=-sg; stack.append(j); size+=1
        M -= 2*sg*size
        mm = M/V
        m2s.append(mm*mm); m4s.append(mm**4); ms.append(mm)
    return m2s,m4s,ms
t=time.time()
m2,m4,m=run(16,2.269,20000,1)
print('time per 20000 clusters',time.time()-t)
