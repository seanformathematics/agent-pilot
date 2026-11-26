
import random, math, time, os, json
os.makedirs("/work", exist_ok=True)
def wolff_run(N, T, nclus, seed, nbin=1000, burn=2000):
    rng = random.Random(seed)
    V = N*N
    nb = [0]*(4*V)
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[4*i+0] = ((x+1)%N)*N + y
            nb[4*i+1] = ((x-1)%N)*N + y
            nb[4*i+2] = x*N + (y+1)%N
            nb[4*i+3] = x*N + (y-1)%N
    s = [1]*V
    M = V  # total spin sum
    padd = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    out = []  # per bin: sums of m^2, m^4, m, count
    acc2=acc4=accm=0.0; cnt=0
    total = burn + nclus
    for step in range(total):
        i0 = int(rnd()*V)
        sd = s[i0]
        s[i0] = -sd
        stack = [i0]
        size = 1
        while stack:
            j = stack.pop()
            for k in (nb[4*j], nb[4*j+1], nb[4*j+2], nb[4*j+3]):
                if s[k] == sd and rnd() < padd:
                    s[k] = -sd
                    stack.append(k)
                    size += 1
        M += -2*sd*size
        if step >= burn:
            m = M/V
            m2 = m*m
            acc2 += m2; acc4 += m2*m2; accm += m; cnt += 1
            if cnt == nbin:
                out.append((acc2, acc4, accm, cnt))
                acc2=acc4=accm=0.0; cnt=0
    return out

t=time.time()
o = wolff_run(16, 2.3, 20000, 1)
print("time", time.time()-t, len(o))
