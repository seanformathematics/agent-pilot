
import random, math, time, json
import numpy as np
def run(N, T, nsteps, seed):
    rng = random.Random(seed)
    V = N*N
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
    out = np.empty((nsteps,2))
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
        out[step,0]=mm*mm; out[step,1]=mm**4
    return out
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    t=time.time()
    d=run(16,T,400000,1000+k)
    np.save(f'/work/pilot_T{T}.npy',d)
    d=d[20000:]
    m2=d[:,0].mean(); m4=d[:,1].mean()
    U=1-m4/(3*m2*m2)
    # blocked error
    nb_=50; L=len(d)//nb_
    Ub=[]
    for b in range(nb_):
        seg=d[b*L:(b+1)*L]
        a2=seg[:,0].mean(); a4=seg[:,1].mean()
        Ub.append(1-a4/(3*a2*a2))
    Ub=np.array(Ub)
    err=Ub.std(ddof=1)/math.sqrt(nb_)
    print(T, round(U,5), 'blockerr',round(err,5), 'time',round(time.time()-t,1), flush=True)
print('total',time.time()-t0)
