
import random, math, time
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
    s = [-1]*V if T<2.269 else [1 if rng.random()<0.5 else -1 for _ in range(V)]
    M = sum(s)
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
def jk_U(d, nb_=100):
    L=len(d)//nb_
    blocks=[d[b*L:(b+1)*L] for b in range(nb_)]
    S2=np.array([b[:,0].sum() for b in blocks]); S4=np.array([b[:,1].sum() for b in blocks]); n=np.array([len(b) for b in blocks])
    T2=S2.sum(); T4=S4.sum(); NT=n.sum()
    Ujk=[]
    for b in range(nb_):
        a2=(T2-S2[b])/(NT-n[b]); a4=(T4-S4[b])/(NT-n[b])
        Ujk.append(1-a4/(3*a2*a2))
    Ujk=np.array(Ujk)
    U=1-(T4/NT)/(3*(T2/NT)**2)
    err=math.sqrt((nb_-1)/nb_*np.sum((Ujk-Ujk.mean())**2))
    return U,err
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for k,T in enumerate(Ts):
    reps=[]
    for seed in [2000+k,3000+k]:
        reps.append(run(16,T,400000,seed))
    pil=np.load(f'/work/pilot_T{T}.npy')[20000:]
    reps.append(pil)
    # replicas: estimate U per replica and spread
    Us=[]
    for r in reps:
        r2 = r[20000:] if r is not reps[-1] else r
        Us.append(jk_U(r2)[0])
    Ua=np.array(Us)
    # combined jackknife over all data (concatenate)
    comb=np.vstack([r[20000:] for r in reps[:2]]+[reps[2]])
    Uc,ec=jk_U(comb,150)
    print(T,'U_comb',round(Uc,5),'jk_err',round(ec,5),'replica Us',np.round(Ua,5),'replica spread/sqrt(3)',round(Ua.std(ddof=1)/math.sqrt(3),5),flush=True)
    np.save(f'/work/comb_T{T}.npy',comb)
