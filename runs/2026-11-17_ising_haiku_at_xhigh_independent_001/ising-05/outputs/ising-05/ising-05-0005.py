
import os, time, math, random, json
import numpy as np
os.makedirs("/work", exist_ok=True)

def build(N):
    nb=[]
    for x in range(N):
        for y in range(N):
            nb.append((((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N))
    return nb

def run_wolff(N, T, nmeas, seed, nburn):
    rng=random.Random(seed)
    nb=build(N); V=N*N
    spins=[1 if rng.random()<0.5 else -1 for _ in range(V)]
    M=sum(spins)
    p=1.0-math.exp(-2.0/T)
    rnd=rng.random; rint=rng.randrange
    out=np.empty(nmeas, dtype=np.float64)
    for it in range(nburn+nmeas):
        i=rint(V); s0=spins[i]; spins[i]=-s0
        stack=[i]; c=1
        while stack:
            j=stack.pop()
            for k in nb[j]:
                if spins[k]==s0 and rnd()<p:
                    spins[k]=-s0; stack.append(k); c+=1
        M-=2*s0*c
        if it>=nburn:
            out[it-nburn]=M/V
    return out

def U_of(m):
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2*m2)

def analyze(chains, nbins=50):
    allm=np.concatenate(chains)
    U=U_of(allm)
    Uc=np.array([U_of(c) for c in chains])
    err_chain=Uc.std(ddof=1)/np.sqrt(len(chains))
    blocks=[]
    for c in chains:
        bl=len(c)//nbins
        blocks.append(c[:bl*nbins].reshape(nbins,bl))
    blocks=np.concatenate(blocks,axis=0)
    nb_=blocks.shape[0]
    s2=(blocks**2).sum(1); s4=(blocks**4).sum(1); cnt=blocks.shape[1]
    S2=s2.sum(); S4=s4.sum(); Ntot=cnt*nb_
    Ujk=np.empty(nb_)
    for b in range(nb_):
        m2=(S2-s2[b])/(Ntot-cnt); m4=(S4-s4[b])/(Ntot-cnt)
        Ujk[b]=1-m4/(3*m2*m2)
    err_jk=math.sqrt((nb_-1)/nb_*np.sum((Ujk-Ujk.mean())**2))
    return dict(U=float(U), err_jk=float(err_jk), err_chain=float(err_chain),
                Uchains=[float(x) for x in Uc], nsamp=int(len(allm)),
                mabs=float(np.mean(np.abs(allm))))

Ts=[1.5,1.6,1.7,1.8,1.9,2.0]
tag="part1"
nchain=8; nmeas=100000; nburn=5000
res={}
t0=time.time()
for T in Ts:
    chains=[]
    seeds=[]
    for ch in range(nchain):
        seed=int(round(T*1000))*100+ch+7
        seeds.append(seed)
        chains.append(run_wolff(16,T,nmeas,seed,nburn))
    np.savez_compressed(f"/work/m_T{T:.2f}.npz", *[c.astype(np.float32) for c in chains])
    r=analyze(chains)
    r["seeds"]=seeds
    res[f"{T:.2f}"]=r
    print(f"T={T:.2f} U={r['U']:.5f} err_jk={r['err_jk']:.5f} err_chain={r['err_chain']:.5f} <|m|>={r['mabs']:.4f} time={time.time()-t0:.1f}", flush=True)
json.dump(res, open(f"/work/res_{tag}.json","w"), indent=1)
