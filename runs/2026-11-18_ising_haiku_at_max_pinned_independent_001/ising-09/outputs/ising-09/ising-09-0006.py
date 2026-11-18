
import sys, os, time
sys.path.insert(0, '/work')
import numpy as np
import wolff
L=16; N2=L*L; NSTEPS=500000; NBURN=10000
TEMPS=[round(1.5+0.1*k,1) for k in range(14)]
def seed_for(T,c): return 100000 + int(round(T*10))*100 + c
def fname(T,c): return f"/work/data/M_T{T:.1f}_c{c}.npy"
tasks=[]
for T in TEMPS:
    if T>=2.4: cs=range(8)
    elif T>=2.2: cs=range(4)
    else: cs=[]
    for c in cs: tasks.append((T,c))
t_start=time.time(); done=[]
for T,c in tasks:
    if os.path.exists(fname(T,c)): continue
    if time.time()-t_start > 230: break
    t0=time.time()
    M,_=wolff.wolff_run(L,T,NSTEPS,seed=seed_for(T,c),nburn=NBURN)
    np.save(fname(T,c), M.astype(np.int16))
    m=M/N2; m2=m*m
    print(f"T={T} chain={c} time={time.time()-t0:.1f}s U_chain={1-(m2*m2).mean()/(3*m2.mean()**2):.5f}", flush=True)
print("production elapsed", round(time.time()-t_start,1))

# ---- independent check: checkerboard Metropolis, vectorized over replicas ----
def metropolis_U(T, R=2000, nburn=300, nsweep=700, seed=777):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8), size=(R,L,L))
    ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    black=((ii+jj)%2==0)
    acc_tab={d: np.exp(-d/T) for d in (4,8)}
    def half(s, mask):
        h = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE = 2*s*h   # energy change for flipping s (J=1)
        u = rng.random(s.shape)
        p = np.where(dE<=0, 1.0, np.where(dE==4, acc_tab[4], np.where(dE==8, acc_tab[8], 0.0)))
        flip = mask[None,:,:] & (u < p)
        s[flip] = -s[flip]
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for sw in range(nburn+nsweep):
        half(s, black); half(s, ~black)
        if sw>=nburn:
            M=s.sum(axis=(1,2)).astype(np.float64)/N2
            m2=M*M
            S2+=m2; S4+=m2*m2; n+=1
    a2=S2/n; a4=S4/n            # per-replica time averages (replicas are independent)
    U=1-a4.mean()/(3*a2.mean()**2)
    Uj=np.empty(R)
    tot2=a2.sum(); tot4=a4.sum()
    for r in range(R):
        b2=(tot2-a2[r])/(R-1); b4=(tot4-a4[r])/(R-1)
        Uj[r]=1-b4/(3*b2*b2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U, err, a2.mean()

for T in [2.5, 2.8]:
    t0=time.time()
    U,err,m2=metropolis_U(T)
    print(f"Metropolis check T={T}: U={U:.5f} +- {err:.5f} <m^2>={m2:.5f} time {time.time()-t0:.1f}s", flush=True)
