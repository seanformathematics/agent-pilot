
import sys, time, json
sys.path.insert(0,'/work')
import numpy as np, run

N=16
def metro_run(N, R, T, seed, nsweeps, nth, start):
    rng = np.random.default_rng(seed)
    if start=='cold':
        s = np.ones((R,N,N), np.int8)
    else:
        s = (rng.integers(0,2,size=(R,N,N),dtype=np.int8)*2-1).astype(np.int8)
    ii,jj = np.indices((N,N)); par = (ii+jj)&1
    masks = [(par==0)[None], (par==1)[None]]
    table = np.minimum(1.0, np.exp(-np.arange(-8,9,4)/T))
    m2sum = np.zeros(R); m4sum = np.zeros(R); n=0
    for it in range(nth+nsweeps):
        for p in (0,1):
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE = 2*s*nn
            prob = table[((dE+8)>>2)]
            flip = (rng.random(s.shape) < prob) & masks[p]
            s = np.where(flip, -s, s)
        if it >= nth:
            m = s.sum(axis=(1,2), dtype=np.int32)/(N*N)
            m2sum += m*m; m4sum += m**4; n+=1
    return m2sum/n, m4sum/n

# timing
t=time.time(); metro_run(N,256,2.3,1,50,0,'hot'); dt=(time.time()-t)/50
print(f"Metropolis: {dt*1e3:.2f} ms per sweep for R=256 ({256/dt:.0f} replica-sweeps/s)", flush=True)

prod = {o['T']:o for o in json.load(open('/work/final_results.json'))['points']}
R=256
budget=70.0   # seconds per temperature
cases=[(1.5,'cold'),(2.0,'cold'),(2.3,'hot'),(2.8,'hot')]
for k,(T,start) in enumerate(cases):
    nsw=int(budget/dt); nth=500
    t=time.time()
    a,b = metro_run(N,R,T,555000+k,nsw,nth,start)
    # jackknife over replicas
    Rn=len(a); sA,sB=a.sum(),b.sum()
    U=run.U_from(sA/Rn,sB/Rn)
    Uj=run.U_from((sA-a)/(Rn-1),(sB-b)/(Rn-1))
    err=np.sqrt((Rn-1)/Rn*np.sum((Uj-Uj.mean())**2))
    Usw=prod[T]['U']; esw=prod[T]['U_err']
    z=(U-Usw)/np.hypot(err,esw)
    print(f"T={T} ({start}, {nsw} sweeps x {R} replicas, {time.time()-t:.0f}s): U_metro={U:.6f}+-{err:.6f}   U_SW={Usw:.6f}+-{esw:.6f}   z={z:+.2f}", flush=True)
