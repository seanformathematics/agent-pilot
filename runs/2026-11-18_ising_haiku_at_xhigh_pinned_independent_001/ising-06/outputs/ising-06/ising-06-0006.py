import numpy as np, time, json, os
N=16
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks = [((ii+jj)%2==c) for c in (0,1)]
def make_table(T):
    tab = np.zeros(9, dtype=np.float32)
    for p in range(-4,5):
        tab[p+4] = 1.0 if p<=0 else np.exp(-2.0*p/T)
    return tab
def simulate(T, R, burn, meas, seed, nblk=10):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((R,N,N))<0.5, -1, 1).astype(np.int8)
    tab = make_table(T)
    def sweep(s):
        for c in (0,1):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            pt = tab[(s*nb + 4)]
            r = rng.random((R,N,N), dtype=np.float32)
            flip = (r < pt) & masks[c]
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(burn):
        s = sweep(s)
    S2 = np.zeros(R); S4=np.zeros(R)
    per = meas//nblk
    for sw in range(meas):
        s = sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2=m*m; m4=m2*m2
        S2+=m2; S4+=m4
    return S2, S4
def estimates(S2, S4, meas):
    R=len(S2)
    S2n=S2/meas; S4n=S4/meas
    U = 1 - S4n.mean()/(3*S2n.mean()**2)
    tot2=S2n.sum(); tot4=S4n.sum()
    loo2=(tot2-S2n)/(R-1); loo4=(tot4-S4n)/(R-1)
    Uloo = 1 - loo4/(3*loo2**2)
    jk = (R-1)/R*np.sum((Uloo-Uloo.mean())**2)
    return U, np.sqrt(jk)

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=1000; burn=1000; meas=5000
out = {}
if os.path.exists('/work/resB.json'): out=json.load(open('/work/resB.json'))
t00=time.time()
for T in Ts:
    key=str(T)
    if key in out: continue
    seed = 5000+int(round(T*100))
    t0=time.time()
    S2,S4 = simulate(T,R,burn,meas,seed)
    U,err = estimates(S2,S4,meas)
    out[key]=dict(T=T,U=float(U),err=float(err),seed=seed)
    json.dump(out, open('/work/resB.json','w'), indent=1)
    print(T, U, err, round(time.time()-t0,1), flush=True)
    if time.time()-t00 > 500: break
print("done", time.time()-t00)
