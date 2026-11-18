
import numpy as np, time, json
exec(open("/work/ising.py").read())
def run_blocks(T, R, N, nburn, nblocks, nper, nsep, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    acc = np.zeros(17, dtype=np.float32)
    for dE in (-8,-4,0,4,8):
        acc[dE+8] = min(1.0, np.exp(-dE/T))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==p) for p in (0,1)]
    def sweep(s):
        for p in (0,1):
            nn = (np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2)).astype(np.int8)
            dE = (2*s*nn).astype(np.int8)
            pr = acc[dE.astype(np.int16)+8]
            u = rng.random(s.shape, dtype=np.float32)
            flip = (u < pr) & masks[p][None]
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nburn): s = sweep(s)
    out2=[];out4=[]
    for b in range(nblocks):
        S2=np.zeros(R);S4=np.zeros(R)
        for k in range(nper):
            for _ in range(nsep): s=sweep(s)
            m = s.reshape(R,-1).mean(axis=1,dtype=np.float64); m2=m*m
            S2+=m2; S4+=m2*m2
        out2.append(S2/nper); out4.append(S4/nper)
    return np.array(out2), np.array(out4)
def U_jk(m2,m4):
    # m2,m4: arrays over replicas (R,)
    R=len(m2)
    U = 1-np.mean(m4)/(3*np.mean(m2)**2)
    tot2=m2.sum(); tot4=m4.sum()
    jk = 1-((tot4-m4)/(R-1))/(3*((tot2-m2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U, err
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
R=512; res={}
t0=time.time()
for i,T in enumerate(Ts):
    a2,a4 = run_blocks(T, R, 16, 2000, 4, 500, 10, 1000+i)
    m2=a2.mean(0); m4=a4.mean(0)
    U,err = U_jk(m2,m4)
    # block consistency: U from first half vs second half
    Uh1,_ = U_jk(a2[:2].mean(0),a4[:2].mean(0)); Uh2,_=U_jk(a2[2:].mean(0),a4[2:].mean(0))
    res[T]=dict(U=float(U),err=float(err),seed=1000+i,halves=[float(Uh1),float(Uh2)])
    np.savez(f"/work/raw_T{T}.npz", m2=m2, m4=m4)
    print(T, U, err, Uh1, Uh2, "elapsed", round(time.time()-t0,1), flush=True)
json.dump(res, open("/work/res_part1.json","w"), indent=1)
