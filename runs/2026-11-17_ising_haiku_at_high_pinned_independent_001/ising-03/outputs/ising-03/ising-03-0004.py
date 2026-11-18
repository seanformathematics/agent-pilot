
import numpy as np, time
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
    out2=[];out4=[];outE=[]
    for b in range(nblocks):
        S2=np.zeros(R);S4=np.zeros(R);SE=np.zeros(R)
        for k in range(nper):
            for _ in range(nsep): s=sweep(s)
            m = s.reshape(R,-1).mean(axis=1,dtype=np.float64); m2=m*m
            S2+=m2; S4+=m2*m2
        out2.append(S2/nper); out4.append(S4/nper)
    return np.array(out2), np.array(out4)
def U_of(m2,m4):
    return 1 - np.mean(m4,axis=-1)/(3*np.mean(m2,axis=-1)**2)
t=time.time()
a2,a4 = run_blocks(2.27, 1024, 16, 1000, 8, 250, 10, 12345)
print("time", time.time()-t)
Ub = 1 - a4/(3*a2**2)  # per replica per block
for b in range(8):
    print(b, np.mean(Ub[b]), np.std(Ub[b])/np.sqrt(1024))
# jackknife-free: replica-level U using totals
m2 = a2.mean(0); m4=a4.mean(0)
U = 1-np.mean(m4)/(3*np.mean(m2)**2)
# replica-level jackknife
R=1024
jk=[]
for r in range(R):
    mask=np.ones(R,bool); mask[r]=False
    jk.append(1-np.mean(m4[mask])/(3*np.mean(m2[mask])**2))
jk=np.array(jk)
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print("U",U,"err",err)
