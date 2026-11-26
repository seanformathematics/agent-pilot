
import numpy as np, time, os, json
N=16
def run(T, R, n_eq, n_meas, every, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    a4 = np.exp(-4/T); a8 = np.exp(-8/T)
    ii, jj = np.indices((N,N))
    masks = [((ii+jj)%2==p) for p in (0,1)]
    def sweep(s):
        for p in (0,1):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nb
            u = rng.random(s.shape)
            prob = np.where(dE==4, a4, np.where(dE==8, a8, 1.0))
            flip = masks[p] & (u < prob)
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s = sweep(s)
    # two halves to check stationarity
    res = {}
    halves = []
    for h in range(2):
        M2 = np.zeros(R); M4=np.zeros(R); cnt=0
        for k in range(n_meas*every//2):
            s = sweep(s)
            if (k+1)%every==0:
                m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
                M2 += m*m; M4 += m**4; cnt+=1
        halves.append((M2/cnt, M4/cnt))
    M2 = (halves[0][0]+halves[1][0])/2; M4=(halves[0][1]+halves[1][1])/2
    return M2, M4, halves

os.makedirs("/work/data", exist_ok=True)
for i,T in enumerate([1.9, 2.0, 2.1]):
    t=time.time()
    M2,M4,halves = run(T, 1000, 3000, 2000, 10, 1000+i)
    np.savez(f"/work/data/T{T}.npz", M2=M2, M4=M4, h0_2=halves[0][0], h0_4=halves[0][1], h1_2=halves[1][0], h1_4=halves[1][1])
    U = 1-M4.mean()/(3*M2.mean()**2)
    print(T, "U=",U, "halfU:", 1-halves[0][1].mean()/(3*halves[0][0].mean()**2), 1-halves[1][1].mean()/(3*halves[1][0].mean()**2), "time", time.time()-t, flush=True)
