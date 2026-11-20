
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)

def run(T, R=1000, L=16, n_therm=500, n_meas=3000, seed=1, block=500):
    rng = np.random.default_rng(seed)
    S = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,L,L))
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    masks = [((ii+jj)%2==p) for p in (0,1)]
    # acceptance for dE = 4,8 (2*s*nn with nn in {-4..4}); dE<=0 always accepted
    acc = {}
    for dE in (4,8):
        acc[dE] = np.exp(-dE/T)
    def sweep():
        for p in (0,1):
            nn = (np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
            dE = 2*S*nn
            r = rng.random(S.shape)
            flip = masks[p] & ((dE<=0) | ((dE==4)&(r<acc[4])) | ((dE==8)&(r<acc[8])))
            S[flip] *= -1
    for _ in range(n_therm): sweep()
    A = np.zeros(R); B = np.zeros(R)
    blocks=[]
    cnt=0
    bA=np.zeros(R); bB=np.zeros(R); bn=0
    for t in range(n_meas):
        sweep()
        m = S.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        A += m2; B += m2*m2; cnt += 1
        bA += m2; bB += m2*m2; bn+=1
        if bn==block:
            blocks.append(1-(bB/bn)/(3*(bA/bn)**2)); bA[:]=0; bB[:]=0; bn=0
    A/=cnt; B/=cnt
    U = 1 - B.mean()/(3*A.mean()**2)
    return U, np.array(blocks)

t0=time.time()
U, bl = run(2.269, R=1000, n_therm=100, n_meas=200, seed=1, block=100)
print(time.time()-t0, U, bl.mean(axis=1))
