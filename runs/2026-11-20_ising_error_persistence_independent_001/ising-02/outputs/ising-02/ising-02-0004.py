
import numpy as np, time, json

def run(T, R=1000, L=16, n_therm=2000, n_meas=10000, seed=1, block=1000):
    rng = np.random.default_rng(seed)
    S = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,L,L))
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    masks = [((ii+jj)%2==p) for p in (0,1)]
    acc4 = np.exp(-4.0/T); acc8 = np.exp(-8.0/T)
    def sweep():
        for p in (0,1):
            nn = (np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
            dE = 2*S*nn
            r = rng.random(S.shape)
            flip = masks[p] & ((dE<=0) | ((dE==4)&(r<acc4)) | ((dE==8)&(r<acc8)))
            S[flip] *= -1
    for _ in range(n_therm): sweep()
    A = np.zeros(R); B = np.zeros(R)
    for t in range(n_meas):
        sweep()
        m = S.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        A += m2; B += m2*m2
    A/=n_meas; B/=n_meas
    U = 1 - B.mean()/(3*A.mean()**2)
    sA=A.sum(); sB=B.sum()
    Uk = 1 - ((sB-B)/(R-1))/(3*((sA-A)/(R-1))**2)
    Uerr = np.sqrt((R-1)/R*np.sum((Uk-Uk.mean())**2))
    return float(U), float(Uerr)

out={}
t0=time.time()
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    U,Ue = run(T, seed=int(round(T*1000))+7777)
    out[T]=(U,Ue)
    print(T, repr(U), repr(Ue), round(time.time()-t0,1), flush=True)
json.dump({str(k):v for k,v in out.items()}, open('/work/run2.json','w'))
