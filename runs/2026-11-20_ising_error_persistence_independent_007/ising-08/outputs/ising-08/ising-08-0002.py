
import numpy as np, time, json
def run(T, R, nwarm, nmeas, seed):
    rng = np.random.default_rng(seed)
    N=16
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
    beta=1.0/T
    acc = {4: np.exp(-beta*4), 8: np.exp(-beta*8)}
    def sweep(s):
        for m in masks:
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nn
            p = np.where(dE<=0, 1.0, np.where(dE==4, acc[4], np.where(dE==8, acc[8], 0.0)))
            flip = (rng.random(s.shape) < p) & m
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nwarm):
        s = sweep(s)
    H=nmeas//2
    out={}
    for half in range(2):
        M2=np.zeros(R); M4=np.zeros(R); M1=np.zeros(R)
        for _ in range(H):
            s = sweep(s)
            m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
            M2 += m*m; M4 += m**4; M1+=np.abs(m)
        out[f"M2_{half}"]=(M2/H).tolist(); out[f"M4_{half}"]=(M4/H).tolist(); out[f"M1_{half}"]=(M1/H).tolist()
    return out
def U_of(M2,M4):
    return 1-np.mean(M4)/(3*np.mean(M2)**2)
def jk(M2,M4):
    R=len(M2); Us=np.empty(R)
    S2=M2.sum(); S4=M4.sum()
    for k in range(R):
        a=(S2-M2[k])/(R-1); b=(S4-M4[k])/(R-1)
        Us[k]=1-b/(3*a*a)
    return U_of(M2,M4), np.sqrt((R-1)/R*np.sum((Us-Us.mean())**2))
res={}
t0=time.time()
for i,T in enumerate([1.9,2.0,2.1]):
    o=run(T, 2000, 2000, 4000, 1000+i)
    M2=(np.array(o["M2_0"])+np.array(o["M2_1"]))/2
    M4=(np.array(o["M4_0"])+np.array(o["M4_1"]))/2
    U,e=jk(M2,M4)
    U0,e0=jk(np.array(o["M2_0"]),np.array(o["M4_0"]))
    U1,e1=jk(np.array(o["M2_1"]),np.array(o["M4_1"]))
    res[T]=dict(U=U,err=e,U_h0=U0,U_h1=U1,err_h=e0)
    print(T, U, e, "halves", U0, U1, e0, time.time()-t0, flush=True)
    json.dump({str(k):v for k,v in res.items()}, open("/work/res_a.json","w")) if True else None
