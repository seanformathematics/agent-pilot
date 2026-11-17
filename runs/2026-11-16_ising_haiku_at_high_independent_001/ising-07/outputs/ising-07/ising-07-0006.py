
import numpy as np, time, json, os
N=16
ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0 = ((ii+jj)%2==0); mask1 = ~mask0
def run(T, R, nequil, nmeas, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    tab = np.exp(-np.arange(-8,9)/T).astype(np.float32)
    def sweep(s):
        for mask in (mask0, mask1):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s*nb
            u = rng.random(s.shape, dtype=np.float32)
            acc = (dE <= 0) | (u < tab[np.clip(dE+8,0,16)])
            s[acc & mask] *= -1
    for _ in range(nequil): sweep(s)
    S2 = np.zeros(R); S4=np.zeros(R); S2h=np.zeros(R); S4h=np.zeros(R)
    for k in range(nmeas):
        sweep(s)
        m = s.sum(axis=(1,2))/(N*N)
        m2 = m*m
        S2 += m2; S4 += m2*m2
        if k==nmeas//2-1: S2h=S2.copy(); S4h=S4.copy()
    return S2, S4, nmeas, S2h, S4h, nmeas//2

def binder_jk(S2,S4,n):
    R=len(S2)
    def U(a2,a4,cnt):
        return 1 - (a4/cnt)/(3*(a2/cnt)**2)
    Utot = U(S2.sum(),S4.sum(),n*R)
    Ui = np.array([U(S2.sum()-S2[r], S4.sum()-S4[r], n*(R-1)) for r in range(R)])
    err = np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    return float(Utot), float(err)

temps=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
for idx,T in enumerate(temps):
    t=time.time()
    S2,S4,n,S2h,S4h,nh = run(T,512,3000,10000,seed=2000+idx)
    U,e = binder_jk(S2,S4,n)
    Uh,eh = binder_jk(S2h,S4h,nh)
    res[T]=dict(U=U,err=e,U_firsthalf=Uh,err_firsthalf=eh,seed=2000+idx,time=time.time()-t)
    print(T, res[T], flush=True)
json.dump(res,open('/work/part2.json','w'))
