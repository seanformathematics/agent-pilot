
import numpy as np, time
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
    S2 = np.zeros(R); S4=np.zeros(R)
    for k in range(nmeas):
        sweep(s)
        m = s.sum(axis=(1,2))/(N*N)
        m2 = m*m
        S2 += m2; S4 += m2*m2
    return S2, S4, nmeas

def binder_jk(S2,S4,n):
    R=len(S2)
    def U(a2,a4,cnt):
        return 1 - (a4/cnt)/(3*(a2/cnt)**2)
    Utot = U(S2.sum(),S4.sum(),n*R)
    Ui = np.array([U(S2.sum()-S2[r], S4.sum()-S4[r], n*(R-1)) for r in range(R)])
    err = np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
    return Utot, err

t=time.time()
S2,S4,n = run(2.3, 512, 3000, 10000, seed=12345)
print("time",time.time()-t)
print("full: U,err", binder_jk(S2,S4,n))
