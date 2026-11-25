import numpy as np, time
N=16
def make(R, rng):
    return rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T,rng,acc_tab):
    R=S.shape[0]
    for m in masks:
        nn = (np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE = 2*S*nn  # energy change when flipping (in units of J)
        # acceptance: dE<=0 always; dE=4,8 -> exp(-dE/T)
        u = rng.random(S.shape, dtype=np.float32)
        acc = (u < acc_tab[(dE+8)//2]) & m
        S = np.where(acc, -S, S)
    return S
rng=np.random.default_rng(1)
R=2000
S=make(R,rng)
for T in [2.269]:
    acc_tab=np.zeros(9,dtype=np.float32)
    for k in range(9):
        d=2*k-8
        acc_tab[k]= 1.0 if d<=0 else np.exp(-d/T)
    t0=time.time()
    for it in range(200):
        S=sweep(S,T,rng,acc_tab)
    dt=time.time()-t0
    print("time per sweep (R=2000):",dt/200)
    m=S.reshape(R,-1).mean(1)
    print("mean|m|",np.abs(m).mean(), "U", 1-np.mean(m**4)/(3*np.mean(m**2)**2))