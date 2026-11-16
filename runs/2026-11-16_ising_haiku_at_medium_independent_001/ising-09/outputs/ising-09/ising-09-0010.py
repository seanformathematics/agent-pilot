import numpy as np,time
N=16;R=1000;BURN=1000;SWEEPS=4000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T):
    rng=np.random.default_rng(20240601+int(round(T*100)))
    S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    def half(S,mask):
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE=2*S*nb
        prob=np.exp(-dE/T)
        u=rng.random(S.shape,dtype=np.float32)
        S[(u<prob)&mask]*=-1
    S2=np.zeros(R);S4=np.zeros(R);S1=np.zeros(R);n=0
    for sw in range(BURN+SWEEPS):
        half(S,masks[0]);half(S,masks[1])
        if sw>=BURN:
            m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m;S2+=m2;S4+=m2*m2;S1+=np.sqrt(m2);n+=1
    np.savez(f'/work/res_T{T:.2f}.npz',S2=S2/n,S4=S4/n,S1=S1/n,n=n,R=R)
t0=time.time()
for T in [2.5,2.6,2.7,2.8]:
    run(T);print(T,time.time()-t0,flush=True)