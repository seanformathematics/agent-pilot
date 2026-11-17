import numpy as np
R,L=1000,16
ii,jj=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed,burn,meas):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,L,L))
    p={d:np.exp(-d/T) for d in (4,8)}
    def sweep():
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            r=rng.random(s.shape,dtype=np.float32)
            acc=((dE<=0)|((dE==4)&(r<p[4]))|((dE==8)&(r<p[8])))&m
            s[acc]*=-1
    for _ in range(burn): sweep()
    m2=np.zeros(R); m4=np.zeros(R)
    for _ in range(meas):
        sweep(); mm=s.reshape(R,-1).mean(1); m2+=mm**2; m4+=mm**4
    m2/=meas; m4/=meas
    return 1-m4.mean()/(3*m2.mean()**2)
for T in [2.3,2.4]:
    print(T, "burn10000 meas2000 U=", round(run(T,777,10000,2000),5), flush=True)
