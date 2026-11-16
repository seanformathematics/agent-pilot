
import numpy as np, time, sys
N=16
def run(T, R=1000, sweeps=3000, burn=1000, seed=1):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    # acceptance for dE=2*s*nn, nn in {-4,-2,0,2,4}
    acc={}
    for dE in [4,8]:
        acc[dE]=np.exp(-dE/T)
    s1=np.zeros(R); s2=np.zeros(R); s4=np.zeros(R); cnt=0
    for sw in range(sweeps):
        for m in masks:
            nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nn  # in {-8,-4,0,4,8}
            p=np.ones_like(dE,dtype=np.float64)
            p=np.where(dE==4,acc[4],p)
            p=np.where(dE==8,acc[8],p)
            u=rng.random(s.shape)
            flip=m[None]&((dE<=0)|(u<p))
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=burn:
            mag=s.reshape(R,-1).mean(axis=1)
            m2=mag*mag; s1+=m2; s2+=m2*m2; cnt+=1
    m2=s1/cnt; m4=s2/cnt  # per replica averages
    return m2,m4
if __name__=="__main__":
    t=time.time()
    m2,m4=run(2.269,R=200,sweeps=200,burn=50,seed=5)
    print(time.time()-t, m2.mean(), m4.mean())
