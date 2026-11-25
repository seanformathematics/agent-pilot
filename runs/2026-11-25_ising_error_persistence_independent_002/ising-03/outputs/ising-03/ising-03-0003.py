
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks()

def run(T, R, nsweep, nburn, seed, meas_every=1, ):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    # acceptance table: dE = 2*s*nb, nb sum of 4 neighbours
    acc=np.zeros(9)  # index dE+8 for dE in -8..8 (only positive dE needs exp)
    for dE in range(-8,9):
        acc[dE+8]=1.0 if dE<=0 else np.exp(-dE/T)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    def half(s,mask):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=(2*s.astype(np.int16)*nb)
        p=acc[(dE+8)]
        r=rng.random(s.shape)
        flip=(r<p)&mask
        s[flip]*=-1  # int8 multiply
    for t in range(nburn+nsweep):
        half(s,MA); half(s,MB)
        if t>=nburn and (t-nburn)%meas_every==0:
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m
            S2+=m2; S4+=m2*m2; cnt+=1
    return S2,S4,cnt

def binder(S2,S4,cnt):
    R=len(S2)
    def U_of(idx):
        a=S2[idx].sum()/ (cnt*len(idx)); b=S4[idx].sum()/(cnt*len(idx))
        return 1-b/(3*a*a)
    U=U_of(np.arange(R))
    # jackknife over replicas
    jk=np.array([U_of(np.delete(np.arange(R),k)) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err

if __name__=='__main__':
    t0=time.time()
    S2,S4,c=run(2.269,256,20,0,1)
    print('bench 20 sweeps R=256', time.time()-t0)
    t0=time.time()
    S2,S4,c=run(2.269,512,200,200,2)
    print('bench 200 sweeps R=512 time', time.time()-t0)
    print(binder(S2,S4,c))
