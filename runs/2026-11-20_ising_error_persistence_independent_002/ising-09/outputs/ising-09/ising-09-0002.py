import numpy as np, time, os
os.makedirs("/work",exist_ok=True)
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
MA,MB=make_masks()

def run(T, R, n_eq, n_meas, seed, start='hot'):
    rng=np.random.default_rng(seed)
    if start=='hot':
        s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    else:
        s=np.ones((R,N,N),dtype=np.int8)
    # acceptance table indexed by dE in {4,8}: p = exp(-dE/T)
    acc={d:np.exp(-d/T) for d in (4,8)}
    def half(s,mask):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn  # energy change for flipping s (E=-sum ss): dE=2*s*nn
        p=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
        u=rng.random((R,N,N))
        flip=(u<p)&mask
        s[flip]*=-1
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for sw in range(n_eq+n_meas):
        half(s,MA); half(s,MB)
        if sw>=n_eq:
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m
            S2+=m2; S4+=m2*m2; cnt+=1
    return S2/cnt, S4/cnt

t=time.time()
m2,m4=run(2.269,500,50,200,1)
print("time 250 sweeps R=500:",time.time()-t)
print(m2.mean(), m4.mean())
