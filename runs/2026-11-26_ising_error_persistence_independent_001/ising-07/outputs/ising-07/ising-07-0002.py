import numpy as np, time
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
def run(T,R,seed,nburn,nmeas,start='random'):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    m0,m1=make_masks()
    # acceptance table for dE = 2*s*h, h in {-4,-2,0,2,4}; dE in {-8..8}
    acc={dE:min(1.0,np.exp(-dE/T)) for dE in range(-8,9,4)}
    tab=np.zeros(17)
    for dE in range(-8,9,4): tab[dE+8]=acc[dE]
    def sweep(s):
        for mask in (m0,m1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s.astype(np.int16)*h
            p=tab[dE+8]
            r=rng.random(s.shape)
            flip=mask & (r<p)
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R); mabs=np.zeros(R)
    series=[]
    for _ in range(nmeas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        m2+=m*m; m4+=m**4
    return m2/nmeas, m4/nmeas
t=time.time()
a,b=run(2.27,1000,1,10,20)
print('time per sweep',(time.time()-t)/30)
print(a.mean(), b.mean())