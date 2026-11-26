import numpy as np, time, json, os
os.makedirs('/work', exist_ok=True)
N=16; R=1000
def run(T, nsweep, neq, seed, R=R):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    # acceptance table for dE = 2*s*nbsum, nbsum in {-4,-2,0,2,4}
    acc={}
    for dE in [-8,-4,0,4,8]:
        pass
    probs=np.exp(-np.arange(-16,17,4)*0.0)  # unused
    ms=[];m2=[];m4=[]
    def sweep(s):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb  # energy change for flipping s (E=-sum ss)
            p=np.exp(-dE/T)
            r=rng.random(s.shape)
            flip=(r<p)&m[None]
            s[flip]*=-1
        return s
    tot=neq+nsweep
    out=np.zeros((nsweep,3))
    for t in range(tot):
        sweep(s)
        if t>=neq:
            mag=s.reshape(R,-1).mean(1,dtype=np.float64)
            a=mag**2; out[t-neq]=[a.mean(), (a**2).mean(), np.abs(mag).mean()]
    return out
if __name__=="__main__":
    for T in [1.9,2.3]:
        t0=time.time()
        o=run(T,100,200,1)
        m2=o[:,0].mean(); m4=o[:,1].mean()
        print(T, time.time()-t0, 1-m4/(3*m2**2))