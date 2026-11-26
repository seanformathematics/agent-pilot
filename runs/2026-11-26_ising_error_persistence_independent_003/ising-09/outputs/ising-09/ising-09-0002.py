import numpy as np, time
N=16
def run(T, R, nsweep, ntherm, seed, meas_every=1):
    rng=np.random.default_rng(seed)
    beta=1.0/T
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    # acceptance probs for h*s values: dE = 2*s*h, p = exp(-2 beta s h)
    acc=np.zeros((2,5))  # index by s (0:-1,1:+1), h index (h+4)/2
    def sweep(s):
        for c in (0,1):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            # flip prob: exp(-2 beta s h) for s*h>0 
            sh=(s*h).astype(np.float64)
            p=np.exp(-2*beta*sh)
            r=rng.random(s.shape)
            flip=(r<p)&masks[c]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(ntherm): s=sweep(s)
    m2=[];m4=[]
    nm=0
    for k in range(nsweep):
        s=sweep(s)
        if (k+1)%meas_every==0:
            m=s.reshape(R,-1).mean(1)
            m2.append(m*m); m4.append(m**4); nm+=1
    m2=np.array(m2); m4=np.array(m4)  # (nm,R)
    return m2,m4
t=time.time()
m2,m4=run(2.3,1000,200,200,1)
print("time",time.time()-t)
M2=m2.mean(); 
U=1-m4.mean()/(3*m2.mean()**2)
print("U",U, np.mean(m2))
