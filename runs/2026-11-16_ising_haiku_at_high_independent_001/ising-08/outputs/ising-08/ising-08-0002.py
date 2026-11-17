
import numpy as np, time, os
code = r'''
import numpy as np, sys, time
N=16
def run(T, seed, R=400, nequil=2000, nmeas=20000):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    def sweep(S):
        for par in (0,1):
            nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
            dE=2*S*nn
            acc=(dE<=0)|(rng.random(S.shape)<np.exp(-np.clip(dE,0,None)/T))
            S[acc & masks[par]] *= -1
    for _ in range(nequil): sweep(S)
    m2s=np.zeros(R); m4s=np.zeros(R); ms=np.zeros(R)
    series=np.zeros((nmeas,R),dtype=np.float32)
    for t in range(nmeas):
        sweep(S)
        m=S.reshape(R,-1).mean(axis=1)
        m2s+=m*m; m4s+=m**4; ms+=np.abs(m)
        series[t]=m
    return m2s/nmeas, m4s/nmeas, ms/nmeas, series
'''
open('/work/ising.py','w').write(code)
exec(code)
t0=time.time()
m2,m4,am,series=run(2.3, seed=1001, R=400, nequil=2000, nmeas=10000)
print("time",time.time()-t0)
# jackknife over replicas
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
U=U_of(m2,m4)
R=len(m2)
jk=np.array([U_of(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print("U",U,"jk err",err, "<|m|>",am.mean())
# autocorrelation of |m| from series (replica-averaged)
x=np.abs(series).astype(float)
x=x-x.mean(axis=0)
ac=[]
for lag in [1,2,5,10,20,50,100,200]:
    c=np.mean(x[:-lag]*x[lag:])/np.mean(x*x)
    ac.append((lag,round(c,3)))
print(ac)
