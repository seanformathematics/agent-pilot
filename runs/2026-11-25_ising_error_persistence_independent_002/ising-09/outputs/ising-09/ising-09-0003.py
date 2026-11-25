
script = r'''
import numpy as np, sys, time
N=16
i,j=np.indices((N,N))
MASKS=[((i+j)%2==0),((i+j)%2==1)]
def run(T,R,nsweep,nburn,seed,block=10,start='cold'):
    rng=np.random.default_rng(seed)
    if start=='cold': s=np.ones((R,N,N),dtype=np.int8)
    else: s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    thr=np.zeros(9); thr[4]=np.exp(-4/T); thr[8]=np.exp(-8/T)
    nb_blocks=nsweep//block
    M2=np.zeros((nb_blocks,R)); M4=np.zeros((nb_blocks,R))
    tot=nburn+nsweep
    bi=0; acc2=np.zeros(R); acc4=np.zeros(R); c=0
    for sw in range(tot):
        for p in (0,1):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            u=rng.random((R,N,N),dtype=np.float32)
            acc=((dE<=0)|(u<thr[np.clip(dE,0,8)])) & MASKS[p]
            s=np.where(acc,-s,s).astype(np.int8)
        if sw>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            acc2+=m*m; acc4+=m**4; c+=1
            if c==block:
                M2[bi]=acc2/block; M4[bi]=acc4/block; bi+=1
                acc2[:]=0; acc4[:]=0; c=0
    return M2,M4
'''
open('/work/sim.py','w').write(script)
exec(script)
import time
t=time.time()
M2,M4=run(2.3,256,3000,0,12345,block=10,start='cold')
print('time',time.time()-t)
# time series of replica-averaged m2 and m4 block means
ts=M2.mean(1)
x=ts-ts.mean()
def tau_int(x,maxlag=200):
    n=len(x); v=np.dot(x,x)/n
    rho=[np.dot(x[:n-k],x[k:])/(n-k)/v for k in range(maxlag)]
    rho=np.array(rho)
    # sokal window
    t=0.5
    for W in range(1,maxlag):
        t=0.5+np.sum(rho[1:W+1])
        if W>=6*t: return t,W
    return t,maxlag
print('tau_int (blocks of10 sweeps) m2:',tau_int(x))
print('m2 mean by 50-block chunks:',[round(M2[k:k+30].mean(),4) for k in range(0,300,30)])
U=1-M4/(3*M2**2)
print('U first/last chunk', U[:50].mean(), U[-50:].mean())
print('U std over replicas of sweep-avg', U.mean(0).std()/np.sqrt(256))
