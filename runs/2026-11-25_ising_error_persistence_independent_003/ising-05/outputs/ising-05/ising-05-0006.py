
import numpy as np, time, json
module = r'''
import numpy as np
N=16; NS=N*N
idx=np.arange(NS).reshape(N,N)
RIGHT=np.roll(idx,-1,axis=1).ravel()
DOWN=np.roll(idx,-1,axis=0).ravel()
def sw_sweep(s, T, rng):
    K=s.shape[0]
    p=1-np.exp(-2.0/T)
    br = (s==s[:,RIGHT]) & (rng.random((K,NS))<p)
    bd = (s==s[:,DOWN]) & (rng.random((K,NS))<p)
    lab=np.arange(K*NS)
    src_r=np.nonzero(br.ravel())[0]
    dst_r=(src_r//NS)*NS+RIGHT[src_r%NS]
    src_d=np.nonzero(bd.ravel())[0]
    dst_d=(src_d//NS)*NS+DOWN[src_d%NS]
    a=np.concatenate([src_r,src_d]); b=np.concatenate([dst_r,dst_d])
    while True:
        m=np.minimum(lab[a],lab[b])
        new=lab.copy()
        np.minimum.at(new,a,m); np.minimum.at(new,b,m)
        new=new[new]; new=new[new]
        if np.array_equal(new,lab): break
        lab=new
    flip_lab = rng.random(K*NS)<0.5
    f = flip_lab[lab].reshape(K,NS)
    return np.where(f, -s, s).astype(np.int8)
def run(T, K, nburn, nmeas, seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(K,NS))
    for _ in range(nburn):
        s=sw_sweep(s,T,rng)
    S2=np.zeros(K); S4=np.zeros(K); S1=np.zeros(K)
    for _ in range(nmeas):
        s=sw_sweep(s,T,rng)
        m=s.mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    return S2/nmeas, S4/nmeas, S1/nmeas
def binder_jk(M2, M4):
    K=len(M2)
    U=lambda a,b: 1-b.mean()/(3*a.mean()**2)
    u=U(M2,M4)
    jk=np.array([U(np.delete(M2,i),np.delete(M4,i)) for i in range(K)])
    err=np.sqrt((K-1)/K*np.sum((jk-jk.mean())**2))
    return u, err
'''
open('/work/sw.py','w').write(module)
import sys; sys.path.insert(0,'/work')
from sw import *
res={}
seeds={1.9:1001,2.0:1002,2.1:1003}
for T in [1.9,2.0,2.1]:
    t0=time.time()
    M2,M4,M1=run(T,400,200,2000,seeds[T])
    u,e=binder_jk(M2,M4)
    np.savez(f'/work/prod_T{T}.npz',M2=M2,M4=M4,M1=M1)
    print(T, u, e, "time", round(time.time()-t0,1), flush=True)
