code = r'''
import numpy as np, time, os
N=16
I,J=np.indices((N,N))
MA=((I+J)%2==0); MB=~MA
def run(T,R,neq,nmeas,seed,chunk_report=None):
    rng=np.random.default_rng(seed)
    # half replicas all-up, half all-down (symmetric mixture); equilibrate from there
    s=np.ones((R,N,N),dtype=np.int8)
    s[R//2:]=-1
    if T>2.6:  # disordered start for high T
        s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    acc2=np.exp(-4.0/T); acc4=np.exp(-8.0/T)
    S1=np.zeros(R); S2=np.zeros(R); S4=np.zeros(R); S_E=np.zeros(R)
    tot=neq+nmeas
    for sw in range(tot):
        for M in (MA,MB):
            h=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sh=s*h
            p=np.where(sh<=0,1.0,np.where(sh==2,acc2,acc4))
            flip=(rng.random((R,N,N))<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=neq:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S1+=np.abs(m); S2+=m2; S4+=m2*m2
    return S1/nmeas,S2/nmeas,S4/nmeas
def U_from(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
def U_jk(m2,m4):
    R=len(m2)
    U=U_from(m2,m4)
    # delete-one jackknife using sums
    tot2=m2.sum(); tot4=m4.sum()
    Uj=np.empty(R)
    for k in range(R):
        a2=(tot2-m2[k])/(R-1); a4=(tot4-m4[k])/(R-1)
        Uj[k]=1-a4/(3*a2*a2)
    var=(R-1)/R*np.sum((Uj-Uj.mean())**2)
    return U,np.sqrt(var)
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, ising, time
import numpy as np
t=time.time()
a1,a2,a4=ising.run(2.27,2000,1000,4000,12345)
print("time",time.time()-t)
print("U,err",ising.U_jk(a2,a4))
# block halves check: first half replicas vs second
print("U first half reps",ising.U_from(a2[:1000],a4[:1000]),"second",ising.U_from(a2[1000:],a4[1000:]))
np.savez('/work/pilot_227.npz',m1=a1,m2=a2,m4=a4)
