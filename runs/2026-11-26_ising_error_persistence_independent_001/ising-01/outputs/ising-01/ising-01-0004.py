
src = r'''
import numpy as np
N=16
_i,_j=np.indices((N,N))
mA=((_i+_j)%2==0); mB=~mA

def simulate(T, R, seed, n_eq, n_meas, rec_every=1):
    """Checkerboard Metropolis on R independent 16x16 lattices (random start).
    Returns per-replica sums of m^2 and m^4 over measurements, and a
    time series of the replica-averaged m^2, m^4 (for autocorrelation / equilibration checks)."""
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance prob for k=s*h (h = sum of 4 neighbours): dE=2k
    ptab=np.array([1.0,1.0,1.0,np.exp(-4.0/T),np.exp(-8.0/T)])  # index by k+4 for k in -4..4 step 2 -> idx 0,2,4,6,8? handled below
    def sweep(s):
        for m in (mA,mB):
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            k=s*h
            p=np.exp(-2.0*np.maximum(k,0)/T)
            u=rng.random(s.shape)
            flip=(u<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R)
    ts2=[]; ts4=[]
    for it in range(n_meas):
        s=sweep(s)
        if it%rec_every==0:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            S2+=m2; S4+=m2*m2
            ts2.append(m2.mean()); ts4.append((m2*m2).mean())
    n=len(ts2)
    return S2,S4,n,np.array(ts2),np.array(ts4)

def binder_from_sums(S2,S4,n):
    m2=S2.sum()/ (n*len(S2)); m4=S4.sum()/(n*len(S4))
    return 1-m4/(3*m2*m2)

def jackknife_U(S2,S4):
    """Per-replica sums -> U and jackknife error over replicas (replicas are independent)."""
    R=len(S2)
    A=S2.sum(); B=S4.sum()
    U=1-(B/R)/(3*(A/R)**2)
    # leave-one-out
    Aj=A-S2; Bj=B-S4
    Uj=1-(Bj/(R-1))/(3*(Aj/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
'''
open('/work/ising.py','w').write(src)
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from ising import *
t=time.time()
S2,S4,n,t2,t4=simulate(2.3,500,101,500,3000)
print('time',time.time()-t)
U,err=jackknife_U(S2,S4)
print('U',U,'err',err,'n',n)
# time series of U in blocks of 300 sweeps to check equilibration
for b in range(0,3000,300):
    print(b, 1-np.mean(t4[b:b+300])/(3*np.mean(t2[b:b+300])**2))
# autocorrelation of m2 series (replica-averaged)
x=t2-t2.mean()
ac=np.correlate(x,x,'full')[len(x)-1:]/ (x@x)
tau=1+2*np.sum(ac[1:200]*(np.arange(1,200)<np.argmax(ac<0.01)))
print('tau_int m2 approx',tau)
