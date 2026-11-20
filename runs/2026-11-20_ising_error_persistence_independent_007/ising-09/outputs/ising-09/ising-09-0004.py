import sys, numpy as np, time
sys.path.insert(0,'/work')
from ising import run
def ujk(m2,m4):
    R=len(m2)
    U=1-np.mean(m4)/(3*np.mean(m2)**2)
    idx=np.arange(R)
    Uj=np.empty(R)
    tot2=m2.sum(); tot4=m4.sum()
    for k in range(R):
        a2=(tot2-m2[k])/(R-1); a4=(tot4-m4[k])/(R-1)
        Uj[k]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t=time.time()
# equilibration check: compare early vs late block averages
m2a,m4a,_=run(2.3,512,0,2000,11)
U1,e1=ujk(m2a,m4a)
m2b,m4b,_=run(2.3,512,3000,5000,12)
U2,e2=ujk(m2b,m4b)
print("T=2.3 first 2000 sweeps from random start: U",U1,"err",e1)
print("T=2.3 after 3000 eq, 5000 meas: U",U2,"err",e2, "time",time.time()-t)
