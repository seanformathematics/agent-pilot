import sys, numpy as np, time
sys.path.insert(0,'/work')
from ising import run
def ujk(m2,m4):
    R=len(m2)
    U=1-np.mean(m4)/(3*np.mean(m2)**2)
    tot2=m2.sum(); tot4=m4.sum()
    Uj=np.array([1-((tot4-m4[k])/(R-1))/(3*((tot2-m2[k])/(R-1))**2) for k in range(R)])
    return U,np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
t=time.time()
m2,m4,m1=run(2.3,512,20000,5000,2005)
print("T=2.3 eq20000 nm5000:",ujk(m2,m4),"time",time.time()-t)
d=np.load('/work/res_2.3.npz')
print("T=2.3 eq5000 nm20000:",ujk(d['m2'],d['m4']))
