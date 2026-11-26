
import sys
sys.path.insert(0,'/work')
import numpy as np
from ising import simulate
def jackknife_U(S2,S4,n):
    a=S2/n; b=S4/n
    R=len(a)
    A=a.sum(); B=b.sum()
    U=1-(B/R)/(3*(A/R)**2)
    Aj=A-a; Bj=B-b
    Uj=1-(Bj/(R-1))/(3*(Aj/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
for T,seed in [(2.3,7777),(2.4,8888)]:
    S2,S4,n,_,_=simulate(T,2000,seed,500,3000)
    print(T,jackknife_U(S2,S4,n))
