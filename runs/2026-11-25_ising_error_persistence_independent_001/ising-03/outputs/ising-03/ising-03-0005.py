
import sys, time, numpy as np, os
sys.path.insert(0,'/work')
from ising import *
src = r'''
import numpy as np
from ising import run, make_masks
def production(T, seed, R=1000, nburn=2000, nmeas=20000, blk=1000):
    rng=np.random.default_rng(seed)
    S,rng,_=run(T,R,seed,nburn,rng=rng,meas=False)
    s2=np.zeros(R); s4=np.zeros(R); s1=np.zeros(R)
    bl2=[];bl4=[];bl1=[]
    cur2=np.zeros(R);cur4=np.zeros(R);cur1=np.zeros(R)
    for sw in range(nmeas):
        S,rng,_=run(T,R,seed,1,S=S,rng=rng,meas=False)
        m=np.abs(S.reshape(R,-1).mean(axis=1))
        m2=m*m
        s2+=m2; s4+=m2*m2; s1+=m
        cur2+=m2; cur4+=m2*m2; cur1+=m
        if (sw+1)%blk==0:
            bl2.append(cur2.copy());bl4.append(cur4.copy());bl1.append(cur1.copy())
            cur2[:]=0;cur4[:]=0;cur1[:]=0
    np.savez(f'/work/res_{T}_{seed}.npz',s2=s2,s4=s4,s1=s1,bl2=np.array(bl2),bl4=np.array(bl4),bl1=np.array(bl1),n=nmeas,R=R)
    return s2,s4
def ujack(s2,s4,n):
    R=len(s2)
    tot2=s2.sum();tot4=s4.sum()
    U=1-(tot4/(R*n))/(3*(tot2/(R*n))**2)
    a2=(tot2-s2)/((R-1)*n); a4=(tot4-s4)/((R-1)*n)
    Ul=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Ul-Ul.mean())**2))
    return U,err
'''
open('/work/prod.py','w').write(src)
from prod import *
t=time.time()
s2,s4=production(2.2,11,nmeas=20000)
U,e=ujack(s2,s4,20000)
print(2.2,"U=",U,"err=",e,"time",time.time()-t)
