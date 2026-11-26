
import sys, time, numpy as np
sys.path.insert(0,'/work')
import wolff
def U_of(b2,b4): return 1-b4.mean()/(3*b2.mean()**2)
def jk(b2,b4,nb=20):
    n=len(b2); idx=np.arange(n)
    groups=np.array_split(idx,nb)
    Us=[]
    for g in groups:
        mask=np.ones(n,bool); mask[g]=False
        Us.append(U_of(b2[mask],b4[mask]))
    Us=np.array(Us)
    return np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
Us=[];je=[]
t=time.time()
for sd in range(8):
    s,M,b2,b4 = wolff.run(2.3, 1000+sd, 200000, 100)
    Us.append(U_of(b2,b4)); je.append(jk(b2,b4))
Us=np.array(Us)
print('time',time.time()-t)
print('U per seed',Us)
print('seed scatter std (ddof1)',Us.std(ddof=1),' mean',Us.mean(), ' sem',Us.std(ddof=1)/np.sqrt(8))
print('jackknife per run',np.array(je), 'mean',np.mean(je))
