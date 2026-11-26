
import sys, time, numpy as np
sys.path.insert(0,'/work')
import wolff
def U_of(b2,b4): return 1-b4.mean()/(3*b2.mean()**2)
def jk(b2,b4,nb=50):
    n=len(b2); idx=np.arange(n)
    # delete-group jackknife with nb groups
    groups=np.array_split(idx,nb)
    Us=[]
    for g in groups:
        mask=np.ones(n,bool); mask[g]=False
        Us.append(U_of(b2[mask],b4[mask]))
    Us=np.array(Us)
    return np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2)), U_of(b2,b4)
t=time.time()
s,M,b2,b4 = wolff.run(2.3, 12345, 1000000, 100)
print('time',time.time()-t)
for B in [1,10,100]:
    n=len(b2)//B*B
    bb2=b2[:n].reshape(-1,B).mean(1); bb4=b4[:n].reshape(-1,B).mean(1)
    print('block',B, 'nblocks',len(bb2), jk(bb2,bb4,50))
