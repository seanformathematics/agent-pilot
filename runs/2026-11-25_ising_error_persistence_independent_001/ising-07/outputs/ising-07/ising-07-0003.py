
import sys, time, numpy as np
sys.path.insert(0,'/work')
from wolff import wolff_run
t=time.time()
m2,m4 = wolff_run(16, 2.3, 1000000, 12345, nburn=5000)
print("time", time.time()-t)
np.savez('/work/pilot_T2.3.npz', m2=m2, m4=m4)
def U_of(a,b): return 1-b.mean()/(3*a.mean()**2)
print("U", U_of(m2,m4))
# binning
for nb in [10,20,50,100,200,500,1000]:
    L=len(m2)//nb
    us=[]
    for b in range(nb):
        sl=slice(b*L,(b+1)*L)
        us.append(U_of(m2[sl],m4[sl]))
    us=np.array(us)
    print(nb, L, us.std(ddof=1)/np.sqrt(nb))
# autocorrelation of m2
x=m2-m2.mean()
for lag in [1,2,5,10,20,50]:
    print("lag",lag, np.dot(x[:-lag],x[lag:])/np.dot(x,x))
