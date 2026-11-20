import numpy as np, time, sys, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, time, sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
N=16
idx=np.arange(N*N).reshape(N,N)
EA=np.concatenate([idx.ravel(), idx.ravel()])
EB=np.concatenate([np.roll(idx,-1,axis=1).ravel(), np.roll(idx,-1,axis=0).ravel()])
def run(T, nsweep, nequil, seed, out):
    rng=np.random.default_rng(seed)
    p=1-np.exp(-2.0/T)
    s=np.where(rng.random(N*N)<0.5,1,-1).astype(np.int64)
    m2=np.empty(nsweep); m4=np.empty(nsweep)
    tot=nequil+nsweep
    for it in range(tot):
        same=(s[EA]==s[EB])
        keep=same & (rng.random(EA.size)<p)
        a=EA[keep]; b=EB[keep]
        g=coo_matrix((np.ones(a.size,dtype=np.int8),(a,b)),shape=(N*N,N*N))
        nc,lab=connected_components(g,directed=False)
        flip=rng.random(nc)<0.5
        s=np.where(flip[lab],-s,s)
        if it>=nequil:
            m=s.sum()/(N*N)
            m2[it-nequil]=m*m; m4[it-nequil]=m**4
    np.savez(out, m2=m2, m4=m4, T=T, seed=seed, nequil=nequil)
if __name__=="__main__":
    T=float(sys.argv[1]); ns=int(sys.argv[2]); ne=int(sys.argv[3]); seed=int(sys.argv[4]); out=sys.argv[5]
    t=time.time(); run(T,ns,ne,seed,out); print("done",time.time()-t)
'''
open('/work/sw.py','w').write(code)
import subprocess
print(subprocess.run(['python','/work/sw.py','2.3','200000','2000','101','/work/pilot_2.3.npz'],capture_output=True,text=True,cwd='/work'))
d=np.load('/work/pilot_2.3.npz')
m2=d['m2']; m4=d['m4']
def U(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
print("U",U(m2,m4), "mean m2",m2.mean())
# binning errors
for nb in [10,20,50,100,200,500]:
    L=len(m2)//nb
    us=[U(m2[i*L:(i+1)*L],m4[i*L:(i+1)*L]) for i in range(nb)]
    print(nb, L, np.std(us,ddof=1)/np.sqrt(nb))
# autocorrelation of m2
x=m2-m2.mean()
ac=[np.dot(x[:-k],x[k:])/np.dot(x,x) for k in [1,2,5,10,20,50]]
print("acf",np.round(ac,3))