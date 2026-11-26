
code = r'''
import numpy as np
N=16
def make_masks():
    i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    b=((i+j)%2==0)
    return b, ~b
def run(T, R, seed, n_eq, n_meas, save_every=1):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    black,white=make_masks()
    p_tab=np.zeros(9)
    for k in range(-4,5):
        dE=2*k
        p_tab[k+4]=1.0 if dE<=0 else np.exp(-dE/T)
    def sweep(s):
        for mask in (black,white):
            h = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            k = (s*h).astype(np.int16)
            p = p_tab[k+4]
            u = rng.random(s.shape)
            flip = (u < p) & mask
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    series=[]
    for it in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2+=m*m; m4+=m**4; cnt+=1
    return m2/cnt, m4/cnt

def binder_jk(m2, m4, nblocks=None):
    R=len(m2)
    U=1-np.mean(m4)/(3*np.mean(m2)**2)
    # replica jackknife
    idx=np.arange(R)
    Uj=np.empty(R)
    S2=m2.sum(); S4=m4.sum()
    for r in range(R):
        a2=(S2-m2[r])/(R-1); a4=(S4-m4[r])/(R-1)
        Uj[r]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U, err
'''
open('/work/ising.py','w').write(code)
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from ising import run, binder_jk
t=time.time()
m2,m4=run(2.3,1000,12345,2000,5000)
print("time",time.time()-t)
print(binder_jk(m2,m4))
# check stability: split time into halves via separate runs? just report
