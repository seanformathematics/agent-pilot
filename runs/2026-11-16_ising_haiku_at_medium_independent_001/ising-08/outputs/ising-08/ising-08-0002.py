code = r'''
import numpy as np, json, sys, time
from multiprocessing import Pool
N=16
def run(args):
    T, seed, R, n_eq, n_meas = args
    rng = np.random.default_rng(seed)
    beta = 1.0/T
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
    # acceptance table for dE = 2*s*nn, nn in {-4,-2,0,2,4}
    acc = {}
    for s0 in (-1,1):
        for nn in (-4,-2,0,2,4):
            dE = 2*s0*nn
            acc[(s0,nn)] = np.exp(-beta*dE) if dE>0 else 1.0
    # build lookup array indexed by [s+1 //2, (nn+4)//2]
    tab = np.zeros((2,5))
    for s0 in (-1,1):
        for k,nn in enumerate((-4,-2,0,2,4)):
            tab[(s0+1)//2,k]=acc[(s0,nn)]
    m2s=0.0; m4s=0.0; ms=0.0; cnt=0
    m2_rep = np.zeros(R); m4_rep=np.zeros(R)
    def sweep():
        for mk in masks:
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            idx = ((nn+4)//2).astype(np.int8)
            p = tab[((s+1)//2).astype(np.int8), idx]
            flip = (rng.random(s.shape) < p) & mk
            s[flip] *= -1
    for _ in range(n_eq): sweep()
    for _ in range(n_meas):
        sweep()
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        m2_rep += m2; m4_rep += m2*m2
    m2_rep/=n_meas; m4_rep/=n_meas
    return T, m2_rep.tolist(), m4_rep.tolist()
if __name__=="__main__":
    mode = sys.argv[1]
    if mode=="test":
        t=time.time(); r=run((2.3, 1, 1000, 50, 50)); print("time 100 sweeps R=1000:", time.time()-t)
        m2=np.array(r[1]); m4=np.array(r[2]); print("U", 1-m4.mean()/(3*m2.mean()**2))
'''
open('/work/ising.py','w').write(code)
import subprocess
print(subprocess.run(['python3','/work/ising.py','test'],capture_output=True,text=True,timeout=300))