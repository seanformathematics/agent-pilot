
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
MA=((ii+jj)%2==0)
MB=~MA
def run(T, R, nsweep, seed, s=None, rec_every=1):
    rng=np.random.default_rng(seed)
    if s is None:
        s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    pt=np.array([1,1,1,np.exp(-4/T),np.exp(-8/T)])
    out=np.empty((nsweep//rec_every,R,2))
    for sw in range(nsweep):
        for M in (MA,MB):
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            k=(s*nb).astype(np.int16)
            p=pt[(k+4)>>1]
            flip=(rng.random(s.shape)<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        if (sw+1)%rec_every==0:
            m=s.sum(axis=(1,2))/(N*N)
            out[sw//rec_every,:,0]=m*m
            out[sw//rec_every,:,1]=m**4
    return s,out
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time()
s,out=run(2.3,100,200,1)
print("time per sweep R=100:",(time.time()-t)/200)
