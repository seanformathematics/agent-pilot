
import numpy as np
N=16
def energy(s):
    return -(np.sum(s*np.roll(s,1,-2)) + np.sum(s*np.roll(s,1,-1)))
# vectorized version as in module
rng=np.random.default_rng(3)
R=4; T=2.4
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
P=np.zeros((2,5))
for si,s0 in enumerate((-1,1)):
    for hi,h in enumerate((-4,-2,0,2,4)):
        P[si,hi]=min(1.0,np.exp(-2*s0*h/T))
print("P table\n",P)
for sweepno in range(200):
    for mk in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        si=(s>0).astype(np.int8); hi=((h+4)//2).astype(np.int8)
        p=P[si,hi]; u=rng.random((R,N,N))
        s=np.where((u<p)&mk,-s,s).astype(np.int8)
    if sweepno%50==0 or sweepno==199:
        print(sweepno, [energy(s[r])/(2*N*N) for r in range(R)], [s[r].mean() for r in range(R)])
