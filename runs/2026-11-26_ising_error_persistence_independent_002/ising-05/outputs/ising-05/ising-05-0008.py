
import numpy as np, math
N=16; T=2.4
rng=np.random.default_rng(11)
s=rng.choice([-1,1],size=(N,N))
ms=[]
for sw in range(300):
    for i in range(N):
        for j in range(N):
            h=s[(i+1)%N,j]+s[(i-1)%N,j]+s[i,(j+1)%N]+s[i,(j-1)%N]
            dE=2*s[i,j]*h
            if dE<=0 or rng.random()<math.exp(-dE/T):
                s[i,j]=-s[i,j]
    if sw%30==0 or sw==299:
        ms.append(round(s.mean(),3))
print("ref |m| series", ms)
# vectorized single replica for comparison
s2=rng.choice([-1,1],size=(1,N,N)).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
P=np.zeros((2,5))
for si,s0 in enumerate((-1,1)):
    for hi,h in enumerate((-4,-2,0,2,4)):
        P[si,hi]=min(1.0,np.exp(-2*s0*h/T))
ms2=[]
for sw in range(300):
    for mk in masks:
        h=(np.roll(s2,1,1)+np.roll(s2,-1,1)+np.roll(s2,1,2)+np.roll(s2,-1,2))
        si=(s2>0).astype(np.int8); hi=((h+4)//2).astype(np.int8)
        p=P[si,hi]; u=rng.random((1,N,N))
        s2=np.where((u<p)&mk,-s2,s2).astype(np.int8)
    if sw%30==0 or sw==299: ms2.append(round(float(s2.mean()),3))
print("vec |m| series", ms2)
print(np.unique(s2, return_counts=True))
