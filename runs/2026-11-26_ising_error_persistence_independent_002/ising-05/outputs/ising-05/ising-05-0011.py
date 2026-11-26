
import numpy as np
N=16
def run1(T, nsw, seed, start='rand'):
    rng=np.random.default_rng(seed)
    s=rng.choice([-1,1],size=(N,N)).astype(np.int8) if start=='rand' else np.ones((N,N),np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    P=np.zeros((2,5))
    for si,s0 in enumerate((-1,1)):
        for hi,h in enumerate((-4,-2,0,2,4)):
            P[si,hi]=min(1.0,np.exp(-2*s0*h/T))
    am=[]; en=[]
    for sw in range(nsw):
        for mk in masks:
            h=(np.roll(s,1,0)+np.roll(s,-1,0)+np.roll(s,1,1)+np.roll(s,-1,1))
            si=(s>0).astype(np.int8); hi=((h+4)//2).astype(np.int8)
            p=P[si,hi]; u=rng.random((N,N))
            s=np.where((u<p)&mk,-s,s).astype(np.int8)
        if sw>200:
            am.append(abs(s.mean())); en.append(-(np.sum(s*np.roll(s,1,0))+np.sum(s*np.roll(s,1,1)))/(N*N))
    return round(np.mean(am),3), round(np.mean(en),3), round(np.std(am),3)
for T in [5.0, 3.0, 2.4, 2.0]:
    print(T, [run1(T,1500,sd) for sd in (1,2,3)])
