
import numpy as np
N=16; R=200
def make(T, seed, hot):
    rng = np.random.default_rng(seed)
    if hot:
        s = rng.choice([-1,1], size=(R,N,N)).astype(np.int8)
    else:
        sign = rng.choice([-1,1], size=(R,1,1)).astype(np.int8)
        s = np.broadcast_to(sign,(R,N,N)).copy()
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    P=np.zeros((2,5))
    for si,s0 in enumerate((-1,1)):
        for hi,h in enumerate((-4,-2,0,2,4)):
            P[si,hi]=min(1.0,np.exp(-2*s0*h/T))
    def sweep(s):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            si=(s>0).astype(np.int8); hi=((h+4)//2).astype(np.int8)
            p=P[si,hi]; u=rng.random((R,N,N))
            s=np.where((u<p)&mk,-s,s).astype(np.int8)
        return s
    return s, sweep
for T in [2.4, 2.3]:
    for hot in [True, False]:
        s, sweep = make(T, 7, hot)
        out=[]
        for blk in range(8):
            m2s=[]; m4s=[]
            for k in range(250):
                s=sweep(s)
                if blk>0 or k>=0:
                    m=s.reshape(R,-1).mean(axis=1)
                    m2s.append(m*m); m4s.append(m**4)
            M2=np.mean(m2s); M4=np.mean(m4s)
            out.append((blk*250, round(M2,5), round(1-M4/(3*M2**2),4)))
        print("T",T,"hot" if hot else "cold", out, flush=True)
