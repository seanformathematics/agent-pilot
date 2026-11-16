
import numpy as np, time
N=16; R=256
def run(T, nsweep, seed, nequil):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    # acceptance table for dE=2*s*nn, nn in {-4..4} even
    acc={}
    for s0 in (-1,1):
        for nn in (-4,-2,0,2,4):
            dE=2*s0*nn
            acc[(s0,nn)]=1.0 if dE<=0 else np.exp(-dE/T)
    # table indexed by s (0/1) and (nn+4)/2
    tab=np.zeros((2,5))
    for k,s0 in enumerate((-1,1)):
        for q,nn in enumerate((-4,-2,0,2,4)):
            tab[k,q]=acc[(s0,nn)]
    ms=[];
    for sweep in range(nsweep):
        for c in range(2):
            nb=(np.roll(s,1,1).astype(np.int8)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-2+3,2)*0)  # placeholder
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            p=tab[(s>0).astype(np.int8), ((nb+4)//2)]
            u=rng.random(s.shape)
            flip=(u<p)&masks[c]
            s=np.where(flip,-s,s).astype(np.int8)
        if sweep>=nequil:
            m=s.reshape(R,-1).mean(axis=1)
            ms.append(m)
    return np.array(ms)  # shape (T, R)
t=time.time()
ms=run(2.269,200,1,0)
print(time.time()-t)
