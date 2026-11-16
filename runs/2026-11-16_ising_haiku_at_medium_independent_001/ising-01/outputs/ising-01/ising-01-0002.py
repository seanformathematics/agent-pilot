
import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=256
def run(T, nsweep, seed, nequil):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    tab=np.zeros((2,5))
    for k,s0 in enumerate((-1,1)):
        for q,nn in enumerate((-4,-2,0,2,4)):
            dE=2*s0*nn
            tab[k,q]=1.0 if dE<=0 else np.exp(-dE/T)
    s2=np.zeros(R); s4=np.zeros(R); cnt=0; ae=[]
    for sweep in range(nsweep):
        for c in range(2):
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            p=tab[(s>0).astype(np.int8), ((nb+4)//2)]
            flip=(rng.random(s.shape)<p)&masks[c]
            s=np.where(flip,-s,s).astype(np.int8)
        if sweep>=nequil:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            s2+=m**2; s4+=m**4; cnt+=1
    return s2/cnt, s4/cnt
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1]
out={}
for k,T in enumerate(Ts):
    t=time.time()
    a2,a4=run(T,23000,1000+k,3000)
    np.savez(f'/work/T{T}.npz',m2=a2,m4=a4)
    U=1-np.mean(a4)/(3*np.mean(a2)**2)
    print(T,round(U,5),round(time.time()-t,1),flush=True)
