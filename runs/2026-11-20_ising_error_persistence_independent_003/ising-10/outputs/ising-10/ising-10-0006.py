
import numpy as np, time
N=16; R=1024; NSW=30000; BLK=500
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
def run(T,SEED):
    rng=np.random.default_rng(SEED)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    def sweep(s):
        for mask in (mA,mB):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            p=np.exp(-np.clip(dE,0,None)/T)
            acc=(dE<=0)|(rng.random(s.shape)<p)
            s=np.where(acc&mask,-s,s)
        return s
    nb=NSW//BLK
    S2=np.zeros((nb,R)); S4=np.zeros((nb,R)); S1=np.zeros((nb,R))
    for b in range(nb):
        a2=np.zeros(R); a4=np.zeros(R); a1=np.zeros(R)
        for k in range(BLK):
            s=sweep(s)
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            a2+=m2; a4+=m2*m2; a1+=np.abs(m)
        S2[b]=a2; S4[b]=a4; S1[b]=a1
    np.savez(f'/work/raw_T{T}_N{N}.npz',S2=S2,S4=S4,S1=S1,BLK=BLK,R=R,NSW=NSW,SEED=SEED)
t=time.time()
run(2.2,2202); print("T=2.2 done",time.time()-t)
run(2.3,2302); print("T=2.3 done",time.time()-t)
