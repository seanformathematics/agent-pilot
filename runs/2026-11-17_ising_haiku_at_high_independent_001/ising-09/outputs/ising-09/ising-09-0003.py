import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; TS=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
RPT=500
Ts=np.repeat(np.array(TS),RPT); RT=len(Ts)
SEED=12345
ckpt='/work/state.npz'
if os.path.exists(ckpt):
    d=np.load(ckpt); S=d['S']; A2=d['A2']; A4=d['A4']; nsw=int(d['nsw']); rng=np.random.default_rng([SEED,nsw])
else:
    rng=np.random.default_rng(SEED)
    S=(rng.integers(0,2,(RT,N,N))*2-1).astype(np.int8)
    A2=np.zeros(RT); A4=np.zeros(RT); nsw=0
mask_black=((np.add.outer(np.arange(N),np.arange(N))%2)==0)
p4=np.exp(-4.0/Ts).astype(np.float32)[:,None,None]; p8=np.exp(-8.0/Ts).astype(np.float32)[:,None,None]
def half(S,mask):
    nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
    dE=2*S*nn
    thr=np.where(dE<=0,np.float32(2.0),np.where(dE==4,p4,p8))
    flip=(rng.random(S.shape,dtype=np.float32)<thr)&mask
    S[flip]*=-1
def sweep(S):
    half(S,mask_black); half(S,~mask_black)
def run(tmax,nburn_extra=0):
    global S,A2,A4,nsw
    t0=time.time(); 
    while time.time()-t0<tmax:
        sweep(S); nsw+=1
        m=S.sum((1,2),dtype=np.int64)/(N*N)
        if nsw>BURN:
            m2=m*m; A2+=m2; A4+=m2*m2
        if nsw%200==0:
            np.savez(ckpt,S=S,A2=A2,A4=A4,nsw=nsw)
    np.savez(ckpt,S=S,A2=A2,A4=A4,nsw=nsw)
BURN=1000
if __name__=='__main__':
    t=time.time(); 
    for _ in range(20): sweep(S)
    print('time per sweep',(time.time()-t)/20)
