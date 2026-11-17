import numpy as np, time, sys
sys.path.insert(0,"/work")
src=open("/work/ising.py").read()
ns={}
exec(src.split('if __name__')[0],ns)
# custom trace run
def trace(T,R,total,block,seed,start_ordered):
    N=16
    rng=np.random.default_rng(seed)
    if start_ordered:
        s=np.where(rng.random((R,1,1))<0.5,1,-1).astype(np.int8)*np.ones((R,N,N),np.int8)
    else:
        s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    d=np.array([-4,-2,0,2,4]); ptab=np.minimum(1.0,np.exp(-2.0*d/T))
    me=((np.arange(N)[:,None]+np.arange(N)[None,:])%2==0); masks=[me,~me]
    out=[]
    for k in range(total//block):
        acc_m=0; 
        for _ in range(block):
            for mk in masks:
                nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
                delta=(s*nn).astype(np.int8)
                p=ptab[((delta+4)>>1)]
                acc=(rng.random((R,N,N))<p)&mk
                s=np.where(acc,-s,s).astype(np.int8)
            m=s.reshape(R,-1).mean(axis=1)
            acc_m+=np.abs(m).mean()
        out.append(((k+1)*block, acc_m/block))
    return out
t=time.time()
for T in [1.5,2.27]:
    o=trace(T,1000,4000,250,7,T<2)
    print("T",T,[(a,round(b,4)) for a,b in o], time.time()-t)
