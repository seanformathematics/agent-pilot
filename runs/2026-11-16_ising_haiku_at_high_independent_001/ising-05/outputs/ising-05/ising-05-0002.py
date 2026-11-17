import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks(N):
    i,j=np.indices((N,N))
    return ((i+j)%2==0),((i+j)%2==1)
def run(T,R,nsweep,neq,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    A,B=make_masks(N)
    # acceptance table indexed by s*h (h in -4..4 step 2): dE=2*s*h
    acc={}
    for x in [2,4]:
        acc[x]=np.exp(-2.0*x/T)
    Ms=[];E=[]
    tot=neq+nsweep
    for sw in range(tot):
        for M in (A,B):
            h=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sh=(s*h).astype(np.float64)
            p=np.where(sh<=0,1.0,np.where(sh==2,acc[2],acc[4]))
            flip=(rng.random((R,N,N))<p)&M
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=neq:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            Ms.append(m)
    return np.array(Ms)  # (nsweep,R)
def binder(M):
    m2=(M**2).mean(); m4=(M**4).mean()
    return 1-m4/(3*m2*m2)
if __name__=="__main__":
    for T in [2.27,1.5]:
        t=time.time()
        Ms=run(T,1000,300,200,1)
        dt=time.time()-t
        print(T,"time",dt,"per sweep",dt/500)
        print("U all", binder(Ms), "mean|m|",np.abs(Ms).mean())
        # per-replica time-averaged jackknife
        m2=(Ms**2).mean(0); m4=(Ms**4).mean(0)
        print("U from replica-avgs", 1-m4.mean()/(3*m2.mean()**2))
