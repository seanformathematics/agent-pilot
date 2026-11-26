
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_masks():
    i,j=np.indices((N,N))
    return ((i+j)%2==0), ((i+j)%2==1)
mA,mB=make_masks()

def run(T, R, seed, n_eq, n_meas, init='random', snap=None):
    rng=np.random.default_rng(seed)
    if init=='random':
        s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    else:
        s=np.ones((R,N,N),dtype=np.int8)
    # acceptance table indexed by s*h where h in {-4,-2,0,2,4}; dE=2*s*h
    acc={}
    for k in range(-4,5,2):  # k = s*h
        acc[k]=np.exp(-2.0*k/T)
    # lookup: index (s*h+4)//2 in 0..4
    tab=np.array([acc[-4],acc[-2],1.0,1.0,1.0]) # placeholder replaced below
    tab=np.array([np.exp(-2.0*(k)/T) if k>0 else 1.0 for k in [-4,-2,0,2,4]])
    # k = s*h ; flip accepted if dE=2*k<=0 -> k<=0 always accept; k>0 accept w.p. exp(-2k/T)
    def sweep(s):
        for m in (mA,mB):
            h=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            k=(s*h).astype(np.int8)
            p=np.where(k<=0,1.0,np.exp(-2.0*np.maximum(k,0)/T))
            u=rng.random(s.shape)
            flip=(u<p)&m
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq):
        s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); nm=0
    E1=np.zeros(R)
    series=[]
    for _ in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2+=m2; S4+=m2*m2; nm+=1
        series.append(m2)
    return s,S2/nm,S4/nm,np.array(series)

if __name__=='__main__':
    t=time.time()
    s,a2,a4,ser=run(2.269,2000,1,0,50)
    print('time per sweep (R=2000):',(time.time()-t)/50)
