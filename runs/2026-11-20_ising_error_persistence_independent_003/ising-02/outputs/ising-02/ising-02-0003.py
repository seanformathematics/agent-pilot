import numpy as np, time
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def acc_table(T):
    k=np.array([-4,-2,0,2,4]); dE=2*k
    return np.where(dE<=0,1.0,np.exp(-np.maximum(dE,0)/T)).astype(np.float32)
def sweep(s,rng,at):
    for p in (0,1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        pa=at[((s*nb)+4)//2]
        r=rng.random(s.shape,dtype=np.float32)
        s[masks[p]&(r<pa)]*=-1
def run(T,R,burn,meas,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    at=acc_table(T)
    for _ in range(burn): sweep(s,rng,at)
    S2=np.zeros(R); S4=np.zeros(R); S1=np.zeros(R)
    for _ in range(meas):
        sweep(s,rng,at)
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        m2=m*m; S2+=m2; S4+=m2*m2; S1+=np.abs(m)
    return S2/meas,S4/meas,S1/meas
def Ufun(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jk(m2,m4,G=20):
    R=len(m2); idx=np.arange(R)%G
    U=Ufun(m2,m4)
    Us=[]
    for g in range(G):
        k=idx!=g
        Us.append(Ufun(m2[k],m4[k]))
    Us=np.array(Us)
    return U, np.sqrt((G-1)/G*np.sum((Us-Us.mean())**2))
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for T in Ts:
    m2,m4,m1=run(T,400,300,2000,seed=int(T*1000))
    U,e=jk(m2,m4)
    print(T, 'U=%.5f err=%.5f'%(U,e), 'm2=%.4f |m|=%.4f'%(m2.mean(),m1.mean()), 'time',round(time.time()-t0,1))