import time, numpy as np, json
N=16; R=1000
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==k) for k in (0,1)]
def sweep(s,beta,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        p=np.exp(-beta*np.clip(dE,0,None))
        acc=(dE<=0)|(rng.random(s.shape)<p)
        s[acc&m]*=-1
    return s
results={}
BURN=3000; MEAS=5000
for k,T in enumerate(Ts):
    seed=1000+k
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    beta=1/T
    t0=time.time()
    for _ in range(BURN): sweep(s,beta,rng)
    S2=np.zeros(R); S4=np.zeros(R)
    half=np.zeros((2,R,2))
    for sw in range(MEAS):
        sweep(s,beta,rng)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        S2+=m2; S4+=m2*m2
        if sw==MEAS//2-1: h1=(S2.copy(),S4.copy())
    h2=(S2-h1[0],S4-h1[1])
    results[T]=dict(S2=S2.tolist(),S4=S4.tolist(),h1_2=h1[0].tolist(),h1_4=h1[1].tolist())
    # quick estimate
    m2a=S2/MEAS; m4a=S4/MEAS
    U=1-np.mean(m4a)/(3*np.mean(m2a)**2)
    U1=1-np.mean(h1[1]/(MEAS//2))/(3*np.mean(h1[0]/(MEAS//2))**2)
    U2=1-np.mean(h2[1]/(MEAS-MEAS//2))/(3*np.mean(h2[0]/(MEAS-MEAS//2))**2)
    print(T, "U=%.5f  halves %.5f %.5f  time %.1f"%(U,U1,U2,time.time()-t0), flush=True)
json.dump({str(k):v for k,v in results.items()},open('/work/raw.json','w'))