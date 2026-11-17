import numpy as np, time, json
N=16
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R=256; therm=2000; nsw=5000
ii,jj=np.indices((N,N)); black=((ii+jj)%2==0)
out=[]
for idx,T in enumerate(Ts):
    seed=1000+idx
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep():
        for mask in (black,~black):
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nb
            acc=(dE<=0)|(rng.random((R,N,N),dtype=np.float32)<np.exp(-np.clip(dE,0,None)/T).astype(np.float32))
            s[acc&mask]*=-1
    t0=time.time()
    for _ in range(therm): sweep()
    M2=np.zeros((2,R)); M4=np.zeros((2,R))
    for k in range(nsw):
        sweep()
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        h=0 if k<nsw//2 else 1
        M2[h]+=m*m; M4[h]+=m**4
    M2/=nsw/2; M4/=nsw/2
    Ur=1-(M4[0]+M4[1])/2/(3*((M2[0]+M2[1])/2)**2)
    Ur_h=[1-M4[h]/(3*M2[h]**2) for h in (0,1)]
    U=Ur.mean(); SE=Ur.std(ddof=1)/np.sqrt(R)
    rec=dict(T=T,U=float(U),U_err=float(SE),halfdiff=float(Ur_h[0].mean()-Ur_h[1].mean()),
             half_err=float(np.sqrt(Ur_h[0].var(ddof=1)/R+Ur_h[1].var(ddof=1)/R)),
             time=round(time.time()-t0,1), minU=float(Ur.min()), maxU=float(Ur.max()))
    out.append(rec); print(rec, flush=True)
json.dump(out,open('/work/results.json','w'),indent=1)
