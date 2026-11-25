
import numpy as np, time, os, sys, json
N=16; R=2000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0

def run(T, budget_s, burn_sweeps_if_new=1500):
    fn=f"/work/state_T{T:.2f}.npz"
    tab=np.array([min(1.0,np.exp(-2*h/T)) for h in range(-4,5)])
    if os.path.exists(fn):
        d=np.load(fn)
        s=d['s']; S2=d['S2']; S4=d['S4']; n=int(d['n']); seedc=int(d['seedc']); blocks=list(d['blocks'])
        burn_needed=0
    else:
        seedc=0
        base=int(round(T*100))*1000
        rng0=np.random.default_rng(base)
        s=np.where(rng0.random((R,N,N))<0.5,-1,1).astype(np.int8)
        S2=np.zeros(R); S4=np.zeros(R); n=0; blocks=[]
        burn_needed=burn_sweeps_if_new
    rng=np.random.default_rng(int(round(T*100))*1000+seedc+7)
    def half(s,mask):
        nn=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        p=tab[(s*nn)+4]
        acc=(rng.random(s.shape,dtype=np.float32)<p)&mask
        s[acc]*=-1
    t0=time.time()
    for _ in range(burn_needed):
        half(s,mask0); half(s,mask1)
    # measurement
    nb=0; bm=0.0; bcount=0
    sweeps=0
    while time.time()-t0<budget_s:
        for _ in range(200):
            half(s,mask0); half(s,mask1)
            m=s.sum(axis=(1,2))/(N*N)
            m2=m*m
            S2+=m2; S4+=m2*m2; n+=1
            bm+=np.abs(m).mean(); bcount+=1
            sweeps+=1
        blocks.append(bm/bcount); bm=0; bcount=0
    np.savez(fn,s=s,S2=S2,S4=S4,n=n,seedc=seedc+1,blocks=np.array(blocks))
    # estimate
    U_all=1-(S4.sum()/(R*n))/(3*(S2.sum()/(R*n))**2)
    # jackknife over replicas
    tot2=S2.sum(); tot4=S4.sum()
    Uj=np.empty(R)
    for r in range(R):
        a2=(tot2-S2[r])/((R-1)*n); a4=(tot4-S4[r])/((R-1)*n)
        Uj[r]=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return dict(T=T,n=n,sweeps_this_call=sweeps,U=U_all,U_err=err,burn=burn_needed,
                mabs_blocks=[round(float(b),4) for b in blocks[-8:]])

if __name__=="__main__":
    T=float(sys.argv[1]); budget=float(sys.argv[2])
    print(json.dumps(run(T,budget)))
