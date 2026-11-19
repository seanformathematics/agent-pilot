import numpy as np, time, sys
N=16; R=1000; T=float(sys.argv[1]) if len(sys.argv)>1 else 1.9
SEED=1000+int(round(T*100))
EQ=2000; MEAS=18000; BLK=1000
rng=np.random.default_rng(SEED)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
p4=np.exp(-4/T); p8=np.exp(-8/T)
def sweep(s):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        sh=s*h
        p=np.where(sh<=0,1.0,np.where(sh==2,p4,np.where(sh==4,p8,0.0)))
        flip=(rng.random((R,N,N))<p)&m
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t0=time.time()
for _ in range(EQ): s=sweep(s)
S2=np.zeros(R); S4=np.zeros(R); blocks=[]
b2=np.zeros(R); b4=np.zeros(R); nb=0
for k in range(MEAS):
    s=sweep(s)
    m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
    m2=m*m
    S2+=m2; S4+=m2*m2
    b2+=m2; b4+=m2*m2; nb+=1
    if nb==BLK:
        blocks.append(1-(b4/nb)/(3*(b2/nb)**2)); b2[:]=0; b4[:]=0; nb=0
n=MEAS
Ur=1-(S4/n)/(3*(S2/n)**2)
U=np.mean(Ur)  # placeholder
A2=S2.sum()/(R*n); A4=S4.sum()/(R*n)
Utot=1-A4/(3*A2**2)
# jackknife over replicas
jk=np.empty(R)
tot2=S2.sum(); tot4=S4.sum()
for r in range(R):
    a2=(tot2-S2[r])/((R-1)*n); a4=(tot4-S4[r])/((R-1)*n)
    jk[r]=1-a4/(3*a2**2)
Ujk_err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print(f"T={T} U={Utot:.5f} jackknife_err={Ujk_err:.5f} time={time.time()-t0:.1f}s")
print("block U trend (mean over replicas per 1000-sweep block):",np.round(np.array(blocks),4))
print("<|m|> approx", np.sqrt(A2))
np.savez(f"/work/res_T{T}.npz",S2=S2,S4=S4,blocks=np.array(blocks))