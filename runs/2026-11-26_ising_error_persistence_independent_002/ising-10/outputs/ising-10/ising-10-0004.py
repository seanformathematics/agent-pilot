
import numpy as np, time, json
N=16; R=1000; EQ=3000; MEAS=12000
i,j=np.indices((N,N)); MASK=((i+j)%2==0)
def sweep(s,T,rng):
    for m in (MASK,~MASK):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        s[acc & m[None]] *= -1
T=2.4; seed=int(T*1000)+7
rng=np.random.default_rng(seed)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
for k in range(EQ): sweep(s,T,rng)
S2=np.zeros((R,2)); S4=np.zeros((R,2))
for k in range(MEAS):
    sweep(s,T,rng)
    m=s.reshape(R,-1).mean(axis=1)
    h=0 if k<MEAS//2 else 1
    S2[:,h]+=m**2; S4[:,h]+=m**4
n=MEAS//2
tot2=S2.sum(1); tot4=S4.sum(1); nt=MEAS
m2=tot2.sum()/(nt*R); m4=tot4.sum()/(nt*R)
U=1-m4/(3*m2**2)
nb=50; blocks=np.array_split(np.arange(R),nb)
Us=[]
for b in blocks:
    keep=np.setdiff1d(np.arange(R),b)
    a2=tot2[keep].sum()/(nt*len(keep)); a4=tot4[keep].sum()/(nt*len(keep))
    Us.append(1-a4/(3*a2**2))
Us=np.array(Us)
err=np.sqrt((nb-1)/nb*np.sum((Us-Us.mean())**2))
Uh=[1-(S4[:,h].sum()/(n*R))/(3*(S2[:,h].sum()/(n*R))**2) for h in range(2)]
print(dict(U=float(U),err=float(err),halves=[float(x) for x in Uh],m=float(np.sqrt(m2))))
np.savez(f'/work/raw_T{T}.npz',S2=S2,S4=S4)
