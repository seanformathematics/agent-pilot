import numpy as np, time
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0), ((ii+jj)%2==1)]
def sweep(s,rng,acc):
    for m in masks:
        nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE = 2*s*nb
        r = rng.random(s.shape,dtype=np.float32)
        flip = ((dE<=0) | (r < acc[np.clip(dE,0,8)])) & m
        s[flip] *= -1
def U_jk(S2,S4,nm):
    R=len(S2)
    tot2=S2.sum(); tot4=S4.sum()
    U=1-(tot4/(R*nm))/(3*(tot2/(R*nm))**2)
    jk=np.empty(R)
    for r in range(R):
        m2=(tot2-S2[r])/((R-1)*nm); m4=(tot4-S4[r])/((R-1)*nm)
        jk[r]=1-m4/(3*m2*m2)
    return U, np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
def run(T,R=1000,neq=3000,nmeas=10000,seed=1):
    acc=np.array([np.exp(-dE/T) for dE in range(9)],dtype=np.float32)
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for k in range(neq): sweep(s,rng,acc)
    H=nmeas//2
    A2=np.zeros(R);A4=np.zeros(R);B2=np.zeros(R);B4=np.zeros(R)
    for k in range(nmeas):
        sweep(s,rng,acc)
        m=s.reshape(R,-1).mean(1); m2=m*m
        if k<H: A2+=m2; A4+=m2*m2
        else: B2+=m2; B4+=m2*m2
    return A2,A4,B2,B4,H,nmeas-H
t0=time.time()
for T,seed in [(2.1,203),(2.4,204)]:
    A2,A4,B2,B4,na,nb=run(T,seed=seed)
    S2=A2+B2;S4=A4+B4
    U,err=U_jk(S2,S4,na+nb)
    UA,_=U_jk(A2,A4,na); UB,_=U_jk(B2,B4,nb)
    print(f"T={T} U={U:.6f} err={err:.6f} halves: {UA:.6f} {UB:.6f} time={time.time()-t0:.1f}",flush=True)
    np.savez(f'/work/final_T{T}.npz',S2=S2,S4=S4,nm=na+nb,A2=A2,A4=A4,B2=B2,B4=B4)
