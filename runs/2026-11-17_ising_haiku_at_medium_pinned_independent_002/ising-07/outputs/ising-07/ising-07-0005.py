
import numpy as np, json, time
N=16; R=1000
def run(T, seed, n_eq, n_meas, R=R):
    rng=np.random.default_rng(seed)
    if T < 2.269:
        sign=rng.choice([-1,1],size=(R,1,1)).astype(np.int8)
        s=np.ones((R,N,N),dtype=np.int8)*sign
    else:
        s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    def sweep(s):
        for mk in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
            dE=2*s*nn
            r=rng.random(s.shape)
            pa=np.where(dE<=0,1.0,np.exp(-np.clip(dE,0,None)/T))
            flip=(r<pa)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq): s=sweep(s)
    halves=[[np.zeros(R),np.zeros(R)] for _ in range(2)]
    cnt=[0,0]
    for k in range(n_meas):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1); m2=m*m
        h=0 if k<n_meas//2 else 1
        halves[h][0]+=m2; halves[h][1]+=m2*m2; cnt[h]+=1
    a2=(halves[0][0]+halves[1][0])/(cnt[0]+cnt[1]); a4=(halves[0][1]+halves[1][1])/(cnt[0]+cnt[1])
    h1=(halves[0][0]/cnt[0],halves[0][1]/cnt[0]); h2=(halves[1][0]/cnt[1],halves[1][1]/cnt[1])
    return a2,a4,h1,h2
def U_of(x2,x4):
    return 1-x4.mean()/(3*x2.mean()**2)
def jack(a2,a4):
    R=len(a2); tot2=a2.sum(); tot4=a4.sum()
    jk=np.array([U_of((tot2-a2[i])/(R-1),(tot4-a4[i])/(R-1)) for i in range(R)])
    return U_of(a2,a4), np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out={}
t0=time.time()
for k,T in enumerate(Ts):
    a2,a4,h1,h2=run(T,2000+k,2000,4000)
    U,e=jack(a2,a4)
    U1=U_of(h1[0],h1[1]); U2=U_of(h2[0],h2[1])
    out[T]=dict(U=U,err=e,U_half1=U1,U_half2=U2,m2=a2.mean())
    print(T, round(U,5), round(e,5), "halves", round(U1,4), round(U2,4), "time", round(time.time()-t0,1), flush=True)
json.dump(out,open('/work/part2.json','w'))
