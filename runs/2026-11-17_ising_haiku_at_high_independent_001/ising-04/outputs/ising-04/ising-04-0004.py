
import numpy as np, time, json
N=16; R=400
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for mk in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        u=rng.random(s.shape,dtype=np.float32)
        flip=(u<np.exp(-dE/T))&mk
        s[flip]*=-1
def jack_U(m2,m4):
    R=len(m2); S2=m2.sum(); S4=m4.sum()
    U=1-(S4/R)/(3*(S2/R)**2)
    Us=np.array([1-((S4-m4[k])/(R-1))/(3*((S2-m2[k])/(R-1))**2) for k in range(R)])
    return U,np.sqrt((R-1)/R*np.sum((Us-Us.mean())**2))
run1=json.load(open('/work/run1.json'))
out={}
for T,seed in [(1.5,777),(2.2,778),(2.3,779),(2.5,780),(2.8,781)]:
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    for _ in range(3000): sweep(s,T,rng)
    m2s=np.zeros(R); m4s=np.zeros(R); nm=6000
    for _ in range(nm):
        sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1); m2s+=m**2; m4s+=m**4
    U,Ue=jack_U(m2s/nm,m4s/nm)
    print(T,round(U,5),round(Ue,5),' run1:',round(run1[str(T)]['U'],5),round(run1[str(T)]['Uerr'],5),flush=True)
