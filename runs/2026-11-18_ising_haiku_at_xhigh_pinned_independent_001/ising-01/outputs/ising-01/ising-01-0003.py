
import numpy as np, time
from scipy.integrate import quad
from scipy.special import ellipk

def onsager_e(T):
    b=1/T
    k=2*np.sinh(2*b)/np.cosh(2*b)**2
    Kk=ellipk(k**2)
    return -(1/np.tanh(2*b))*(1+(2/np.pi)*(2*np.tanh(2*b)**2-1)*Kk)

# independent check: Onsager via integral representation of energy
def onsager_int(T):
    b=1/T
    # u = -J coth(2K)[1 + (2/pi)(2 tanh^2(2K)-1) K(k)], K(k)=int_0^{pi/2} dθ / sqrt(1-k^2 sin^2 θ)
    k=2*np.sinh(2*b)/np.cosh(2*b)**2
    Kint=quad(lambda th: 1/np.sqrt(1-k*k*np.sin(th)**2),0,np.pi/2)[0]
    return -(1/np.tanh(2*b))*(1+(2/np.pi)*(2*np.tanh(2*b)**2-1)*Kint)

for T in [2.0,2.2,2.25,2.3,2.4,2.6,2.8]:
    print(T, onsager_e(T), onsager_int(T))

# Metropolis checkerboard vectorized over replicas, independent of SW
def metro_energy(R,T,nsweep,nburn,seed):
    rng=np.random.default_rng(seed)
    N=16
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    acc=[]
    for sw in range(nburn+nsweep):
        for c in (0,1):
            nb=(np.roll(s,1,1).astype(np.int16)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb  # energy change if flipped
            prob=np.exp(-dE/T)
            flip=(rng.random(s.shape)<prob)&masks[c][None]
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=nburn:
            e=-(s*np.roll(s,1,1)).sum((1,2))-(s*np.roll(s,1,2)).sum((1,2))
            acc.append(e.mean()/N/N)
    acc=np.array(acc)
    return acc.mean(), acc.std()/np.sqrt(len(acc))

t=time.time()
for T in [2.3]:
    m,e=metro_energy(128,T,1500,500,7)
    print("Metropolis N=16 T",T,"E/site",m,"+-",e, "time",time.time()-t)
