
import sys, time
sys.path.insert(0,'/work')
import numpy as np
import ising_sw as M

T=2.8
t=time.time()
m2b,m4b=M.run(T,seed=777,R=256,burn=300,L=1000,block=10)
M2=m2b.mean(0); M4=m4b.mean(0)
print("SW: <m^2>*V=",M2.mean()*256," U=",1-M2.mean()*0+ (1-M4.mean()/(3*M2.mean()**2)), "time",time.time()-t)

# independent Metropolis at T=2.8
def metro(R,T,nsweep,nburn,seed,N=16):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    m2s=[];m4s=[]
    for sw in range(nburn+nsweep):
        for c in (0,1):
            nb=(np.roll(s,1,1).astype(np.int16)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            prob=np.exp(-dE/T)
            flip=(rng.random(s.shape)<prob)&masks[c][None]
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=nburn:
            m=s.reshape(R,-1).mean(1,dtype=np.float64)
            m2s.append(m*m); m4s.append(m**4)
    m2s=np.array(m2s); m4s=np.array(m4s)
    return m2s.mean(0), m4s.mean(0)
t=time.time()
A2,A4=metro(128,T,1000,300,99)
print("Metro: <m^2>*V=",A2.mean()*256," U=",1-A4.mean()/(3*A2.mean()**2), "time",time.time()-t)

# susceptibility via correlations from SW configurations
rng=np.random.default_rng(5)
sim=M.SW(64,T,rng)
for _ in range(300): sim.sweep()
chi_acc=np.zeros((16,16)); nconf=0
for k in range(400):
    sim.sweep()
    s=sim.s.reshape(64,16,16).astype(float)
    F=np.fft.fft2(s)
    corr=np.fft.ifft2(np.abs(F)**2).real/256
    chi_acc+=corr.mean(0); nconf+=64
chi_acc/=(nconf/64)
print("chi=sum_r <s0 sr> =", chi_acc.sum(), " => <m^2>*V =", chi_acc.sum())
# kappa4 check
m=sim.mags()
print("sample means: m2",np.mean(m*m)*256,"U_from64", 1-np.mean(m**4)/(3*np.mean(m*m)**2))
