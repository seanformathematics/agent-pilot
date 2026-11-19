
import sys, time
import numpy as np
sys.path.insert(0,'/work')
from sw import SWReplicas
from ana import jackknife_U

N=8
nstates=2**N
bits=((np.arange(nstates)[:,None]>>np.arange(N)[None,:])&1)*2-1
mu=bits.sum(axis=1)
e_in=-(bits*np.roll(bits,-1,axis=1)).sum(axis=1)
e_x=-(bits@bits.T)
nM=N*N+1

def exact_P(T):
    K=1.0/T
    w_in=np.exp(-K*(e_in-e_in.min()))
    Tx=np.exp(-K*(e_x-e_x.min()))
    V=np.zeros((nstates,nM,nstates))
    k0=(mu+N*N)//2
    for s0 in range(nstates):
        V[s0,k0[s0],s0]=w_in[s0]
    shifts=mu//2
    sels={d:np.where(shifts==d)[0] for d in range(-N//2,N//2+1)}
    for step in range(N-1):
        Um=(V.reshape(-1,nstates)@Tx).reshape(nstates,nM,nstates)*w_in[None,None,:]
        V=np.zeros_like(Um)
        for d,sel in sels.items():
            if len(sel)==0: continue
            if d>=0:
                V[:,d:,sel]=Um[:,:nM-d,sel]
            else:
                V[:,:nM+d,sel]=Um[:,-d:,sel]
    P=np.einsum('akb,ba->k',V,Tx)
    return P/P.sum()

Mvals=(2*np.arange(nM)-N*N)/(N*N)       # m values
Rr=1000; nburn=100; nmeas=4000
for iT,T in enumerate([2.0,2.269,2.6]):
    t0=time.time()
    P=exact_P(T)
    sym=np.max(np.abs(P-P[::-1]))
    m2=(P*Mvals**2).sum(); m4=(P*Mvals**4).sum(); U_ex=1-m4/(3*m2**2)
    t1=time.time()
    rng=np.random.default_rng(424200+iT)
    sim=SWReplicas(N,Rr,T,rng,'hot')
    for _ in range(nburn): sim.sweep()
    S2=np.zeros(Rr); S4=np.zeros(Rr)
    for _ in range(nmeas):
        sim.sweep()
        mm=sim.magnetization_sum()/N**2
        S2+=mm**2; S4+=mm**4
    U,err,_=jackknife_U(S2,S4,nmeas)
    print(f"N=8 T={T}: exact U={U_ex:.5f} (<m2>={m2:.5f}, symmetry err {sym:.1e}) | SW U={U:.5f}+-{err:.5f}, <m2>_mc={S2.sum()/(Rr*nmeas):.5f}  z={(U-U_ex)/err:+.2f}  [exact {t1-t0:.0f}s, mc {time.time()-t1:.0f}s]")
