
import numpy as np, math, json, glob
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(m):
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2*m2)
def jk(chains,nbins):
    blocks=[]
    for c in chains:
        bl=len(c)//nbins
        blocks.append(c[:bl*nbins].reshape(nbins,bl))
    blocks=np.concatenate(blocks,axis=0)
    nb_=blocks.shape[0]
    s2=(blocks**2).sum(1); s4=(blocks**4).sum(1); cnt=blocks.shape[1]
    S2=s2.sum(); S4=s4.sum(); Ntot=cnt*nb_
    m2=(S2-s2)/(Ntot-cnt); m4=(S4-s4)/(Ntot-cnt)
    Ujk=1-m4/(3*m2*m2)
    return math.sqrt((nb_-1)/nb_*np.sum((Ujk-Ujk.mean())**2))
out={}
for T in Ts:
    d=np.load(f"/work/m_T{T:.2f}.npz")
    chains=[d[k].astype(np.float64) for k in d.files]
    allm=np.concatenate(chains)
    U=U_of(allm)
    Uc=np.array([U_of(c) for c in chains])
    e_chain=Uc.std(ddof=1)/math.sqrt(len(chains))
    e20=jk(chains,20); e50=jk(chains,50); e100=jk(chains,100)
    out[T]=dict(U=U,e20=e20,e50=e50,e100=e100,e_chain=e_chain,n=len(allm))
    print(f"T={T:.2f} U={U:.6f} jk20={e20:.3e} jk50={e50:.3e} jk100={e100:.3e} chain={e_chain:.3e} nsamp={len(allm)}")
# Tc check via pilot-style run already done; print summary
