
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
def jk_err(m2,m4,nb=40):
    n=len(m2)//nb*nb
    A=m2[:n].reshape(nb,-1).mean(1); B=m4[:n].reshape(nb,-1).mean(1)
    SA,SB=A.sum(),B.sum()
    Uj=np.array([1-((SB-B[k])/(nb-1))/(3*((SA-A[k])/(nb-1))**2) for k in range(nb)])
    return np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
for T in Ts:
    ch=[np.load(f'/work/chain_T{T}_c{c}.npz') for c in range(4)]
    m2=np.concatenate([c['m2'] for c in ch]); m4=np.concatenate([c['m4'] for c in ch])
    Ucs=[U(c['m2'],c['m4']) for c in ch]
    # error from chain-to-chain scatter (4 chains) 
    sc=np.std(Ucs,ddof=1)/2
    # jackknife over blocks in each chain, combined: use per-chain jackknife, average
    ej=np.mean([jk_err(c['m2'],c['m4']) for c in ch])
    print(f"T={T} U={U(m2,m4):.5f}  chains={np.round(Ucs,5)}  chain-scatter/sqrt4={sc:.5f}  jackknife(mean chain)={ej:.5f}  nsamp={len(m2)}")
