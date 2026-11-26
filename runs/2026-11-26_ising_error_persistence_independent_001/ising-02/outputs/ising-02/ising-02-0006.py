import numpy as np
def U_of(S2,S4):
    m2=S2.mean(axis=-1); m4=S4.mean(axis=-1)
    return 1-m4/(3*m2**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T:.2f}.npz'); S2=d['S2']; S4=d['S4']
    C=S2.shape[0]
    # per-chain U
    Uc=U_of(S2,S4)
    # full-sample U (pooled)
    U=U_of(S2.mean(0)[None],S4.mean(0)[None])[0]
    # jackknife over chains
    jk=np.array([U_of(np.delete(S2,c,0).mean(0)[None],np.delete(S4,c,0).mean(0)[None])[0] for c in range(C)])
    err=np.sqrt((C-1)/C*np.sum((jk-jk.mean())**2))
    # consistency: first vs second half of blocks
    h=S2.shape[1]//2
    U1=U_of(S2[:,:h].mean(0)[None],S4[:,:h].mean(0)[None])[0]
    U2=U_of(S2[:,h:].mean(0)[None],S4[:,h:].mean(0)[None])[0]
    print(f"T={T} U={U:.5f} err_jk={err:.5f} chain-std/sqrt(C)={Uc.std(ddof=1)/np.sqrt(C):.5f} halves {U1:.5f} {U2:.5f} <m2>={S2.mean():.4f}")
