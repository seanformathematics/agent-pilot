import numpy as np, glob
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
for T in Ts:
    d=np.load(f'/work/res_N16_T{T:.2f}.npz')
    m2,m4=d['m2'],d['m4']; R=len(m2)
    U=U_of(m2,m4)
    # delete-block jackknife over replicas (50 blocks of 20)
    nb=50; idx=np.arange(R)%nb
    Uj=np.array([U_of(m2[idx!=b],m4[idx!=b]) for b in range(nb)])
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    # bootstrap check
    rng=np.random.default_rng(0)
    bs=[U_of(m2[k],m4[k]) for k in (rng.integers(0,R,R) for _ in range(300))]
    print(f"T={T:.2f} U={U:.5f} jack_err={err:.5f} boot_err={np.std(bs):.5f}  <m2>={m2.mean():.4f}")
