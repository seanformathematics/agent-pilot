
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_est(M):
    m2=(M**2).mean(); m4=(M**4).mean()
    return 1-m4/(3*m2*m2)
rng=np.random.default_rng(12345)
for T in Ts:
    M=np.load(f'/work/M_{T}.npy')
    R=M.shape[1]
    U=U_est(M)
    # bootstrap over replicas
    B=[]
    for b in range(400):
        idx=rng.integers(0,R,R)
        B.append(U_est(M[:,idx]))
    B=np.array(B)
    Uh1=U_est(M[:400]); Uh2=U_est(M[400:])
    print(f"T={T} U={U:.5f} err={B.std():.5f} halves={Uh1:.4f},{Uh2:.4f} <m2>={(M**2).mean():.4g}")
