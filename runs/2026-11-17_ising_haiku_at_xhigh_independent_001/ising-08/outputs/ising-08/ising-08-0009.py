import numpy as np, time, sys
sys.path.insert(0,'/work')
import ising
R=400; nsw=12000; W=1000
t0=time.time()
for T in [2.3,2.2,2.4]:
    rng=np.random.default_rng(777+int(T*10))
    s=np.where(rng.random((R,16,16))<0.5,1,-1).astype(np.int8)
    m2s=np.zeros((nsw,)); m4s=np.zeros((nsw,))
    for k in range(nsw):
        s=ising.sweep(s,T,rng)
        m=s.reshape(R,-1).mean(1); m2=m*m
        m2s[k]=m2.mean(); m4s[k]=(m2*m2).mean()
    print("T",T,"time",round(time.time()-t0,1))
    for w in range(0,nsw,W):
        a2=m2s[w:w+W].mean(); a4=m4s[w:w+W].mean()
        print(f"  sweeps {w}-{w+W}: <m2>={a2:.4f} U_win={1-a4/(3*a2*a2):.4f}")
    # block-averaged cumulative from the end
    np.save(f'/work/diag_T{T:.2f}.npy',np.vstack([m2s,m4s]))
