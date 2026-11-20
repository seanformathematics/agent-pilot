
import subprocess, numpy as np
for T,seed in [(2.1,103),(2.2,104)]:
    r=subprocess.run(['python3','/work/ising.py',str(T),'400','50000','5000',str(seed),f'/work/run_{T}.npz'],capture_output=True,text=True,timeout=280)
    print(r.stdout,r.stderr[-800:])
def analyze(fn):
    d=np.load(fn); m2=d['m2']; m4=d['m4']  # (nblocks,R)
    nb=m2.shape[0]
    # per-replica averages
    M2=m2.mean(0); M4=m4.mean(0)
    U=1-M4.mean()/(3*M2.mean()**2)
    R=len(M2)
    jk=np.empty(R)
    S2=M2.sum(); S4=M4.sum()
    for r_ in range(R):
        a2=(S2-M2[r_])/(R-1); a4=(S4-M4[r_])/(R-1)
        jk[r_]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # block-of-time check: U from first and second half
    h=nb//2
    def Ut(sl):
        a2=m2[sl].mean(0).mean(); a4=m4[sl].mean(0).mean()
        return 1-a4/(3*a2**2)
    return U,err,Ut(slice(0,h)),Ut(slice(h,nb)), M2.mean(), M4.mean()
for T in [1.9,2.0,2.1,2.2]:
    try:
        print(T, analyze(f'/work/run_{T}.npz'))
    except Exception as e: print(T,e)
