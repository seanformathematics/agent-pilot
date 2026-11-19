
import numpy as np, subprocess, glob
def analyze(fn):
    d=np.load(fn)
    S2,S4=d['S2'],d['S4']
    # per replica sums over all blocks
    a2=S2.sum(0); a4=S4.sum(0); n=S2.shape[0]*int(d['blk'])
    m2=a2/n; m4=a4/n
    R=len(m2)
    U_r=1-m4/(3*m2**2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas
    Uj=np.empty(R)
    tot2=a2.sum(); tot4=a4.sum()
    for r in range(R):
        M2=(tot2-a2[r])/((R-1)*n); M4=(tot4-a4[r])/((R-1)*n)
        Uj[r]=1-M4/(3*M2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # time-block check: U from first and second half
    h=S2.shape[0]//2
    def Ut(sl):
        mm2=S2[sl].sum(0); mm4=S4[sl].sum(0); nn=S2[sl].shape[0]*int(d['blk'])
        return 1-(mm4.mean()/nn)/(3*(mm2.mean()/nn)**2)
    return U, err, Ut(slice(0,h)), Ut(slice(h,None)), np.mean(m2/1)
for f in sorted(glob.glob('/work/run_*.npz')):
    print(f, analyze(f))
for T,seed in [(2.0,102),(2.1,103),(2.2,104)]:
    r=subprocess.run(['python3','/work/ising.py',str(T),str(seed),'1000','2000','20000','500'],capture_output=True,text=True,timeout=600)
    print(r.stdout.strip(), r.stderr[-500:])
