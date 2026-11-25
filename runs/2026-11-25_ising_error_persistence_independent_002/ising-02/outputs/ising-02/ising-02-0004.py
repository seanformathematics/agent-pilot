import subprocess, numpy as np, glob
for T,seed in [(2.1,103),(2.2,104)]:
    r=subprocess.run(['python','/work/ising.py',str(T),str(seed),'2000','10000'],capture_output=True,text=True,timeout=280)
    print(r.stdout, r.stderr[-1000:])
def jk(f):
    s2,s4,m1=f['s2'],f['s4'],f['m1']
    a2=s2.mean(0); a4=s4.mean(0)  # per-replica means
    R=a2.size
    def U(x2,x4): return 1-x4.mean()/(3*x2.mean()**2)
    full=U(a2,a4)
    loo=np.array([U(np.delete(a2,i),np.delete(a4,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return full,err
for fn in sorted(glob.glob('/work/res_*.npz')):
    f=np.load(fn)
    U,e=jk(f)
    print(fn, f['T'], f['seed'], f"U={U:.5f} err={e:.5f}")