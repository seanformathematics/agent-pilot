import subprocess, numpy as np, glob
for T,seed in [(2.3,105),(2.4,106)]:
    r=subprocess.run(['python','/work/ising.py',str(T),str(seed),'2000','10000'],capture_output=True,text=True,timeout=280)
    print(r.stdout, r.stderr[-1000:])
def U(x2,x4): return 1-x4.mean()/(3*x2.mean()**2)
rng=np.random.default_rng(999)
for fn in sorted(glob.glob('/work/res_*.npz')):
    f=np.load(fn)
    s2,s4=f['s2'],f['s4']
    a2=s2.mean(0); a4=s4.mean(0); R=a2.size
    full=U(a2,a4)
    loo=np.array([U(np.delete(a2,i),np.delete(a4,i)) for i in range(R)])
    jk=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    boot=[]
    for _ in range(300):
        idx=rng.integers(0,R,R)
        boot.append(U(a2[idx],a4[idx]))
    bs=np.std(boot)
    # block-level: 10 time blocks, jackknife over blocks (pooled over replicas)
    bl=np.array([U(np.delete(s2,b,0).mean(0),np.delete(s4,b,0).mean(0)) for b in range(s2.shape[0])])
    nb=s2.shape[0]
    bj=np.sqrt((nb-1)/nb*np.sum((bl-bl.mean())**2))
    print(f"T={float(f['T']):.2f} U={full:.5f} jk_rep={jk:.6f} boot_rep={bs:.6f} jk_timeblocks={bj:.6f}")