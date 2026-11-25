script = r'''
import numpy as np, time, sys, json
N=16
R=2000
def run(T, seed, burn, meas, nblocks=10):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
    acc_tab=np.zeros(9,dtype=np.float32)
    for k in range(9):
        d=2*k-8
        acc_tab[k]= 1.0 if d<=0 else np.exp(-d/T)
    def sweep(S):
        for m in masks:
            nn = (np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE = 2*S*nn
            u = rng.random(S.shape, dtype=np.float32)
            acc = (u < acc_tab[(dE+8)//2]) & m
            S = np.where(acc, -S, S)
        return S
    for _ in range(burn):
        S=sweep(S)
    S2=np.zeros((nblocks,R)); S4=np.zeros((nblocks,R)); M1=np.zeros((nblocks,R))
    per=meas//nblocks
    for b in range(nblocks):
        for _ in range(per):
            S=sweep(S)
            m=S.reshape(R,-1).mean(1,dtype=np.float64)
            m2=m*m
            S2[b]+=m2; S4[b]+=m2*m2; M1[b]+=np.abs(m)
    return S2/per, S4/per, M1/per

if __name__=="__main__":
    T=float(sys.argv[1]); seed=int(sys.argv[2]); burn=int(sys.argv[3]); meas=int(sys.argv[4])
    t0=time.time()
    s2,s4,m1=run(T,seed,burn,meas)
    np.savez(f"/work/res_T{T:.2f}_s{seed}.npz", s2=s2, s4=s4, m1=m1, T=T, seed=seed, burn=burn, meas=meas)
    U=1-s4.mean()/(3*s2.mean()**2)
    print(f"T={T} seed={seed} time={time.time()-t0:.1f}s U={U:.5f}")
    print("block U:", [round(float(1-s4[b].mean()/(3*s2[b].mean()**2)),4) for b in range(s2.shape[0])])
    print("block <|m|>:", [round(float(m1[b].mean()),4) for b in range(m1.shape[0])])
'''
open('/work/ising.py','w').write(script)
import subprocess
for T,seed in [(1.9,101),(2.0,102)]:
    r=subprocess.run(['python','/work/ising.py',str(T),str(seed),'2000','10000'],capture_output=True,text=True,timeout=280)
    print(r.stdout, r.stderr[-2000:])