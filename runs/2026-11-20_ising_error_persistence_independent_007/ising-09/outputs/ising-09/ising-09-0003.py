code = r'''
import numpy as np, sys, time, json
def run(T, R, n_eq, n_meas, seed, N=16):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==c) for c in (0,1)]
    # acceptance table indexed by dE+8, dE in {-8,-4,0,4,8}
    tab = np.zeros(17)
    for dE in range(-8,9,4):
        tab[dE+8] = 1.0 if dE<=0 else np.exp(-dE/T)
    def sweep():
        for c in (0,1):
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = (2*s*nn).astype(np.int8)
            acc = tab[dE.astype(np.int16)+8]
            r = rng.random((R,N,N))
            flip = masks[c] & (r < acc)
            s[flip] *= -1
    for _ in range(n_eq):
        sweep()
    S2 = np.zeros(R); S4 = np.zeros(R); S1=np.zeros(R)
    V = N*N
    for _ in range(n_meas):
        sweep()
        m = s.sum(axis=(1,2),dtype=np.int64)/V
        m2 = m*m
        S2 += m2; S4 += m2*m2; S1 += np.abs(m)
    return S2/n_meas, S4/n_meas, S1/n_meas
if __name__ == "__main__":
    T=float(sys.argv[1]); R=int(sys.argv[2]); neq=int(sys.argv[3]); nm=int(sys.argv[4]); seed=int(sys.argv[5])
    t=time.time()
    m2,m4,m1 = run(T,R,neq,nm,seed)
    n=len(m2)
    U = 1 - np.mean(m4)/(3*np.mean(m2)**2)
    print("time",time.time()-t, "U",U, "m2",np.mean(m2), "m1", np.mean(m1))
'''
open('/work/ising.py','w').write(code) if __import__('os').path.isdir('/work') else None
import os
os.makedirs('/work',exist_ok=True)
open('/work/ising.py','w').write(code)
import subprocess
print(subprocess.run(['python3','/work/ising.py','2.3','512','0','200','1'],capture_output=True,text=True,cwd='/work'))
print(subprocess.run(['python3','/work/ising.py','2.0','512','0','200','1'],capture_output=True,text=True,cwd='/work').stdout)
