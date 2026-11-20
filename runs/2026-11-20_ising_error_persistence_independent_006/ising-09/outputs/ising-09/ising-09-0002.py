import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, time, sys, json
def run(T, R, nequil, nmeas, seed, N=16, start=None):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N)) if start is None else start
    ii,jj = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    mask = ((ii+jj)%2==0)
    # acceptance table for h = s*nb in {-4,-2,0,2,4}; dE = 2h
    ptab = np.zeros(9)
    for h in (-4,-2,0,2,4):
        ptab[h+4] = min(1.0, np.exp(-2*h/T))
    def sweep(s):
        for m in (mask, ~mask):
            nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            h = (s*nb).astype(np.int16)
            p = ptab[h+4]
            u = rng.random(s.shape)
            flip = (u < p) & m
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nequil):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R); S1=np.zeros(R)
    for _ in range(nmeas):
        s = sweep(s)
        M = s.sum(axis=(1,2), dtype=np.int64)
        m = M / (N*N)
        m2 = m*m
        S2 += m2; S4 += m2*m2; S1 += np.abs(m)
    return S2/nmeas, S4/nmeas, S1/nmeas, s
if __name__ == "__main__":
    T=float(sys.argv[1]); R=int(sys.argv[2]); ne=int(sys.argv[3]); nm=int(sys.argv[4]); seed=int(sys.argv[5])
    t=time.time()
    m2,m4,m1,s = run(T,R,ne,nm,seed)
    print("time",time.time()-t)
'''
open('/work/ising.py','w').write(code)
import subprocess
r=subprocess.run(['python','/work/ising.py','2.269','500','0','50','1'],capture_output=True,text=True)
print(r.stdout,r.stderr)
r=subprocess.run(['python','/work/ising.py','2.269','500','0','200','1'],capture_output=True,text=True)
print(r.stdout,r.stderr)