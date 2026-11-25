
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, sys, time, json
N=16
def run(T, R, nsweep, nburn, seed, outname):
    rng = np.random.default_rng(seed)
    # half initial replicas all-up (ordered), half random
    s = np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    s[:R//2]=1
    # acceptance table indexed by dE+8, dE=2*s*nb in {-8..8}
    dEs=np.arange(-8,9)
    pacc=np.where(dEs<=0,1.0,np.exp(-np.maximum(dEs,0)/T))
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    sum2=np.zeros(R); sum4=np.zeros(R); cnt=0
    tot=nburn+nsweep
    for sw in range(tot):
        for p in (0,1):
            nb = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE = (2*s*nb).astype(np.int16)
            pr = pacc[dE+8]
            acc = rng.random((R,N,N)) < pr
            flip = acc & masks[p][None]
            s = np.where(flip, -s, s).astype(np.int8)
        if sw>=nburn:
            m = s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            m2=m*m
            sum2+=m2; sum4+=m2*m2; cnt+=1
    np.savez(outname, sum2=sum2, sum4=sum4, cnt=cnt)
    return sum2,sum4,cnt
if __name__=='__main__':
    T=float(sys.argv[1]); R=int(sys.argv[2]); ns=int(sys.argv[3]); nb=int(sys.argv[4]); seed=int(sys.argv[5])
    t0=time.time()
    run(T,R,ns,nb,seed,sys.argv[6])
    print('time',time.time()-t0)
'''
open('/work/ising_meta.py','w').write(code)
import subprocess
r=subprocess.run(['python3','/work/ising_meta.py','2.269','1024','20','0','1','/tmp/test.npz'],capture_output=True,text=True)
print(r.stdout,r.stderr)
