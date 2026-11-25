code = r'''
import numpy as np, time, sys, json
N=16
def run(T, seed, R=2000, burn=1000, meas=15000, block=100):
    rng=np.random.default_rng(seed)
    i,j=np.indices((N,N))
    masks=[((i+j)%2==0),((i+j)%2==1)]
    dEs=np.array([-8,-4,0,4,8])
    acc=np.minimum(1.0,np.exp(-dEs/T))
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    def sweep():
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            p=acc[(dE//4)+2]
            flip=(rng.random(s.shape)<p)&mk
            s[flip]*=-1
    for _ in range(burn): sweep()
    nb_blocks=meas//block
    m2b=np.zeros((nb_blocks,R)); m4b=np.zeros((nb_blocks,R)); mabs=np.zeros((nb_blocks,R))
    for b in range(nb_blocks):
        a2=np.zeros(R); a4=np.zeros(R); aa=np.zeros(R)
        for _ in range(block):
            sweep()
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m
            a2+=m2; a4+=m2*m2; aa+=np.abs(m)
        m2b[b]=a2/block; m4b[b]=a4/block; mabs[b]=aa/block
    return m2b,m4b,mabs
if __name__=="__main__":
    T=float(sys.argv[1]); seed=int(sys.argv[2]); R=int(sys.argv[3]); meas=int(sys.argv[4])
    t=time.time()
    m2b,m4b,mabs=run(T,seed,R=R,meas=meas)
    np.savez(f"/work/run_T{T}_s{seed}.npz",m2b=m2b,m4b=m4b,mabs=mabs,T=T,seed=seed)
    print("done",T,seed,time.time()-t)
'''
open('/work/ising.py','w').write(code)
import subprocess
r=subprocess.run(["python3","/work/ising.py","2.3","101","2000","15000"],capture_output=True,text=True,timeout=580,cwd="/work")
print(r.stdout,r.stderr[-2000:])