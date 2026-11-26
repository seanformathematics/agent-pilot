
script = r'''
import numpy as np, time, os, pickle, sys
N=16
R=int(os.environ.get("R","1000"))
i,j=np.indices((N,N))
MA=((i+j)%2==0); MB=~MA
# acceptance table indexed by dE (0,4,8) for positive dE
def run(T, nsweeps, seed, tlimit, tag="", discard=0):
    fn=f"/work/state_T{T:.3f}_R{R}_{tag}.pkl"
    beta=1.0/T
    if os.path.exists(fn):
        st=pickle.load(open(fn,"rb"))
        s=st["s"]; rng=np.random.default_rng(); rng.bit_generator.state=st["rng"]
        acc=st["acc"]; done=st["sweeps"]
    else:
        rng=np.random.default_rng(seed)
        s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
        acc={"n":0,"S2":np.zeros(R),"S4":np.zeros(R),"S1":np.zeros(R),"Sm":np.zeros(R)}
        done=0
    ptab=np.array([1.0,1.0,np.exp(-4*beta),np.exp(-8*beta)])  # index dE/4 for dE in {0,4,8} (dE<=0 ->1)
    t0=time.time(); k=0
    while k<nsweeps and time.time()-t0<tlimit:
        for mask in (MA,MB):
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb  # in {-8..8}
            p=np.where(dE<=0,1.0,np.exp(-beta*np.maximum(dE,0)))
            flip=(rng.random(s.shape)<p)&mask
            s[flip]*=-1
        k+=1; done+=1
        if done>discard:
            M=s.reshape(R,-1).sum(1).astype(np.float64)
            acc["n"]+=1; acc["S2"]+=M*M; acc["S4"]+=M**4; acc["S1"]+=np.abs(M); acc["Sm"]+=M
    st={"s":s,"rng":rng.bit_generator.state,"acc":acc,"sweeps":done}
    pickle.dump(st,open(fn,"wb"))
    return done, acc
if __name__=="__main__":
    T=float(sys.argv[1]); ns=int(sys.argv[2]); seed=int(sys.argv[3]); tl=float(sys.argv[4]); tag=sys.argv[5] if len(sys.argv)>5 else ""
    disc=int(sys.argv[6]) if len(sys.argv)>6 else 0
    done,acc=run(T,ns,seed,tl,tag,disc)
    n=acc["n"]
    if n>0:
        m2=acc["S2"]/n/N**4; m4=acc["S4"]/n/N**8
        print("T",T,"sweeps_done",done,"meas",n,"U_pooled",1-m4.mean()/(3*m2.mean()**2))
'''
open('/work/ising.py','w').write(script)
import subprocess
r=subprocess.run(["python","/work/ising.py","2.269","3000","11","200","pilot","500"],capture_output=True,text=True,cwd="/work")
print(r.stdout,r.stderr[-2000:])
