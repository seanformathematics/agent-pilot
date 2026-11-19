
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from sw import SWBatch
from prod import Ufun

def lean_run(N,T,R,neq,steps,seed,nb=20):
    Ns=N*N
    sim = SWBatch(N,R,T,seed=seed,start=np.arange(R)%2)
    for _ in range(neq): sim.step()
    L=steps//nb; blocks=np.zeros((nb,R,4))
    for b in range(nb):
        acc=np.zeros((R,4))
        for t in range(L):
            M,S2,S4 = sim.step()
            m2=(M/Ns)**2
            acc[:,0]+=m2; acc[:,1]+=m2*m2
            acc[:,2]+=S2/Ns**2; acc[:,3]+=(3*S2**2-2*S4)/Ns**4
        blocks[b]=acc/L
    return blocks

def report(blocks, exactU=None, label=""):
    chain = blocks.mean(axis=0); n=chain.shape[0]; tot=chain.sum(axis=0)
    loo=(tot[None,:]-chain)/(n-1)
    res={}
    for name,(c2,c4) in (("direct",(0,1)),("improved",(2,3))):
        U=Ufun(tot[c2]/n,tot[c4]/n); Ul=Ufun(loo[:,c2],loo[:,c4])
        err=np.sqrt((n-1)/n*((Ul-Ul.mean())**2).sum()); res[name]=(U,err,Ul)
    dif=res['improved'][2]-res['direct'][2]; dife=np.sqrt((n-1)/n*((dif-dif.mean())**2).sum())
    s="%s direct %.5f+-%.5f | improved %.5f+-%.5f | impr-dir %.5f+-%.5f (z=%.2f)"%(label,res['direct'][0],res['direct'][1],res['improved'][0],res['improved'][1],res['improved'][0]-res['direct'][0],dife,(res['improved'][0]-res['direct'][0])/dife)
    if exactU is not None:
        s+=" | exact %.5f: dev direct %.2fσ, improved %.2fσ"%(exactU,(res['direct'][0]-exactU)/res['direct'][1],(res['improved'][0]-exactU)/res['improved'][1])
    print(s, flush=True)

# exact N=4
N=4; Ns=16
states = np.arange(2**Ns, dtype=np.uint32)
bits = ((states[:,None] >> np.arange(Ns, dtype=np.uint32)[None,:]) & 1).astype(np.int8)*2-1
S = bits.reshape(-1,N,N)
E = -(S*np.roll(S,-1,axis=2)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=1)).sum(axis=(1,2))
m = S.sum(axis=(1,2)).astype(np.float64)/Ns
def exactU(T):
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    return 1-(w*m**4).sum()/(3*(w*m**2).sum()**2)

t0=time.time()
for T,seed in ((2.269,900101),(5.0,900102)):
    blocks = lean_run(4,T,R=2000,neq=100,steps=10000,seed=seed)
    report(blocks, exactU(T), "N=4 T=%.3f (20M samples):"%T)
    print("   [%.0fs]"%(time.time()-t0), flush=True)

blocks = lean_run(16,2.8,R=500,neq=300,steps=10000,seed=900103)
report(blocks, None, "N=16 T=2.8 (5M samples):")
print("   [%.0fs]"%(time.time()-t0))
