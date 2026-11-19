
import sys, time
sys.path.insert(0,'/work')
import numpy as np
from metro import MetroBatch
from prod import analyze

def pooled_SW(T):
    z=np.load("/work/prod_T%.1f.npz"%T); blocks=z['blocks']; K,nb,R,_=blocks.shape
    pooled=blocks.transpose(1,0,2,3).reshape(nb,K*R,4)
    return analyze(pooled,'i')

N=16; Ns=N*N; R=1000; NEQ=1000; NS=10000
t0=time.time()
metro_out={}
for T,coldonly,seed in ((1.7,True,810001),(2.0,True,810002),(2.3,False,810003),(2.6,False,810004),(2.8,False,810005)):
    start = np.ones(R,bool) if coldonly else (np.arange(R)%2)
    sim=MetroBatch(N,R,T,seed=seed,start=start)
    for _ in range(NEQ): sim.sweep()
    acc=np.zeros((R,2)); half=np.zeros((2,R,2)); 
    for t in range(NS):
        M=sim.sweep()
        mm=(M/Ns)**2
        v=np.stack([mm,mm*mm],axis=1)
        acc+=v; half[t*2//NS]+=v
    c=acc/NS; n=R; tot=c.sum(axis=0)
    U=1-tot[1]/n/(3*(tot[0]/n)**2)
    loo=(tot[None,:]-c)/(n-1); Ul=1-loo[:,1]/(3*loo[:,0]**2)
    err=np.sqrt((n-1)/n*((Ul-Ul.mean())**2).sum())
    # first-half vs second-half drift test
    h=half/(NS/2); Uh=[]
    for j in range(2):
        tt=h[j].sum(axis=0); Uh.append(1-tt[1]/n/(3*(tt[0]/n)**2))
    sw=pooled_SW(T)
    z=(U-sw['U'])/np.hypot(err,sw['err_chain'])
    metro_out[T]=(U,err)
    print("T=%.1f Metropolis U=%.5f+-%.5f (halves: %.5f %.5f) | SW U=%.5f+-%.5f | diff z=%.2f  [%.0fs]"%(T,U,err,Uh[0],Uh[1],sw['U'],sw['err_chain'],z,time.time()-t0),flush=True)
