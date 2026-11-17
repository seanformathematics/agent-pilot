code = r'''
import numpy as np, time, os, sys
N=16; TS=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
RPT=500
Ts=np.repeat(np.array(TS),RPT); RT=len(Ts)
SEED=12345
BURN=1000
ckpt='/work/state.npz'
if os.path.exists(ckpt):
    d=np.load(ckpt); S=d['S']; A2=d['A2']; A4=d['A4']; nsw=int(d['nsw'])
else:
    r0=np.random.default_rng(SEED)
    S=(r0.integers(0,2,(RT,N,N))*2-1).astype(np.int8)
    A2=np.zeros(RT); A4=np.zeros(RT); nsw=0
rng=np.random.default_rng([SEED,nsw,7])
mask_black=((np.add.outer(np.arange(N),np.arange(N))%2)==0)
p4=np.exp(-4.0/Ts).astype(np.float32)[:,None,None]; p8=np.exp(-8.0/Ts).astype(np.float32)[:,None,None]
def half(S,mask):
    nn=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
    dE=2*S*nn
    thr=np.where(dE<=0,np.float32(2.0),np.where(dE==4,p4,p8))
    flip=(rng.random(S.shape,dtype=np.float32)<thr)&mask
    S[flip]*=-1
def sweep(S):
    half(S,mask_black); half(S,~mask_black)
def run(tmax):
    global S,A2,A4,nsw
    t0=time.time()
    while time.time()-t0<tmax:
        sweep(S); nsw+=1
        m=S.sum((1,2),dtype=np.int64)/(N*N)
        if nsw>BURN:
            m2=m*m; A2+=m2; A4+=m2*m2
        if nsw%500==0:
            np.savez(ckpt,S=S,A2=A2,A4=A4,nsw=nsw)
    np.savez(ckpt,S=S,A2=A2,A4=A4,nsw=nsw)
def analyze(A2,A4,nmeas,G=50):
    out=[]
    for k,T in enumerate(TS):
        sl=slice(k*RPT,(k+1)*RPT)
        a2=A2[sl]/nmeas; a4=A4[sl]/nmeas   # per-replica time averages
        def U(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
        u=U(a2,a4)
        # jackknife over replicas in G groups
        idx=np.arange(RPT)%G
        js=[]
        for g in range(G):
            keep=idx!=g
            js.append(U(a2[keep],a4[keep]))
        js=np.array(js)
        err=np.sqrt((G-1)/G*np.sum((js-js.mean())**2))
        out.append((T,u,err))
    return out
if __name__=='__main__':
    tmax=float(sys.argv[1]) if len(sys.argv)>1 else 400
    run(tmax)
    nmeas=nsw-BURN
    print('nsw',nsw,'nmeas',nmeas)
    for T,u,e in analyze(A2,A4,nmeas):
        print(f'T={T:.2f} U={u:.5f} err={e:.5f}')
'''
open('/work/sim.py','w').write(code)
import subprocess
r=subprocess.run(['python3','/work/sim.py','400'],capture_output=True,text=True,timeout=590,cwd='/work')
print(r.stdout, r.stderr[-2000:])
