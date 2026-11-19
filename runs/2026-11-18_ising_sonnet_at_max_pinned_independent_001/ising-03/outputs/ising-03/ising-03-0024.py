
import sys, pickle, json
sys.path.insert(0,'/work')
import numpy as np
import sw, exact
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))
def tau_int(series):
    x=series-series.mean(axis=1,keepdims=True); n=x.shape[1]
    f=np.fft.rfft(x,2*n,axis=1); ac=np.fft.irfft(f*np.conj(f),axis=1)[:,:n].mean(axis=0); ac/=ac[0]
    tau=0.5
    for w in range(1,n//2):
        tau=0.5+ac[1:w+1].sum()
        if w>=6*tau: break
    return tau
def boot_err(s2,s4,nb=4000,seed=99):
    rng=np.random.default_rng(seed); R=len(s2)
    idx=rng.integers(0,R,(nb,R))
    a=s2[idx].mean(axis=1); b=s4[idx].mean(axis=1)
    return (1-b/(3*a**2)).std(ddof=1)
points=[]; zs=[]
print("   T      U             U_err        U_err(boot)   n_samples  tau_int(m2)  e_z(vs exact)   U_impr+-err")
for k,T in enumerate(Ts):
    A=load('A',k); B=load('B',k)
    assert A['T']==T and B['T']==T and A['N']==16
    blk=np.concatenate([A['blk'],B['blk']],axis=1)
    R=blk.shape[1]; nmeas=A['n_meas']
    s2=blk[0].mean(axis=1); s4=blk[1].mean(axis=1)
    U,e=sw.U_jack(s2,s4); eb=boot_err(s2,s4)
    Ui,ei=sw.U_jack(blk[2].mean(axis=1),blk[3].mean(axis=1))
    eblk=np.concatenate([A['eblk'],B['eblk']],axis=0).mean(axis=1)
    ee=exact.energy_per_site_exact(T,16); ze=(eblk.mean()-ee)/(eblk.std(ddof=1)/np.sqrt(len(eblk))); zs.append(ze)
    tau=tau_int(np.concatenate([A['series'],B['series']],axis=0))
    ns=R*nmeas
    print(f"{T:4.1f}  {U:.8f}  {e:.3e}  {eb:.3e}  {ns:9d}   {tau:5.2f}   {ze:+.2f}   {Ui:.6f}+-{ei:.1e}")
    points.append(dict(T=T,status="ok",U=float(U),U_err=float(e),n_samples=int(ns)))
zs=np.array(zs); print("pooled energy z-scores chi2=%.1f for 14 dof"%(zs**2).sum())
print("max U_err = %.2e (target 5e-3)"%max(p['U_err'] for p in points))
json.dump(points,open('/work/final_points.json','w'),indent=1)
print(json.dumps(points))
