
import sys, pickle
sys.path.insert(0,'/work')
import numpy as np
import sw, exact
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def load(label,k): return pickle.load(open(f'/work/prod{label}_k{k}.pkl','rb'))

def Ujk_blocks(m2b, m4b):
    # jackknife over all (replica, block) units; m2b,m4b arrays (R,nb) of block means (equal block size)
    x=m2b.ravel(); y=m4b.ravel(); n=len(x)
    A=x.sum(); Bs=y.sum()
    U0=1-(Bs/n)/(3*(A/n)**2)
    a=(A-x)/(n-1); b=(Bs-y)/(n-1)
    Ul=1-b/(3*a**2)
    return U0, np.sqrt((n-1)/n*((Ul-Ul.mean())**2).sum())

def boot_err(s2,s4,nb=2000,seed=99):
    rng=np.random.default_rng(seed); R=len(s2)
    idx=rng.integers(0,R,(nb,R))
    a=s2[idx].mean(axis=1); b=s4[idx].mean(axis=1)
    return (1-b/(3*a**2)).std(ddof=1)

def tau_int(series):
    x=series-series.mean(axis=1,keepdims=True); n=x.shape[1]
    f=np.fft.rfft(x,2*n,axis=1); ac=np.fft.irfft(f*np.conj(f),axis=1)[:,:n].mean(axis=0); ac/=ac[0]
    tau=0.5
    for w in range(1,n//2):
        tau=0.5+ac[1:w+1].sum()
        if w>=6*tau: break
    return tau

rows=[]
zAB=[]; zeA=[]; zeB=[]
print("  T   | U_A +- err      | U_B +- err      | zAB  | U_pool +- err(jk)  err(boot) err(blkjk) | z_e(A) z_e(B) | hot-cold z | early-late z | plain-impr dU | tau(m2) | err*sqrt(n)")
for k,T in enumerate(Ts):
    A=load('A',k); B=load('B',k)
    out=[]
    for r in (A,B):
        blk=r['blk']; s2=blk[0].mean(axis=1); s4=blk[1].mean(axis=1)
        out.append(sw.U_jack(s2,s4))
    (UA,eA),(UB,eB)=out
    zab=(UA-UB)/np.hypot(eA,eB); zAB.append(zab)
    # pooled
    blk=np.concatenate([A['blk'],B['blk']],axis=1)      # (4, 2R, nb)
    s2=blk[0].mean(axis=1); s4=blk[1].mean(axis=1)
    U,e=sw.U_jack(s2,s4)
    eb=boot_err(s2,s4)
    Ub,ebk=Ujk_blocks(blk[0],blk[1])
    # improved estimator pooled
    Ui,ei=sw.U_jack(blk[2].mean(axis=1),blk[3].mean(axis=1))
    # energy vs exact
    ee=exact.energy_per_site_exact(T,16)
    ze=[]
    for r in (A,B):
        x=r['eblk'].mean(axis=1); ze.append((x.mean()-ee)/(x.std(ddof=1)/np.sqrt(len(x))))
    zeA.append(ze[0]); zeB.append(ze[1])
    # hot vs cold halves (replicas 0..R/2-1 start hot, R/2.. start cold) in each set
    R=A['R']; half=R//2
    hot=np.concatenate([A['blk'][:, :half],B['blk'][:, :half]],axis=1)
    cold=np.concatenate([A['blk'][:, half:],B['blk'][:, half:]],axis=1)
    Uh,eh=sw.U_jack(hot[0].mean(axis=1),hot[1].mean(axis=1)); Uc,ec=sw.U_jack(cold[0].mean(axis=1),cold[1].mean(axis=1))
    zhc=(Uh-Uc)/np.hypot(eh,ec)
    # early vs late (first 20 blocks vs last 20 blocks), pooled replicas
    nb=blk.shape[2]
    Ue_,ee_=sw.U_jack(blk[0][:,:nb//2].mean(axis=1),blk[1][:,:nb//2].mean(axis=1))
    Ul_,el_=sw.U_jack(blk[0][:,nb//2:].mean(axis=1),blk[1][:,nb//2:].mean(axis=1))
    zel=(Ue_-Ul_)/np.hypot(ee_,el_)
    tau=tau_int(np.concatenate([A['series'],B['series']],axis=0))
    nsamp=blk.shape[1]*A['n_meas']
    print(f"{T:4.2f} | {UA:.5f} {eA:.5f} | {UB:.5f} {eB:.5f} | {zab:5.2f} | {U:.5f} {e:.5f}  {eb:.5f}  {ebk:.5f} | {ze[0]:5.2f} {ze[1]:5.2f} | {zhc:5.2f} | {zel:5.2f} | {Ui-U:+.5f}({ei:.5f}) | {tau:.2f} | {e*np.sqrt(nsamp):.2f}")
zAB=np.array(zAB); print("\nA-vs-B z-scores: chi2 =",(zAB**2).sum(),"for 14 dof; mean z",zAB.mean().round(2))
zeA=np.array(zeA); zeB=np.array(zeB)
print("energy z (A): chi2 =",(zeA**2).sum().round(1)," energy z (B): chi2 =",(zeB**2).sum().round(1)," (14 dof each); mean zA,zB:",zeA.mean().round(2),zeB.mean().round(2))
print("energy z A:",zeA.round(2)); print("energy z B:",zeB.round(2))
