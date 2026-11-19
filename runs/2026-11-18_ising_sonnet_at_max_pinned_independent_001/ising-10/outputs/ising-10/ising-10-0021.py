
import sys, numpy as np
sys.path.insert(0,'/work')
from ana import jackknife_U, U_from_moments
from exact import exact_e_C
np.set_printoptions(linewidth=220)
N=16; nsite=N*N

def jk_diff_U(Ba2,Ba4,na,Bb2,Bb4,nb_):
    R=len(Ba2)
    f=lambda a2,a4,n,r: U_from_moments(a2/(r*n),a4/(r*n))
    full=f(Ba2.sum(),Ba4.sum(),na,R)-f(Bb2.sum(),Bb4.sum(),nb_,R)
    di=f(Ba2.sum()-Ba2,Ba4.sum()-Ba4,na,R-1)-f(Bb2.sum()-Bb2,Bb4.sum()-Bb4,nb_,R-1)
    return full,np.sqrt((R-1)/R*np.sum((di-di.mean())**2))

wins=[(0,1),(1,2),(2,4),(4,8),(8,16),(16,24),(24,32),(32,40)]
for iT in range(2):
    d=np.load(f"/work/relax_{iT}.npz"); T=float(d['T']); Rh=int(d['Rh'])
    B2=d['B2'];B4=d['B4'];BE=d['BE']; R=B2.shape[0]
    e_ex,_=exact_e_C(N,T)
    print(f"===== T={T}  (exact e = {e_ex:.5f})")
    print("window(sweeps)   group  <m2>            U                e")
    for (a,b) in wins:
        n=(b-a)*100
        row=[]
        for name,sl in [('hot',slice(0,Rh)),('cold',slice(Rh,None))]:
            S2=B2[sl,a:b].sum(axis=1);S4=B4[sl,a:b].sum(axis=1);SE=BE[sl,a:b].sum(axis=1)/n
            U,eU,_=jackknife_U(S2,S4,n)
            m2=S2.mean()/n; em2=(S2/n).std(ddof=1)/np.sqrt(len(S2))
            e=SE.mean(); ee=SE.std(ddof=1)/np.sqrt(len(SE))
            row.append((m2,em2,U,eU,e,ee))
            print(f"[{a*100:4d},{b*100:4d})     {name:4s}   {m2:.4f}+-{em2:.4f}  {U:.5f}+-{eU:.5f}  {e:.5f}+-{ee:.5f}")
        (m2h,eh,Uh,eUh,eh_,eeh),(m2c,ec,Uc,eUc,ec_,eec)=row
        print(f"                 hot-cold: dm2={m2h-m2c:+.4f} (z={(m2h-m2c)/np.hypot(eh,ec):+.2f}), dU={Uh-Uc:+.5f} (z={(Uh-Uc)/np.hypot(eUh,eUc):+.2f}), de={eh_-ec_:+.5f} (z={(eh_-ec_)/np.hypot(eeh,eec):+.2f})")
    # paired trends over late windows (pooled hot+cold)
    print("-- pooled paired late-window comparisons of U (blocks of 1000 sweeps after first 200):")
    late=[(2,12),(12,22),(22,32),(32,40)]
    Us=[]
    for (a,b) in late:
        n=(b-a)*100
        U,eU,_=jackknife_U(B2[:,a:b].sum(axis=1),B4[:,a:b].sum(axis=1),n); Us.append((a,b,U,eU))
        print(f"   [{a*100},{b*100}): U={U:.6f}+-{eU:.6f}")
    for i in range(len(late)):
        for j in range(i+1,len(late)):
            a1,b1=late[i];a2,b2=late[j]
            f,e=jk_diff_U(B2[:,a1:b1].sum(axis=1),B4[:,a1:b1].sum(axis=1),(b1-a1)*100,B2[:,a2:b2].sum(axis=1),B4[:,a2:b2].sum(axis=1),(b2-a2)*100)
            print(f"   U[{a1*100},{b1*100}) - U[{a2*100},{b2*100}) = {f:+.6f} +- {e:.6f}  z={f/e:+.2f}")
    # energy trend (all replicas)
    for (a,b) in [(2,21),(21,40)]:
        n=(b-a)*100; SE=BE[:,a:b].sum(axis=1)/n
        print(f"   energy [{a*100},{b*100}): {SE.mean():.5f} +- {SE.std(ddof=1)/np.sqrt(R):.5f}")
