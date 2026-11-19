
import sys, time
sys.path.insert(0,'/work')
import numpy as np
import sw

# ---------- exact enumeration for N=4 ----------
def exact_N4(Ts):
    N=4
    n=N*N
    states = np.arange(2**n, dtype=np.uint32)
    bits = ((states[:,None] >> np.arange(n,dtype=np.uint32)[None,:]) & 1).astype(np.int8)
    s = (2*bits-1).reshape(-1,N,N).astype(np.int16)
    E = -(s*np.roll(s,-1,axis=2) + s*np.roll(s,-1,axis=1)).sum(axis=(1,2)).astype(np.float64)
    m = s.sum(axis=(1,2))/float(n)
    out={}
    for T in Ts:
        w = np.exp(-(E-E.min())/T); Z=w.sum()
        m2=(w*m**2).sum()/Z; m4=(w*m**4).sum()/Z; e=(w*E).sum()/Z/n
        out[T]=(m2,m4,1-m4/(3*m2**2),e)
    return out, (E,)

Ts4=[1.5,2.0,2.27,2.8,4.0]
ex4,_=exact_N4(Ts4)

def jackknife_U(S2,S4):
    # S2,S4: per-replica sums (equal number of samples per replica)
    R=len(S2)
    A=S2.sum(); B=S4.sum()
    U=lambda a,b: 1-(b/ (R*1.0))/(3*(a/(R*1.0))**2)  # a,b are means*R ; constant factors handled below
    # compute with means (per-sample means m2bar=S2.sum()/(R*M)) -> U independent of M only through ratios, so use M=1 normalisation after dividing
    m2=S2.sum()/R; m4=S4.sum()/R
    U0=1-m4/(3*m2**2)
    Ul=np.empty(R)
    for r in range(R):
        a=(S2.sum()-S2[r])/(R-1); b=(S4.sum()-S4[r])/(R-1)
        Ul[r]=1-b/(3*a**2)
    err=np.sqrt((R-1)/R*((Ul-Ul.mean())**2).sum())
    return U0,err

print("N=4 exact vs SW (plain and improved estimators)")
for T in Ts4:
    R=1000; M=1000
    S=sw.IsingSW(4,R,T,seed=123)
    for _ in range(100): S.sweep()
    s2=np.zeros(R); s4=np.zeros(R); i2=np.zeros(R); i4=np.zeros(R); es=np.zeros(R)
    for _ in range(M):
        a,b=S.sweep(improved=True)
        m=S.magnetization()
        s2+=m**2; s4+=m**4; i2+=a; i4+=b; es+=S.energy_per_site()
    s2/=M; s4/=M; i2/=M; i4/=M; es/=M
    Up,ep=jackknife_U(s2,s4); Ui,ei=jackknife_U(i2,i4)
    m2e,m4e,Ue,ee=ex4[T]
    print(f"T={T}: exact U={Ue:.5f} m2={m2e:.5f} m4={m4e:.5f} e={ee:.5f} | SW plain U={Up:.5f}+-{ep:.5f} m2={s2.mean():.5f} m4={s4.mean():.5f} | improved U={Ui:.5f}+-{ei:.5f} m2={i2.mean():.5f} m4={i4.mean():.5f} | e={es.mean():.5f}+-{es.std()/np.sqrt(R):.5f}")
