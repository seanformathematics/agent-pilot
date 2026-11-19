
import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def Ujk(m2,m4):
    # m2,m4: arrays (blocks, R); replica-level averages
    A2=m2.mean(0); A4=m4.mean(0); R=len(A2)
    tot2=A2.sum(); tot4=A4.sum()
    U=1-(tot4/R)/(3*(tot2/R)**2)
    # leave-one-out
    loo=np.array([1-((tot4-A4[i])/(R-1))/(3*((tot2-A2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return U,err
for T in Ts:
    d=np.load(f'/work/run_{T}.npz')
    m2=d['m2']; m4=d['m4']
    U,e=Ujk(m2,m4)
    # first half vs second half
    U1,e1=Ujk(m2[:10],m4[:10]); U2,e2=Ujk(m2[10:],m4[10:])
    print(T, f"U={U:.5f} err={e:.5f}  half1={U1:.4f} half2={U2:.4f}  blocks(2000sw)-mean-m2={m2.mean():.4f}")
