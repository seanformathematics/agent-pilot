import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in Ts:
    d=np.load(f'/work/T{T:.2f}.npz')
    b2=d['b2']; b4=d['b4']; R,nb=b2.shape; blk=int(d['blk'])
    T2=b2.sum(); T4=b4.sum()
    Ut=[]
    for k in range(nb):
        m2=(T2-b2[:,k].sum())/(R*(nb-1)*blk)
        m4=(T4-b4[:,k].sum())/(R*(nb-1)*blk)
        Ut.append(1-m4/(3*m2*m2))
    Ut=np.array(Ut); errt=np.sqrt((nb-1)/nb*np.sum((Ut-Ut.mean())**2))
    m2=T2/(R*nb*blk); m4=T4/(R*nb*blk); U=1-m4/(3*m2*m2)
    print(f"T={T:.2f} U={U:.5f} timeBlockJK_err={errt:.5f}")