import numpy as np, glob
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_from(s2,s4):
    m2=s2.mean(); m4=s4.mean()
    return 1-m4/(3*m2*m2)
def jk(s2,s4):
    R=len(s2); tot2=s2.sum(); tot4=s4.sum()
    m2=tot2/R; m4=tot4/R
    Ui=np.empty(R)
    Ui=1-((tot4-s4)/(R-1))/(3*((tot2-s2)/(R-1))**2)
    Uf=1-m4/(3*m2*m2)
    var=(R-1)/R*np.sum((Ui-Ui.mean())**2)
    return Uf, np.sqrt(var)
for T in Ts:
    d=np.load(f'/work/T{T:.2f}.npz')
    n=int(d['meas'])
    s2=d['s2']/n; s4=d['s4']/n
    U,err=jk(s2,s4)
    # block-based: blocks over replicas and time (bootstrap-like): jackknife over time blocks across replicas
    b2=d['b2']; b4=d['b4']; blk=int(d['blk']); nb=b2.shape[1]
    # time-block jackknife: drop one time block across all replicas
    T2=b2.sum(); T4=b4.sum(); Ut=[]
    for k in range(nb):
        m2=(T2-b2[:,k].sum())/(R:=b2.shape[0])/(nb-1)
        m4=(T4-b4[:,k].sum())/R/(nb-1)
        Ut.append(1-m4/(3*m2*m2))
    Ut=np.array(Ut); errt=np.sqrt((nb-1)/nb*np.sum((Ut-Ut.mean())**2))
    # first-half vs second-half check
    h=nb//2
    Uh1=U_from(b2[:,:h].sum(1)/(h*blk), b4[:,:h].sum(1)/(h*blk))
    Uh2=U_from(b2[:,h:].sum(1)/((nb-h)*blk), b4[:,h:].sum(1)/((nb-h)*blk))
    print(f"T={T:.2f} U={U:.5f} jkErr={err:.5f} timeBlkErr={errt:.5f} U_h1={Uh1:.4f} U_h2={Uh2:.4f} <|m|>={d['s1'].sum()/(2000*n):.4f}")