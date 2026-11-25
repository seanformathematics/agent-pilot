
import numpy as np
files={1.9:21,2.0:31,2.1:41,2.2:11,2.3:51,2.4:61}
for T,seed in sorted(files.items()):
    d=np.load(f'/work/res_{T}_{seed}.npz')
    b2,b4=d['bl2'],d['bl4']  # (nblk,R)
    nb=b2.shape[0]; R=b2.shape[1]
    def U_of(idx):
        s2=b2[idx].sum(0); s4=b4[idx].sum(0); n=len(idx)*1000
        return 1-(s4.sum()/(R*n))/(3*(s2.sum()/(R*n))**2)
    early=U_of(list(range(0,5))); late=U_of(list(range(nb-5,nb))); allu=U_of(list(range(nb)))
    ub=[U_of([i]) for i in range(nb)]
    print(T, "U_all=%.5f early5=%.5f late5=%.5f  blockU mean=%.5f sd=%.5f"%(allu,early,late,np.mean(ub),np.std(ub)))
    print("   <m^2> all=%.5f  <|m|> all=%.5f"%(b2.sum()/(R*nb*1000), d['bl1'].sum()/(R*nb*1000)))
