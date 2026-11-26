
import numpy as np
files={1.9:'/work/T1.9_s1901.npz',2.0:'/work/T2.0_s2001.npz',2.1:'/work/T2.1_s2101.npz',2.2:'/work/T2.2_s2201.npz',2.3:'/work/T2.3_s2301.npz',2.4:'/work/T2.4_s2401.npz'}
for T,f in files.items():
    d=np.load(f); S2,S4,cnt=d['S2'],d['S4'],d['cnt']; R=S2.shape[1]
    row=[]
    for b in range(S2.shape[0]):
        m2=S2[b].sum()/cnt[b]/R; m4=S4[b].sum()/cnt[b]/R
        row.append(1-m4/(3*m2*m2))
    print(T, ' '.join(f'{u:.4f}' for u in row))
    for cut in (1,2,5,10):
        blocks=list(range(cut,S2.shape[0]))
        tot2=S2[blocks].sum(0); tot4=S4[blocks].sum(0); n=cnt[blocks].sum()
        A2=tot2.sum(); A4=tot4.sum()
        m2=A2/(n*R); m4=A4/(n*R); U0=1-m4/(3*m2*m2)
        Uj=np.array([1-((A4-tot4[r])/(n*(R-1)))/(3*((A2-tot2[r])/(n*(R-1)))**2) for r in range(R)])
        err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
        print(f'   cut={cut}: U={U0:.6f} err={err:.6f}')
