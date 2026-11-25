
import numpy as np
d=np.load("/work/res_T1.9_N16.npz")
S2,S4,M,R=d['S2'],d['S4'],int(d['M']),int(d['R'])
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
m2=S2.sum(0)/M; m4=S4.sum(0)/M
U=U_of(m2,m4)
# jackknife over replicas
idx=np.arange(R); G=20
groups=np.array_split(idx,G)
Uj=[]
for g in groups:
    keep=np.ones(R,bool); keep[g]=False
    Uj.append(U_of(m2[keep],m4[keep]))
Uj=np.array(Uj)
err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
print("U",U,"err",err)
for h in (0,1):
    print("half",h,U_of(S2[h]/(M/2),S4[h]/(M/2)))
