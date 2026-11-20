import numpy as np
d=np.load('/work/raw_T2.0.npz'); S2,S4=d['S2'],d['S4']
print(S2.shape)
m2=S2.mean(0); m4=S4.mean(0)
Ur=1-m4/(3*m2**2)
print("per-replica U mean,std:",Ur.mean(),Ur.std(ddof=1), " -> sem", Ur.std(ddof=1)/np.sqrt(len(Ur)))
# time-block scatter of U using all replicas per block
Ub=np.array([1-S4[b].mean()/(3*S2[b].mean()**2) for b in range(S2.shape[0])])
print("block U:",Ub.round(5))
print("std of block U:",Ub.std(ddof=1), "sem over blocks", Ub.std(ddof=1)/np.sqrt(len(Ub)))
print("m2 sem over replicas", m2.std(ddof=1)/np.sqrt(len(m2)), "m4 sem", m4.std(ddof=1)/np.sqrt(len(m4)))