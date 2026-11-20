
import numpy as np, glob
def U_from(a2,a4,nmeas):
    M2=a2.sum()/(nmeas*len(a2)); M4=a4.sum()/(nmeas*len(a4))
    return 1-M4/(3*M2**2)
for f in sorted(glob.glob('/work/sums_*.npz'), key=lambda x: float(x.split('_T')[1].split('_')[0])):
    d=np.load(f); S=d['S']; T=float(d['T']); R=S.shape[-1]
    A2=S[0,0]+S[1,0]; A4=S[0,1]+S[1,1]
    U=U_from(A2,A4,10000)
    Uj=np.array([U_from(np.delete(A2,i),np.delete(A4,i),10000) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    U1=U_from(S[0,0],S[0,1],5000); U2=U_from(S[1,0],S[1,1],5000)
    # block jackknife over 10 blocks of replicas for a sanity check
    blocks=np.array_split(np.arange(R),10)
    Ub=np.array([U_from(np.delete(A2,b),np.delete(A4,b),10000) for b in blocks])
    errb=np.sqrt((10-1)/10*np.sum((Ub-Ub.mean())**2))
    print(f"T={T} U={U:.5f} err_repjk={err:.5f} err_10blk={errb:.5f} U_half1={U1:.5f} U_half2={U2:.5f} <m^2>={A2.sum()/(10000*R):.5f}")
