
import numpy as np, glob
fs=sorted(glob.glob('/work/data/T2.30_c*.npy'))
D=[np.load(f) for f in fs]
B=1000
def U_of(S2,S4,nt):
    m2=S2/nt; m4=S4/nt
    return 1-m4/(3*m2*m2)
A=np.vstack(D); n=len(A); nt=n*B
S2=A[:,0].sum(); S4=A[:,1].sum()
U=U_of(S2,S4,nt)
Uj=np.array([U_of(S2-A[i,0],S4-A[i,1],nt-B) for i in range(n)])
err_j=np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
print('U',U,'jackknife err',err_j,'nblocks',n)
for d in D:
    print('chain U', U_of(d[:,0].sum(),d[:,1].sum(),d.shape[0]*B), 'mean|m|', d[:,2].sum()/(d.shape[0]*B))
print('<m^2>',S2/nt,'<m^4>',S4/nt)
