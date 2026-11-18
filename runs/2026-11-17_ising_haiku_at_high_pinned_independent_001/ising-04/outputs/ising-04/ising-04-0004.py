
import numpy as np, glob
fs=sorted(glob.glob('/work/data/T2.30_c*.npy'))
D=[np.load(f) for f in fs]
def U_of(a2,a4):
    return 1-(a4/a2**2)/3.0
A=np.vstack(D)  # blocks pooled
S2=A[:,0].sum(); S4=A[:,1].sum(); n=len(A)
U=U_of(S2,S4)
# jackknife over blocks
idx=np.arange(n)
Uj=np.array([U_of(S2-A[i,0], S4-A[i,1]) for i in range(n)])
# proper: delete-one jackknife
err_j=np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
print('U',U,'jackknife err (blocks=1000 steps)',err_j, 'nblocks',n)
for d in D:
    print('chain U', U_of(d[:,0].sum(), d[:,1].sum()), 'mean|m|', d[:,2].sum()/d.shape[0]/1000)
# block-level autocorr of U-related quantity
x=A[:,0]/1000
def ac(x,l):
    x=x-x.mean(); return (x[:-l]*x[l:]).mean()/x.var()
print('autocorr m2 lags', [round(ac(A[:,0],l),3) for l in [1,2,5,10]])
