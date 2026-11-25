
import numpy as np
def jk(A,B):
    R=A.shape[0]
    Ar=A.mean(1); Br=B.mean(1)
    sa=Ar.sum(); sb=Br.sum()
    a=(sa-Ar)/(R-1); b=(sb-Br)/(R-1)
    Us=1-b/(3*a*a)
    U=1-(sb/R)/(3*(sa/R)**2)
    err=np.sqrt((R-1)/R*np.sum((Us-Us.mean())**2))
    return U,err
for T in ['1.900','2.000','2.100','2.200','2.300','2.400']:
    d=np.load(f'/work/state_T{T}.npz')
    A,B,C=d['A'],d['B'],d['C']
    nb=A.shape[1]
    U,e=jk(A,B)
    parts=[jk(A[:,k*nb//3:(k+1)*nb//3],B[:,k*nb//3:(k+1)*nb//3])[0] for k in range(3)]
    print(T, 'blocks',nb,'U=%.5f err=%.5f'%(U,e),'thirds',np.round(parts,4),'<|m|>=%.4f'%C.mean(), 'sweeps',int(d['sweeps']))
