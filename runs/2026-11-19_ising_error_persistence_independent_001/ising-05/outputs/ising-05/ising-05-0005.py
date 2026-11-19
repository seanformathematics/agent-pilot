import numpy as np
d=np.load('/work/T2.2_N16.npz')
M2,M4=d['M2'],d['M4']
nb,R=M2.shape
print("blocks",nb)
m2t=M2.mean(1); print("m2 vs block (first 10):",np.round(m2t[:10],4))
print("m2 block means, quarters:",[round(m2t[i*nb//4:(i+1)*nb//4].mean(),4) for i in range(4)])
U=1-M4.mean()/(3*M2.mean()**2)
# jackknife over replicas
G=50
groups=np.array_split(np.arange(R),G)
Ug=[]
for g in groups:
    keep=np.setdiff1d(np.arange(R),g)
    a2=M2[:,keep].mean(); a4=M4[:,keep].mean()
    Ug.append(1-a4/(3*a2*a2))
Ug=np.array(Ug)
err=np.sqrt((G-1)/G*np.sum((Ug-Ug.mean())**2))
print("U",U,"jack err",err)
# per-replica-group std check
# autocorrelation of block series of replica-averaged m2
x=m2t-m2t.mean()
ac=[np.dot(x[:-k],x[k:])/np.dot(x,x) for k in range(1,15)]
print("ac of block-avg m2 (block=100 sweeps):",np.round(ac,3))
