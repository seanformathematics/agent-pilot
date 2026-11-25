import numpy as np
d=np.load("/work/run_T2.3_s101.npz")
m2b,m4b=d['m2b'],d['m4b']
def U_of(m2,m4): return 1-np.mean(m4,axis=0).mean()/(3*np.mean(m2,axis=0).mean()**2)
def ujack(m2,m4):
    # per-replica time averages, jackknife over replicas
    a2=m2.mean(0); a4=m4.mean(0); R=len(a2)
    U=1-a4.mean()/(3*a2.mean()**2)
    S2=a2.sum(); S4=a4.sum()
    loo=np.empty(R)
    for r in range(R):
        x2=(S2-a2[r])/(R-1); x4=(S4-a4[r])/(R-1)
        loo[r]=1-x4/(3*x2*x2)
    err=np.sqrt((R-1)/R*np.sum((loo-loo.mean())**2))
    return U,err
U,e=ujack(m2b,m4b)
print("U",U,"err",e)
# stationarity: first vs second half
h=m2b.shape[0]//2
print("halves", ujack(m2b[:h],m4b[:h])[0], ujack(m2b[h:],m4b[h:])[0])
# autocorrelation of block-level U-ish estimate: lag correlation of m2 across blocks
x=m2b.mean(1); x=x-x.mean()
print("lag-1 corr of block <m2>:", np.corrcoef(x[:-1],x[1:])[0,1])