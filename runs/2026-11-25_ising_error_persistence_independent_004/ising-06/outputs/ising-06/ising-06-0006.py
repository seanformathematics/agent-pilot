import numpy as np
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/res_{T}.npz"); ts=d['ts']
    m2=ts[:,0]; U_t=1-ts[:,1]/(3*ts[:,0]**2)
    a=m2[:2000].mean(); b=m2[-2000:].mean()
    # integrated autocorrelation time of m2 (replica-averaged series)
    x=m2-m2.mean(); c=np.correlate(x,x,'full')[len(x)-1:]/ (len(x)*x.var())
    tau=1+2*np.sum(c[1:200][c[1:200]>0.0]) if True else 0
    print(T, "early<m2>=%.5f late=%.5f"%(a,b), "tau_int(m2)~%.1f"%tau, "U_series_mean=%.4f"%U_t.mean())
