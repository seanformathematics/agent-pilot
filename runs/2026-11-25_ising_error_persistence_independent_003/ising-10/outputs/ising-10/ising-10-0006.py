import numpy as np
for T in [1.9,2.0,2.3]:
    d=np.load(f'/work/T{T}.npz'); a=d['a']; b=d['b']
    print(T, "a quantiles", np.round(np.quantile(a,[0,0.001,0.01,0.5,0.99,1]),4), "n<0.1:", (a<0.1).sum())
    # U excluding outliers with a<0.5*median
    med=np.median(a); keep=a>0.5*med
    print("  U all", round(1-b.mean()/(3*a.mean()**2),5), "U kept", round(1-b[keep].mean()/(3*a[keep].mean()**2),5), "n dropped", (~keep).sum())