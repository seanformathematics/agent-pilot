import numpy as np, glob
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def binder(a2,a4): return 1-a4/(3*a2**2)
pilot={1.5:0.66645,1.6:0.66630,1.7:0.66604,1.8:0.66556,1.9:0.66462,2.0:0.66246,2.1:0.65676,2.2:0.63950,2.3:0.59271,2.4:0.50040,2.5:0.38151,2.6:0.27704,2.7:0.19752,2.8:0.14751}
pilot_err={1.5:0.0,1.6:0.0,1.7:0.0,1.8:0.0,1.9:0.00001,2.0:0.00004,2.1:0.00010,2.2:0.00024,2.3:0.00057,2.4:0.00087,2.5:0.00113,2.6:0.00126,2.7:0.00127,2.8:0.00133}
rng=np.random.default_rng(0)
print("T      U        err_jack   err_group50  n_rep  pilotU  diff/sig   ")
out={}
for T in Ts:
    d=np.load(f"/work/res/T{T:.2f}.npz")
    a2=d["a2"]; a4=d["a4"]; R=len(a2)
    U=binder(a2.mean(),a4.mean())
    tot2=a2.sum(); tot4=a4.sum()
    Uj=binder((tot2-a2)/(R-1),(tot4-a4)/(R-1))
    ej=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # group jackknife
    K=50; idx=np.arange(R); rng.shuffle(idx); groups=np.array_split(idx,K)
    Ug=[]
    for g in groups:
        mask=np.ones(R,bool); mask[g]=False
        Ug.append(binder(a2[mask].mean(),a4[mask].mean()))
    Ug=np.array(Ug)
    eg=np.sqrt((K-1)/K*np.sum((Ug-Ug.mean())**2))
    # bootstrap check
    bs=[]
    for b in range(500):
        s=rng.integers(0,R,R)
        bs.append(binder(a2[s].mean(),a4[s].mean()))
    eb=np.std(bs)
    sig=np.sqrt(ej**2+pilot_err[T]**2)+1e-12
    print(f"{T:.1f}  {U:.5f}  {ej:.5f}   {eg:.5f}   {eb:.5f}  {R}  {pilot[T]:.5f}  {(U-pilot[T])/sig if sig>1e-6 else float('nan'):.2f}")
    out[T]=(U,ej,eb)
np.save("/work/final_results.npy", np.array([[T,out[T][0],out[T][1],out[T][2]] for T in Ts]))
