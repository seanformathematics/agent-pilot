
import numpy as np, pickle
d=pickle.load(open('/work/all.pkl','rb'))
b2=d['b2']; b4=d['b4']   # (nblocks, 6000)
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]; R=1000
nb,_=b2.shape; sweeps_per_block=100
def U_of(idx_rep, blocks):
    # idx_rep: array of replica indices within a T
    s2=b2[blocks][:,idx_rep].sum(); s4=b4[blocks][:,idx_rep].sum()
    n=len(blocks)*len(idx_rep)*sweeps_per_block
    m2=s2/n; m4=s4/n
    return 1-m4/(3*m2**2)
allb=np.arange(nb)
for k,T in enumerate(Ts):
    reps=np.arange(k*R,(k+1)*R)
    U=U_of(reps,allb)
    # jackknife over 50 groups of 20 replicas
    G=50; groups=np.array_split(reps,G)
    jk=np.array([U_of(np.setdiff1d(reps,g),allb) for g in groups])
    err=np.sqrt((G-1)/G*np.sum((jk-jk.mean())**2))
    # block-jackknife over time blocks (all replicas together) as a cross-check
    tg=np.array_split(allb,20)
    jt=np.array([U_of(reps,np.setdiff1d(allb,g)) for g in tg])
    errt=np.sqrt(19/20*np.sum((jt-jt.mean())**2))
    h1=U_of(reps,allb[:nb//2]); h2=U_of(reps,allb[nb//2:])
    print(f"T={T} U={U:.5f} err_repjk={err:.5f} err_timejk={errt:.5f} U_firsthalf={h1:.5f} U_secondhalf={h2:.5f}")
