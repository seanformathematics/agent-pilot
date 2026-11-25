import numpy as np, glob
files={1.9:'res_T1.900_s191',2.0:'res_T2.000_s201',2.1:'res_T2.100_s211',2.2:'res_T2.200_s221',2.3:'res_T2.300_s231',2.4:'res_T2.400_s241'}
for T,f in files.items():
    d=np.load(f'/work/{f}.npz')
    S2=d['S2']; S4=d['S4']; n=int(d['nmeas'])
    a2=S2/n; a4=S4/n   # per-replica time averages
    def U(idx):
        m2=a2[idx].mean(); m4=a4[idx].mean()
        return 1-m4/(3*m2*m2)
    R=len(a2)
    full=U(np.arange(R))
    # delete-one jackknife (replicas independent)
    jk=np.array([U(np.delete(np.arange(R),i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # block jackknife with 20 blocks for robustness
    blocks=np.array_split(np.arange(R),20)
    jb=np.array([U(np.setdiff1d(np.arange(R),b)) for b in blocks])
    errb=np.sqrt((len(blocks)-1)/len(blocks)*np.sum((jb-jb.mean())**2))
    print(f"T={T} U={full:.5f} err_jk={err:.5f} err_block20={errb:.5f} R={R} n={n}")
