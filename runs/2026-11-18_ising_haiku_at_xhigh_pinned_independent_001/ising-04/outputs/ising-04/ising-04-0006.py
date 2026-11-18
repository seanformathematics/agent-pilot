import sys, time, glob
sys.path.insert(0,'/work')
import numpy as np, ising
prod = {}
for f in sorted(glob.glob('/work/res/*.npz')):
    d = np.load(f)
    T = float(f.split('/')[-1][1:5])
    prod[T] = d
# 1) time-block check on production: split measurement series into 8 blocks, jackknife over blocks (replica-averaged series)
print("T     U_prod    err_repl   U_firsthalf  U_secondhalf  err_timeblock(replica-avg series)")
for T in sorted(prod):
    d = prod[T]
    A,B,K = d['A'],d['B'],int(d['K'])
    U,err = ising.binder_jk(A,B,K)
    m2 = d['ser_m2']; m4 = d['ser_m4']
    # halves (replica-averaged series; U from ratio of means)
    h = len(m2)//2
    U1 = 1 - m4[:h].mean()/(3*m2[:h].mean()**2)
    U2 = 1 - m4[h:].mean()/(3*m2[h:].mean()**2)
    # block jackknife on time series (8 blocks)
    nb=8; L=len(m2)//nb
    Ub=[]
    for b in range(nb):
        mask=np.ones(len(m2),bool); mask[b*L:(b+1)*L]=False
        Ub.append(1-m4[mask].mean()/(3*m2[mask].mean()**2))
    Ub=np.array(Ub)
    errt = np.sqrt((nb-1)/nb*np.sum((Ub-Ub.mean())**2))
    print(f"{T:.2f}  {U:.5f}  {err:.5f}   {U1:.5f}      {U2:.5f}       {errt:.5f}")
