import sys, json, time
sys.path.insert(0,'/work')
from ising import run
import numpy as np
Ts = [1.9,2.0,2.1,2.2,2.3,2.4]
res = {}
def ujack(m2, m4, nblk):
    n = len(m2)//nblk*nblk
    a = m2[:n].reshape(nblk,-1).mean(1); b = m4[:n].reshape(nblk,-1).mean(1)
    A = a.sum(); B = b.sum()
    Uall = 1 - (B/nblk) / (3*(A/nblk)**2)
    Ujk = []
    for k in range(nblk):
        A_ = (A-a[k])/(nblk-1); B_ = (B-b[k])/(nblk-1)
        Ujk.append(1 - B_/(3*A_**2))
    Ujk = np.array(Ujk)
    err = np.sqrt((nblk-1)/nblk*np.sum((Ujk-Ujk.mean())**2))
    return Uall, err
out = {}
for k,T in enumerate(Ts):
    n, m2, m4 = run(16, T, 1000+k, 75)
    np.savez(f'/work/T{T}_s{1000+k}.npz', m2=m2, m4=m4)
    U1, e1 = ujack(m2, m4, 100)
    U2, e2 = ujack(m2, m4, 400)
    out[T] = dict(n=n, U=U1, err100=e1, err400=e2)
    print(T, n, U1, e1, e2, flush=True)
