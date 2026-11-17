
import os, sys, time, math
os.chdir('/work'); sys.path.insert(0, '/work')
import numpy as np
import wolff_py as W

for T in [3.5, 5.0, 10.0, 1e9]:
    print('exact 4x4 T=%g U=%.5f' % (T, W.exact_U(4, T)))

S2, S4, nf = W.wolff_run(4, 1e9, 99, 1000, 300000, 1)
print('Wolff 4x4 T=inf U=%.5f +- %.5f (expect %.5f)' % (*W.jk_U(S2, S4, 1, 50), 2/(3*16)))
S2, S4, nf = W.wolff_run(16, 1e9, 99, 1000, 300000, 1)
print('Wolff 16x16 T=inf U=%.5f +- %.5f (expect %.5f)' % (*W.jk_U(S2, S4, 1, 50), 2/(3*256)))

# 4x4 distribution at T=3.5 from exact enumeration
L = 4; N = 16
nbrs = W.make_nbrs(L)
idx = np.arange(2**N)
conf = ((idx[:, None] >> np.arange(N)[None, :]) & 1)*2 - 1
E = np.zeros(2**N)
for i in range(N):
    for j in (nbrs[i][0], nbrs[i][2]):
        E -= conf[:, i]*conf[:, j]
M = conf.sum(axis=1)
for T in [3.5]:
    w = np.exp(-(E-E.min())/T); w /= w.sum()
    print('4x4 T=3.5: <E>/N=%.4f  <M^2>/N=%.4f' % ((w*E).sum()/N, (w*M**2).sum()/N))
    for Mv in range(-16, 17, 2):
        pm = w[M == Mv].sum()
        if pm > 1e-4:
            print('   P(M=%d)=%.4f' % (Mv, pm))

# 16x16 at T=2.8: moments
S2, S4, nf = W.wolff_run(16, 2.8, 2024, 5000, 200000, 1)
m2 = S2.mean()/256**2; m4 = S4.mean()/256**4
print('16x16 T=2.8: <m^2>=%.6g <m^4>=%.6g  <M^2>/N=%.4f  U=%.5f' % (m2, m4, S2.mean()/256, 1-m4/(3*m2*m2)))
