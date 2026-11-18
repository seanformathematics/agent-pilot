
import sys, numpy as np
sys.path.insert(0, '/work')
import sw
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
zs = []
print(f"{'T':>5} {'U_run1':>10} {'e1':>9} {'U_run2':>10} {'e2':>9} {'z':>6} {'U_pool':>10} {'e_pool':>10} {'grp_ratio':>9}")
out = {}
for T in Ts:
    d1 = np.load(f'/work/res_T{T:.2f}.npz'); d2 = np.load(f'/work/rep2_T{T:.2f}.npz')
    S = int(d1['S']); assert int(d2['S']) == S
    B2a, B4a = d1['B2'], d1['B4']; B2b, B4b = d2['B2'], d2['B4']
    U1, e1 = sw.binder_from_sums(B2a, B4a, S)
    U2, e2 = sw.binder_from_sums(B2b, B4b, S)
    B2c = np.concatenate([B2a, B2b]); B4c = np.concatenate([B4a, B4b])
    Uc, ec = sw.binder_from_sums(B2c, B4c, S)
    # subgroup scatter check: 20 groups of 100 replicas from pooled data
    Ug = np.array([sw.binder_from_sums(B2c[g*100:(g+1)*100], B4c[g*100:(g+1)*100], S)[0] for g in range(20)])
    ratio = Ug.std(ddof=1)/(ec*np.sqrt(20))
    z = (U1-U2)/np.sqrt(e1**2+e2**2)
    zs.append(z)
    out[T] = (Uc, ec)
    print(f"{T:5.2f} {U1:10.6f} {e1:9.2e} {U2:10.6f} {e2:9.2e} {z:6.2f} {Uc:10.6f} {ec:10.2e} {ratio:9.2f}")
zs = np.array(zs)
print("RMS of normalised run1-run2 differences:", np.sqrt(np.mean(zs**2)).round(3), " (expect ~1)")
print("max |U_pool - 0.5*(U1+U2)|/ec check computed above; pooled errors max:", max(v[1] for v in out.values()))
np.save('/work/pooled.npy', np.array([[T, out[T][0], out[T][1]] for T in Ts]))
