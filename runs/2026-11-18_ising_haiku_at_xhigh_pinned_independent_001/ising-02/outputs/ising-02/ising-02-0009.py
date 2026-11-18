import json, glob, math
import numpy as np

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rows = []
zs = []
for T in Ts:
    r1 = json.load(open("/work/res_T%.2f.json" % T))
    r2 = json.load(open("/work/run2_T%.2f.json" % T))
    # conservative per-run error: larger of jackknife and chain-to-chain estimates
    e1 = max(r1["err_jk"], r1["err_chain"])
    e2 = max(r2["err_jk"], r2["err_chain"])
    w1, w2 = 1/e1**2, 1/e2**2
    U = (w1*r1["U"] + w2*r2["U"])/(w1+w2)
    eU = math.sqrt(1/(w1+w2))
    z = (r1["U"]-r2["U"])/math.sqrt(e1**2+e2**2)
    zs.append(z)
    n_tot = r1["nsamples"] + r2["nsamples"]
    rows.append(dict(T=T, U=U, err=eU, U1=r1["U"], e1=e1, U2=r2["U"], e2=e2, z=z, n=n_tot,
                     seeds=r1["seeds"]+r2["seeds"], tau=float(np.mean(r1["tau_m2"]+r2["tau_m2"]))))
    print("T=%.2f U=%.6f err=%.6f | run1 %.6f(%.6f) run2 %.6f(%.6f) z=%.2f  nsamp=%d tau~%.2f" % (T, U, eU, r1["U"], e1, r2["U"], e2, z, n_tot, rows[-1]["tau"]))
zs = np.array(zs)
print("z rms (should be ~1):", np.sqrt(np.mean(zs**2)))
json.dump(rows, open("/work/combined.json","w"))
