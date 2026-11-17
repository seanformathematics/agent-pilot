import json, glob, math
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rows = []
for T in Ts:
    d = json.load(open('/work/res/summary_rep0_T%.2f.json' % T))
    rows.append(d)
    ok = math.isfinite(d['U']) and math.isfinite(d['U_err']) and d['U_err'] > 0 and d['U'] <= 2/3 + 1e-12
    print("T=%.2f  U=%.7f  U_err=%.3e  groupcheck=%.3e  n=%d  seed=%d  ok=%s" % (
        T, d['U'], d['U_err'], d['U_err_groupcheck'], d['n_samples'], d['seed'], ok))
print("all ok:", all(math.isfinite(r['U']) and 0 < r['U_err'] <= 0.005 and r['U'] <= 2/3 for r in rows))
print("max U_err: %.3e" % max(r['U_err'] for r in rows))
