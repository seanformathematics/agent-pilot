import json, glob, math
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rows = []
for T in Ts:
    r = json.load(open(f'/work/results/T_{T:.2f}.json'))
    rows.append(r)
    print(f"T={r['T']:.2f} seed={r['seed']} U={r['U']:.7f} U_bc={r['U_jack_bc']:.7f} se_rep={r['se_replica_jack']:.3e} "
          f"se_blk={r['se_block_jack']:.3e} n={r['n_meas']} R={r['R']} steps={r['n_steps']} burn={r['n_burn']} tau_m2={r['tau_int_m2']:.2f} "
          f"<|m|>={r['mean_abs_m']:.5f} <m2>={r['mean_m2']:.6f} <m4>={r['mean_m4']:.6f}")
ok = all(math.isfinite(r['U']) and math.isfinite(r['se_replica_jack']) and r['se_replica_jack'] > 0 and r['U'] <= 2/3 for r in rows)
print("all finite, err>0, U<=2/3:", ok)
print("max se_rep:", max(r['se_replica_jack'] for r in rows), " max |U-U_bc|:", max(abs(r['U']-r['U_jack_bc']) for r in rows))
print("max |U - U_block|/se:", max(abs(r['U']-r['U_block'])/r['se_replica_jack'] for r in rows))
