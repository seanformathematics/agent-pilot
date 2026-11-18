import json
d = json.load(open("/work/final.json"))
rows = []
for r in d['final']:
    ue = max(r['U_err'], r['chain_sd_pooled'])
    rows.append(dict(T=r['T'], U=r['U'], U_err=ue, U_err_jk=r['U_err'], U_err_chain=r['chain_sd_pooled'],
                     n_samples=r['n_samples'], chains=r['chains']))
print(f"{'T':>5} {'U':>13} {'U_err':>12} {'jk':>12} {'chain':>12} {'n':>9} {'K':>3}")
for r in rows:
    print(f"{r['T']:5.2f} {r['U']:13.9f} {r['U_err']:12.9f} {r['U_err_jk']:12.9f} {r['U_err_chain']:12.9f} {r['n_samples']:9d} {r['chains']:3d}")
json.dump(dict(rows=rows, seeds=d['seeds']), open("/work/submission_table.json", "w"), indent=1)
print("max U_err:", max(r['U_err'] for r in rows))
print("seeds:", d['seeds'])
