
import json;r=json.load(open('/work/res.json'))
pts=[dict(T=float(T),status="ok",U=v[0],U_err=max(v[1],1e-6),n_samples=int(v[2])) for T,v in r.items()]
print(json.dumps(pts))
