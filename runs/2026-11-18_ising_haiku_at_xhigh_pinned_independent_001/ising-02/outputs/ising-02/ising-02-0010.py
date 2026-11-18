import json
rows = json.load(open("/work/combined.json"))
for r in rows:
    print(r["T"], repr(r["U"]), repr(r["err"]), r["n"], r["seeds"])
