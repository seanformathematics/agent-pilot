import json
final = json.load(open("/work/final.json"))
for d in final:
    print(repr(d["T"]), repr(d["U"]), repr(d["U_err"]), d["n_samples"], d["k"], d["empirical_se"], d["seeds"])
allseeds = sorted({s for d in final for s in d["seeds"]})
print("ALL SEEDS", allseeds, len(allseeds))
