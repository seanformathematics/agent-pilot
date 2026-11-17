import json, numpy as np
d={}
for f in ['/work/part1.json','/work/part2.json']:
    for k,v in json.load(open(f)).items(): d[float(k)]=v
print("T, U, Uerr, n")
for T in sorted(d):
    m2=np.array(d[T][0]); m4=np.array(d[T][1]); R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # delete-one jackknife over replicas
    S2=m2.sum(); S4=m4.sum()
    Uj=np.array([1-((S4-m4[i])/(R-1))/(3*((S2-m2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # also split-half check
    h=R//2
    Ua=1-m4[:h].mean()/(3*m2[:h].mean()**2); Ub=1-m4[h:].mean()/(3*m2[h:].mean()**2)
    print(f"{T:.1f} {U:.5f} {err:.5f} {R}  halves {Ua:.4f} {Ub:.4f}")
