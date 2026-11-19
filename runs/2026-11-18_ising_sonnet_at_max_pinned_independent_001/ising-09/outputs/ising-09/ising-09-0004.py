
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from sw import SW

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R = 2000
nT = 60
wins = [(1,2),(3,5),(6,10),(11,20),(21,40),(41,60)]
t0 = time.time()
print("T   | window: hot-cold diff in m2 (in units of its stat err) ... [m2 hot / m2 cold at last window]")
for T in Ts:
    rec = {}
    for start in ["hot","cold"]:
        sim = SW(16, R, seed=777 + int(T*100) + (0 if start=="hot" else 5000), start=start)
        m2s = np.zeros((nT, R)); m4s = np.zeros((nT, R))
        for t in range(nT):
            sim.sweep(T)
            m = sim.mag()
            m2s[t] = m*m; m4s[t] = m**4
        rec[start] = (m2s, m4s)
    line = f"{T:.1f} |"
    for (a,b) in wins:
        d = []
        for k in range(2):
            h = rec["hot"][k][a-1:b].mean(axis=0); c = rec["cold"][k][a-1:b].mean(axis=0)
            diff = h.mean()-c.mean()
            err = np.sqrt(h.var(ddof=1)/R + c.var(ddof=1)/R)
            d.append(diff/err)
        line += f" [{a}-{b}] z(m2)={d[0]:+.1f} z(m4)={d[1]:+.1f} |"
    h = rec["hot"][0][-20:].mean(); c = rec["cold"][0][-20:].mean()
    line += f"  m2 hot/cold={h:.4f}/{c:.4f}"
    print(line)
print("time", time.time()-t0)
