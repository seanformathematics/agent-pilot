import numpy as np, glob
files = sorted(glob.glob("/work/res_T*.npz"), key=lambda f: float(f.split("_T")[1].split("_")[0]))
def U_of(M2, M4):
    m2 = M2.mean(); m4 = M4.mean()
    return 1 - m4 / (3 * m2 * m2)
def jack_rep(M2, M4):
    R = M2.shape[0]
    a2 = M2.sum(axis=1); a4 = M4.sum(axis=1)
    S2 = a2.sum(); S4 = a4.sum(); n = M2.size
    Ui = np.empty(R)
    for r in range(R):
        s2 = (S2 - a2[r]) / (n - M2.shape[1]); s4 = (S4 - a4[r]) / (n - M2.shape[1])
        Ui[r] = 1 - s4 / (3 * s2 * s2)
    U = U_of(M2, M4)
    err = np.sqrt((R - 1) / R * np.sum((Ui - Ui.mean())**2))
    return U, err, Ui
rows = []
for f in files:
    d = np.load(f)
    T = float(d["T"]); M2 = d["M2"]; M4 = d["M4"]
    U, err, Ui = jack_rep(M2, M4)
    # block-level stationarity: U per time block (over replicas)
    ub = [U_of(M2[:, b], M4[:, b]) for b in range(M2.shape[1])]
    # block jackknife (over time blocks, all replicas) as cross-check
    nb = M2.shape[1]
    Ub = np.array([1 - (np.delete(M4,b,1).mean())/(3*(np.delete(M2,b,1).mean())**2) for b in range(nb)])
    errb = np.sqrt((nb-1)/nb*np.sum((Ub-Ub.mean())**2))
    rows.append((T, U, err, np.std(Ui), np.mean(ub[:5]), np.mean(ub[5:]), errb))
    print(f"T={T:.2f} U={U:.5f} jackrep_err={err:.5f} sd_rep={np.std(Ui):.4f} | first5blk={np.mean(ub[:5]):.5f} last5blk={np.mean(ub[5:]):.5f} | blockjack_err={errb:.5f}  blocks={np.round(ub,4).tolist()}")