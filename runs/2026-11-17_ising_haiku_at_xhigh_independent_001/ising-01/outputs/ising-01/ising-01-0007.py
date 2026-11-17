import numpy as np, glob
files = sorted(glob.glob("/work/res_T*.npz"), key=lambda f: float(f.split("_T")[1].split("_")[0]))
def U_of(M2, M4):
    return 1 - M4.mean() / (3 * M2.mean()**2)
def jack_rep(M2, M4):
    R, nb = M2.shape
    a2 = M2.sum(1); a4 = M4.sum(1); S2 = a2.sum(); S4 = a4.sum(); n = R*nb
    Ui = np.empty(R)
    for r in range(R):
        s2 = (S2 - a2[r]) / (n - nb); s4 = (S4 - a4[r]) / (n - nb)
        Ui[r] = 1 - s4 / (3*s2*s2)
    return np.sqrt((R-1)/R*np.sum((Ui-Ui.mean())**2))
for f in files:
    d = np.load(f)
    T=float(d["T"]); U=U_of(d["M2"], d["M4"]); e=jack_rep(d["M2"], d["M4"])
    print(f"{T:.2f}  U={U:.6f}  U_err={e:.2e}  seed={int(d['seed'])}")