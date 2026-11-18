
import numpy as np
p = np.load('/work/pooled.npy')
for T, U, e in p:
    print(f"{T:.2f}  U={U:.10f}  U_err={e:.4e}")
print("n_samples per T = 2000 replicas x 2500 sweeps =", 2000*2500)
