
import numpy as np
f=np.load('/work/final.npy')
for T,U,e in f:
    print(f"{T:.2f} {U:.10f} {e:.6e}")
