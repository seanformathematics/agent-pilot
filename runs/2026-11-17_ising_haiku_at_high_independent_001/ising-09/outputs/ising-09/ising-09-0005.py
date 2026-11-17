import sys; sys.argv=['x']
import numpy as np
d=np.load('/work/state.npz'); A2=d['A2']; A4=d['A4']; nsw=int(d['nsw'])
import importlib.util
spec=importlib.util.spec_from_file_location('sim','/work/sim.py'); sim=importlib.util.module_from_spec(spec); spec.loader.exec_module(sim)
nmeas=nsw-sim.BURN
print('nsw',nsw,'nmeas',nmeas)
for T,u,e in sim.analyze(A2,A4,nmeas,G=100):
    print(f'{T:.2f} {u:.7f} {e:.3e}')
