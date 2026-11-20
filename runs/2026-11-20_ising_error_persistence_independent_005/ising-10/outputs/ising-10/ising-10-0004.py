
import sys, time; sys.path.insert(0,'/work')
import ising_lib as L
for T,seed in [(2.3,105),(2.4,106)]:
    t=time.time(); L.run(T,seed,2000,15000)
    print(T, L.analyze(f'/work/acc_T{T}_s{seed}.npz'), time.time()-t)
