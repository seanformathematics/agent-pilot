
import sys
sys.path.insert(0,'/work')
import sim
temps = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
nsw, nb, dt = sim.run(seed=101, temps=temps, R=64, equil=1000, block=100, time_budget=480, out='/work/prod1.npz', L=16)
print("sweeps", nsw, "blocks", nb, "time", round(dt,1))
