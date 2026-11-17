import sys, glob
import numpy as np
sys.path.insert(0, '/work')
import wolff_prod as wp
files = sorted(glob.glob('/work/prod/blocks_*.npz'))
M2=[];M4=[];EE=[];CNT=[]
for f in files:
    d = np.load(f); M2.append(d['m2']); M4.append(d['m4']); EE.append(d['E']); CNT.append(d['cnt'])
M2=np.concatenate(M2); M4=np.concatenate(M4); EE=np.concatenate(EE); CNT=np.concatenate(CNT)
# blocks are 20 steps each (last partial block of each call is shorter); cumulative step index
cum = np.cumsum(CNT)
early = cum <= 500           # first 500 recorded steps
late = cum > 4000            # second half of the run
for g in [0, 5, 9, 11, 13]:
    sl = slice(g*wp.NC, (g+1)*wp.NC)
    e_early = EE[early][:, sl].sum(0)/CNT[early].sum(); e_late = EE[late][:, sl].sum(0)/CNT[late].sum()
    m_early = M2[early][:, sl].sum(0)/CNT[early].sum(); m_late = M2[late][:, sl].sum(0)/CNT[late].sum()
    de = e_early.mean() - e_late.mean(); de_err = np.sqrt(e_early.var(ddof=1)/wp.NC + e_late.var(ddof=1)/wp.NC)
    dm = m_early.mean() - m_late.mean(); dm_err = np.sqrt(m_early.var(ddof=1)/wp.NC + m_late.var(ddof=1)/wp.NC)
    print(f"T={wp.Ts[g]:.2f}: E/N early(0-500)={e_early.mean():.5f} late(>4000)={e_late.mean():.5f} diff={de:+.5f}+-{de_err:.5f} | "
          f"<m2> early={m_early.mean():.5f} late={m_late.mean():.5f} diff={dm:+.5f}+-{dm_err:.5f}")
