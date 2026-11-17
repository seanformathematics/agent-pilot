module_src = r'''
import numpy as np, time, os, glob, json

L = 16
Ts = np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
NC = 256
NT = len(Ts)
R = NT*NC
DIR = '/work/prod'
INIT_SEED = 2000

def wolff_step(S, pb, rng):
    R_ = S.shape[0]
    HO = (rng.random((R_, L, L)) < pb) & (S == np.roll(S, -1, axis=2))
    VO = (rng.random((R_, L, L)) < pb) & (S == np.roll(S, -1, axis=1))
    seed = rng.integers(0, L*L, size=R_)
    C = np.zeros((R_, L*L), dtype=bool)
    C[np.arange(R_), seed] = True
    C = C.reshape(R_, L, L)
    F = C.copy()
    while True:
        new = (np.roll(F & HO, 1, axis=2) | (np.roll(F, -1, axis=2) & HO)
               | np.roll(F & VO, 1, axis=1) | (np.roll(F, -1, axis=1) & VO))
        new &= ~C
        if not new.any():
            break
        C |= new
        F = new
    return np.where(C, -S, S).astype(np.int8)

def energy_per_site(S):
    S32 = S.astype(np.int32)
    return -((S32*np.roll(S32, -1, axis=1)).sum(axis=(1,2)) + (S32*np.roll(S32, -1, axis=2)).sum(axis=(1,2)))/(L*L)

def setup_pb():
    Tr = np.repeat(Ts, NC)
    return (1-np.exp(-2.0/Tr)).reshape(R,1,1)

def run_call(call_idx, seed, n_burn, n_rec_target, time_budget, block=20):
    os.makedirs(DIR, exist_ok=True)
    pb = setup_pb()
    rng = np.random.default_rng(seed)
    st_path = f'{DIR}/state.npz'
    if os.path.exists(st_path):
        st = np.load(st_path)
        S = st['S'].copy(); burn_done = int(st['burn_done']); rec_done = int(st['rec_done'])
    else:
        S = np.random.default_rng(INIT_SEED).choice(np.array([-1,1],dtype=np.int8), size=(R,L,L))
        burn_done = 0; rec_done = 0
    t0 = time.time()
    b0 = burn_done
    while burn_done < n_burn and (time.time()-t0) < time_budget:
        S = wolff_step(S, pb, rng); burn_done += 1
    bm2=[]; bm4=[]; bE=[]; bcnt=[]
    c2 = np.zeros(R); c4 = np.zeros(R); cE = np.zeros(R); cc = 0
    r0 = rec_done
    while burn_done >= n_burn and rec_done < n_rec_target and (time.time()-t0) < time_budget:
        S = wolff_step(S, pb, rng)
        m = S.reshape(R,-1).sum(1)/(L*L)
        c2 += m*m; c4 += m**4; cE += energy_per_site(S); cc += 1; rec_done += 1
        if cc == block:
            bm2.append(c2); bm4.append(c4); bE.append(cE); bcnt.append(cc)
            c2 = np.zeros(R); c4 = np.zeros(R); cE = np.zeros(R); cc = 0
    if cc > 0:
        bm2.append(c2); bm4.append(c4); bE.append(cE); bcnt.append(cc)
    elapsed = time.time()-t0
    if len(bm2) > 0:
        np.savez(f'{DIR}/blocks_{call_idx:02d}.npz', m2=np.array(bm2), m4=np.array(bm4),
                 E=np.array(bE), cnt=np.array(bcnt), seed=seed)
    np.savez(st_path, S=S, burn_done=burn_done, rec_done=rec_done)
    info = dict(call_idx=call_idx, seed=seed, burn_done=burn_done, rec_done=rec_done,
                new_burn=burn_done-b0, new_rec=rec_done-r0, elapsed_s=round(elapsed,2))
    with open(f'{DIR}/log.jsonl','a') as f:
        f.write(json.dumps(info)+'\n')
    return info

def jackknife(a2, a4):
    n = len(a2)
    A = a4.sum(); B = a2.sum()
    U = 1 - (A/n)/(3*(B/n)**2)
    Uj = np.array([1 - ((A-a4[i])/(n-1))/(3*((B-a2[i])/(n-1))**2) for i in range(n)])
    err = np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
    return U, err

def analyze():
    files = sorted(glob.glob(f'{DIR}/blocks_*.npz'))
    M2=[];M4=[];EE=[];CNT=[]
    for f in files:
        d = np.load(f)
        M2.append(d['m2']); M4.append(d['m4']); EE.append(d['E']); CNT.append(d['cnt'])
    M2 = np.concatenate(M2); M4 = np.concatenate(M4); EE = np.concatenate(EE); CNT = np.concatenate(CNT)
    ntot = CNT.sum()
    nb = len(CNT); half = nb//2
    cnt1 = CNT[:half].sum(); cnt2 = CNT[half:].sum()
    s2 = (M2.sum(0)/ntot).reshape(NT, NC)
    s4 = (M4.sum(0)/ntot).reshape(NT, NC)
    sE = (EE.sum(0)/ntot).reshape(NT, NC)
    s2a = (M2[:half].sum(0)/cnt1).reshape(NT, NC); s4a = (M4[:half].sum(0)/cnt1).reshape(NT, NC)
    s2b = (M2[half:].sum(0)/cnt2).reshape(NT, NC); s4b = (M4[half:].sum(0)/cnt2).reshape(NT, NC)
    rows = []
    for g in range(NT):
        U, Ue = jackknife(s2[g], s4[g])
        Ua, _ = jackknife(s2a[g], s4a[g]); Ub, _ = jackknife(s2b[g], s4b[g])
        rows.append(dict(T=float(Ts[g]), U=float(U), U_err=float(Ue), n_samples=int(NC*ntot),
                         U_half1=float(Ua), U_half2=float(Ub), m2=float(s2[g].mean()),
                         E=float(sE[g].mean()), E_err=float(sE[g].std(ddof=1)/np.sqrt(NC))))
    return rows, int(ntot), len(files)
'''
os.makedirs('/work', exist_ok=True) if False else None
import os
os.makedirs('/work/prod', exist_ok=True)
with open('/work/wolff_prod.py', 'w') as f:
    f.write(module_src)
import sys
sys.path.insert(0, '/work')
import wolff_prod as wp

info = wp.run_call(call_idx=1, seed=2001, n_burn=1000, n_rec_target=8000, time_budget=270.0)
print(info)
rows, ntot, nfiles = wp.analyze()
print("recorded steps per chain so far:", ntot, "block files:", nfiles)
for r in rows:
    print(f"T={r['T']:.2f} U={r['U']:.5f} Uerr={r['U_err']:.5f} n={r['n_samples']} U_h1={r['U_half1']:.5f} U_h2={r['U_half2']:.5f} <m2>={r['m2']:.5f} E={r['E']:.5f}+-{r['E_err']:.5f}")
