import time, multiprocessing as mp, sys
sys.path.insert(0, '/work')

def job(seed):
    import ising
    t0 = time.time()
    M, _ = ising.wolff_chain(16, 2.27, 40000, 1000, seed=seed)
    return time.time() - t0

if __name__ == "__main__":
    t0 = time.time(); job(1); single = time.time() - t0
    print(f"single job wall: {single:.2f}s", flush=True)
    for nproc in [4, 8]:
        t0 = time.time()
        with mp.Pool(nproc) as pool:
            times = pool.map(job, list(range(100, 100+nproc)))
        wall = time.time() - t0
        print(f"{nproc} parallel jobs: wall {wall:.2f}s, per-job times mean {sum(times)/len(times):.2f}s", flush=True)
