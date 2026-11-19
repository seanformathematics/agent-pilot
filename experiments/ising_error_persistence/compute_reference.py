import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime,timezone
import subprocess, platform
import logging


SEED = 137
N = 16

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True, cwd=Path(__file__).parent).stdout.strip()


# set up logging
log = logging.getLogger("compute_reference")
def setupLogging(path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                    logging.FileHandler(path, mode="w", encoding="utf-8")])

class IsingModel( object ):
    def __init__( self , size , seed ):
        self.size = size
        self.seed = seed
        self.randomNumberGenerator = np.random.default_rng(seed)
        self.lattice = self.generateSpinLattice()

        self.testWolff = lambda T : 1 - np.exp( -2.0 / T ) > self.randomNumberGenerator.random()
        self.testMetropolis = lambda E,T : np.exp( -1 * E / T ) > self.randomNumberGenerator.random()
    
    def generateSpinLattice( self , hot = True ):
        '''Produce a square lattice either (cold) uniform or (hot) randomly assign lattice sites with +1/-1 spins.'''
        if hot:
            lattice = self.randomNumberGenerator.integers( low=0 , high=2, size = (self.size,self.size) )
            return 2*lattice - 1
        else:
            return np.ones( shape=(self.size,self.size) )
        

    def resetLattice( self , hot = True ):
        self.lattice = self.generateSpinLattice( hot = hot )

    ## ------- informational tools -------- ##

    def getMagnetization( self ):
        return abs( self.lattice.sum() / self.lattice.size )

    def getMagnetizationPowers( self , n ):
        return (self.lattice.sum() / self.lattice.size) ** n

    ## ------- basic tools -------- ##
    
    def getRandomIndex( self ):
        return self.randomNumberGenerator.integers(self.size)
        
    def getRandomSite( self ):
        return ( self.getRandomIndex() , self.getRandomIndex() )

    def getNeighbors( self , i, j ):
        original = [i,j]
        N = original + np.array([0,1])
        S = original + np.array([0,-1])
        E = original + np.array([1,0])
        W = original + np.array([-1,0])
        return [tuple(np.mod(a,self.size)) for a in [N,S,E,W]]

    def getSiteEnergy( self , i , j ):
        return sum([ 2 * self.lattice[site] * self.lattice[(i,j)] for site in self.getNeighbors(i,j)])


    ## ------- MC tools -------- ##
    
    def metropolisBySite( self , T ):
        sites = [self.getRandomSite() for _ in range(self.lattice.size)]
        for site in sites:
            energy = self.getSiteEnergy( *site )
            if energy <= 0:
                self.lattice[site] *= -1
            else:
                if self.testMetropolis( energy , T ):
                    self.lattice[site] *= -1
    
    ## TODO: Metropolis Grid Scan vs Checkerboard Scan?
        
    def wolff( self , T ):
        
        stack = []
        site = self.getRandomSite()
        initialSpin = int(self.lattice[site])
        self.lattice[site] *= -1
        stack.append(site)
        while len(stack) > 0:
            s = stack.pop()
            neighbors = self.getNeighbors( *s )
            for n in neighbors:
                if (self.lattice[n] == initialSpin) and self.testWolff(T):
                    self.lattice[n] *= -1
                    stack.append( n )
                    

class Simulation( object ):
    def __init__( self, temperature ):
        self.T = temperature
        self.records = []
        self.round = 0

        # burn in round
        self.burnIn()
        self.wolffRound()

    def run( self , rounds ):
        for _ in range(rounds):
            self.wolffRound()

    def getObservations( self ):
        data = pd.DataFrame( self.records )
        return data

    @property
    def data( self ):
        return pd.DataFrame(self.records)

    def wolffRound( self ):

        self.round += 1
        self.lattice.wolff( self.T )
        
        row = {}
        row["round"] = self.round
        row["m"]  = self.lattice.getMagnetization()
        
        self.records.append(row)

    def burnIn( self ):
        hotLattice = IsingModel(N,[SEED,int(self.T * 10),1])
        coldLattice = IsingModel(N,[SEED,int(self.T * 10),0])
        coldLattice.resetLattice( hot = False )

        count = 0
        diff = 1
        records = []
        
        while diff > 0:
            count += 1
            hot = hotLattice.getMagnetization()
            cold = coldLattice.getMagnetization()
            diff = cold - hot
            records.append({ "round" : count , "hot" : hot , "cold" : cold , "diff" : diff } )
        
            hotLattice.wolff( self.T )
            coldLattice.wolff( self.T )

        self.burnTime = count
        self.burnRecords = {"record":"magnetization during burn in", "temperature" : self.T , "data": records , "burn_in_clusters" : 5*count }
        self.lattice = coldLattice
        self.otherLattice = hotLattice

        # finish the burn in at 5x the timescale
        for _ in range( 4 * self.burnTime ):
            self.lattice.wolff( self.T )
            self.otherLattice.wolff( self.T )


    def jackknifeBinder( self , numberOfBlocks ):

        m = np.asarray(self.data.m)

        # make sure we have an integral number of entries for each block
        n = numberOfBlocks * (len(m) // numberOfBlocks)
    
        # get a number of blocks
        b = m[:n].reshape(numberOfBlocks, -1)
        m2b = (b**2).mean(axis=1)
        m4b = (b**4).mean(axis=1)
        m2t, m4t = m2b.sum(), m4b.sum()
        
        # leave-one-block-out estimates
        m2j = (m2t - m2b) / (numberOfBlocks - 1)
        m4j = (m4t - m4b) / (numberOfBlocks - 1)
        
        Uj = 1 - m4j / (3 * m2j**2)
        U = 1 - m4b.mean() / (3 * m2b.mean()**2)
        err = np.sqrt((numberOfBlocks - 1) * np.mean((Uj - Uj.mean())**2))
        
        return pd.Series({"numberOfBlocks":numberOfBlocks,"U":U, "U_err": err})

    def jackknifeBinderStable( self , numberOfBlocks , errorLimit = 0.0005 , blockRange = 10 ):
        
        main = self.jackknifeBinder( numberOfBlocks )
        above = self.jackknifeBinder( numberOfBlocks + blockRange )
        below = self.jackknifeBinder( numberOfBlocks - blockRange )
        
        tests = {}
        tests["below"] = bool(below["U_err"] < errorLimit)
        tests["main"] = bool(main["U_err"] < errorLimit)
        tests["above"] = bool(above["U_err"] < errorLimit)
        tests["all"] = len(set(tests.values())) == 1


        tests["u_err"] = [{"blocks": numberOfBlocks - blockRange , "U": float(below["U"]) , "U_err" : float(below["U_err"])},
                          {"blocks": numberOfBlocks              , "U": float(main["U"])  , "U_err" : float(main["U_err"])},
                          {"blocks": numberOfBlocks + blockRange , "U": float(above["U"]) , "U_err" : float(above["U_err"])}]
        return tests
        
    
    def blockSizeAnalysis( self , plot = False ):

        blockSizes = [5*x for x in range(1,100)]
        analysis = pd.Series(blockSizes).apply( lambda x : self.jackknifeBinder(x) )
        if plot:
            analysis.plot(x='numberOfBlocks',y='U_err',kind='scatter')
        return analysis
        
class WolffRuns( object ):
    
    def __init__( self , temperatures , startingRounds , blockCount , errorLimit = 0.0005 , blockRange = 10 ):
        self.temperatures = temperatures
        self.rounds = startingRounds
        self.blocks = blockCount
        self.errorResolution = errorLimit
        self.blockRange = blockRange

        log.info("=== START OF WOLFF CLUSTER RUN ===")
        log.info("\n")
        log.info(f"running for T in {temperatures}, jackknife block count = {blockCount} with block range (+/- {blockRange}), starting with {startingRounds} clusters")
        self.run()
        log.info("\n")
        log.info("=== END OF WOLFF CLUSTER RUN ===")

    def run( self ):
        records = []
        
        for t in self.temperatures:
            log.info(f"--- Start of run for T={t} ---")
            S = Simulation( t )

            log.info(f"burn-in: t*={S.burnTime}, then {4 * S.burnTime} more clusters")

            S.run(self.rounds)

            data = {}
            data["temp"] = t
            data["rounds"] = self.rounds
            data["seeds"] = {"cold": S.lattice.seed , "hot":S.otherLattice.seed}
    
            log.info(f"cold start seed: {S.lattice.seed}, hot start seed {S.otherLattice.seed}")

            data["jackknife_tests"] = []
            U = S.jackknifeBinder(self.blocks)
            test = S.jackknifeBinderStable( self.blocks , errorLimit = self.errorResolution , blockRange = self.blockRange )
            data["jackknife_tests"].append({"clusters": data["rounds"], **test})
            
            data["U"] = U
            data["test"] = test
            
            log.info(f"binder cumulant: {U[["U","U_err"]].to_json()}.")

            errs = ", ".join(f"{r['blocks']}: {r['U_err']:.5f}" for r in test["u_err"])
            log.info(f"U_err by blocks: {errs}   all below limit: {test['main'] and test['all']}")


            check = test["main"] and test["all"]

            while not check:
                
                extra = data["rounds"]
                log.info(f"Running {extra} clusters.")
                S.run(extra)
                data["rounds"] += extra

                U = S.jackknifeBinder(self.blocks)
                test = S.jackknifeBinderStable( self.blocks , errorLimit = self.errorResolution , blockRange = self.blockRange )
                data["jackknife_tests"].append({"clusters": data["rounds"], **test})

                data["U"] = U
                data["test"] = test
                check = test["main"] and test["all"]
                log.info(f"binder cumulant: {U[["U","U_err"]].to_json()}.")

                # append jackknife test data to records.
                log.info(test)

            log.info(f"clusters: {data["rounds"]}\n")
            
            ## append the burn in records for clarity
            data["burn_records"] = S.burnRecords

            ## record total cluster count (rounds) for the actual simulation (exclusive of burn in)
            data["total_clusters"] = data["rounds"]
            ## record all records for this simulation run
            records.append(data)
            self.records = records
            log.info(f"--- End of run for T={t} ---")

    def report( self , commit ):

        data = pd.DataFrame(self.records)
        t = data["temp"]
        u = data["U"].apply(lambda x : x["U"])
        uerr = data["U"].apply(lambda x : x["U_err"])
        seeds = data["seeds"]
        burn = data["burn_records"]
        total = data["total_clusters"]    
        jackknife_tests = data["jackknife_tests"]
    
        points = pd.DataFrame({"T":t,"U":u,"U_err":uerr, "seeds":seeds,"burn_in": burn, "total_clusters":total, "jackknife_check":jackknife_tests}).apply(dict,axis=1).to_list()
        

        output = {}
        output["N"] = N
        output["method"] = (
            "Single-cluster Wolff updates on an L=16 periodic square lattice "
            "(H = -J sum s_i s_j, J = k_B = 1, bond probability 1 - exp(-2/T)). "
            "Equilibration: cold (all +1) and hot (random) starts evolved in parallel "
            "until cold |m| <= hot |m| at step t*, then 4t* further clusters. "
            f"Measurement: |m| recorded after every cluster flip; runs start at {self.rounds} "
            "clusters per T and are doubled as needed until the "
            f"jackknife error on U is below {self.errorResolution} at {self.blocks-self.blockRange}, {self.blocks} and {self.blocks+self.blockRange} blocks. "
            f"U = 1 - <m^4>/(3<m^2>^2), errors by blocked jackknife over {self.blocks} contiguous "
            f"blocks. Seeds are triples: [{SEED},int(10*T),0] (cold start), [{SEED},int(10*T),1] (hot start)."
        )     
        output["script"] = "experiments/ising/compute_reference.py"
        output["computed_at"] = datetime.now(timezone.utc).isoformat()
        output["seeds"] = { "base": 137,
                            "rule": "np.random.default_rng([base, int(10*T), 0]) for the cold start, [base, int(10*T), 1] for the hot start."
                            }
        output["points"] = points
        output["status"] = "final"
        output["commit"] = commit
        output["versions"] = {"python": platform.python_version(),
                      "numpy": np.__version__, "pandas": pd.__version__}

        return output

if __name__ == "__main__":
    # record which version of the code we ran
    here = Path(__file__).resolve()
    if git("status", "--porcelain", "--", str(here)):
        raise SystemExit(f"{here.name} is modified or untracked; commit it before running.")
    commit = git("rev-parse", "HEAD")

    rounds = 1000
    blockCount = 50
    errorLimit = 0.0005
    temperatures = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

    # set up logging
    setupLogging(here.parent/"N16.log")
    log.info(f"commit {commit}; python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}")
    log.info(f"start {rounds} clusters, {blockCount} blocks, error limit {errorLimit}, T = {temperatures}")

    
    try:
        W = WolffRuns( temperatures, rounds , blockCount , errorLimit=errorLimit)
        text = json.dumps(W.report(commit = commit), indent=2)
        (here.parent / "N16.json").write_text(text, encoding="utf-8")
        log.info("Output written to N16.json.")
    except Exception:
        log.exception("run failed")
        raise
    
        

