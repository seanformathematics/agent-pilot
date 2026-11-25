import sys; sys.path.insert(0,'/work')
import importlib.util, time, json
spec=importlib.util.spec_from_file_location('ising','/tmp/run_python/ising_lib.py')
