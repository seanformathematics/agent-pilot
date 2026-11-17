
import shutil, subprocess, sys, importlib.util, os
for prog in ['gcc', 'cc', 'clang', 'tcc', 'g++', 'make']:
    print(prog, shutil.which(prog))
for mod in ['numba', 'cython', 'cffi', 'scipy', 'numpy', 'jax', 'torch', 'pythran']:
    print(mod, importlib.util.find_spec(mod) is not None)
print(sys.version)
try:
    r = subprocess.run([sys.executable, '-m', 'pip', 'download', 'numba', '-d', '/tmp/x', '--no-deps', '-q'], capture_output=True, text=True, timeout=60)
    print('pip download rc', r.returncode, r.stdout[-300:], r.stderr[-300:])
except Exception as e:
    print('pip failed', e)
print(os.cpu_count())
