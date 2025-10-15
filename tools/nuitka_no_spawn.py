# tools/nuitka_no_spawn.py
import runpy
import subprocess
import sys

# Force CPython to use fork/exec instead of posix_spawn
subprocess._USE_POSIX_SPAWN = False

# Make argv look like "nuitka <args>"
sys.argv[0] = "nuitka"

# Run Nuitka's CLI in this same interpreter
runpy.run_module("nuitka", run_name="__main__")
