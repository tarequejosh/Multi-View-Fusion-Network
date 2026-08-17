"""
Run any Python script in the 'research' conda environment,
capturing output to a log file to avoid cp932 encoding issues.
Usage: python run_in_research.py <script.py> [args...]
"""
import subprocess, sys, os, pathlib

env_python = r"C:\Users\tareq\miniconda3\envs\research\python.exe"
script = sys.argv[1]
extra_args = sys.argv[2:]

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

result = subprocess.run(
    [env_python, script] + extra_args,
    cwd=os.getcwd(),
    env=env,
    capture_output=False,
)
sys.exit(result.returncode)
