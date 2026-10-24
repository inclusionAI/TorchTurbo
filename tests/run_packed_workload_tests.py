"""One-command packed-workload correctness and performance tests."""
import os
from pathlib import Path
import shlex
import subprocess
import sys


def run_phase(name: str, command: list[str], env: dict[str,str]) -> bool:
    print("\nPHASE_START "+name,flush=True)
    print("COMMAND "+shlex.join(command),flush=True)
    result=subprocess.run(command,env=env,stderr=subprocess.STDOUT)
    print(f"PHASE_RESULT {'PASS' if result.returncode==0 else 'FAIL'} {name} exit_code={result.returncode}",flush=True)
    return result.returncode==0

def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,"PYTHONPATH":str(root/"src")+os.pathsep+os.environ.get("PYTHONPATH",""),"PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"1"}
    print("TEST_SUITE packed-workload-balance",flush=True)
    results=[run_phase("correctness",[sys.executable,"-m","pytest","-vv","--color=no","-p","no:cacheprovider",str(root/"tests/engine/test_packed_workload.py")],env),
             run_phase("sampler_performance",[sys.executable,str(root/"tests/performance/packed_workload_benchmark.py")],env)]
    print(f"\nTEST_SUMMARY {'PASS' if all(results) else 'FAIL'} phases={len(results)} passed={sum(results)} failed={len(results)-sum(results)}",flush=True)
    return int(not all(results))

if __name__=="__main__":
    raise SystemExit(main())
