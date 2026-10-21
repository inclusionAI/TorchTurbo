"""Measure deterministic full-epoch sampler CPU cost against a locked floor."""
import argparse
import json
import platform
import os
from pathlib import Path
import random
import statistics
import time
import torch
from torchturbo.engine.core import DPBlockDistributedSampler
from torchturbo.engine.packed_workload import PackedCost,PackedWorkloadSampler


def measure() -> dict:
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    rng=random.Random(42)
    lengths=[rng.randint(33501,131072) for _ in range(969)]
    costs=[PackedCost(n,n*n,(n+63)//64,n,131072) for n in lengths]
    data=list(range(len(costs)))
    sampler=DPBlockDistributedSampler(data,num_replicas=64,rank=0,dp_size=8,dp_rank=0,rank_block=4,seed=42,drop_last=False)
    candidate=PackedWorkloadSampler(sampler,costs,1)
    samples=[]
    for i in range(25):
        start=time.process_time()
        for _ in range(10):list(candidate)
        elapsed=(time.process_time()-start)/10
        if i>=5:samples.append(elapsed)
    return {"machine":platform.machine(),"packs":969,"dp":8,"rank_block":4,"clock":"process_time","samples":samples,"median_seconds":statistics.median(samples),"p95_seconds":sorted(samples)[18]}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,default=Path(__file__).with_name("packed_workload_floor.json"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args();result=measure()
    print("PERFORMANCE_RESULT "+json.dumps(result),flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(result,indent=2))
    floor=json.loads(args.baseline.read_text())
    assert result["machine"]==floor["machine"],"Performance baseline requires the recorded CPU architecture"
    assert result["median_seconds"]<=floor["max_median_seconds"],result
    print(f"PERFORMANCE_GATE PASS median_ms={result['median_seconds']*1000:.6f} limit_ms={floor['max_median_seconds']*1000:.6f}",flush=True)

if __name__=="__main__":main()
