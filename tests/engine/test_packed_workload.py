"""Independent sample, optimizer-batch, SP and precision contracts."""
from collections import Counter
import random
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from torchturbo.engine.config import EngineConfig
from torchturbo.engine.core import DPBlockDistributedSampler, TorchTurboEngine
from torchturbo.engine.packed_workload import (
    PackedCost, PackedWorkloadSampler, balance_step, packed_costs, supports_packed_costs,
)

class CachedPacks(Dataset):
    def __init__(self) -> None:
        self.dataset={"lengths":[[10],[20],[30],[40],[50],[60]]}
        self.packed_idx=[[0,1],[2],[3,4],[5]]
        self.packing_length=100
    def __len__(self) -> int:
        return len(self.packed_idx)
    def __getitem__(self,index: int) -> object:
        raise AssertionError("Scheduling must not load or alter sample tensors")

def test_metadata_and_wrappers():
    data=CachedPacks();costs=packed_costs(data)
    assert costs[0]==PackedCost(30,500,2,20,100)
    assert packed_costs(ConcatDataset([data,data]))==costs+costs
    assert packed_costs(Subset(data,[2,0]))==[costs[2],costs[0]]

@pytest.mark.parametrize("seed",[0,42,137])
@pytest.mark.parametrize("epoch",[0,3])
@pytest.mark.parametrize("shuffle",[False,True])
@pytest.mark.parametrize("drop_last",[False,True])
@pytest.mark.parametrize("batch_size",[1,2])
def test_batch_membership_and_sp(seed,epoch,shuffle,drop_last,batch_size):
    data=list(range(193));rng=random.Random(19)
    costs=[PackedCost(n,n*n,(n+63)//64,n,131072) for n in [rng.randint(1,131072) for _ in data]]
    block=4*batch_size;before=[];after=[]
    for dp_rank in range(4):
        options=dict(dataset=data,num_replicas=8,rank=dp_rank*2,dp_size=4,dp_rank=dp_rank,rank_block=block,shuffle=shuffle,seed=seed,drop_last=drop_last)
        base=DPBlockDistributedSampler(**options);base.set_epoch(epoch)
        candidate=PackedWorkloadSampler(base,costs,batch_size)
        before.append(list(base));after.append(list(candidate))
        assert len(candidate)==len(base)
        assert list(candidate)==after[-1]
        replica=DPBlockDistributedSampler(**{**options,"rank":dp_rank*2+1});replica.set_epoch(epoch)
        assert list(PackedWorkloadSampler(replica,costs,batch_size))==after[-1]
    for start in range(0,len(before[0]),block):
        original=Counter(tuple(row[i:i+batch_size]) for row in before for i in range(start,start+block,batch_size))
        balanced=Counter(tuple(row[i:i+batch_size]) for row in after for i in range(start,start+block,batch_size))
        assert original==balanced
    candidate.set_epoch(epoch+1)
    assert base.epoch==epoch+1

def test_optimizer_update_equivalence():
    torch.manual_seed(42)
    x=torch.randn(32,5,dtype=torch.float64);target=torch.randn(32,3,dtype=torch.float64)
    first=torch.nn.Linear(5,3,dtype=torch.float64);second=torch.nn.Linear(5,3,dtype=torch.float64);second.load_state_dict(first.state_dict())
    costs=[PackedCost(i+1,(i+1)**2,1,i+1,64) for i in range(32)]
    original=[list(range(rank*4,(rank+1)*4)) for rank in range(8)]
    balanced=balance_step(original,costs,1)
    for model,groups in [(first,original),(second,balanced)]:
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        for row in groups:
            for index in row:((model(x[index])-target[index]).square().sum()/32).backward()
        optimizer.step()
    for a,z in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,z,rtol=1e-12,atol=1e-12)

def test_wave_cost_floor():
    rng=random.Random(71)
    for _ in range(200):
        costs=[PackedCost(n,n*n,(n+63)//64,n,131072) for n in [rng.randint(1,131072) for _ in range(32)]]
        original=[list(range(rank*4,(rank+1)*4)) for rank in range(8)]
        candidate=balance_step(original,costs,1)
        def waits(groups):return sum(max(costs[row[i]].score for row in groups) for i in range(4))
        assert waits(candidate)<=waits(original)+1e-8

def test_recognizes_only_cached_packs():
    data=CachedPacks()
    assert supports_packed_costs(data)
    assert supports_packed_costs(Subset(data,[0,2]))
    assert supports_packed_costs(ConcatDataset([data,data]))
    assert not supports_packed_costs(list(range(8)))
    assert not supports_packed_costs(ConcatDataset([data,list(range(8))]))


@pytest.mark.parametrize("kind,gbs,dp_size,expected",[
    ("packed",16,4,True),
    ("subset",16,4,True),
    ("concat",16,4,True),
    ("ordinary",16,4,False),
    ("packed",None,4,False),
    ("packed",16,1,False),
])
def test_automatic_dataloader_selection(monkeypatch,kind,gbs,dp_size,expected):
    from torchturbo.engine import core
    engine=TorchTurboEngine.__new__(TorchTurboEngine)
    engine.config=EngineConfig(gbs=gbs)
    monkeypatch.setattr(engine,"get_dp_info",lambda:{"dp_size":dp_size,"dp_rank":0})
    monkeypatch.setattr(engine,"get_sp_info",lambda:{"sp_size":2,"sp_rank":0})
    monkeypatch.setattr(core.dist,"get_world_size",lambda:dp_size*2)
    monkeypatch.setattr(core.dist,"get_rank",lambda:0)
    dataset=CachedPacks()
    if kind=="subset":dataset=Subset(dataset,[0,2])
    if kind=="concat":dataset=ConcatDataset([dataset,dataset])
    if kind=="ordinary":dataset=list(range(64))
    loader=engine.create_dataloader(dataset,batch_size=1,pin_memory=False)
    assert isinstance(loader.sampler,PackedWorkloadSampler) is expected
