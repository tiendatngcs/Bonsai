from .frontend import PureRockmate
from .frontend import from_config


giga = 1024 **3 

def getRockmateModel(net, sample, budget):
    budget = budget * giga
    assert budget is not None, "Please provide a memory budget in bytes"
    rkModel = PureRockmate(net, sample, budget, cache_sched=True)
    # rkModel = PureCheckmate(net, sample, budget)
    return rkModel

def getCheckmateModel(net, sample, budget):
    budget = budget * giga
    rkModel = from_config(net, sample, budget, config_type="checkmate", memory_mode="checkmate", cache_sched=True)
    return rkModel

def getSegmentILPModel(net, sample, budget):
    budget = budget * giga
    rkModel = from_config(net, sample, budget, config_type="segment_ilp", memory_mode="segment_ilp", cache_sched=True)
    return rkModel

def getGreedySearchModel(net, sample, budget):
    budget = budget * giga
    rkModel = from_config(net, sample, budget, config_type="greedy-search", memory_mode="greedy-search")
    return rkModel

def getRecompOpListModel(net, sample, budget):
    budget = budget * giga
    rkModel = from_config(net, sample, budget, config_type="recompute-op-list", memory_mode="recompute-op-list")
    return rkModel

def getNoRecomputeModel(net, sample, budget):
    budget = budget * giga
    rkModel = from_config(net, sample, budget, config_type="no-recompute", memory_mode="no-recompute", cache_sched=True)
    return rkModel

def getOffloadAllModel(net, sample, budget):
    budget = budget * giga
    rkModel = from_config(net, sample, budget, config_type="offload_all", memory_mode="offload_all")
    return rkModel

def getOffmateModel(net, sample, budget):
    budget = budget * giga
    # rkModel =  Offmate(net, sample, budget)
    rkModel = from_config(net, sample, budget, config_type="offmate")
    return rkModel
