from typing import Dict, Any
import time
import numpy as np
from copy import deepcopy
from ...op_schedule import (
    Activation,
    Parameter,
    Buffer,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    PrepareOp,
    ExecCodeOp,
    OpSchedule,
)
from ..main import get_sched, add_sched, translate
from .ilp_model import ModelPULP
from .ilp_offload import ModelPULPOffload

from .ilp_segment_model import CheckpointSolverHandler

# Dat mod
from rkgb.core.backward import (
    ComputationNode,
    AllocationNode,
    ForwardAndBackwardGraph
)


class knapsack:
    def __init__(self, parameter_sizes: list, pre_solve_size=10):
        size = [s[1] for s in parameter_sizes]
        self.parameter_sizes = parameter_sizes
        self.sizes = [s / sum(size) for s in size]
        self.pre_solve_size = pre_solve_size

    def get_size(self, indices, sizes):
        return sum(sizes[i] for i in indices)

    # @lru_cache(maxsize=4096 * 4096)
    def solve(self, frac: float, i: int = 0, sizes=[]):
        sizes = sizes or self.sizes
        if frac < 0:
            return []
        if i == len(sizes):
            return list(range(i))
        res1 = self.solve(frac, i + 1, sizes)
        res2 = self.solve(frac - sizes[i], i + 1, sizes)
        res2 = [i] + res2
        if self.get_size(res1, sizes) <= self.get_size(res2, sizes):
            return res1
        else:
            return res2

    def select(self, frac: float):
        sizes = self.sizes.copy()
        parameter_sizes = self.parameter_sizes.copy()
        selections = []
        while len(sizes) > self.pre_solve_size and frac > 0:
            sel_i = self.presolve(frac, sizes)
            if sel_i is None:
                break
            selections.append(parameter_sizes.pop(sel_i)[0])
            frac -= sizes.pop(sel_i)
        indices = self.solve(frac, sizes=sizes)
        selections += [parameter_sizes[i][0] for i in indices]
        return selections

    def select_size(self, size: int):
        if not self.parameter_sizes:
            return []
        return self.select(size / sum(s[1] for s in self.parameter_sizes))

    def presolve(self, frac, sizes):
        array = np.array(sizes)
        sel_i = np.argmax(array * (array < frac))
        if array[sel_i] > frac:
            return np.argmin(array)
        return sel_i


def schedule(md: ModelPULP, hgraph=None, check_valid=False):
    """
    Given the solution from HILP, we want to translate the result
    to a OpSchedule that can be used in a higher level.
    """
    hgraph = hgraph if hgraph else md.hgraph

    init_op_list = []
    restore_op_list = []
    init_alive_status = {}
    loss_op = ComputeOp(md.hgraph.cluster.loss_cnode, disabled=True)
    if isinstance(md, ModelPULPOffload):
        W = len(md.parameter_size)
        (
            op_list,
            init_alive_status,
            init_op_list,
            restore_op_list,
        ) = schedule_offload(md, hgraph)
        op_name_list = [op.name for op in op_list]

        init_ops = {op.target.name: op for op in init_op_list}
        for pnode in md.hgraph.cluster.parameter_nodes:
            if pnode.mem < md.minor_offload_size or pnode.is_buffer:
                device = "cuda"
            else:
                device = "cpu"
            alloc = Parameter(pnode)
            alloc_grad = Parameter(pnode, is_grad=True)
            if alloc.name not in init_ops:
                init_op_list.append(
                    PrepareOp(
                        alloc,
                        device=device,
                        cpu_optimize=OffloadOp(alloc_grad).name
                        in op_name_list,
                        pin_memory=OffloadOp(alloc).name in op_name_list,
                    )
                )

    else:
        op_list = []
        for pnode in md.hgraph.cluster.parameter_nodes:
            init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))
        for t in range(md.T):
            for k in md.krange(t):
                if t == md.loss_idx and k == md.loss_idx:
                    op_list.append(loss_op)
                op_list += schedule_compute(md, t, k, hgraph)

    # print("finish scheduling")
    for anode in md.hgraph.cluster.interfaces["input_data_anodes"]:
        init_alive_status[anode.name] = True  # anode share the name as alloc

    op_sched = OpSchedule(
        op_list,
        loss_idx=op_list.index(loss_op),
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=restore_op_list,
        with_parameters=isinstance(md, ModelPULPOffload),
        optimizer_states_factor=(
            md.optimize_metrics["optimizer_states_factor"]
            if hasattr(md, "optimize_metrics")
            else None
        ),
    )
    # check_valid = True
    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError

    # print("Generated op_sched")
    # for op in op_sched.op_list:
    #     print(f"  Op is type {type(op)}")
    #     print(f"  Op in sched: {op}")
    return op_sched


def schedule_compute(md: ModelPULP, t, k, hgraph):
    op_list = []
    sol = md.sol

    j = md.hcn2sub_c[k]
    # if md.sumComp[t, k].value() == 1:
    if sol(md.sumComp[t, k]):
        hcn = hgraph.list_HCNs[k]
        opt = -1
        for o in range(md.nSched[k]):
            if sol(md.Comp[t, k, o]):
                opt = o
                break
        if opt > -1:
            h_obj = md.list_list_sched[j][opt]
            if hcn.is_fwd:
                sub_op_list = h_obj.op_list[: h_obj.loss_idx]
            else:
                sub_op_list = h_obj.op_list[h_obj.loss_idx + 1 :]

                # if md.sumAliveP[(j, t + 1)].value() == 0:
                # sub_op_list.append()
            sub_op_list = deepcopy(sub_op_list)

            if (
                not hcn.is_fwd
                # and md.sumAliveP[(j, t + 1)].value() > 0
                and sol(md.sumAliveP[t + 1, j])
            ):  # phantoms should be kept
                phantoms_to_keep = h_obj.phantoms
                # for op in sub_op_list[::-1]:
                #     if (
                #         op.is_del
                #         and not op.disabled
                #         and op.kn in phantoms_to_keep
                #     ):
                #         # Only the last del should be disabled
                #         op.disabled = True
                #         phantoms_to_keep.remove(op.kn)
            # translating sub_op_list
            if hcn.sub_cluster is not hcn.sub_cluster.representee_cluster:
                sub_op_list = translate(hcn.sub_cluster, sub_op_list)
        else:
            h_obj = hcn
            sub_op_list = deepcopy(h_obj.ff_op_list)

        anode_to_protect = []
        if not hcn.sub_cluster is None:
            for anode in hcn.sub_cluster.all_interfaces:
                i_ = [hdn.anode.name for hdn in md.hgraph.list_HANs].index(anode.name)
                if (k, i_) in md.delete_list and k in md.han_users_real[i_]:
                    eidx = md.delete_list.index((k, i_))
                    if not sol(md.delete[t, eidx]):
                        anode_to_protect.append(anode.name)

        for op in sub_op_list:
            if isinstance(op, DeleteOp) and op.target.anode.name in anode_to_protect:
                op.disabled = True
        op_list += sub_op_list

    for eidx, (k_, i) in enumerate(md.delete_list):
        # print(k_, i)
        # if k == k_ and md.delete[t, eidx].value()==1:
        if k == k_ and sol(md.delete[t, eidx]):
            han = hgraph.list_HANs[i]
            op_list.append(DeleteOp(Activation(han.anode)))
    return op_list


def schedule_offload(md: ModelPULPOffload, hgraph=None):
    """
    V1: md.grouping = False:
    merge every cluster, ofl/prf/del partially, high memory overhead
    """
    hgraph = hgraph if hgraph else md.hgraph

    ### Handle multiplier
    md.params_vars = [
        md.AliveW,
        md.OflWProg,
        md.OflW,
        md.PrfW,
        md.PrfWProg,
        md.OptC,
        md.AliveG,
        md.OflGProg,
        md.OflG,
        md.PrfG,
        md.PrfGProg,
        md.AliveO,
        md.OflO,
        md.OflOProg,
        md.PrfO,
        md.PrfOProg,
    ]
    if isinstance(md.req_w, float):
        multiplier = 1 - md.req_w
    else:
        multiplier = 1 - md.req_w.value()
    for p in md.params_vars:
        for k, v in p.items():
            p[k] = v * 1 / (1 - multiplier)
            pass

    if md.activation_offload:
        md.selected_phantoms = {}

    def add_op(op_dict, op_list):
        for op in op_list:
            op_dict[op[:2]].append(op[2])

    md.ofl_ops = {step: [] for step in md.active_steps}
    md.prf_ops = {step: [] for step in md.active_steps}
    md.del_ops = {step: [] for step in md.active_steps}
    md.opt_ops = {step: [] for step in md.active_steps}
    md.cpu_optimized_params = {}
    md.cpu_optimized_steps = {step: [] for step in md.active_steps}
    init_op_list = []
    restore_op_list = []
    init_alive_status = dict()
    if md.grouping:
        for w in range(md.W)[::-1]:
            o_l, p_l, d_l, t_l, i_l, r_l, init_alive = group(md, w)
            add_op(md.ofl_ops, o_l)
            add_op(md.prf_ops, p_l)
            add_op(md.del_ops, d_l)
            add_op(md.opt_ops, t_l)
            init_op_list.extend([ops[2] for ops in i_l])
            restore_op_list.extend([ops[2] for ops in r_l])
            for alloc in init_alive:
                init_alive_status[alloc.name] = True

        for j in range(md.J):
            ofl_ops, prf_ops, del_ops = group_activation_offload(md, j)
            add_op(md.ofl_ops, ofl_ops)
            add_op(md.prf_ops, prf_ops)
            add_op(md.del_ops, del_ops)
    # else:
    #     init_op_list = md.schedule_init_op_list()

    sol = md.sol
    # offload_buffers = {w:[] for w in range(W)}
    op_list = []

    # for op in init_op_list:
    #     if isinstance(op, AllocateOp):
    #         init_alive_status[op.target] = True

    for t in range(md.T):
        for k in md.krange(t):
            op_list.extend(schedule_step(md, t, k))

    return op_list, init_alive_status, init_op_list, restore_op_list

def schedule_step(md: ModelPULP, t, k):
    op_list = []
    if not md.sol(md.sumComp[t, k]):
        return op_list
    op_list.append(SynchronizeOp(f"{(t,k)}"))
    if t == md.loss_idx and k == md.loss_idx:
        op_list.append(ComputeOp(md.hgraph.cluster.loss_cnode, disabled=True))
    
    j = md.hcn2sub_c[k]
    op_list += schedule_compute(md, t, k, md.hgraph)
    last_op = op_list[-1]
    last_op.record_event = True
    self_ops = []
    self_targets = [
        p.param_name for w in md.hcn2param[k] for p in md.parameters[w]
    ]
    if md.hcn2sub_c[k] is not None and md.activation_offload:
        self_targets += md.selected_phantoms[j]
    for op in md.opt_ops[t, k]:
        op_list.append(op)
        if op.target.target_name in self_targets:
            op.record_event = True
    for op in md.ofl_ops[t, k]:
        if op.target.target_name in self_targets:
            self_ops.append(op)
        else:
            op_list.append(op)
    for op in md.del_ops[t, k]:
        if op.target.target_name in self_targets:
            self_ops.append(op)
        else:
            op_list.append(op)
    for op in md.prf_ops[t, k]:
        op_list.append(op)

    op_list.extend(self_ops)

    occurrences = dict()
    for i, op in enumerate(op_list):
        if op.name not in occurrences:
            occurrences[op.name] = []
        occurrences[op.name].append(i)
    for op in self_ops:
        if isinstance(op, OffloadOp):
            op.record_event = True
            opt_op_name = OptimizeOp([op.target.pnode.param_name],
                        alloc=op.target).name
            if opt_op_name in occurrences:
                opt_op = op_list[max(occurrences[opt_op_name])]
                op.wait_events.append((opt_op.op_type, opt_op.target.name))

        if isinstance(op, DeleteOp):
            ofl_op_name = OffloadOp(alloc=op.target).name
            if ofl_op_name in occurrences:
                ofl_op = op_list[max(occurrences[ofl_op_name])]
                op.wait_events.append((ofl_op.op_type, ofl_op.target.name))
    return op_list

def group(md: ModelPULPOffload, w, tol=1):
    # Group the parameters of each block for the task
    fwd_i = min(md.param2hcn[w])
    bwd_i = max(md.param2hcn[w])
    early_fwd = []
    for t in range(bwd_i, md.T):
        if not md.single_fwd and md.sol(md.sumComp[t, fwd_i]):
            early_fwd.append(t)  # if recompute fwd after bwd
    hcn = md.hgraph.list_HCNs[fwd_i]
    parameters = {pnode.param_name: pnode for pnode in md.parameters[w]}
    parameter_size = sum(pnode.mem for pnode in parameters.values())

    Alive = {p: 1 for p in parameters.keys()}
    Offloaded = {p: False for p in parameters.keys()}

    ofl_ops = []
    prf_ops = []
    del_ops = []
    opt_ops = []
    init_ops = []
    restore_ops = []
    init_alive = []
    cpu_optimize_candidates = {p: 0 for p in parameters.keys()} if md.optimize_metrics else {}

    def apply_cpu_optimize(p):
        for t, k, op in ofl_ops:
            if op.target.target_name == p:
                op.target.is_grad = True
                # op.grad = True
                break
        # for (t,k,op) in del_ops:
        #     if op.target.name == p:
        #         op.grad = True
        del_ops.append((t, k, DeleteOp(Parameter(parameters[p], is_grad=True))))
        i = (
            md.active_steps.index((t, k)) + 1
        )  # TODO: distribute cpu optimization based on time
        p_alloc = Parameter(parameters[p])
        op = OptimizeOp(
            list_params=[p],
            cpu=True,
            alloc=p_alloc,
            time=parameters[p].mem / md.cpu_optimize_speed / md.gcd,
        )
        opt_ops.append((*md.active_steps[i], op))
        md.cpu_optimized_steps[md.active_steps[i]].append(p)
        # del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p]))))

        # if cpu optimize, do not keep w after bwd

    def apply_gpu_optimize(p):
        p_alloc = Parameter(parameters[p])
        op = OptimizeOp(
            list_params=[p],
            alloc=p_alloc,
            time=parameters[p].mem / md.gpu_optimize_speed / md.gcd,
            overhead=parameters[p].mem * md.optimizer_overhead_factor,
        )
        opt_ops.append((bwd_i, bwd_i, op))  # optimize after bwd
        del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p], is_grad=True))))

    assert (bwd_i, bwd_i) in md.active_steps
    idx = md.active_steps.index((bwd_i, bwd_i))
    for t, k in md.active_steps[idx:] + md.active_steps[:idx]:
        t_, k_ = md.next_idx(t, k)
        current_alive_size = sum(parameters[p].mem * a for p, a in Alive.items())
        current_offloaded_size = sum(
            parameters[p].mem * a for p, a in Offloaded.items()
        )
        next_alive_size = round(
            (md.AliveG[(t_, k_, w)] + md.AliveW[(t_, k_, w)]).value() * parameter_size
        )
        next_offloaded_size = round(
            (md.OflGProg[(t_, k_, w)] + md.OflWProg[(t_, k_, w)]).value()
            * parameter_size
        )

        # assert current_alive_size <= round(md.AliveW[(t, k, w)].value() * parameter_size)

        if (t, k) == (0, 0):  # init
            for p, a in Alive.items():
                if a:
                    p_alloc = Parameter(parameters[p])
                    # init_ops.append((t, k, AllocateOp(p_alloc)))
                    init_ops.append(
                        (t, k, PrepareOp(p_alloc, device="cuda", cpu_optimize=True))
                    )
                    init_alive.append(p_alloc)
                    # op = PrefetchOp(
                    #     alloc=p_alloc, indices=(0, None),
                    #     time=parameters[p].mem/md.bandwidthPrf/md.gcd
                    # )
                    # init_ops.append((t, k, op))
                    op = OffloadOp(
                        alloc=p_alloc,
                        indices=(0, None),
                        time=parameters[p].mem / md.bandwidthOfl / md.gcd,
                    )
                    restore_ops.append((t, k, op))
                    restore_ops.append((t, k, DeleteOp(p_alloc)))

        if next_offloaded_size > current_offloaded_size:
            # print(t,k, next_offloaded_size, current_offloaded_size)
            ofl_size = next_offloaded_size - current_offloaded_size
            candidates = {
                p: parameters[p].mem * (1 - o) for p, o in Offloaded.items() if o < 1
            }
            if not candidates:
                if ofl_size < 1024:
                    ofl_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(ofl_size)
            # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
            #     pass
            for p in select_paras:
                op = OffloadOp(
                    alloc=Parameter(parameters[p]),
                    indices=(0, None),
                    time=parameters[p].mem / md.bandwidthOfl / md.gcd,
                )
                ofl_ops.append((t, k, op))
                Offloaded[p] = 1

        if current_alive_size > next_alive_size:
            del_size = current_alive_size - next_alive_size
            candidates = {}
            for p, o in Offloaded.items():
                if Alive[p] > 0 and o > 0:
                    candidates[p] = min(o, Alive[p]) * parameters[p].mem
            if not candidates:
                if del_size < 1024:
                    del_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(del_size)
            # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
            #     pass
            for p in select_paras:
                del_ops.append((t, k, DeleteOp(Parameter(parameters[p]))))
                Alive[p] = 0
        if current_alive_size < next_alive_size:
            # prefetch should be smaller than solution
            prf_size = next_alive_size - current_alive_size
            candidates = {
                p: parameters[p].mem * (1 - a) for p, a in Alive.items() if a < 1
            }
            if not candidates:
                if prf_size < 1024:
                    prf_size = 0
                else:
                    raise ValueError
            if md.sol(md.AliveW[(t_, k_, w)]):
                select_paras = list(candidates.keys())
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
            else:
                selector = knapsack(list(candidates.items()))
                unselect_paras = selector.select_size(
                    sum(candidates.values()) - prf_size
                )

                select_paras = [p for p in candidates.keys() if p not in unselect_paras]
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
            #     pass
            for p in select_paras:
                prf_ops.append((t, k, AllocateOp(Parameter(parameters[p]))))
                op = PrefetchOp(
                    alloc=Parameter(parameters[p]),
                    indices=(0, None),
                    time=parameters[p].mem / md.bandwidthPrf / md.gcd,
                )
                prf_ops.append((t, k, op))
                Alive[p] = 1
                if (t > bwd_i and t < min(early_fwd + [md.T + 1])) or t < fwd_i:
                    # cpu optimize only if prefetch before fwd
                    if parameters[p].info.requires_grad and md.optimize_metrics:
                        # only trainable parameters will be optimize candidate
                        cpu_optimize_candidates[p] = 1

    if not md.optimize_metrics:
        return ofl_ops, prf_ops, del_ops, opt_ops, init_ops, restore_ops, init_alive

    candidates = {
        p: parameters[p].mem * a for p, a in cpu_optimize_candidates.items() if a > 0
    }
    select_paras = []
    # assert sum(candidates.values())/parameter_size >= md.sumOptC[w].value()-0.01

    # cpu_optimize_size = md.sumOptC[w].value()*parameter_size# size by subgraph
    if isinstance(md.req_w, float):
        multiplier = 1 - md.req_w
    else:
        multiplier = 1 - md.req_w.value()
    # cpu_optimize_size = (sum(md.sumOptC[w_].value() *
    #                         md.parameter_gradient_size[w_] *md.gcd
    #                         for w_ in range(w, md.W)) / (1-multiplier)
    #                     - sum(md.cpu_optimized_params.values()))# size by all graphs
    cpu_optimize_size = min(
        sum(candidates.values()),
        (md.sumOptC[w].value() * md.parameter_gradient_size[w] * md.gcd)
        / (1 - multiplier),
    )

    if candidates and cpu_optimize_size > 0:
        # print(candidates, cpu_optimize_size)
        selector = knapsack(list(candidates.items()))
        select_paras = selector.select_size(cpu_optimize_size)
        if cpu_optimize_size > sum(candidates[p] for p in select_paras):
            raise ValueError
        # print(select_paras)

    # Optimize parameters which requires grad
    gpu_optimze_param = []
    for p, pnode in parameters.items():
        if not pnode.info.requires_grad:
            continue
        if p in select_paras:
            md.cpu_optimized_params[p] = parameters[p].mem
            apply_cpu_optimize(p)
        else:
            apply_gpu_optimize(p)
            gpu_optimze_param.append(pnode)
    if md.optimize_metrics and gpu_optimze_param:
        ofl_ops_os, prf_ops_os, del_ops_os, init_alive_os = group_optimizer_states(
            md, w, gpu_optimze_param
        )
        ofl_ops += ofl_ops_os
        prf_ops += prf_ops_os
        del_ops += del_ops_os
        init_alive += init_alive_os
    return ofl_ops, prf_ops, del_ops, opt_ops, init_ops, restore_ops, init_alive


def group_optimizer_states(md, w, gpu_optimize_param):
    # To offload and prefetch optimizer states witin the gpu_optimize_param
    ofl_ops = []
    prf_ops = []
    del_ops = []
    init_alive = []
    fwd_i = min(md.param2hcn[w])
    bwd_i = max(md.param2hcn[w])
    hcn = md.hgraph.list_HCNs[fwd_i]
    parameters = {
        pnode.param_name: pnode for pnode in md.parameters[w] if pnode.requires_grad
    }
    parameter_size = sum(pnode.mem for pnode in parameters.values())
    gpu_optimize_size = sum(pnode.mem for pnode in gpu_optimize_param)

    Alive = {pnode.param_name: 1 for pnode in gpu_optimize_param}
    Offloaded = {pnode.param_name: False for pnode in gpu_optimize_param}
    assert (bwd_i, bwd_i) in md.active_steps
    idx = md.active_steps.index((bwd_i, bwd_i))
    for t, k in md.active_steps[idx:] + md.active_steps[:idx]:
        if (t, k) == (0, 0):  # init
            for p, a in Alive.items():
                if a:
                    init_alive.append(Parameter(parameters[p], is_optim_states=True))

        t_, k_ = md.next_idx(t, k)
        current_alive_size = sum(parameters[p].mem * a for p, a in Alive.items())
        current_offloaded_size = sum(
            parameters[p].mem * a for p, a in Offloaded.items()
        )
        next_alive_size = min(
            gpu_optimize_size, round((md.AliveO[(t_, k_, w)]).value() * parameter_size)
        )
        next_offloaded_size = min(
            gpu_optimize_size,
            round((md.OflOProg[(t_, k_, w)]).value() * parameter_size),
        )
        if parameter_size * (1 - md.sumOptC[w]).value() < gpu_optimize_size:
            next_offloaded_size += (
                gpu_optimize_size - parameter_size * (1 - md.sumOptC[w]).value()
            )

        # assert current_alive_size <= round(md.AliveW[(t, k, w)].value() * parameter_size)
        if next_offloaded_size > current_offloaded_size:
            # print(t,k, next_offloaded_size, current_offloaded_size)
            ofl_size = next_offloaded_size - current_offloaded_size
            candidates = {
                p: parameters[p].mem * (1 - o) for p, o in Offloaded.items() if o < 1
            }
            if not candidates:
                if ofl_size < 1024:
                    ofl_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(ofl_size)
            # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
            #     pass
            for p in select_paras:
                op = OffloadOp(
                    alloc=Parameter(parameters[p], is_optim_states=True),
                    indices=(0, None),
                    time=parameters[p].mem
                    / md.bandwidthOfl
                    / md.gcd
                    * md.optimizer_states_factor,
                )
                ofl_ops.append((t, k, op))
                Offloaded[p] = 1

        if current_alive_size > next_alive_size:
            if k_ == bwd_i:
                continue
            del_size = current_alive_size - next_alive_size
            candidates = {}
            for p, o in Offloaded.items():
                if Alive[p] > 0 and o > 0:
                    candidates[p] = min(o, Alive[p]) * parameters[p].mem
            if not candidates:
                if del_size < 1024:
                    del_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(del_size)
            # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
            #     pass
            for p in select_paras:
                del_ops.append(
                    (t, k, DeleteOp(Parameter(parameters[p], is_optim_states=True)))
                )
                Alive[p] = 0
        if current_alive_size < next_alive_size or k_ == bwd_i:
            # if w == 15:print(md.active_steps[k_]==bwd_i)
            # prefetch should be smaller than solution
            prf_size = next_alive_size - current_alive_size
            candidates = {
                p: parameters[p].mem * (1 - a) for p, a in Alive.items() if a < 1
            }
            if not candidates:
                if prf_size < 1024:
                    prf_size = 0
                else:
                    raise ValueError
            if (
                md.sol(md.AliveO[(t_, k_, w)] + md.sumOptC[w] - md.req_w)
                or k_ == bwd_i
            ):

                select_paras = list(candidates.keys())
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
            else:
                selector = knapsack(list(candidates.items()))
                unselect_paras = selector.select_size(
                    sum(candidates.values()) - prf_size
                )

                select_paras = [p for p in candidates.keys() if p not in unselect_paras]
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
            #     pass
            for p in select_paras:
                prf_ops.append(
                    (t, k, AllocateOp(Parameter(parameters[p], is_optim_states=True)))
                )
                op = PrefetchOp(
                    alloc=Parameter(parameters[p], is_optim_states=True),
                    indices=(0, None),
                    time=parameters[p].mem
                    / md.bandwidthPrf
                    / md.gcd
                    * md.optimizer_states_factor,
                )
                prf_ops.append((t, k, op))
                Alive[p] = 1
        if k_ == bwd_i:
            assert 0 not in Alive.values()

    return ofl_ops, prf_ops, del_ops, init_alive


def group_activation_offload(md: ModelPULPOffload, j):
    ofl_ops = []
    prf_ops = []
    del_ops = []

    if (
        not md.activation_offload or not md.single_fwd
    ):  # we assume activation is based on single fwd
        return ofl_ops, prf_ops, del_ops

    fwd_i = min(md.sub_c2hcn[j])
    bwd_i = max(md.sub_c2hcn[j])
    hcn = md.hgraph.list_HCNs[fwd_i]
    sub_cluster = hcn.sub_cluster

    if md.sol(md.sumComp[fwd_i, fwd_i]):
        opt = -1
        for o in range(md.nSched[fwd_i]):
            if md.sol(md.Comp[fwd_i, fwd_i, o]):
                opt = o
                break

    assert opt > -1

    # phantoms = {}
    # for re_anode in md.list_list_sched[j][opt].phantoms:
    #     anode = sub_cluster.translate_representee_node(re_anode)
    #     phantoms[anode.name] = anode
    phantoms = {anode.name: anode for anode in md.phantoms[j, opt]}
    md.selected_phantoms[j] = [anode.main_target for anode in phantoms.values()]
    phantom_size = sum(anode.mem for anode in phantoms.values())

    Alive = {anode.name: 1 for anode in phantoms.values()}
    # Offloaded = {anode.name: False for anode in phantoms.values()}
    assert (bwd_i, bwd_i) in md.active_steps

    idx_f = md.active_steps.index((fwd_i, fwd_i))
    idx_b = md.active_steps.index((bwd_i, bwd_i))
    for t, k in md.active_steps[idx_f:] + md.active_steps[:idx_b]:
        t_, k_ = md.next_idx(t, k)

        # current_alive_size = sum(phantoms[n].mem * a for n, a in Alive.items())
        current_alive_size = sum(phantoms[n].mem * a for n, a in Alive.items() if a > 0)

        next_alive_size = phantom_size - md.OflPProg[t, k, j].value() * md.gcd

        # next_offloaded_size = min(gpu_optimize_size,
        #     round((md.OflOProg[(t_, k_, w)]).value() * parameter_size))
        # if parameter_size * (1-md.sumOptC[w]).value()<gpu_optimize_size:
        #     next_offloaded_size += gpu_optimize_size - parameter_size * (1-md.sumOptC[w]).value()

        # assert current_alive_size <= round(md.AliveW[(t, k, w)].value() * parameter_size)
        if current_alive_size > next_alive_size:
            # print(t,k, next_offloaded_size, current_offloaded_size)
            ofl_size = current_alive_size - next_alive_size
            candidates = {n: phantoms[n].mem * a for n, a in Alive.items() if a > 0}
            if not candidates:
                if ofl_size < 1024:
                    ofl_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(ofl_size)
            # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
            #     pass
            for n in select_paras:
                op = OffloadOp(
                    alloc=Activation(phantoms[n]),
                    indices=(0, None),
                    time=phantoms[n].mem / md.bandwidthOfl / md.gcd,
                )
                ofl_ops.append((t, k, op))
                del_ops.append((t, k, DeleteOp(Activation(phantoms[n]))))
                Alive[n] = 0

        # if current_alive_size > next_alive_size:
        #     if k_ ==bwd_i:continue
        #     del_size = current_alive_size - next_alive_size
        #     candidates = {}
        #     for p, o in Offloaded.items():
        #         if Alive[p]>0 and o>0:
        #             candidates[p] = min(o, Alive[p])*parameters[p].mem
        #     if not candidates:
        #         if del_size<1024:
        #             del_size = 0
        #         else:
        #             raise ValueError
        #     selector = knapsack(list(candidates.items()))
        #     select_paras = selector.select_size(del_size)
        #     # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
        #     # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
        #     #     pass
        #     for p in select_paras:
        #         del_ops.append((t, k, DeleteOp(Parameter(parameters[p],
        #                                                     is_optim_states=True)
        #                                         )))
        #         Alive[p] = 0
        if current_alive_size < next_alive_size or k_ == bwd_i:
            # if w == 15:print(md.active_steps[k_]==bwd_i)
            # prefetch should be smaller than solution
            prf_size = next_alive_size - current_alive_size
            candidates = {
                n: phantoms[n].mem * (1 - a) for n, a in Alive.items() if a < 1
            }
            if not candidates:
                if prf_size < 1024:
                    prf_size = 0
                else:
                    raise ValueError
            if k_ == bwd_i:
                select_paras = list(candidates.keys())
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
            else:
                selector = knapsack(list(candidates.items()))
                unselect_paras = selector.select_size(
                    sum(candidates.values()) - prf_size
                )

                select_paras = [p for p in candidates.keys() if p not in unselect_paras]
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
            #     pass
            for n in select_paras:
                alloc = Activation(phantoms[n])
                prf_ops.append((t, k, AllocateOp(alloc)))
                op = PrefetchOp(
                    alloc=alloc,
                    indices=(0, None),
                    time=alloc.mem / md.bandwidthPrf / md.gcd,
                )
                prf_ops.append((t, k, op))
                Alive[n] = 1
        if k_ == bwd_i:
            assert 0 not in Alive.values()

    return ofl_ops, prf_ops, del_ops

def schedule_alt(md: ModelPULP, hgraph=None, check_valid=False):
    """
    Given the solution from HILP, we want to translate the result
    to a OpSchedule that can be used in a higher level.
    """
    hgraph = hgraph if hgraph else md.hgraph

    init_op_list = []
    restore_op_list = []
    init_alive_status = {}
    loss_op = ComputeOp(md.hgraph.cluster.loss_cnode, disabled=True)

    op_list = []
    for pnode in md.hgraph.cluster.parameter_nodes:
        init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))
    for t in range(md.T):
        for k in md.krange(t):
            if t == md.loss_idx and k == md.loss_idx:
                op_list.append(loss_op)
            op_list += schedule_compute_alt(md, t, k, hgraph)

    # print("finish scheduling")
    for anode in md.hgraph.cluster.interfaces["input_data_anodes"]:
        init_alive_status[anode.name] = True  # anode share the name as alloc

    op_sched = OpSchedule(
        op_list,
        loss_idx=op_list.index(loss_op),
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=restore_op_list,
        with_parameters=isinstance(md, ModelPULPOffload),
        optimizer_states_factor=(
            md.optimize_metrics["optimizer_states_factor"]
            if hasattr(md, "optimize_metrics")
            else None
        ),
    )
    # check_valid = True
    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched


def schedule_compute_alt(md: ModelPULP, t, k, hgraph):
    op_list = []
    sol = md.sol

    j = md.hcn2sub_c[k]
    # if md.sumComp[t, k].value() == 1:
    if sol(md.sumComp[t, k]):
        hcn = hgraph.list_HCNs[k]
        opt = -1
        for o in range(md.nSched[k]):
            if sol(md.Comp[t, k, o]):
                opt = o
                break
        if opt > -1:
            h_obj = md.list_list_sched[j][opt]
            if hcn.is_fwd:
                sub_op_list = h_obj.op_list[: h_obj.loss_idx]
            else:
                sub_op_list = h_obj.op_list[h_obj.loss_idx + 1 :]

                # if md.sumAliveP[(j, t + 1)].value() == 0:
                # sub_op_list.append()
            sub_op_list = deepcopy(sub_op_list)

            if (
                not hcn.is_fwd
                # and md.sumAliveP[(j, t + 1)].value() > 0
                and sol(md.sumAliveP[t + 1, j])
            ):  # phantoms should be kept
                phantoms_to_keep = h_obj.phantoms
                # for op in sub_op_list[::-1]:
                #     if (
                #         op.is_del
                #         and not op.disabled
                #         and op.kn in phantoms_to_keep
                #     ):
                #         # Only the last del should be disabled
                #         op.disabled = True
                #         phantoms_to_keep.remove(op.kn)
            # translating sub_op_list
            if hcn.sub_cluster is not hcn.sub_cluster.representee_cluster:
                sub_op_list = translate(hcn.sub_cluster, sub_op_list)
        else:
            h_obj = hcn
            sub_op_list = deepcopy(h_obj.ff_op_list)

        anode_to_protect = []
        if not hcn.sub_cluster is None:
            for anode in hcn.sub_cluster.all_interfaces:
                i_ = [hdn.anode.name for hdn in md.hgraph.list_HANs].index(anode.name)
                if (k, i_) in md.delete_list and k in md.han_users_real[i_]:
                    eidx = md.delete_list.index((k, i_))
                    if not sol(md.delete[t, eidx]):
                        anode_to_protect.append(anode.name)

        for op in sub_op_list:
            if isinstance(op, DeleteOp) and op.target.anode.name in anode_to_protect:
                op.disabled = True
        op_list += sub_op_list

    for eidx, (k_, i) in enumerate(md.delete_list):
        # print(k_, i)
        # if k == k_ and md.delete[t, eidx].value()==1:
        if k == k_ and sol(md.delete[t, eidx]):
            han = hgraph.list_HANs[i]
            op_list.append(DeleteOp(Activation(han.anode)))
    return op_list


def schedule_alt_2(md: ModelPULP, check_valid=False):
    print("Generating non-checkpointing schedule...")
    fb_graph = md.hgraph.fb_graph
    op_list = []
    init_op_list = []

    for pnode in fb_graph.parameter_nodes:
        init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))


    pending_fwd_cnodes = list(fb_graph.dict_fwd_cnodes.values())
    pending_bwd_cnodes = list(fb_graph.dict_bwd_cnodes.values())
    pending_bwd_cnodes.reverse()

    data_anodes = list(fb_graph.dict_data_anodes.values())
    grad_anodes = list(fb_graph.dict_grad_anodes.values())
    phantoms_anodes = list(fb_graph.dict_phantoms_anodes.values())

    all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)

    # pending_fwd_cnodes = md.hgraph.list_HCNs

    deleted_anodes = set()

    executed_fwd_cnodes = set()
    executed_bwd_cnodes = set()
    loss_idx = -1

    while len(pending_fwd_cnodes) != 0:
        cnode = pending_fwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)
        executed_fwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        if cnode.name == md.hgraph.cluster.loss_cnode.name:
            loss_idx = len(executed_fwd_cnodes) + len(deleted_anodes) -1

        print(f"appending compute op for fwd cnode {cnode.name}")
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not (used_by_fwd or used_by_bwd)
            if no_longer_used and adep not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)

        for anode in all_anodes:
            if "source" in anode.name:
                continue
            # assert isinstance(anode, AllocationNode)
            used_by_fwd = False
            used_by_bwd = False
            for cuser in anode.users_real:
                # assert isinstance(cuser, ComputationNode)
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not (used_by_fwd or used_by_bwd)
            if no_longer_used and anode not in deleted_anodes:
                op_list.append(DeleteOp(Activation(anode)))
                deleted_anodes.add(anode)
        all_anodes = all_anodes.difference(deleted_anodes)

    assert loss_idx != -1

    while len(pending_bwd_cnodes) != 0:
        cnode = pending_bwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)
        executed_bwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        print(f"appending compute op for bwd cnode {cnode.name}")
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            # used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not used_by_bwd
            if no_longer_used and adep.name not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
        # for anode in all_anodes:
        #     if "source" in anode.name:
        #         continue
        #     # assert isinstance(anode, AllocationNode)
        #     used_by_bwd = False
        #     for cuser in anode.users_real:
        #         # assert isinstance(cuser, ComputationNode)
        #         if cuser in pending_bwd_cnodes:
        #             used_by_bwd = True
        #     no_longer_used = not used_by_bwd
        #     if no_longer_used and anode not in deleted_anodes:
        #         op_list.append(DeleteOp(Activation(anode)))
        #         deleted_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)

    init_alive_status = {}
    init_alive_status[fb_graph.source_data_anode.name] = True

    op_sched = OpSchedule(
        op_list,
        loss_idx=loss_idx,
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=[],
        with_parameters=False,
    )

    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched

node_to_rockmate_name_map = {
    "convolution": "convolution",
    "native_batch_norm": "batchnorm",
    "cudnn_batch_norm": "batchnorm",
    "relu": "relu",
    "maxpool": "maxpool",
    "add": "add",
    "linear": "linear",
    "flatten": "flatten",
    "dropout": "dropout",
    "_softmax": "softmax",
    "AvgPool2d": "avgpool",
    "AdaptiveAvgPool2d": "adaptiveavgpool",
    "View": "view",
    "Transpose": "transpose",
    "Upsample": "upsample",
    "cat": "cat",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "LeakyReLU": "leakyrelu",
    "clone": "clone",
    "nll_loss_forward": "loss",
    "native_layer_norm": "getitem",
    "baddbmm": "baddbmm",
    "where": "where",
    "bmm": "bmm",
    "mm": "mm",
    "addmm": "addmm",
    "native_dropout": "native_dropout",
    "gelu": "gelu",
    "mul": "mul",
    "neg": "neg",
    "div": "div",
    "embedding": "embedding",
    "maximum": "maximum",
    "minimum": "minimum",
    "avg_pool2d": "avg_pool2d",
    "max_pool2d_with_indices": "max_pool2d_with_indices",
}

# rockmate_to_node_name_map = {
#     rockmate_name: node_name
#     for node_name, rockmate_name in node_to_rockmate_name_map.items()
# }

for node_name, rockmate_name in list(node_to_rockmate_name_map.items()):
    node_to_rockmate_name_map[f"{node_name}_"] = rockmate_name

rockmate_to_segmentilp_map = {}
segmentilp_to_rockmate_map = {}

def extract_rockmate_name_from_node(node):
    node_name = node.name
    node_name = node_name.replace("FWD", "").replace("BWD", "")
    node_name = node_name.replace("[", "").replace("]", "")
    
    # from this point, some examples __129_avg_pool2d_1 --> avg_pool2d, __129_avg_pool2d --> avg_pool2d, __114_relu_15 --> relu
    
    # Remove the first two underscores if present
    if node_name.startswith("__"):
        node_name = node_name[2:]
        
    # split by underscore, if first component is number, remove it, if last component is number, remove it
    node_name_components = node_name.split("_")
    if node_name_components[0].isdigit():
        assert len(node_name_components) >=2, f"Node name {node_name} is not valid."
        node_name_components = node_name_components[1:]
    if node_name_components[-1].isdigit():
        assert len(node_name_components) >=2, f"Node name {node_name} is not valid."
        node_name_components = node_name_components[:-1]
    
    # now the remaining components form the rockmate name, join them back with underscore
    rockmate_name = "_".join(node_name_components)
    return rockmate_name

def get_rockmate_name(node_name):
    special_case_1 = ["native_batch_norm", "maxpool", "native_layer_norm", "max_pool2d_with_indices", "cudnn_batch_norm"]
    special_case_2 = ["native_dropout"]
    assert node_name in node_to_rockmate_name_map or node_name in special_case_1 or node_name in special_case_2, f"Node name {node_name} not found in mapping."
    if node_name in special_case_1:
        return "getitem"
    if node_name in special_case_2:
        return "clone"
    return node_to_rockmate_name_map[node_name]

# def process_checkpointed_list(fwd_node_list, cp_solver_handler: CheckpointSolverHandler):
#     for i, fwd_node in enumerate(fwd_node_list):
#         print(f"Processing node {i} {fwd_node.name} has {fwd_node.get_all_standard_deps()} deps: {[cnode.name for cnode in fwd_node.get_all_standard_deps()]}")

#     checkpointed_nodes = cp_solver_handler.checkpointed_nodes
#     root_nodes = cp_solver_handler.root_nodes
#     output_nodes = cp_solver_handler.output_nodes

#     # for segment in checkpointed_segments:
#     #     print(f"Checkpointed segment: {segment}")
#     #     for node in segment.nodes:
#     #         print(f"Checkpointed node: {node.name}")
#     for checkpointed_node in checkpointed_nodes:
#         print(f"checkpointed node: {checkpointed_node.name}")

#     checkpointed_nodes_names = [n.name.split("-")[0] for n in checkpointed_nodes]
#     checkpointed_nodes_idx = [int(n.name.split("-")[1]) for n in checkpointed_nodes]

#     root_nodes_names = [n.name.split("-")[0] for n in root_nodes]
#     root_nodes_idx = [int(n.name.split("-")[1]) for n in root_nodes]

#     output_nodes_names = [n.name.split("-")[0] for n in output_nodes]
#     output_nodes_idx = [int(n.name.split("-")[1]) for n in output_nodes]
#     print(f"Checkpointed nodes idx: {checkpointed_nodes_idx}")
#     print(f"Root nodes idx: {root_nodes_idx}")
#     print(f"Output nodes idx: {output_nodes_idx}")
#     # for cp_node_name in checkpointed_nodes_names:

#     for i, fwd_node in enumerate(fwd_node_list):
#         if i in checkpointed_nodes_idx:
#             # assert i not in root_nodes_idx, f"Node {fwd_node.name} cannot be both checkpointed and root."
#             print(f"Marking node {i} {fwd_node.name} as checkpointed")
#             fwd_node.is_checkpointed = True
        
#         if i in root_nodes_idx:
#             assert i in checkpointed_nodes_idx, f"Root Node {fwd_node.name} must be checkpointed."
#             print(f"Marking node {i} {fwd_node.name} as root (save its inputs)")
#             fwd_node.is_segment_head = True

#         # output nodes must be checkpointed, not all checkpointed nodes are output nodes
#         if i in output_nodes_idx:
#             assert i in checkpointed_nodes_idx, f"Output node {fwd_node.name} must be checkpointed."
#             print(f"Marking node {i} {fwd_node.name} as output node (segment tail)")
#             fwd_node.is_segment_tail = True
    
#     # print(f"exit for now")
#     # exit(1)

def is_matching_node(rockmate_node, segmentilp_node):
    segmentilp_node_name = segmentilp_node.name.split("-")[0]
    rockmate_node_name = extract_rockmate_name_from_node(rockmate_node)
    
    assert len(rockmate_node_name) > 0, f"Rockmate node name {rockmate_node.name} is not valid."
    assert len(segmentilp_node_name) > 0, f"SegmentILP node name {segmentilp_node.name} is not valid."
    
    if segmentilp_node_name.lower() in rockmate_node_name or get_rockmate_name(segmentilp_node_name.lower()) in rockmate_node_name:
        return True
    return False
    
def process_checkpointed_list(fwd_node_list, cp_solver_handler: CheckpointSolverHandler):
    checkpointed_nodes = cp_solver_handler.checkpointed_nodes
    root_nodes = cp_solver_handler.root_nodes
    output_nodes = cp_solver_handler.output_nodes
    
    for checkpointed_node in checkpointed_nodes:
        print(f"checkpointed node: {checkpointed_node.name}")
        
        
    for i, fwd_node in enumerate(fwd_node_list):
        if "clone" in fwd_node.name or "zeros" in fwd_node.name:
            # Skipping for now
            continue
        segmentilp_node = rockmate_to_segmentilp_map.get(fwd_node, None)
        if segmentilp_node in checkpointed_nodes:
            print(f"Marking node {i} {fwd_node.name} as checkpointed")
            fwd_node.is_checkpointed = True
        
        if segmentilp_node in root_nodes:
            assert segmentilp_node in checkpointed_nodes, f"Root Node {fwd_node.name} must be checkpointed."
            print(f"Marking node {i} {fwd_node.name} as root (save its inputs)")
            fwd_node.is_segment_head = True
            
        # output nodes must be checkpointed, not all checkpointed nodes are output nodes
        if segmentilp_node in output_nodes:
            assert segmentilp_node in checkpointed_nodes, f"Output node {fwd_node.name} must be checkpointed."
            print(f"Marking node {i} {fwd_node.name} as output node (segment tail)")
            fwd_node.is_segment_tail = True
            
    def process_clone_and_zeros_nodes(node):
        nonlocal fwd_node_list
        assert "clone" in node.name or "zeros" in node.name, f"Node {node.name} is not clone or zeros."
        idx = fwd_node_list.index(node)
        # these nodes are checkpointed if the subsequent cnode is checkpointed
        if idx+1 < len(fwd_node_list):
            next_node = fwd_node_list[idx+1]
            if "clone" in next_node.name or "zeros" in next_node.name:
                process_clone_and_zeros_nodes(next_node)
            if next_node.is_checkpointed:
                print(f"Marking node {idx} {node.name} as checkpointed because next node {next_node.name} is checkpointed")
                node.is_checkpointed = True
                # if the next node is a root node, then this node should also be a root node
                if next_node.is_segment_head:
                    print(f"Marking node {idx} {node.name} as root because next node {next_node.name} is root")
                    node.is_segment_head = True
        
    # process clone and zeros nodes
    for i, fwd_node in enumerate(fwd_node_list):
        if "clone" in fwd_node.name or "zeros" in fwd_node.name:
            process_clone_and_zeros_nodes(fwd_node)
        
    
    # base on the mapping the list of checkpointed nodes, apply checkpointing status to rockmate nodes
    
    
    
        
def remove_first_occurrence_from_list(original_list, items_to_remove_set):
    """
    Removes only the first occurrence of items present in a set from a list,
    preserving other duplicates.

    Args:
        original_list (list): The list from which to remove items.
        items_to_remove_set (set): The set of items to remove.

    Returns:
        list: A new list with the first occurrences of specified items removed.
    """
    new_list = []
    # Create a mutable copy of the set to track removed items
    temp_remove_set = items_to_remove_set.copy() 

    for item in original_list:
        if item in temp_remove_set:
            temp_remove_set.remove(item)  # Remove only the first occurrence
        else:
            new_list.append(item)
    return new_list

def schedule_alt_3(md: ModelPULP, cp_solver_handler: CheckpointSolverHandler, check_valid=False):
    print("Generating checkpointing schedule...")
    fb_graph = md.hgraph.fb_graph

    assert cp_solver_handler.cp_solver.solved, "Checkpoint solver has not solved the problem yet."

    # Call my solver
    op_list = []
    init_op_list = []

    for pnode in fb_graph.parameter_nodes:
        init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))


    pending_fwd_cnodes = list(fb_graph.dict_fwd_cnodes.values())
    pending_bwd_cnodes = list(fb_graph.dict_bwd_cnodes.values())
    pending_bwd_cnodes.reverse()

    pending_recomputation_cnodes = set()

    segment_head = set()

    process_checkpointed_list(pending_fwd_cnodes, cp_solver_handler)

    data_anodes = list(fb_graph.dict_data_anodes.values())
    grad_anodes = list(fb_graph.dict_grad_anodes.values())
    phantoms_anodes = list(fb_graph.dict_phantoms_anodes.values())

    # all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)
    all_anodes = data_anodes + grad_anodes + phantoms_anodes

    # pending_fwd_cnodes = md.hgraph.list_HCNs

    deleted_anodes = set()

    executed_fwd_cnodes = set()
    executed_bwd_cnodes = set()
    loss_idx = -1

    while len(pending_fwd_cnodes) != 0:
        cnode = pending_fwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)
        executed_fwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        if cnode.is_segment_head or cnode.is_checkpointed:
            pending_recomputation_cnodes.add(cnode)

        # record loss idx
        if cnode.name == md.hgraph.cluster.loss_cnode.name:
            loss_idx = len(executed_fwd_cnodes) + len(deleted_anodes) -1

        # print(f"appending compute op for fwd cnode {cnode.name}")
        # delete input tho cnode?
        if cnode.is_segment_head:
            print(f"cnode {cnode.name} is segment head, skip deleting its inputs")
            # note, might cause a problem for add operator where one of its input is from another segment
            segment_head.add(cnode)
            
            for anode in all_anodes:
                if "source" in anode.name:
                    continue
                # assert isinstance(anode, AllocationNode)
                used_by_fwd = False
                used_by_bwd = False
                used_by_recomputation = False
                for cuser in anode.users_real:
                    # assert isinstance(cuser, ComputationNode)
                    if cuser in pending_fwd_cnodes:
                        used_by_fwd = True
                    if cuser in pending_bwd_cnodes:
                        used_by_bwd = True
                    if cuser in segment_head:
                        used_by_recomputation = True
                no_longer_used = not (used_by_fwd or used_by_bwd or used_by_recomputation)
                if no_longer_used and anode not in deleted_anodes:
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
            # all_anodes = all_anodes.difference(deleted_anodes)
            all_anodes = remove_first_occurrence_from_list(all_anodes, deleted_anodes)
            continue
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            dep_no_longer_used = not (used_by_fwd or used_by_bwd)
            dep_is_checkpointed = not used_by_fwd and used_by_bwd and adep.is_checkpointed

            if dep_is_checkpointed: print(f"adep {adep.name} is_checkpointed: {adep.is_checkpointed}")

            if (dep_no_longer_used or dep_is_checkpointed) and adep not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)

        for anode in all_anodes:
            if "source" in anode.name:
                continue
            # assert isinstance(anode, AllocationNode)
            used_by_fwd = False
            used_by_bwd = False
            used_by_recomputation = False
            for cuser in anode.users_real:
                # assert isinstance(cuser, ComputationNode)
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
                if cuser in segment_head:
                    used_by_recomputation = True
            no_longer_used = not (used_by_fwd or used_by_bwd or used_by_recomputation)
            if no_longer_used and anode not in deleted_anodes:
                op_list.append(DeleteOp(Activation(anode)))
                deleted_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)
        all_anodes = remove_first_occurrence_from_list(all_anodes, deleted_anodes)

    assert loss_idx != -1
    print(f"Currently live anodes: ")
    for anode in all_anodes:
        print(f"\tAnode {anode.name} is used by {[cnode.name for cnode in anode.users_real]}")

    # BWD pass

    rematerialized_anodes = set()
    redeleted_anodes = set()

    reexecuted_fwd_nodes = set()

    all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)

    while len(pending_bwd_cnodes) != 0:
        cnode = pending_bwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)

        def compute_cnode_input(cnode):
            nonlocal all_anodes
            assert isinstance(cnode, ComputationNode)
            # for anode in cnode.deps_real:
            #     if anode in deleted_anodes:
            #         print(f"Recomputing input anode {anode.name} for bwd cnode {cnode.name}")
            #         if cnode.is_fwd: pending_recomputation_cnodes.add(cnode)
            #         for cdep in anode.deps:
            #             assert isinstance(cdep, ComputationNode)
            #             if cdep.is_fwd:
            #                 compute_cnode_input(cdep)

            #         rematerialized_anodes.add(anode)

            # if cnode is bwd then it is not checkpointed, all of required activations exists, execute it then delete the inputs if no longer used
            # Check its fwd counterpart (one of its deps), execute such fwd dep if it is checkpointed
            # Ignore any deps that is not its fwd counter part
            # if cnode is fwd then it is checkpointed, check its deps and execute it

            if not cnode.is_fwd:    # is bwd
                # Check its fwd counterpart
                for cdep in cnode.get_all_standard_deps():
                    if cdep.main_target == cnode.main_target \
                        and cdep.is_fwd \
                        and cdep.is_checkpointed \
                        and cdep not in reexecuted_fwd_nodes:
                            print(f"Recomputing fwd cnode {cdep.name} for bwd cnode {cnode.name}")
                            # pending_recomputation_cnodes.add(cdep)
                            compute_cnode_input(cdep)

                # execute the bwd cnode itself
                executed_bwd_cnodes.add(cnode)
                op_list.append(ComputeOp(cnode))

                # delete its inputs if no longer used
                for adep in cnode.get_all_standard_deps():
                    # auser is anode
                    # used_by_fwd = False
                    used_by_bwd = False
                    for cuser in adep.get_all_standard_users():
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not used_by_bwd
                    if no_longer_used and adep.name not in deleted_anodes:
                        for anode in adep.users:
                            assert isinstance(anode, AllocationNode)
                            op_list.append(DeleteOp(Activation(anode)))
                            deleted_anodes.add(anode)

                # for anode in all_anodes:
                #     if "source" in anode.name:
                #         continue
                #     # assert isinstance(anode, AllocationNode)
                #     used_by_bwd = False
                #     used_by_recomputation = False
                #     for cuser in anode.users_real:
                #         # assert isinstance(cuser, ComputationNode)
                #         if cuser in pending_bwd_cnodes:
                #             used_by_bwd = True
                #         if cuser in segment_head:
                #             used_by_recomputation = True
                #     no_longer_used = not (used_by_bwd or used_by_recomputation)
                #     if no_longer_used and anode not in deleted_anodes:
                #         op_list.append(DeleteOp(Activation(anode)))
                #         deleted_anodes.add(anode)
                # all_anodes = all_anodes.difference(deleted_anodes)


            if cnode.is_fwd:
                # check its deps
                print(f"Cnode {cnode.name} has {len(cnode.get_all_standard_deps())} deps: {[d.name for d in cnode.get_all_standard_deps()]}")
                if not cnode.is_segment_head:
                    # reveresed_deps = list(cnode.get_all_standard_deps())
                    # reveresed_deps.reverse()
                    # print(f"Processing deps for cnode {cnode.name}: {[d.name for d in reveresed_deps]}")
                    for cdep in cnode.get_all_standard_deps():
                        assert isinstance(cdep, ComputationNode)
                        cross_segment_dep = cnode.is_fwd and cdep.is_segment_tail
                        if cdep.is_fwd and (cdep.is_checkpointed or cdep.is_segment_head) and cdep not in reexecuted_fwd_nodes and not cross_segment_dep:
                            print(f"Recomputing fwd cnode {cdep.name} for cnode {cnode.name}")
                            # pending_recomputation_cnodes.add(cdep)
                            compute_cnode_input(cdep)
                
                # Execute recomputation fwd
                executed_bwd_cnodes.add(cnode)
                if cnode in pending_recomputation_cnodes: pending_recomputation_cnodes.remove(cnode)
                op_list.append(ComputeOp(cnode))
                reexecuted_fwd_nodes.add(cnode)

                for adep in cnode.get_all_standard_deps():
                    # adep is cnode
                    used_by_fwd = False
                    used_by_bwd = False
                    for cuser in adep.get_all_standard_users():
                        if cuser in pending_recomputation_cnodes:
                            used_by_fwd = True
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not (used_by_fwd or used_by_bwd)
                    if no_longer_used and adep not in redeleted_anodes:
                        for anode in adep.users:
                            assert isinstance(anode, AllocationNode)
                            op_list.append(DeleteOp(Activation(anode)))
                            redeleted_anodes.add(anode)


                for anode in all_anodes:
                    if "source" in anode.name:
                        continue
                    # assert isinstance(anode, AllocationNode)
                    used_by_fwd = False
                    used_by_bwd = False
                    for cuser in anode.users_real:
                        # assert isinstance(cuser, ComputationNode)
                        if cuser in pending_recomputation_cnodes:
                            used_by_fwd = True
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not (used_by_fwd or used_by_bwd)
                    # if no_longer_used and anode not in redeleted_anodes:
                    if no_longer_used:
                        op_list.append(DeleteOp(Activation(anode)))
                        redeleted_anodes.add(anode)
                all_anodes = all_anodes.difference(redeleted_anodes)

            # for anode in rematerialized_anodes:
            #     used_by_recomputation_fwd = False
            #     used_by_bwd = False
            #     for cuser in anode.users_real:
            #         assert isinstance(cuser, ComputationNode)
            #         if cuser in pending_recomputation_cnodes:
            #             used_by_recomputation_fwd = True
            #         if cuser in pending_bwd_cnodes:
            #             used_by_bwd = True
            #     no_longer_used = not (used_by_recomputation_fwd or used_by_bwd)
            #     if no_longer_used and anode not in redeleted_anodes:
            #         # print(f"Deleting rematerialized anode {anode.name} after bwd cnode {cnode.name}")
            #         op_list.append(DeleteOp(Activation(anode)))
            #         # rematerialized_anodes.remove(anode)
            #         redeleted_anodes.add(anode)
            # rematerialized_anodes.difference_update(redeleted_anodes)
        # executed_bwd_cnodes.add(cnode)
        # op_list.append(ComputeOp(cnode))

        compute_cnode_input(cnode)

        print(f"appending compute op for bwd cnode {cnode.name}")

        # repending_fwd_cnodes = []
        # def recompute_cnode(cnode):
        #     assert isinstance(cnode, ComputationNode)
        #     assert cnode.is_fwd
        #     for adep in cnode.get_all_standard_deps():
        #         if adep.is_checkpointed:
        #             recompute_cnode(adep)
            
        #     executed_bwd_cnodes.add(cnode)
        #     op_list.append(ComputeOp(cnode))
        #     for adep in cnode.get_all_standard_deps():
        #         # auser is anode
        #         used_by_fwd = False
        #         used_by_bwd = False
        #         for cuser in adep.get_all_standard_users():
        #             if cuser in pending_fwd_cnodes:
        #                 used_by_fwd = True
        #             if cuser in pending_bwd_cnodes:
        #                 used_by_bwd = False
        #         no_longer_used = not (used_by_fwd or used_by_bwd)
        #         # op_is_checkpointed = not used_by_fwd and used_by_bwd and adep.is_checkpoined
        #         if no_longer_used and adep not in redeleted_anodes:
        #             for anode in adep.users:
        #                 assert isinstance(anode, AllocationNode)
        #                 op_list.append(DeleteOp(Activation(anode)))
        #                 redeleted_anodes.add(anode)
        
        # for adep in cnode.get_all_standard_deps():
        #     if adep.is_checkpointed:
        #         adep_users = adep.users
        #         tensors_were_deleted = all(adep_user in deleted_anodes for adep_user in adep_users)
        #         assert tensors_were_deleted, "Tensors for checkpointed adep should have been deleted before recomputation."
        #         recompute_cnode(adep)

        # Delete input tensors of cnode if no longer used
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            # used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not used_by_bwd
            if no_longer_used and adep.name not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
        # for anode in all_anodes:
        #     if "source" in anode.name:
        #         continue
        #     # assert isinstance(anode, AllocationNode)
        #     used_by_bwd = False
        #     for cuser in anode.users_real:
        #         # assert isinstance(cuser, ComputationNode)
        #         if cuser in pending_bwd_cnodes:
        #             used_by_bwd = True
        #     no_longer_used = not used_by_bwd
        #     if no_longer_used and anode not in deleted_anodes:
        #         op_list.append(DeleteOp(Activation(anode)))
        #         deleted_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)

    init_alive_status = {}
    init_alive_status[fb_graph.source_data_anode.name] = True

    op_sched = OpSchedule(
        op_list,
        loss_idx=loss_idx,
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=[],
        with_parameters=False,
    )

    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched

def  schedule_alt_4(md: ModelPULP, cp_solver_handler: CheckpointSolverHandler, check_valid=False):
    print("Generating checkpointing schedule...")
    fb_graph = md.hgraph.fb_graph

    assert cp_solver_handler.cp_solver.solved, "Checkpoint solver has not solved the problem yet."

    # Call my solver
    op_list = []
    init_op_list = []

    for pnode in fb_graph.parameter_nodes:
        init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))


    pending_fwd_cnodes = list(fb_graph.dict_fwd_cnodes.values())
    pending_bwd_cnodes = list(fb_graph.dict_bwd_cnodes.values())
    pending_bwd_cnodes.reverse()

    process_checkpointed_list(pending_fwd_cnodes, cp_solver_handler)

    data_anodes = list(fb_graph.dict_data_anodes.values())
    grad_anodes = list(fb_graph.dict_grad_anodes.values())
    phantoms_anodes = list(fb_graph.dict_phantoms_anodes.values())

    all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)

    # pending_fwd_cnodes = md.hgraph.list_HCNs

    deleted_anodes = set()

    executed_fwd_cnodes = set()
    executed_bwd_cnodes = set()
    loss_idx = -1

    while len(pending_fwd_cnodes) != 0:
        cnode = pending_fwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)
        executed_fwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        if cnode.name == md.hgraph.cluster.loss_cnode.name:
            loss_idx = len(executed_fwd_cnodes) + len(deleted_anodes) -1

        print(f"appending compute op for fwd cnode {cnode.name}")
        # delete input tho cnode?
        if cnode.is_segment_head:
            print(f"cnode {cnode.name} is segment head, skip deleting its inputs")
            continue
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not (used_by_fwd or used_by_bwd)
            op_is_checkpointed = not used_by_fwd and used_by_bwd and (adep.is_checkpointed or adep.is_segment_head)

            if op_is_checkpointed: print(f"adep {adep.name} is_checkpointed: {adep.is_checkpointed}")

            if (no_longer_used or op_is_checkpointed) and adep not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
    assert loss_idx != -1

    while len(pending_bwd_cnodes) != 0:
        cnode = pending_bwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)
        executed_bwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        print(f"appending compute op for bwd cnode {cnode.name}")
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            # used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not used_by_bwd
            if no_longer_used and adep.name not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
        # for anode in all_anodes:
        #     if "source" in anode.name:
        #         continue
        #     # assert isinstance(anode, AllocationNode)
        #     used_by_bwd = False
        #     for cuser in anode.users_real:
        #         # assert isinstance(cuser, ComputationNode)
        #         if cuser in pending_bwd_cnodes:
        #             used_by_bwd = True
        #     no_longer_used = not used_by_bwd
        #     if no_longer_used and anode not in deleted_anodes:
        #         op_list.append(DeleteOp(Activation(anode)))
        #         deleted_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)

    init_alive_status = {}
    init_alive_status[fb_graph.source_data_anode.name] = True

    op_sched = OpSchedule(
        op_list,
        loss_idx=loss_idx,
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=[],
        with_parameters=False,
    )

    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched

def find_segmentilp_node_by_name(segmentilp_nodes, from_index: int, rockmate_node):
    # If could not find from from_index to end, return -1
    # If found, return the index
    rockmate_node_name = extract_rockmate_name_from_node(rockmate_node)
    for idx in range(from_index, len(segmentilp_nodes)):
        segmentilp_node = segmentilp_nodes[idx]
        segmentilp_node_name = segmentilp_node.name.split("-")[0]
        if segmentilp_node_name.lower() in rockmate_node_name or get_rockmate_name(segmentilp_node_name.lower()) in rockmate_node_name:
            return idx
    return -1
        

def node_mapping(md: ModelPULP, cp_solver_handler: CheckpointSolverHandler):
    fb_graph = md.hgraph.fb_graph
    fwd_node_list = list(fb_graph.dict_fwd_cnodes.values())
    
    rockmate_node_stack = fwd_node_list.copy()
    segmentilp_node_stack = cp_solver_handler.nodes.copy()
    
    print("SegmentILP nodes before filtering:")
    for node in segmentilp_node_stack:
        print(f"{node.name}")
    
    # iterate through segmentilp nodes and ignore the following nodes
    ignore_names = ["view", "_unsafe_view", "split", "permute", "transpose", "expand", "slice", "t", "_log_softmax"]
    ignore_nodes = []
    
    for segmentilp_node in segmentilp_node_stack:
        node_name = segmentilp_node.name.split("-")[0]
        if node_name in ignore_names:
            # print(f"Ignoring node {segmentilp_node.name} in mapping")
            ignore_nodes.append(segmentilp_node)
            
    print("Ignoreing the following nodes from SegmentILP node stack:")
    for node in ignore_nodes:
        print(f"{node.name}")
    
    segmentilp_node_stack = remove_first_occurrence_from_list(segmentilp_node_stack, set(ignore_nodes))
    
    
    # print rockmate nodes
    print("Rockmate nodes:")
    for rockmate_node in rockmate_node_stack:
        print(f"{rockmate_node.name}")
        
    # print segmentilp nodes
    print("SegmentILP nodes:")
    for segmentilp_node in segmentilp_node_stack:
        print(f"{segmentilp_node.name}")
        
    # Some clone operators in rockmate node stack does not have segmentilp equivalent, we remove them and decide if they are recomputed later
    to_remove_rockmate_nodes = []
    for rockmate_node in rockmate_node_stack:
        node_name = extract_rockmate_name_from_node(rockmate_node)
        nodes_to_avoid = ["clone", "zeros", "new_zeros", "new_empty", "empty", "bernoulli"]
        if node_name in nodes_to_avoid:
            # print(f"Removing clone node {rockmate_node.name} from rockmate node stack for mapping")
            to_remove_rockmate_nodes.append(rockmate_node)
    
    rockmate_node_stack = remove_first_occurrence_from_list(rockmate_node_stack, set(to_remove_rockmate_nodes))
    
    print(f"rockmate_node_stack size: {len(rockmate_node_stack)}")
    print(f"segmentilp_node_stack size: {len(segmentilp_node_stack)}")
    # assert len(rockmate_node_stack) <= len(segmentilp_node_stack), "Rockmate node stack should not be larger than SegmentILP node stack."
        
    # temporary end the program here to inspect the nodes
    # exit(1)
    # establish mapping from rockmate nodes to segmentilp nodes and vice versa
    rockmate_stack_idx = 0
    segmentilp_stack_idx = 0
    while rockmate_stack_idx < len(rockmate_node_stack):
        rockmate_node = rockmate_node_stack[rockmate_stack_idx]
        
        new_segmentilp_stack_idx = find_segmentilp_node_by_name(segmentilp_node_stack, segmentilp_stack_idx, rockmate_node)
        
        if new_segmentilp_stack_idx == -1:
            # Could not find mapping, skip this rockmate node
            print(f"Could not find mapping for rockmate node {rockmate_node.name}, skipping it.")
            rockmate_stack_idx += 1
            continue
        else:
            segmentilp_stack_idx = new_segmentilp_stack_idx
            # establish mapping
            segmentilp_node = segmentilp_node_stack[segmentilp_stack_idx]
            # segmentilp_node_name = segmentilp_node.name.split("-")[0]
            rockmate_to_segmentilp_map[rockmate_node] = segmentilp_node
            segmentilp_to_rockmate_map[segmentilp_node] = rockmate_node
            print(f"Mapping found: {rockmate_node.name} -> {segmentilp_node.name}")
            rockmate_stack_idx += 1
            segmentilp_stack_idx += 1
            
    # print the mapping
    print("Rockmate to SegmentILP node mapping:")
    for rockmate_node, segmentilp_node in rockmate_to_segmentilp_map.items():
        print(f"{rockmate_node.name} -> {segmentilp_node.name}")
        
    return

def schedule_alt_5(md: ModelPULP, cp_solver_handler: CheckpointSolverHandler, check_valid=False):
    print("Generating checkpointing schedule...")
    fb_graph = md.hgraph.fb_graph

    if isinstance(cp_solver_handler, CheckpointSolverHandler):
        assert cp_solver_handler.cp_solver.solved, "Checkpoint solver has not solved the problem yet."

    # Call my solver
    op_list = []
    init_op_list = []

    for pnode in fb_graph.parameter_nodes:
        init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))


    pending_fwd_cnodes = list(fb_graph.dict_fwd_cnodes.values())
    pending_bwd_cnodes = list(fb_graph.dict_bwd_cnodes.values())
    pending_bwd_cnodes.reverse()

    pending_recomputation_cnodes = set()

    segment_head = set()

    process_checkpointed_list(pending_fwd_cnodes, cp_solver_handler)

    data_anodes = list(fb_graph.dict_data_anodes.values())
    grad_anodes = list(fb_graph.dict_grad_anodes.values())
    # phantoms_anodes = list(fb_graph.dict_phantoms_anodes.values())

    # all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)
    # all_anodes = data_anodes + grad_anodes + phantoms_anodes
    all_anodes = data_anodes + grad_anodes

    # pending_fwd_cnodes = md.hgraph.list_HCNs

    deleted_anodes = set()

    executed_fwd_cnodes = set()
    executed_bwd_cnodes = set()
    loss_idx = -1

    while len(pending_fwd_cnodes) != 0:
        cnode = pending_fwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)
        executed_fwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        if cnode.is_segment_head or cnode.is_checkpointed:
            pending_recomputation_cnodes.add(cnode)

        # record loss idx
        if cnode.name == md.hgraph.cluster.loss_cnode.name:
            loss_idx = len(executed_fwd_cnodes) + len(deleted_anodes) -1

        # print(f"appending compute op for fwd cnode {cnode.name}")
        # delete input tho cnode?
        if cnode.is_segment_head:
            print(f"cnode {cnode.name} is segment head, skip deleting its inputs")
            # note, might cause a problem for add operator where one of its input is from another segment
            segment_head.add(cnode)
            
            for anode in all_anodes:
                if "source" in anode.name:
                    continue
                # assert isinstance(anode, AllocationNode)
                used_by_fwd = False
                used_by_bwd = False
                used_by_recomputation = False
                for cuser in anode.users_real:
                    # assert isinstance(cuser, ComputationNode)
                    if cuser in pending_fwd_cnodes:
                        used_by_fwd = True
                    if cuser in pending_bwd_cnodes:
                        used_by_bwd = True
                    if cuser in segment_head:
                        used_by_recomputation = True
                no_longer_used = not (used_by_fwd or used_by_bwd or used_by_recomputation)
                if no_longer_used and anode not in deleted_anodes:
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
            # all_anodes = all_anodes.difference(deleted_anodes)
            all_anodes = remove_first_occurrence_from_list(all_anodes, deleted_anodes)
            continue
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            dep_no_longer_used = not (used_by_fwd or used_by_bwd)
            dep_is_checkpointed = not used_by_fwd and used_by_bwd and adep.is_checkpointed

            if dep_is_checkpointed: print(f"adep {adep.name} is_checkpointed: {adep.is_checkpointed}")

            if (dep_no_longer_used or dep_is_checkpointed) and adep not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)

        for anode in all_anodes:
            if "source" in anode.name:
                continue
            # assert isinstance(anode, AllocationNode)
            used_by_fwd = False
            used_by_bwd = False
            used_by_recomputation = False
            for cuser in anode.users_real:
                # assert isinstance(cuser, ComputationNode)
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
                if cuser in segment_head:
                    used_by_recomputation = True
            no_longer_used = not (used_by_fwd or used_by_bwd or used_by_recomputation)
            if no_longer_used and anode not in deleted_anodes:
                op_list.append(DeleteOp(Activation(anode)))
                deleted_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)
        all_anodes = remove_first_occurrence_from_list(all_anodes, deleted_anodes)

    assert loss_idx != -1
    print(f"Currently live anodes: ")
    for anode in all_anodes:
        print(f"\tAnode {anode.name} is used by {[cnode.name for cnode in anode.users_real]}")

    # BWD pass

    rematerialized_anodes = set()
    deleted_in_fwd_anodes = deleted_anodes.copy()
    redeleted_anodes = set()
    bwd_to_be_redeleted_anodes = set()

    reexecuted_fwd_nodes = set()

    # all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)
    all_anodes = set(data_anodes + grad_anodes)

    while len(pending_bwd_cnodes) != 0:
        cnode = pending_bwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)

        def compute_cnode_input(cnode):
            nonlocal all_anodes
            nonlocal deleted_anodes
            assert isinstance(cnode, ComputationNode)
            # for anode in cnode.deps_real:
            #     if anode in deleted_anodes:
            #         print(f"Recomputing input anode {anode.name} for bwd cnode {cnode.name}")
            #         if cnode.is_fwd: pending_recomputation_cnodes.add(cnode)
            #         for cdep in anode.deps:
            #             assert isinstance(cdep, ComputationNode)
            #             if cdep.is_fwd:
            #                 compute_cnode_input(cdep)

            #         rematerialized_anodes.add(anode)

            # if cnode is bwd then it is not checkpointed, all of required activations exists, execute it then delete the inputs if no longer used
            # Check its fwd counterpart (one of its deps), execute such fwd dep if it is checkpointed
            # Ignore any deps that is not its fwd counter part
            # if cnode is fwd then it is checkpointed, check its deps and execute it

            if not cnode.is_fwd:    # is bwd
                # Check its fwd counterpart
                for cdep in cnode.get_all_standard_deps():
                    if cdep.main_target == cnode.main_target \
                        and cdep.is_fwd \
                        and cdep.is_checkpointed \
                        and cdep not in reexecuted_fwd_nodes:
                            print(f"Recomputing fwd cnode {cdep.name} for bwd cnode {cnode.name}")
                            # pending_recomputation_cnodes.add(cdep)
                            compute_cnode_input(cdep)

                # execute the bwd cnode itself
                executed_bwd_cnodes.add(cnode)
                op_list.append(ComputeOp(cnode))

                # delete its inputs if no longer used
                for adep in cnode.get_all_standard_deps():
                    # auser is anode
                    # used_by_fwd = False
                    used_by_bwd = False
                    for cuser in adep.get_all_standard_users():
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not used_by_bwd
                    if no_longer_used and adep.name not in deleted_anodes:
                        for anode in adep.users:
                            if anode not in deleted_anodes or anode in bwd_to_be_redeleted_anodes:
                                assert isinstance(anode, AllocationNode)
                                op_list.append(DeleteOp(Activation(anode)))
                                deleted_anodes.add(anode)
                                bwd_to_be_redeleted_anodes.discard(anode)

                # for anode in all_anodes:
                #     if "source" in anode.name:
                #         continue
                #     # assert isinstance(anode, AllocationNode)
                #     used_by_bwd = False
                #     used_by_recomputation = False
                #     for cuser in anode.users_real:
                #         # assert isinstance(cuser, ComputationNode)
                #         if cuser in pending_bwd_cnodes:
                #             used_by_bwd = True
                #         if cuser in segment_head:
                #             used_by_recomputation = True
                #     no_longer_used = not (used_by_bwd or used_by_recomputation)
                #     if no_longer_used and anode not in deleted_anodes:
                #         op_list.append(DeleteOp(Activation(anode)))
                #         deleted_anodes.add(anode)
                # all_anodes = all_anodes.difference(deleted_anodes)


            if cnode.is_fwd:
                # check its deps
                print(f"Cnode {cnode.name} has {len(cnode.get_all_standard_deps())} deps: {[d.name for d in cnode.get_all_standard_deps()]}")
                if not cnode.is_segment_head:
                    # reveresed_deps = list(cnode.get_all_standard_deps())
                    # reveresed_deps.reverse()
                    # print(f"Processing deps for cnode {cnode.name}: {[d.name for d in reveresed_deps]}")
                    for cdep in cnode.get_all_standard_deps():
                        assert isinstance(cdep, ComputationNode)
                        cross_segment_dep = cnode.is_fwd and cdep.is_segment_tail
                        if cdep.is_fwd and (cdep.is_checkpointed or cdep.is_segment_head) and cdep not in reexecuted_fwd_nodes and not cross_segment_dep:
                            print(f"Recomputing fwd cnode {cdep.name} for cnode {cnode.name}")
                            # pending_recomputation_cnodes.add(cdep)
                            compute_cnode_input(cdep)
                
                # Execute recomputation fwd
                executed_bwd_cnodes.add(cnode)
                if cnode in pending_recomputation_cnodes: pending_recomputation_cnodes.remove(cnode)
                op_list.append(ComputeOp(cnode))
                reexecuted_fwd_nodes.add(cnode)

                for adep in cnode.get_all_standard_deps():
                    # adep is cnode
                    used_by_fwd = False
                    used_by_bwd = False
                    for cuser in adep.get_all_standard_users():
                        if cuser in pending_recomputation_cnodes:
                            used_by_fwd = True
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not (used_by_fwd or used_by_bwd)
                    if no_longer_used and adep not in redeleted_anodes:
                        for anode in adep.users:
                            if anode not in redeleted_anodes and anode in deleted_in_fwd_anodes:
                                assert isinstance(anode, AllocationNode)
                                op_list.append(DeleteOp(Activation(anode)))
                                redeleted_anodes.add(anode)
                                
                    if used_by_bwd:
                        for anode in adep.users:
                            bwd_to_be_redeleted_anodes.add(anode)
                            
                for auser in cnode.get_all_standard_users():
                    # auser is cnode 
                    assert isinstance(auser, ComputationNode)
                    used_by_bwd = False
                    for cuser in auser.get_all_standard_users():
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    if used_by_bwd:
                        for anode in cnode.users:
                            bwd_to_be_redeleted_anodes.add(anode)


                for anode in all_anodes:
                    if "source" in anode.name:
                        continue
                    # assert isinstance(anode, AllocationNode)
                    used_by_fwd = False
                    used_by_bwd = False
                    for cuser in anode.users_real:
                        # assert isinstance(cuser, ComputationNode)
                        if cuser in pending_recomputation_cnodes:
                            used_by_fwd = True
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not (used_by_fwd or used_by_bwd)
                    if no_longer_used and anode not in redeleted_anodes and anode in deleted_in_fwd_anodes:
                    # if no_longer_used:
                        op_list.append(DeleteOp(Activation(anode)))
                        redeleted_anodes.add(anode)
                all_anodes = all_anodes.difference(redeleted_anodes)

            # for anode in rematerialized_anodes:
            #     used_by_recomputation_fwd = False
            #     used_by_bwd = False
            #     for cuser in anode.users_real:
            #         assert isinstance(cuser, ComputationNode)
            #         if cuser in pending_recomputation_cnodes:
            #             used_by_recomputation_fwd = True
            #         if cuser in pending_bwd_cnodes:
            #             used_by_bwd = True
            #     no_longer_used = not (used_by_recomputation_fwd or used_by_bwd)
            #     if no_longer_used and anode not in redeleted_anodes:
            #         # print(f"Deleting rematerialized anode {anode.name} after bwd cnode {cnode.name}")
            #         op_list.append(DeleteOp(Activation(anode)))
            #         # rematerialized_anodes.remove(anode)
            #         redeleted_anodes.add(anode)
            # rematerialized_anodes.difference_update(redeleted_anodes)
        # executed_bwd_cnodes.add(cnode)
        # op_list.append(ComputeOp(cnode))

        compute_cnode_input(cnode)

        print(f"appending compute op for bwd cnode {cnode.name}")

        # repending_fwd_cnodes = []
        # def recompute_cnode(cnode):
        #     assert isinstance(cnode, ComputationNode)
        #     assert cnode.is_fwd
        #     for adep in cnode.get_all_standard_deps():
        #         if adep.is_checkpointed:
        #             recompute_cnode(adep)
            
        #     executed_bwd_cnodes.add(cnode)
        #     op_list.append(ComputeOp(cnode))
        #     for adep in cnode.get_all_standard_deps():
        #         # auser is anode
        #         used_by_fwd = False
        #         used_by_bwd = False
        #         for cuser in adep.get_all_standard_users():
        #             if cuser in pending_fwd_cnodes:
        #                 used_by_fwd = True
        #             if cuser in pending_bwd_cnodes:
        #                 used_by_bwd = False
        #         no_longer_used = not (used_by_fwd or used_by_bwd)
        #         # op_is_checkpointed = not used_by_fwd and used_by_bwd and adep.is_checkpoined
        #         if no_longer_used and adep not in redeleted_anodes:
        #             for anode in adep.users:
        #                 assert isinstance(anode, AllocationNode)
        #                 op_list.append(DeleteOp(Activation(anode)))
        #                 redeleted_anodes.add(anode)
        
        # for adep in cnode.get_all_standard_deps():
        #     if adep.is_checkpointed:
        #         adep_users = adep.users
        #         tensors_were_deleted = all(adep_user in deleted_anodes for adep_user in adep_users)
        #         assert tensors_were_deleted, "Tensors for checkpointed adep should have been deleted before recomputation."
        #         recompute_cnode(adep)

        # Delete input tensors of cnode if no longer used
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            # used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not used_by_bwd
            if no_longer_used and adep.name not in deleted_anodes:
                for anode in adep.users:
                    if anode not in deleted_anodes:
                        assert isinstance(anode, AllocationNode)
                        op_list.append(DeleteOp(Activation(anode)))
                        deleted_anodes.add(anode)
        # for anode in all_anodes:
        #     if "source" in anode.name:
        #         continue
        #     # assert isinstance(anode, AllocationNode)
        #     used_by_bwd = False
        #     for cuser in anode.users_real:
        #         # assert isinstance(cuser, ComputationNode)
        #         if cuser in pending_bwd_cnodes:
        #             used_by_bwd = True
        #     no_longer_used = not used_by_bwd
        #     if no_longer_used and anode not in deleted_anodes:
        #         op_list.append(DeleteOp(Activation(anode)))
        #         deleted_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)
        
    # lastly, delete all remaining anodes that are not source
    # all_anodes = set(data_anodes + grad_anodes)
    # deleted_anodes = set()
    # for anode in all_anodes:
    #     if "source" in anode.name:
    #         continue
    #     if anode not in deleted_anodes:
    #         op_list.append(DeleteOp(Activation(anode)))
    #         deleted_anodes.add(anode)

    init_alive_status = {}
    init_alive_status[fb_graph.source_data_anode.name] = True

    op_sched = OpSchedule(
        op_list,
        loss_idx=loss_idx,
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=[],
        with_parameters=False,
    )

    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched

def schedule_offload_all(md: ModelPULP, check_valid=False):
    print("Generating checkpointing schedule...")
    fb_graph = md.hgraph.fb_graph

    # Call my solver
    op_list = []
    init_op_list = []

    for pnode in fb_graph.parameter_nodes:
        init_op_list.append(ExecCodeOp(pnode.param_name, pnode.get_code()))


    pending_fwd_cnodes = list(fb_graph.dict_fwd_cnodes.values())
    pending_bwd_cnodes = list(fb_graph.dict_bwd_cnodes.values())
    pending_bwd_cnodes.reverse()

    pending_recomputation_cnodes = set()

    segment_head = set()

    data_anodes = list(fb_graph.dict_data_anodes.values())
    grad_anodes = list(fb_graph.dict_grad_anodes.values())
    # phantoms_anodes = list(fb_graph.dict_phantoms_anodes.values())

    # all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)
    # all_anodes = data_anodes + grad_anodes + phantoms_anodes
    all_anodes = data_anodes + grad_anodes

    # pending_fwd_cnodes = md.hgraph.list_HCNs

    deleted_anodes = set()
    offloaded_anodes = set()

    executed_fwd_cnodes = set()
    executed_bwd_cnodes = set()
    loss_idx = -1

    while len(pending_fwd_cnodes) != 0:
        cnode = pending_fwd_cnodes.pop(0)
        cnode_unique_id = cnode.unique_id
        assert isinstance(cnode, ComputationNode)
        executed_fwd_cnodes.add(cnode)
        op_list.append(ComputeOp(cnode))

        if cnode.is_segment_head or cnode.is_checkpointed:
            pending_recomputation_cnodes.add(cnode)

        # record loss idx
        if cnode.name == md.hgraph.cluster.loss_cnode.name:
            loss_idx = len(executed_fwd_cnodes) + len(deleted_anodes) + len(offloaded_anodes) -1

        # print(f"appending compute op for fwd cnode {cnode.name}")
        # delete input tho cnode?
        if cnode.is_segment_head:
            print(f"cnode {cnode.name} is segment head, skip deleting its inputs")
            # note, might cause a problem for add operator where one of its input is from another segment
            segment_head.add(cnode)
            
            for anode in all_anodes:
                if "source" in anode.name:
                    continue
                # assert isinstance(anode, AllocationNode)
                used_by_fwd = False
                used_by_bwd = False
                used_by_recomputation = False
                for cuser in anode.users_real:
                    # assert isinstance(cuser, ComputationNode)
                    if cuser in pending_fwd_cnodes:
                        used_by_fwd = True
                    if cuser in pending_bwd_cnodes:
                        used_by_bwd = True
                    if cuser in segment_head:
                        used_by_recomputation = True
                # no_longer_used = not (used_by_fwd or used_by_bwd or used_by_recomputation)
                to_offload = not used_by_fwd and used_by_bwd and cnode_unique_id < 400 and cnode_unique_id%2 == 0
                to_delete = not (used_by_fwd or used_by_bwd or used_by_recomputation)
                if to_offload: assert not to_delete, "Anode cannot be both offloaded and deleted."
                if to_delete and anode not in deleted_anodes:
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
                elif to_offload and anode not in offloaded_anodes:
                    op_list.append(OffloadOp(Activation(anode)))
                    offloaded_anodes.add(anode)
            # all_anodes = all_anodes.difference(deleted_anodes)
            all_anodes = remove_first_occurrence_from_list(all_anodes, deleted_anodes)
            continue
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            dep_no_longer_used = not (used_by_fwd or used_by_bwd)
            dep_is_checkpointed = not used_by_fwd and used_by_bwd and adep.is_checkpointed
            to_offload = not used_by_fwd and used_by_bwd and not adep.is_checkpointed and cnode_unique_id < 400 and cnode_unique_id%2 == 0
            
            if to_offload: assert not dep_no_longer_used and not dep_is_checkpointed, "Anode cannot be both offloaded and deleted or checkpointed."

            if dep_is_checkpointed: print(f"adep {adep.name} is_checkpointed: {adep.is_checkpointed}")

            if (dep_no_longer_used or dep_is_checkpointed) and adep not in deleted_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(DeleteOp(Activation(anode)))
                    deleted_anodes.add(anode)
            elif to_offload and adep not in offloaded_anodes:
                for anode in adep.users:
                    assert isinstance(anode, AllocationNode)
                    op_list.append(OffloadOp(Activation(anode)))
                    offloaded_anodes.add(anode)

        for anode in all_anodes:
            if "source" in anode.name:
                continue
            # assert isinstance(anode, AllocationNode)
            used_by_fwd = False
            used_by_bwd = False
            used_by_recomputation = False
            for cuser in anode.users_real:
                # assert isinstance(cuser, ComputationNode)
                if cuser in pending_fwd_cnodes:
                    used_by_fwd = True
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
                if cuser in segment_head:
                    used_by_recomputation = True
            # no_longer_used = not (used_by_fwd or used_by_bwd or used_by_recomputation)
            to_offload = not used_by_fwd and used_by_bwd and cnode_unique_id < 400 and cnode_unique_id%2 == 0
            to_delete = not (used_by_fwd or used_by_bwd or used_by_recomputation)
            if to_offload: assert not to_delete, "Anode cannot be both offloaded and deleted."
            # if no_longer_used and anode not in deleted_anodes:
            if to_delete and anode not in deleted_anodes:
                op_list.append(DeleteOp(Activation(anode)))
                deleted_anodes.add(anode)
            # elif to_offload and anode not in offloaded_anodes:
            #     op_list.append(OffloadOp(Activation(anode)))
            #     offloaded_anodes.add(anode)
        # all_anodes = all_anodes.difference(deleted_anodes)
        all_anodes = remove_first_occurrence_from_list(all_anodes, deleted_anodes)

    assert loss_idx != -1
    print(f"Currently live anodes: ")
    for anode in all_anodes:
        print(f"\tAnode {anode.name} is used by {[cnode.name for cnode in anode.users_real]}")

    # >>> BWD pass

    rematerialized_anodes = set()
    deleted_in_fwd_anodes = deleted_anodes.copy()
    redeleted_anodes = set()
    bwd_to_be_redeleted_anodes = set()

    reexecuted_fwd_nodes = set()

    # all_anodes = set(data_anodes + grad_anodes + phantoms_anodes)
    all_anodes = set(data_anodes + grad_anodes)

    while len(pending_bwd_cnodes) != 0:
        cnode = pending_bwd_cnodes.pop(0)
        assert isinstance(cnode, ComputationNode)

        def compute_cnode_input(cnode):
            nonlocal all_anodes
            nonlocal deleted_anodes
            assert isinstance(cnode, ComputationNode)

            if not cnode.is_fwd:    # is bwd
                # Check its fwd counterpart
                for cdep in cnode.get_all_standard_deps():
                    if cdep.main_target == cnode.main_target \
                        and cdep.is_fwd \
                        and cdep.is_checkpointed \
                        and cdep not in reexecuted_fwd_nodes:
                            print(f"Recomputing fwd cnode {cdep.name} for bwd cnode {cnode.name}")
                            # pending_recomputation_cnodes.add(cdep)
                            compute_cnode_input(cdep)

                # fetch in the offloaded anodes that are needed by this bwd cnode
                for adep in cnode.get_all_standard_deps():
                    for anode in adep.users:
                        if anode in offloaded_anodes:
                            print(f"Fetching offloaded anode {anode.name} for bwd cnode {cnode.name}")
                            op_list.append(PrefetchOp(Activation(anode)))
                            offloaded_anodes.remove(anode)
                # execute the bwd cnode itself
                executed_bwd_cnodes.add(cnode)
                op_list.append(ComputeOp(cnode))

                # delete its inputs if no longer used
                for adep in cnode.get_all_standard_deps():
                    # used_by_fwd = False
                    used_by_bwd = False
                    for cuser in adep.get_all_standard_users():
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not used_by_bwd
                    if no_longer_used and adep.name not in deleted_anodes:
                        for anode in adep.users:
                            if anode not in deleted_anodes or anode in bwd_to_be_redeleted_anodes:
                                assert isinstance(anode, AllocationNode)
                                op_list.append(DeleteOp(Activation(anode)))
                                deleted_anodes.add(anode)
                                bwd_to_be_redeleted_anodes.discard(anode)

            if cnode.is_fwd:
                # check its deps
                print(f"Cnode {cnode.name} has {len(cnode.get_all_standard_deps())} deps: {[d.name for d in cnode.get_all_standard_deps()]}")
                if not cnode.is_segment_head:
                    # reveresed_deps = list(cnode.get_all_standard_deps())
                    # reveresed_deps.reverse()
                    # print(f"Processing deps for cnode {cnode.name}: {[d.name for d in reveresed_deps]}")
                    for cdep in cnode.get_all_standard_deps():
                        assert isinstance(cdep, ComputationNode)
                        cross_segment_dep = cnode.is_fwd and cdep.is_segment_tail
                        if cdep.is_fwd and (cdep.is_checkpointed or cdep.is_segment_head) and cdep not in reexecuted_fwd_nodes and not cross_segment_dep:
                            print(f"Recomputing fwd cnode {cdep.name} for cnode {cnode.name}")
                            # pending_recomputation_cnodes.add(cdep)
                            compute_cnode_input(cdep)
                
                # Fetch in the offloaded anodes that are needed by this fwd cnode
                for adep in cnode.get_all_standard_deps():
                    for anode in adep.users:
                        if anode in offloaded_anodes:
                            print(f"Fetching offloaded anode {anode.name} for fwd cnode {cnode.name}")
                            op_list.append(PrefetchOp(Activation(anode)))
                            offloaded_anodes.remove(anode)
                
                # Execute recomputation fwd
                executed_bwd_cnodes.add(cnode)
                if cnode in pending_recomputation_cnodes: pending_recomputation_cnodes.remove(cnode)
                op_list.append(ComputeOp(cnode))
                reexecuted_fwd_nodes.add(cnode)

                for adep in cnode.get_all_standard_deps():
                    # adep is cnode
                    used_by_fwd = False
                    used_by_bwd = False
                    for cuser in adep.get_all_standard_users():
                        if cuser in pending_recomputation_cnodes:
                            used_by_fwd = True
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not (used_by_fwd or used_by_bwd)
                    if no_longer_used and adep not in redeleted_anodes:
                        for anode in adep.users:
                            if anode not in redeleted_anodes and anode in deleted_in_fwd_anodes:
                                assert isinstance(anode, AllocationNode)
                                op_list.append(DeleteOp(Activation(anode)))
                                redeleted_anodes.add(anode)
                                
                    if used_by_bwd:
                        for anode in adep.users:
                            bwd_to_be_redeleted_anodes.add(anode)
                            
                for auser in cnode.get_all_standard_users():
                    # auser is cnode 
                    assert isinstance(auser, ComputationNode)
                    used_by_bwd = False
                    for cuser in auser.get_all_standard_users():
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    if used_by_bwd:
                        for anode in cnode.users:
                            bwd_to_be_redeleted_anodes.add(anode)


                for anode in all_anodes:
                    if "source" in anode.name:
                        continue
                    # assert isinstance(anode, AllocationNode)
                    used_by_fwd = False
                    used_by_bwd = False
                    for cuser in anode.users_real:
                        # assert isinstance(cuser, ComputationNode)
                        if cuser in pending_recomputation_cnodes:
                            used_by_fwd = True
                        if cuser in pending_bwd_cnodes:
                            used_by_bwd = True
                    no_longer_used = not (used_by_fwd or used_by_bwd)
                    if no_longer_used and anode not in redeleted_anodes and anode in deleted_in_fwd_anodes:
                    # if no_longer_used:
                        op_list.append(DeleteOp(Activation(anode)))
                        redeleted_anodes.add(anode)
                all_anodes = all_anodes.difference(redeleted_anodes)

        compute_cnode_input(cnode)

        print(f"appending compute op for bwd cnode {cnode.name}")

        # Delete input tensors of cnode if no longer used
        for adep in cnode.get_all_standard_deps():
            # auser is anode
            # used_by_fwd = False
            used_by_bwd = False
            for cuser in adep.get_all_standard_users():
                if cuser in pending_bwd_cnodes:
                    used_by_bwd = True
            no_longer_used = not used_by_bwd
            if no_longer_used and adep.name not in deleted_anodes:
                for anode in adep.users:
                    if anode not in deleted_anodes:
                        assert isinstance(anode, AllocationNode)
                        op_list.append(DeleteOp(Activation(anode)))
                        deleted_anodes.add(anode)

    init_alive_status = {}
    init_alive_status[fb_graph.source_data_anode.name] = True

    op_sched = OpSchedule(
        op_list,
        loss_idx=loss_idx,
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=[],
        with_parameters=False,
    )

    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched