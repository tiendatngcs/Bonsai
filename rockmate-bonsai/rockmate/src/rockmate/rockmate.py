# ============
# = ROCKMATE =
# ============
import pickle
import sys
import torch
from torch import tensor
import time
from datetime import datetime
import warnings
import gc
import rkgb
from rkgb.lowlevel.preprocess_samples import ExampleInputs
from rkgb.lowlevel.measure import tensor_memory_size, get_Timer
from rkgb.lowlevel.constants import ref_verbose, ExceptionModuleDoesNotReqGrad
from rkgb.lowlevel.ast_add_on import ast_to_str, make_str_list_assign
from rkgb.core.partitioned import (
    PartitionerBottomToTop,
    PartitionerSequence,
    Partitioner,
)
from rkgb.lowlevel.constants import init_target_string

from .utils import get_optimize_metrics, measure_bandwidth
from .op_schedule import *
from .simulation import Simulator
from .solvers.main import preprocess, solve_recursive, FastSolver, add_sched
from .solvers import HILP, CheapSolver, RK_rotor
from .compiler import Compiler, RK_Storage, make_gd, Fct_record_event
import psutil

import os

class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        out = self.model(**inputs)
        return out.logits   # remove HF dict → required for stable dynamo export

fragmentations = []
allocated_memory_list = []
current_iteration = 0   # first iteration is 1

iterations_to_print_allocated_memory = [2]

sys.setrecursionlimit(10000)

def get_memory_fragmentation(device=None):
    """
    Estimates memory fragmentation ratio for the given CUDA device.
    Returns a float between 0 and 1 (higher = more fragmented).
    """
    device = device or torch.cuda.current_device()
    stats = torch.cuda.memory_stats(device)
    reserved = stats["reserved_bytes.all.current"]
    allocated = stats["allocated_bytes.all.current"]
    if reserved == 0:
        return 0.0
    return 1 - (allocated / reserved)

def update_current_iteration():
    global current_iteration
    current_iteration += 1
    # print(f"\n=== Starting iteration {current_iteration}\n")
    
def print_allocated_memory_before(op):
    global current_iteration
    if current_iteration not in iterations_to_print_allocated_memory:
        return
    if os.environ.get('PRINT_ALLOCATED_MEMORY', '0') == '1':
        stats = torch.cuda.memory_stats(None)
        allocated = stats["allocated_bytes.all.current"]
        print(f"Memory usage before {op}: \t\t{allocated / (1024**2):.2f} MB")

def print_allocated_memory(op):
    global current_iteration
    if current_iteration not in iterations_to_print_allocated_memory:
        return
    if os.environ.get('PRINT_ALLOCATED_MEMORY', '0') == '1':
        stats = torch.cuda.memory_stats(None)
        allocated = stats["allocated_bytes.all.current"]
        print(f"Memory usage after {op}: \t\t{allocated / (1024**2):.2f} MB")

class Rockmate(torch.nn.Module):
    compiler = None
    autograd_Function = None
    backward_stop = False
    backward_add_output_grad = True
    op_sched = None
    module_does_not_req_grad = False

    def __init__(
        self,
        original_mod,
        model_inputs,
        budget=None,
        top_solvers=None,
        bottom_solvers=None,
        partitioners=None,
        rkgb_res=None,
        solve_sched=True,
        verbose=False,
        model_kwargs=None,
        cpu_optim=torch.optim.Adam,
        gpu_optim=torch.optim.Adam,
        optim_kwargs={},
        minor_offload_size=10 * 1024**2,
        keep_outputs=False,
        dynamic_batch_dim=None,
        solve_recursive=True,
        cache_sched=False,
        memory_mode=None
    ):
        super().__init__()
        ref_verbose[0] = verbose
        self.top_solvers = top_solvers
        self.bottom_solvers = bottom_solvers
        if self.top_solvers is None and self.bottom_solvers is None:
            self.top_solvers = [HILP()]
            self.bottom_solvers = [HILP()]
        self.partitioners = partitioners
        if self.partitioners is None:
            self.partitioners = [PartitionerBottomToTop(can_use_rotor=False)]
        self.device = torch.device("cuda")  # Not obtaining from model
        self.exec_with_record_mem = False
        self.exec_with_record_time = False
        self.budget = budget
        self.rkgb_res = rkgb_res
        self.keep_outputs = keep_outputs
        self.list_solutions = []
        self.dynamic_batch_dim = dynamic_batch_dim
        self.timer = get_Timer(self.device)
        object.__setattr__(self, "original_mod", original_mod)
        
        self.cache_sched = cache_sched
        
        self.model_name = original_mod.__class__.__name__
        if hasattr(original_mod, 'model_name'):
            self.model_name = original_mod.model_name
        
        # self.memory_mode = memory_mode
        # if not hasattr(self, 'memory_mode'):
        self.memory_mode = memory_mode
        assert hasattr(self, 'memory_mode'), "Rockmate instance must have memory_mode attribute"
        assert self.memory_mode is not None, "Rockmate memory_mode attribute must not be None"
        print(f"Memory_mode: {self.memory_mode}")
            
        self.batch_size = None
        if isinstance(model_inputs, (list, tuple)):
            self.batch_size = model_inputs[0].shape[0]
        elif isinstance(model_inputs, dict):
            input_ids = model_inputs.get('input_ids', None)
            assert input_ids is not None, "model_inputs dict must contain 'input_ids' key to get batch size"
            self.batch_size = input_ids.shape[0]
        else:
            self.batch_size = model_inputs.shape[0]
        
        print(f"Solving for model {self.model_name} with batch size {self.batch_size}")
        
        self.schedule_dir = "./schedules"
        # create schedule dir if does not exist
        if not os.path.exists(self.schedule_dir):
            os.makedirs(self.schedule_dir)
            
        gpu_name = os.environ.get('GPU_NAME', 'UnknownGPU')
        
        # self.file_name = os.path.join(self.schedule_dir, f"{self.memory_mode}_{self.model_name}_batch{self.batch_size}_{int(self.budget)}_sched_{gpu_name}.pkl")
        self.file_name = os.environ.get('SCHEDULE_FILE', os.path.join(self.schedule_dir, f"{self.memory_mode}_{self.model_name}_batch{self.batch_size}_{int(self.budget)}_sched_{gpu_name}.pkl"))
        
        # If schedule dir does not exist, create it
        if not os.path.exists(self.schedule_dir):
            os.makedirs(self.schedule_dir)

        self.config_partitioner()


        self.optimize_metrics = get_optimize_metrics(
            list(original_mod.parameters())[0],
            optim=gpu_optim,
            cpu_optim=cpu_optim,
            optim_kwargs=optim_kwargs,
            minor_offload_size=minor_offload_size,
        )

        self.get_rkgb_result(model_inputs, model_kwargs, minor_offload_size)

        # TODO: only measure bandwidth if needed
        self.bandwidth = measure_bandwidth()

        self.global_dict = make_gd(
            self.device,
            self.original_mod,
            self.dict_constants,
            optimize_metrics=self.optimize_metrics,
        )
        
        self.fix_solver_config()

        if solve_sched:
            self.solve_sched(recursive=solve_recursive)
            # print(f"Op list before compilation:")
            # for op in self.op_list:
            #     print(f"  {op}")
            self.get_compiled_fct()

        # recomputed_size = self.get_recomputed_size()
        # recomputed_ops = self.get_recomputed_ops()

        # print(f"Total recomputed size: {recomputed_size / (1024**2):.2f} MB")
        # print(f"Total recomputed ops: {recomputed_ops}")

    def get_rkgb_result(self, model_inputs, model_kwargs, minor_offload_size):
        # -- use rkGB --
        try:
            if self.rkgb_res is None:
                self.rkgb_res = rkgb.rkgb.Result(
                    self.original_mod,
                    model_args=model_inputs,
                    model_kwargs=model_kwargs,
                    # verbose=verbose,
                    # wanted_graphs={"FB"},
                    partitioners=self.partitioners,
                    inspection_device=torch.device("cuda"),
                    print_time_in_each_stage=True,
                    dynamic_batch_dim=self.dynamic_batch_dim,
                )
            else:
                self.rkgb_res = self.rkgb_res

        except ExceptionModuleDoesNotReqGrad:
            self.module_does_not_req_grad = True
        
        # print(f"Forward graph:")
        # # for node in self.rkgb_res.simplified_graph.nodes:
        # for node in self.rkgb_res.forward_graph.nodes:
        #     print(f"  {node}")

        self.dict_constants = self.rkgb_res.forward_and_backward_graph.dict_constants
        self.init_code = ast_to_str(self.rkgb_res.forward_and_backward_graph.init_code)
        self.output_names = [
            o.name
            for o in self.rkgb_res.forward_and_backward_graph.list_output_data_anodes
        ]

        self.minor_param_nodes = []
        for pnode in self.rkgb_res.hierarchical_cluster.parameter_nodes:
            if pnode.mem < minor_offload_size and not pnode.is_buffer:
                self.minor_param_nodes.append(pnode)
        for pnode in self.rkgb_res.S.init_node.required_parameter_nodes:
            if not pnode.is_buffer:
                self.minor_param_nodes.append(pnode)

    def config_partitioner(self):
        # ensure HILP config match partitioner config
        for partitioner in self.partitioners:
            if isinstance(partitioner, PartitionerBottomToTop):
                for solver in self.bottom_solvers:
                    if isinstance(solver, HILP):
                        if solver.config.nb_total_nodes < partitioner.config.max_estimate_per_sub_graph:
                            print(f"Warning, bottom solver HILP has nb_total_nodes {solver.config.nb_total_nodes}, "
                                  f"smaller than partitioner max_estimate_per_sub_graph {partitioner.config.max_estimate_per_sub_graph}."
                                  " This may result in failure to find schedules")
                for solver in self.top_solvers:
                    if isinstance(solver, HILP):
                        if solver.config.nb_total_nodes < partitioner.config.max_estimate_for_main_graph:
                            print(f"Warning, top solver HILP has nb_total_nodes {solver.config.nb_total_nodes}, "
                                  "smaller than partitioner max_estimate_for_main_graph "
                                  f"{partitioner.config.max_estimate_for_main_graph}. This may result in failure to find schedules")

    def fix_solver_config(self):
        # Set some options whoe values can only be known at runtime
        for solver in self.top_solvers:
            if isinstance(solver, HILP):
                solver.config.model_kwargs = solver.config.model_kwargs.copy()
                solver.config.protected_names = solver.config.protected_names.copy()
                solver.config.model_kwargs["optimize_metrics"] = self.global_dict["optimize_metrics"]
                solver.config.protected_names.extend([f"{init_target_string} data", f"{init_target_string} grad"])
                if self.keep_outputs:
                    solver.config.protected_names.extend(self.output_names)
                if self.dynamic_batch_dim is not None:
                    solver.config.model_kwargs["dynamic_batch_size"] = True

        for solver in self.bottom_solvers:
            if isinstance(solver, HILP):
                solver.config.protected_names.extend([f"{init_target_string} data", f"{init_target_string} grad"])
                if self.keep_outputs:
                    solver.config.protected_names.extend(self.output_names)


    def preprocess(self, solver = None):
        if solver is None:
            solver = FastSolver()
        for cluster in self.rkgb_res.hierarchical_structure.all_clusters:
            if not cluster.is_bottom:
                solver.preprocess(
                    cluster,
                    no_del_names=[
                        f"{init_target_string} data",
                        f"{init_target_string} grad",
                    ]
                    + self.output_names,
                )

    def fast_solve(self):
        solver = FastSolver()
        cluster = self.rkgb_res.hierarchical_cluster

        op_scheds = solver.solve(
                    cluster,
                    no_del_names=[
                        f"{init_target_string} data",
                        f"{init_target_string} grad",
                    ]
                    + self.output_names,
                )
        self.op_sched = op_scheds[0]

    def solver_recursive(self, list_solvers=None, only_preprocess=False):
        list_solvers = list_solvers or self.bottom_solvers

        solve_recursive(
            self.rkgb_res.hierarchical_cluster,
            list_solvers=list_solvers,
            skip_self=True,
        )

    def get_recomputed_size(self):
        assert self.op_sched is not None
        in_bwd = False
        total_recomputed = 0
        for op in self.op_sched.op_list:
            node = op.target
            if isinstance(node, ComputationNode):
                if node.is_fwd:
                    if in_bwd:
                        anodes = node.users
                        for anode in anodes:
                            gpu_tensor = self.compiler.storage.get_val(anode.main_target).data
                            tensor_size_bytes = gpu_tensor.numel() * gpu_tensor.element_size()
                            total_recomputed += tensor_size_bytes
                else:
                    in_bwd = True
        return total_recomputed

    def get_recomputed_ops(self):
        assert self.op_sched is not None
        in_bwd = False
        total_recomputed_ops = 0
        for op in self.op_sched.op_list:
            node = op.target
            if isinstance(node, ComputationNode):
                if node.is_fwd:
                    if in_bwd:
                        total_recomputed_ops += 1
                else:
                    in_bwd = True
        return total_recomputed_ops

    def solve_sched(self, budget: float = None, list_solvers=None, recursive=True) -> None:
        assert budget is not None or self.budget is not None, "Please provide a memory budget in bytes"
        if self.cache_sched and os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.op_sched = pickle.load(f)
            assert self.op_sched is not None
            print(f"Loaded schedule from {self.file_name}")
            return
        # else:
        #     print(f"No cached schedule found at {self.file_name}, solving schedule...")
        #     exit(1)
        
        # if multiple solvers in list_solvers,
        # will choose the one with minimum time
        budget = budget or self.budget
        # budget -= self.minor_size
        assert budget is not None, "Please provide a memory budget in bytes"
        list_solvers = list_solvers or self.top_solvers

        # Set some options whoe values can only be known at runtime
        for solver in list_solvers:
            if isinstance(solver, HILP):
                # TODO: if no partitioning is allowed, update solver max nodes
                hilp_solver = True
                solver.config.optimize_metrics = self.global_dict["optimize_metrics"]
                if solver.config.offload:
                    if self.bandwidth is None:
                        self.bandwidth = measure_bandwidth()
                    solver.config.model_kwargs["bandwidth"] = self.bandwidth
        for solver in list_solvers:
            if isinstance(solver, HILP):
                solver.config.protected_names.extend([f"{init_target_string} data", f"{init_target_string} grad"])
                if self.keep_outputs:
                    solver.config.protected_names.extend(self.output_names)
                if self.dynamic_batch_dim is not None:
                    solver.config.model_kwargs["dynamic_batch_size"] = True

        self.preprocess()
        if self.bottom_solvers and recursive:
            self.solver_recursive()

        list_solutions = []
        for solver in list_solvers:
            list_solutions.extend(
                solver(self.rkgb_res.hierarchical_cluster, [budget])
            )

        self.rkgb_res.hierarchical_cluster.list_schedules.extend(list_solutions)
        print(f"Number of found schedules: {len(list_solutions)}")
        if not list_solutions:
            warnings.warn("no feasible schedule is found")
            return False
        else:
            self.list_solutions.extend(list_solutions)
            self.op_sched = list_solutions[
                np.argmin([sum(op_sched.time) for op_sched in list_solutions])
            ]
            print(f"Selected schedule is:")
            for op in self.op_sched.op_list:
                print(f"  {op}")
        
            # pickled the op_sched
            if self.cache_sched and not os.path.exists(self.file_name):
                with open(self.file_name, 'wb') as f:
                    pickle.dump(self.op_sched, f)
                print(f"Saved schedule to {self.file_name}")
            return True

            # self.op_list = self.op_sched.op_list

    def get_compiled_fct(self, new_compiler=True, simulate=True):
        if self.op_sched is None:
            return 
        if simulate:
            self.op_sched.simulate_update(Simulator, refine_optimize=True)
        if new_compiler:
            storage = RK_Storage()
            storage.init(self.global_dict)
            storage.dict_info = self.rkgb_res.forward_graph.dict_info
            self.compiler = Compiler(storage)
            if self.dynamic_batch_dim is not None:
                storage.dynamic_shapes = self.rkgb_res.raw_graph.dynamic_shapes
            
        self.minor_parameters = []
        if self.minor_param_nodes:
            self.minor_parameters = [
                pnode.get_value(self.original_mod)
                for pnode in self.minor_param_nodes
            ]

            def optimize():
                self.compiler.storage.ld["Optimize_minors"].step()
                for p in self.minor_parameters:
                    p.grad = None
                for pnode in self.minor_param_nodes:
                    code = make_str_list_assign(pnode.view_code, suffix=".data")
                    exec(code, self.global_dict, self.compiler.storage.ld)

            self.op_list[-1].add_fct(optimize)

        self.autograd_Function = define_autograd_Function(self)
        self.inherits_original_mod_attributes_and_methods()
        self.compiler.compile_preparation(
            self.rkgb_res.hierarchical_cluster,
            self.op_sched,
            self.minor_param_nodes,
            self.rkgb_res.forward_graph.output_nodes,
        )

        self.compiler.compile_sched(self.op_sched)
        
    def _exec(self, op: Op, disable=False):
        # print(f"Executing op: {op}")
        try:
            if self.exec_with_record_mem:
                torch.cuda.reset_peak_memory_stats()
                self.mem_before = torch.cuda.memory_allocated()
                self.max_before = torch.cuda.max_memory_allocated()
                if not disable:
                    op()
                
                torch.cuda.synchronize()
                allo_mem = torch.cuda.memory_allocated() - self.mem_before
                peak_mem = torch.cuda.max_memory_allocated() - self.max_before
                self.max_mem.append(peak_mem - allo_mem)
                self.allo_mem.append(allo_mem)
                op.allo_mem = allo_mem
                op.peak_mem = peak_mem - allo_mem
            else:
                if not disable:
                    op()
                else:
                    for f in op.fct_list:
                        if isinstance(f, Fct_record_event):
                            f()
        except Exception as e:
            print(f"Failed to execute {op}")
            raise e

    def exec_step(self, step):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
        ) as p:
            self.timer.start()
            for op in step.op_list:
                # print_allocated_memory_before(op)
                # print(f"Executing step op: {op}")
                self._exec(op)
                if os.environ.get('ROCKMATE_RECORD_FRAGMENTATION', '0') == '1':
                    fragmentation = get_memory_fragmentation()
                    fragmentations.append(fragmentation)
                    allocated_memory_list.append(torch.cuda.memory_allocated())
                print_allocated_memory(op)
            # torch.cuda.synchronize()
            # timer.end()
            # self.step_alloc_time.append(timer.elapsed())

            # timer.start()
            # for op in (
            #         step.ofl_ops
            #         +step.prf_ops
            #         +step.comp_ops
            #         +step.opt_ops
            #         +step.del_ops):
            #     self._exec(op)
            torch.cuda.synchronize()
            self.timer.end()
        p.export_chrome_trace(f"profiles/{len(self.step_profiles)}.json")
        self.step_time.append(self.timer.elapsed())
        self.step_memory_stats.append(torch.cuda.memory_stats())
        self.step_profiles.append(
            p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        )

    def init_fwd_exec(self):
        # TODO: rename to first forward; also do the first backward differently incase grad not exist
        """
        The first forward execution to:
        1. allocate parameters in RK_storage;
        2. set placeholders for activations;
        3. run forward the first time so no parameter gradients.
        """
        
        # print(f"Init op list includes:")
        # for op in self.op_sched.init_op_list:
        #     print(f"  {op}")

        for op in self.op_sched.init_op_list:
            # print_allocated_memory_before(op)
            # print(f"Executing init fwd op: {op}")
            self._exec(op)
            if os.environ.get('ROCKMATE_RECORD_FRAGMENTATION', '0') == '1' and current_iteration not in iterations_to_print_allocated_memory:
                fragmentation = get_memory_fragmentation()
                fragmentations.append(fragmentation)
                allocated_memory_list.append(torch.cuda.memory_allocated())
            print_allocated_memory(op)
        torch.cuda.synchronize()
        with torch.enable_grad():
            exec(self.init_code, self.global_dict, self.compiler.storage.ld)
            for op in self.op_list[: self.op_sched.loss_idx + 1]:
                disable = False
                if isinstance(op, OffloadOp) and isinstance(op.target, Parameter):
                    if op.target.is_grad or op.target.is_optim_states:
                        disable = True  # first iteration without grad offload
                if isinstance(op, OptimizeOp):
                    disable = True  # first iteration no need to optimize
                # print(f"Executing init fwd op: {op}, disable={disable}")
                self._exec(op, disable=disable)
                if not disable: print_allocated_memory(op)
                # torch.cuda.synchronize()

    def restore_exec(self, keep_grad=False):
        """
        After the training iteration, restore parameters to original_mod.
        All the updated parameters of the original_mod will be kept on CPU.
        """

        if self.compiler.storage.ld == {}:
            return None
        self.zero_grad()
        for l in self.restore_fct_list:
            self._exec(l)
        for p in self.minor_parameters:
            p.data = p.data.to("cpu")

        for k, v in self.op_sched.dict_alloc_param.items():
            if v.pnode.is_buffer or v.pnode in self.minor_param_nodes:
                target = v.pnode.get_value(self.original_mod)
                target.data = target.data.to("cpu")
                continue

            if v.is_grad or v.is_optim_states:
                continue

            target = v.pnode.get_value(self.original_mod)
            target.data = self.compiler.storage.ld[v.target_name].data
            if keep_grad:
                target.grad = self.compiler.storage.ld[v.target_name].grad

        self.compiler.storage.ld = {}
        torch.cuda.synchronize()

    #  === nn.module's forward method wrapping self.autograd_Function.forward ===
    def forward(self, *args, record_mem=False, record_time=False, **kwargs):
        if self.module_does_not_req_grad:
            return self.original_mod(*args, **kwargs)
        self.exec_with_record_mem = record_mem
        self.exec_with_record_time = record_time
        self.max_mem = []
        self.allo_mem = []
        self.step_time = []
        self.step_alloc_time = []
        self.step_memory_stats = []
        self.step_profiles = []
        if not self.training:
            self.original_mod.eval()
            return self.original_mod(*args, **kwargs)
        else:
            if self.compiler is None:
                raise Exception(
                    "No schedule compiled, no solution to exec,\n"
                    "To be able to use the RkMod you first need "
                    "to call get_compiled_fct(_HILP)."
                )
            elif self.autograd_Function is None:
                raise Exception(
                    "The custom forward and backward functions haven't "
                    "been generated yet, please call the function: "
                    "define_autograd_Function"
                )
            # -> Send the inputs to Function.forward via the buffer (Rem 1)
            self.inputs_buffer = ExampleInputs(self.original_mod, args, kwargs)
            self.dict_inputs_buffer = dict_inputs = self.inputs_buffer.dict

            # -> Pass the inputs which req grad to prepare their backward (Rem 1)
            inputs_which_req_grad = []
            name_of_inputs_which_req_grad = []
            for k, v in dict_inputs.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    inputs_which_req_grad.append(v)
                    name_of_inputs_which_req_grad.append(k)
            self.name_of_inputs_which_req_grad_buffer = name_of_inputs_which_req_grad

            dummy_input = torch.ones(1).requires_grad_()
            output_mt_values = self.autograd_Function.apply(
                dummy_input, *inputs_which_req_grad
            )

            self.out_code = []
            self.output_targets = []
            # for output_node in self.rkgb_res.simplified_graph.output_nodes:
            # for main_target, set_output_targets in self.rkgb_res.simplified_graph.dict_output_mt_to_targets_sent.items():
            for main_target in self.rkgb_res.forward_and_backward_graph.output_targets:
                # anode = self.rkgb_res.hierarchical_cluster.dict_nodes[main_target]
                cnode = self.rkgb_res.hierarchical_cluster.dict_nodes[
                    f"FWD[{main_target}]"
                ]
                code = make_str_list_assign(cnode.body_code)
                for out_target in cnode.all_targets:
                    code = code.replace(out_target, f"out_{out_target}")
                    self.output_targets.append(f"out_{out_target}")

                # for output_target in set_output_targets:
                #     # ast_code = self.rkgb_res.simplified_graph.dict_output_viewing_code[main_target]
                #     # code = ""
                #     # for assign in ast_code.body:
                #     #     if ast_to_str(assign.targets) == output_target:
                #     #         code += f"{ast_to_str(assign)}\n"

                #     code = code.replace(output_target, f"out_{output_target}")
                #     if main_target != output_target:
                #         code = code.replace(main_target, f"out_{main_target}")
                #     else:
                #         code = ""
                self.out_code.append(code)
                exec(code, self.compiler.storage.gd, self.compiler.storage.ld)

            # -> Output what's defined in module
            output_nodes = self.rkgb_res.forward_graph.output_nodes
            final_outputs = [
                self.compiler.get_val(f"out_{output.main_target}")
                for output in output_nodes
            ]
            return tuple(final_outputs)

    def zero_grad(self, set_to_none=True, remain_for_offload=True):
        if remain_for_offload:
            remains = []
            # for k,v in self.op_sched.dict_alloc_param.items():
            #     if v.is_grad and self.op_sched.alive_list[-1][k]:
            #     # if v.is_grad and self.op_sched.init_alive_status[k]:
            #         remains.append(v.pnode.param_name)
            for k, p in self.original_mod.named_parameters():
                if k not in remains and p.grad is not None:
                    # if set_to_none:
                    #     p.grad = None
                    # else:
                    p.grad.zero_()
        else:
            self.original_mod.zero_grad(set_to_none=set_to_none)

    @property
    def minor_size(self):
        optimizer_states_factor = 0
        if self.optimize_metrics:
            optimizer_states_factor += self.optimize_metrics["optimizer_states_factor"]
        return sum([pnode.mem for pnode in self.minor_param_nodes]) * (optimizer_states_factor+1)

    @property
    def op_list(self):
        return self.op_sched.op_list

    def inherits_original_mod_attributes_and_methods(self):
        for k, v in self.original_mod.__dict__.items():
            if not "forward" in k and not "backward" in k and k not in ["training"]:
                self.__dict__[k] = v

    def save_to_local(self, path, id=None):
        if not id:
            id = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")    ## rkMod.save_sched_to_local("dump")
        with open(f"{path}/{id}_rkgb_res.pkl", "wb") as f:
            pickle.dump(self.rkgb_res, f)

        
    def get_mean_fragmentations(self):
        return np.mean(np.array(fragmentations))

    def get_mean_allocated_memory(self):
        return np.mean(np.array(allocated_memory_list))

def define_autograd_Function(RkMod: Rockmate):
    class RK_autograd_Function(torch.autograd.Function):
        # === OUR FORWARD FUNCTION ===
        @staticmethod
        def forward(ctx, dummy_input, *args):
            update_current_iteration()  # first iteration is 1
            print_allocated_memory("None")
            if RkMod.compiler.storage.ld != {}:
                ctx.RK_Storage = storage = RkMod.compiler.storage
                ctx.name_of_inputs_which_req_grad = (
                    RkMod.name_of_inputs_which_req_grad_buffer
                )
                dict_inputs = (
                    RkMod.dict_inputs_buffer
                )  # args are passed in RkMod.forward
                #  -> Detach input tensors (Rem 3) and store all the inputs
                dict_input_tensors_detach = dict()  #  dict : input -> detached input
                for k, v in dict_inputs.items():
                    if isinstance(v, torch.Tensor):
                        v_d = v.detach().requires_grad_(v.requires_grad)
                        dict_input_tensors_detach[v] = v_d
                        storage.ld[k] = v_d
                    #  TODO elif iterables of Tensors ?
                    else:
                        storage.ld[k] = v

                with torch.enable_grad():
                    exec(
                        RkMod.init_code, RkMod.global_dict, storage.ld
                    )  # is compiler.global_dict
                    if RkMod.exec_with_record_time and RkMod.op_sched.loss_step > 0:
                        for step in RkMod.op_sched.steps[: RkMod.op_sched.loss_step]:
                            RkMod.exec_step(step)
                    else:
                        # print(f"Fwd op list includes:")
                        # for op in RkMod.op_list[: RkMod.op_sched.loss_idx + 1]:
                        #     print(f"  {op}")
                        for op in RkMod.op_list[: RkMod.op_sched.loss_idx]:
                            # print_allocated_memory_before(op)
                            # print(f"Executing fwd op: {op}")
                            RkMod._exec(op)
                            if os.environ.get('ROCKMATE_RECORD_FRAGMENTATION', '0') == '1' and current_iteration not in iterations_to_print_allocated_memory:
                                fragmentation = get_memory_fragmentation()
                                fragmentations.append(fragmentation)
                                allocated_memory_list.append(torch.cuda.memory_allocated())
                            print_allocated_memory(op)
            else:
                # *** INITIALIZATION PART ***
                #  -> Get the inputs using the buffer (Rem 1)
                dict_inputs = RkMod.dict_inputs_buffer
                RkMod.dict_inputs_buffer = None

                #  -> Create the RK_Storage for this run, and store it in ctx
                ctx.RK_Storage = storage = RkMod.compiler.storage
                storage.init(RkMod.global_dict)
                RkMod.compiler.storage = storage

                #  -> Store what we need to return inputs' grad (Rem 1)
                ctx.name_of_inputs_which_req_grad = (
                    RkMod.name_of_inputs_which_req_grad_buffer
                )

                #  -> Detach input tensors (Rem 3) and store all the inputs
                dict_input_tensors_detach = dict()  #  dict : input -> detached input
                for k, v in dict_inputs.items():
                    if isinstance(v, torch.Tensor):
                        v_d = v.detach().requires_grad_(v.requires_grad)
                        dict_input_tensors_detach[v] = v_d
                        storage.ld[k] = v_d
                        if RkMod.dynamic_batch_dim is not None:
                            inspect_batch = storage.dict_info[k].tensor_size[
                                RkMod.dynamic_batch_dim
                            ]
                            batch_multiplier = (
                                v_d.shape[RkMod.dynamic_batch_dim] / inspect_batch
                            )
                            storage.batch_multiplier = batch_multiplier

                    #  TODO elif iterables of Tensors ?
                    else:
                        storage.ld[k] = v

                torch.cuda.synchronize()
                with torch.enable_grad():
                    RkMod.init_fwd_exec()
                torch.cuda.synchronize()

            #  *** EXECUTION PART ***
            # -> Autograd turns off itself before giving use the control.
            # -> But we need it to forward/backward each node.

            # -> Get the output
            outs = []
            for (
                out_node
            ) in RkMod.rkgb_res.forward_and_backward_graph.list_output_data_anodes:
                out_mt = out_node.main_target
                # if out_mt == RkMod.rkgb_res.forward_graph.nodes[-1].main_target:
                #     # Last output node; its grad_fn will not be called again
                #     outs.append(RkMod.compiler.get_val(out_mt))
                #     continue
                RkMod.compiler.get_val(f"out_{out_mt}").data = RkMod.compiler.get_val(
                    out_mt
                )
                o = RkMod.compiler.get_val(f"out_{out_mt}")
                outs.append(o)

            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)
            # -> Remember that out have been detached from the rest during exec
            """
            ctx.set_materialize_grads(True) # as the default
            # -> so we don't have to check if grad_output is None
            # -> if outputs' grad is None Autograd fill them with zeros
            """

        # === END OF FORWARD FUNCTION ===

        """
        @staticmethod
        def setup_context(ctx,inputs,outputs):
            pass
        # PyTorch prefer to handle the ctx in a separate method,
        # but it makes more sense for us to handle ctx during the forward.
        # (Also, setup_context can only access to inputs and outputs,
        # so they suggest to pass intermediate values as outputs,
        # but outputs can only be tensors)
        # Anyway, it's not important, Autograd is originally designed
        # to handle the ctx during the forward, it's just that PyTorch 2.0
        # now prefers to use this separate method.
        """

        # === OUR BACKWARD FUNCTION ===
        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grad_outs):
            #  -> Reload the storage and out
            # storage = ctx.RK_Storage
            # RkMod.compiler.storage = storage
            # -> Put grad_out in out.grad (Rem 4)
            for out_node, out_grad in zip(
                RkMod.rkgb_res.forward_and_backward_graph.list_output_data_anodes,
                grad_outs,
            ):
                out = RkMod.compiler.get_val(out_node.main_target)
                # if out_grad.mean() != 0:
                out.grad = out_grad.data.as_strided_(
                    out.shape, out.stride(), out.storage_offset()
                )
                # print(out_node.main_target, out.grad.mean)
                out_grad.data = torch.empty(0)

            if not RkMod.keep_outputs:
                for target in RkMod.output_targets:
                    if hasattr(RkMod.compiler.storage.ld[target], 'data'):
                        RkMod.compiler.storage.ld[target].data = torch.empty(0)
                    if hasattr(RkMod.compiler.storage.ld[target], 'grad'):
                        RkMod.compiler.storage.ld[target].grad = None

            loss_idx = RkMod.op_sched.loss_idx
            #  * record_mem stuff *
            # if RkMod.exec_with_record_mem:
            #     RkMod.output_size = tensor_memory_size(
            #         storage.ld[RkMod.output.main_target]
            #     )
            #     # TODO: record output grad if needed

            with torch.enable_grad():
                stop = RkMod.backward_stop
                if stop:
                    assert False, "Dat: Should not reach here"
                    for i, op in enumerate(
                        RkMod.op_list[loss_idx + 1 : loss_idx + stop + 1]
                    ):
                            # print(f"Executing bwd1 op: {op}")
                            RkMod._exec(op)
                            if os.environ.get('ROCKMATE_RECORD_FRAGMENTATION', '0') == '1' and current_iteration not in iterations_to_print_allocated_memory:
                                fragmentation = get_memory_fragmentation()
                                fragmentations.append(fragmentation)
                                allocated_memory_list.append(torch.cuda.memory_allocated())
                            print_allocated_memory(op)
                else:
                    if RkMod.exec_with_record_time:
                        for step in RkMod.op_sched.steps[RkMod.op_sched.loss_step :]:
                            RkMod.exec_step(step)
                    else:
                        # print(f"bwd2 op list includes:")
                        # for op in RkMod.op_list[loss_idx + 1 :]:
                        #     print(f"  {op}")
                        for op in RkMod.op_list[loss_idx + 1 :]:
                            # print(f"Executing bwd2 op: {op}")
                            RkMod._exec(op)
                            if os.environ.get('ROCKMATE_RECORD_FRAGMENTATION', '0') == '1' and current_iteration not in iterations_to_print_allocated_memory:
                                fragmentation = get_memory_fragmentation()
                                fragmentations.append(fragmentation)
                                allocated_memory_list.append(torch.cuda.memory_allocated())
                            print_allocated_memory(op)
                    # if RkMod.exec_with_record_mem and RkMod.backward_add_output_grad:
                    #     RkMod.allo_mem[loss_idx] += RkMod.output_size
                    #  -> return grad of dummy input + inputs' which req grad (Rem 1)
                    grad_inputs = tuple(
                        RkMod.compiler.get_val(inp).grad
                        for inp in ctx.name_of_inputs_which_req_grad
                    )
                    grads = (torch.ones(1),) + grad_inputs

                    return grads

        # === END OF BACKWARD FUNCTION ===

    return RK_autograd_Function
