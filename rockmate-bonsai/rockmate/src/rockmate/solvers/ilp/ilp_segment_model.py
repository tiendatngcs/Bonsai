from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, FrozenSet

import argparse

from collections import defaultdict
from itertools import combinations, product
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, LpStatusOptimal, value, PULP_CBC_CMD, getSolver, GUROBI_CMD
from tqdm import tqdm

import time
import math

@dataclass
class Node:
    name: str
    id: int = -1
    tensors: List['Tensor'] = field(default_factory=list)
    execution_time: int = 0
    backward_execution_time: int = 0
    # level: int = -1  # topological level in the DAG

    # Forward pass
    allocated_pre_forward: Optional[int] = None
    allocated_post_forward: Optional[int] = None
    input_tensors: List['Tensor'] = field(default_factory=list)
    output_tensors: List['Tensor'] = field(default_factory=list)
    generated_tensors: List['Tensor'] = field(default_factory=list)
    saved_tensors: List['Tensor'] = field(default_factory=list)
    allocated_tensors: List['Tensor'] = field(default_factory=list)
    tmp_tensors: List['Tensor'] = field(default_factory=list)   # Temptensors = allocated tensors - output_tensors

    # Backward pass
    allocated_pre_backward: Optional[int] = None
    allocated_post_backward: Optional[int] = None
    grad_input_tensors: List['Tensor'] = field(default_factory=list)
    grad_output_tensors: List['Tensor'] = field(default_factory=list)
    grad_generated_tensors: List['Tensor'] = field(default_factory=list)
    grad_allocated_tensors: List['Tensor'] = field(default_factory=list)
    grad_tmp_tensors: List['Tensor'] = field(default_factory=list)   # GradTemptensors = grad allocated tensors - grad_output_tensors

    # Graph links
    previous_nodes: List['Node'] = field(default_factory=list)
    next_nodes: List['Node'] = field(default_factory=list)
    
    recursive_previous_nodes: Set['Node'] = field(default_factory=set)
    recursive_next_nodes: Set['Node'] = field(default_factory=set)
    
    # Other properties
    is_large_node: bool = False
    compute_intensity = None
    is_compute_intensive: bool = False
    
    def set_compute_intensity(self, all_saved_tensors: Set['Tensor']):
        # calculate this metric for each node
        # sigma = volume of memory saved and allocated / execution time
        # if sigma is low, the node is compute intensive
        allocated_tensors = set(self.allocated_tensors)
        relevant_tensors = allocated_tensors.intersection(all_saved_tensors)
        total_size = sum(t.size for t in relevant_tensors)
        if self.execution_time == 0:
            self.compute_intensity = float('inf')
            return
        self.compute_intensity = total_size / self.execution_time
    
    def __hash__(self):
        return hash(self.id)
    
    def __str__(self):
        return f"Node(id={self.id}, name={self.name})"
    
    def fwd_temporary_tensors(self) -> List['Tensor']:
        return self.tmp_tensors
    
    def bwd_temporary_tensors(self) -> List['Tensor']:
        return self.grad_tmp_tensors
    
def previous_nodes_recursive(node: Node, visited: Set[Node]) -> Set[Node]:
    if node in visited:
        return set()
    visited.add(node)
    result = set(node.previous_nodes)
    for prev in node.previous_nodes:
        result.update(previous_nodes_recursive(prev, visited))
    return result

def next_nodes_recursive(node: Node, visited: Set[Node]) -> Set[Node]:
    if node in visited:
        return set()
    visited.add(node)
    result = set(node.next_nodes)
    for nxt in node.next_nodes:
        result.update(next_nodes_recursive(nxt, visited))
    return result

def find_large_nodes(nodes: List[Node]): 
    # threshold is at 90th percentile of allocated_size.union(output_size)
    sizes = []
    for node in nodes:
        allocated_tensors = set(node.allocated_tensors)
        output_tensors = set(node.output_tensors)
        relevant_tensors = allocated_tensors.union(output_tensors)
        total_size = sum(t.size for t in relevant_tensors)
        sizes.append(total_size)
        
    sizes.sort()
    threshold_index = int(0.9 * len(sizes))
    threshold_size = sizes[threshold_index]
    for node in nodes:
        allocated_tensors = set(node.allocated_tensors)
        output_tensors = set(node.output_tensors)
        relevant_tensors = allocated_tensors.union(output_tensors)
        total_size = sum(t.size for t in relevant_tensors)
        if total_size >= threshold_size:
            node.is_large_node = True
            
def find_compute_intensive_nodes(nodes):
    # print out the most compute intensive nodes
    compute_intensive_nodes = []
    compute_intensivity_values = []
    all_saved_tensors = set()
    for node in nodes:
        all_saved_tensors.update(node.saved_tensors)
    for node in nodes:
        node.set_compute_intensity(all_saved_tensors)
        print(f"Node ID: {node.id}, Name: {node.name}, Compute Intensity: {node.compute_intensity}")
        compute_intensivity_values.append(node.compute_intensity)
    compute_intensivity_values.sort()
    threshold_index = int(0.1 * len(compute_intensivity_values))
    threshold_value = compute_intensivity_values[threshold_index]
    for node in nodes:
        if node.compute_intensity <= threshold_value:
            node.is_compute_intensive = True
            compute_intensive_nodes.append(node)
    
    # print them out
    print("Most compute intensive nodes:")
    for node in compute_intensive_nodes:
        print(f"Node ID: {node.id}, Name: {node.name}, Compute Intensity: {node.compute_intensity}")
    return compute_intensive_nodes

@dataclass
class Tensor:
    time_stamp: int
    name: str
    size: int
    address: int
    fringe: Optional[Node] = field(default=None, compare=False)
    first_save_node: Optional[Node] = field(default=None, compare=False)
    
    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return (self.size, self.address) == \
               (other.size, other.address)

    def __hash__(self):
        return hash((self.size, self.address))
    
    def __str__(self):
        return f"Tensor(name={self.name}, size={self.size}, address={self.address})"
    
    
def persisting_tensors(nodes: List[Node]) -> List[Tensor]:
    # persisting tensors are set of tensors exists before the iteration execution.
    # persisting tensors are not deallocated in the bacward pass after it is used.
    # one example of persisting tensor is the input image tensor, the weight tensors.
    persisting_tensors = []
    for node in nodes:
        if len(node.previous_nodes) == 0:
            persisting_tensors.extend(node.input_tensors)
    return persisting_tensors

@dataclass
class Segment:
    nodes: List[Node]
    nodes_set: Set[Node] = field(init=False)
    # lvl_start: int = -1
    # lvl_end: int = -1
    node_ids: List[int] = field(default=None)
    name: str = field(default=None)
    execution_time: int = field(default=None)
    input_tensors: List[Tensor] = field(default=None)
    _input_size: int = field(default=None)
    # saved_size_cache: Dict[int, int] = field(default_factory=dict)

    select: LpVariable = field(init=False)
    cp: LpVariable = field(init=False)
    z: LpVariable = field(init=False)  # selected but not checkpointed

    _saved_tensors: List[Tensor] = field(default_factory=list)
    
    def __post_init__(self):
        self.nodes = sorted(self.nodes, key=lambda node: node.id)
        self.nodes_set = set(self.nodes)
        # self.lvl_start = min(node.level for node in self.nodes)
        # self.lvl_end = max(node.level for node in self.nodes)
        self.node_ids = [node.id for node in self.nodes]
        self.execution_time = sum(node.execution_time for node in self.nodes)
        node_ids_tuple = tuple(self.node_ids)
        # self.name = "_".join(str(nid) for nid in self.node_ids)
        self.name = str(hash(node_ids_tuple))
        # Collect saved tensors from all nodes in the segment
        saved_tensor_set = set()
        p_tensors = persisting_tensors(self.nodes)
        for node in self.nodes:
            if node not in p_tensors:
                saved_tensor_set.update(node.saved_tensors)
        self._saved_tensors = list(saved_tensor_set)
        
        self.select = LpVariable(f'select_{self.name}', cat='Binary')
        self.cp = LpVariable(f'cp_{self.name}', cat='Binary')
        self.z = LpVariable(f'z_{self.name}', cat='Binary') # selected but not checkpointed

        self.input_tensors = self.find_input_tensors()
        self._input_size = sum(t.size for t in self.input_tensors)

        self.lineage_set_to_tensors = {}

        # self.saved_size_cache = {}

    def __str__(self):
        node_ids = [node.id for node in self.nodes]
        return f"Segment(nodes={node_ids})"
    
    def saved_tensors(self, current_node:Node) -> Set[Tensor]:
        if current_node is None:
            saved_tensors_set = set()
            for node in self.nodes:
                saved_tensors_set.update(node.saved_tensors)
            return saved_tensors_set
        # previous_set = current_node.recursive_previous_nodes.union({current_node})
        next_set = current_node.recursive_next_nodes
        assert current_node not in next_set, f"Current node {current_node.id} should not be in its own next_set."
        # lineage_set = frozenset(self.nodes_set.intersection(previous_set))
        executed_set = frozenset(self.nodes_set.difference(next_set))
        # if lineage_set in self.lineage_set_to_tensors:
        #     return self.lineage_set_to_tensors[lineage_set]
        if executed_set in self.lineage_set_to_tensors:
            return self.lineage_set_to_tensors[executed_set]
        saved_tensors = set()
        # for node in lineage_set:
        #     saved_tensors.update(node.saved_tensors)
        # self.lineage_set_to_tensors[lineage_set] = saved_tensors
        for node in executed_set:
            saved_tensors.update(node.saved_tensors)
        self.lineage_set_to_tensors[executed_set] = saved_tensors
        return saved_tensors        # return lineage_set
        # if self.nodes_set.isdisjoint(previous_set):
        #     return []
        # else:
        #     # parts of the current segment is on the lineage of current_node
            
        # if current_node.level < self.lvl_start:
        #     return []
        # elif current_node.level > self.lvl_end:
        #     return self._saved_tensors
        # else:
        #     # return union of saved tensors of nodes in the segment up to and including current_node
        #     saved_tensor_set = set()
        #     p_tensors = persisting_tensors(self.nodes)
        #     for node in self.nodes:
        #         if node.level <= current_node.level:
        #             saved_tensor_set.update(node.saved_tensors)
        #     # exclude tensors from persisting tensors
        #     saved_tensor_set.difference_update(p_tensors)
        #     return list(saved_tensor_set)
        
    def saved_size(self, current_node:Node, exclude=set()) -> int:
        if current_node is None:
            tensors_after_exclusion = self.saved_tensors(current_node=None).difference(exclude)
            return sum(t.size for t in tensors_after_exclusion)
        
        # if current_node.id in self.saved_size_cache:
        #     return self.saved_size_cache[current_node.id]
        tensors_after_exclusion = self.saved_tensors(current_node).difference(exclude)
        total = sum(t.size for t in tensors_after_exclusion)
        # print(f"Segment {self.name} saved_size at Node {current_node.id}: {total}")
        # self.saved_size_cache[current_node.id] = total
        return total
    
    def saved_diff_input_size(self, current_node:Node) -> int:
        if current_node is None:
            saved_tensors = self.saved_tensors(current_node=None)
            input_tensors = self.input_tensors
            saved_diff_input = set(saved_tensors).difference(set(input_tensors))
            return sum(t.size for t in saved_diff_input)
        # if current_node.id in self.saved_size_cache:
        #     return self.saved_size_cache[current_node.id]
        saved_tensors = self.saved_tensors(current_node)
        input_tensors = self.input_tensors
        saved_diff_input = set(saved_tensors).difference(set(input_tensors))
        total = sum(t.size for t in saved_diff_input)
        # print(f"Segment {self.name} saved_size at Node {current_node.id}: {total}")
        # self.saved_size_cache[current_node.id] = total
        return total
    
    def input_tensors(self) -> List[Tensor]:
        input_tensor_set = set()
        p_tensors = persisting_tensors(self.nodes)
        minimum_level = min(node.level for node in self.nodes)
        for node in self.nodes:
            if node.level == minimum_level:
                input_tensor_set.update(node.input_tensors)
        # exclude tensors from persisting tensors
        input_tensor_set.difference_update(p_tensors)
        return list(input_tensor_set)
    
    def fwd_lineage_node_set(self, current_node:Node) -> FrozenSet[Node]:
        previous_set = current_node.recursive_previous_nodes.union({current_node})
        lineage_set = frozenset(self.nodes_set.intersection(previous_set))
        return lineage_set
    
    def bwd_lineage_node_set(self, current_node:Node) -> FrozenSet[Node]:
        next_set = current_node.recursive_next_nodes.union({current_node})
        lineage_set = frozenset(self.nodes_set.intersection(next_set))
        return lineage_set
    
    def fwd_intermediate_tensors(self, future_nodes: List[Node], current_node = None) -> Set[Tensor]:
        # return this segment's generated tensors that are used by future nodes
        to_be_used_tensors = set()
        for node in future_nodes:
            for t in node.input_tensors:
                to_be_used_tensors.add(t)
        if not current_node:
            segment_generated_tensors = set()
            for node in self.nodes:
                segment_generated_tensors.update(node.generated_tensors)
            return to_be_used_tensors.intersection(segment_generated_tensors)
        
        previous_set = current_node.recursive_previous_nodes.union({current_node})
        lineage_set = frozenset(self.nodes_set.intersection(previous_set))
        segment_generated_tensors = set()
        for node in lineage_set:
            segment_generated_tensors.update(node.generated_tensors)
        return to_be_used_tensors.intersection(segment_generated_tensors)

    
    def bwd_intermediate_tensors(self, future_nodes: List[Node], current_node=None) -> Set[Tensor]:
        # return this segment's grad_generated tensors that are used by future nodes
        to_be_used_tensors = set()
        for node in future_nodes:
            for t in node.grad_input_tensors:
                to_be_used_tensors.add(t)
        if not current_node:
            segment_grad_generated_tensors = set()
            for node in self.nodes:
                segment_grad_generated_tensors.update(node.grad_generated_tensors)
            return to_be_used_tensors.intersection(segment_grad_generated_tensors)
        
        next_set = current_node.recursive_next_nodes.union({current_node})
        lineage_set = frozenset(self.nodes_set.intersection(next_set))
        segment_grad_generated_tensors = set()
        for node in lineage_set:
            segment_grad_generated_tensors.update(node.grad_generated_tensors)
        return to_be_used_tensors.intersection(segment_grad_generated_tensors)
    
    def grad_generated_tensors(self, current_node) -> Set[Tensor]:
        grad_generated_tensors = set()
        lineage_set = self.bwd_lineage_node_set(current_node)
        for node in lineage_set:
            grad_generated_tensors.update(node.grad_generated_tensors)
        # assert(len(grad_generated_tensors) > 0), f"Segment {self.name} has no grad_generated_tensors at Node {current_node.id}."
        return grad_generated_tensors

    def root_nodes(self) -> List[Node]:
        return [node for node in self.nodes if len(node.previous_nodes) == 0 or any(prev not in self.nodes_set for prev in node.previous_nodes)]
    
    def output_nodes(self) -> List[Node]:
        return [node for node in self.nodes if len(node.next_nodes) == 0 or all(nxt not in self.nodes_set for nxt in node.next_nodes)]

    def find_input_tensors(self) -> List[Tensor]:
        # from self.nodes, find all root nodes (i.e., nodes with no previous_nodes in self.nodes)
        roots = self.root_nodes()
        input_tensors = set()
        for root in roots:
            input_tensors.update(root.input_tensors)

        # generated_tensors = set()  # set of all tensors outputed by all nodes in this segment
        # for node in self.nodes:
        #     generated_tensors.update(node.generated_tensors)
        
        # input tensors are those inputed to root nodes but not outputed by any node in this segment
        # input_tensors.difference_update(output_tensors)
        # new_input_tensors = input_tensors.difference(generated_tensors)
        # if len(new_input_tensors) == 0:
        #     print(f"Debug Segment {self.name}:")
        #     print("  No input tensors found.")
        #     print(f"  Input Tensors: {[t.address for t in input_tensors]}")
            # print(f"  Generated Tensors: {[t.address for t in generated_tensors]}")

        # assert(len(new_input_tensors) > 0), f"Segment {self.name} has no input tensors."
        return list(input_tensors)

    def input_size(self, exclude=set()) -> int:
        # return sum(t.size for t in self.input_tensors)
        if exclude:
            tensors_after_exclusion = set(self.input_tensors).difference(exclude)
            return sum(t.size for t in tensors_after_exclusion)
        return self._input_size

    def __eq__(self, other):
        if not isinstance(other, Segment):
            return NotImplemented
        return self.name == other.name
        # return len(self.nodes) == len(other.nodes) and all(n1.id == n2.id for n1, n2 in zip(self.nodes, other.nodes))
    def __hash__(self):
        return hash(self.name)

def parse_trace_with_links(file_path: str) -> List[Node]:
    nodes: List[Node] = []
    bwd_nodes: List[Node] = []
    tensor_pool: Dict[Tuple[int, int], Tensor] = {}
    tensor_producers: Dict[Tuple[int, int], Node] = {}
    nodes_by_id: Dict[int, Node] = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_node = None
    current_phase = "forward"  # forward or backward
    for line in lines:
        line = line.strip()
        if line.startswith("------>"):
            current_node = Node(name="")
        elif line.startswith("<------"):
            if current_node:
                if current_phase == "forward":
                    for p_tensor in current_node.input_tensors:
                        if p_tensor.fringe and p_tensor.fringe != current_node:
                            if p_tensor.fringe not in current_node.previous_nodes:
                                current_node.previous_nodes.append(p_tensor.fringe)
                            if current_node not in p_tensor.fringe.next_nodes:
                                p_tensor.fringe.next_nodes.append(current_node)
                            
                    # update fringe to current node
                    for o_tensor in current_node.output_tensors:
                        o_tensor.fringe = current_node
                
                # Generated tensors are outputs that are not inputs
                if len(current_node.generated_tensors) == 0: current_node.generated_tensors = [t for t in current_node.output_tensors if t not in current_node.input_tensors]
                if len(current_node.grad_generated_tensors) == 0: current_node.grad_generated_tensors = [t for t in current_node.grad_output_tensors if t not in current_node.grad_input_tensors]
                if len(current_node.tmp_tensors) == 0: current_node.tmp_tensors = [t for t in current_node.allocated_tensors if t not in current_node.output_tensors]
                if len(current_node.grad_tmp_tensors) == 0: current_node.grad_tmp_tensors = [t for t in current_node.grad_allocated_tensors if t not in current_node.grad_output_tensors]
                    
                if current_phase == "forward": nodes.append(current_node)
                else: bwd_nodes.append(current_node)
                current_node = None
        elif line.startswith("Execution Time"):
            if current_phase == "forward":
                current_node.execution_time = int(line.split(":")[1])
            else:
                current_node.backward_execution_time = int(line.split(":")[1])
        elif current_node is not None:
            if not current_node.name:
                _, id = line.split('-')
                id = int(id)
                if id in nodes_by_id.keys():
                    current_node = nodes_by_id[id]
                    current_phase = "backward"
                else:
                    current_node.name = line
                    current_node.id = id
                    nodes_by_id[id] = current_node
            elif line.startswith(">Allocated[PRE]:"):
                if current_phase == "forward":
                    current_node.allocated_pre_forward = int(line.split(":")[1])
                else:
                    current_node.allocated_pre_backward = int(line.split(":")[1])
            elif line.startswith(">Allocated[POST]:"):
                if current_phase == "forward":
                    current_node.allocated_post_forward = int(line.split(":")[1])
                else:
                    current_node.allocated_post_backward = int(line.split(":")[1])
            #     current_node.allocated_pre = int(line.split(":")[1])
            # elif line.startswith(">Allocated[POST]:"):
            #     current_node.allocated_post = int(line.split(":")[1])
            else:
                parts = line.split(',')
                if len(parts) != 5:
                    continue
                time_stamp = int(parts[0])
                status = parts[1]
                name = parts[2]
                size = int(parts[3])
                address = int(parts[4])
                key = (size, address)

                if key not in tensor_pool:
                    tensor_pool[key] = Tensor(time_stamp, name, size, address)
                tensor = tensor_pool[key]
                current_node.tensors.append(tensor)

                # Track producer relationships
                if status == "output":
                    # tensor.fringe = current_node
                    current_node.output_tensors.append(tensor)
                    tensor_producers[key] = current_node
                elif status == "input":
                    # producer = tensor_producers.get(key)
                    current_node.input_tensors.append(tensor)
                elif status == "save":
                    current_node.saved_tensors.append(tensor)
                    if tensor.first_save_node is None:
                        tensor.first_save_node = current_node
                elif status == "allocated":
                    if current_phase == "forward":
                        current_node.allocated_tensors.append(tensor)
                    else:
                        current_node.grad_allocated_tensors.append(tensor)
                elif status == "grad_input":
                    current_node.grad_input_tensors.append(tensor)
                elif status == "grad_output":
                    current_node.grad_output_tensors.append(tensor)
    
    for node in nodes:
        node.recursive_previous_nodes = previous_nodes_recursive(node, set())
        node.recursive_next_nodes = next_nodes_recursive(node, set())
    return nodes, bwd_nodes

def print_parsed_trace(nodes: List[Node]):
    for node in nodes:
        print(f"Node: {node.name}, FWD Allocated: {node.allocated_pre_forward} - {node.allocated_post_forward}, BWD Allocated: {node.allocated_pre_backward} - {node.allocated_post_backward}, PrevCount: {len(node.previous_nodes)}, NextCount: {len(node.next_nodes)}, SaveCount: {len(node.saved_tensors)}")
        # for tensor in node.tensors:
        #     print(f"  Tensor: time_stamp={tensor.time_stamp}, name={tensor.name}, size={tensor.size}, address={tensor.address}")
        print()

def simulate_memory_time_series(nodes: List[Node], bwd_nodes: List[Node], checkpointed_segments: List[Segment], starting_mem=0) -> Tuple[List[int], List[int]]:
    forward_series = []
    backward_series = []
    saved_tensors_list = []
    execution_time = 0
    persisting_tensor_list = persisting_tensors(nodes)
    mem = starting_mem
    
    segment_is_executed = dict()
    for seg in checkpointed_segments:
        segment_is_executed[seg] = False
    
    def execute_segment_forward(seg: Segment, with_grad: bool):
        nonlocal mem
        nonlocal execution_time
        print(f"Executing segment {seg.name} forward, with_grad={with_grad}")
        regional_saved_tensors_list = []
        
        if not with_grad:
            input_tensors = seg.input_tensors
            for t in input_tensors:
                if t not in saved_tensors_list:
                    saved_tensors_list.extend(input_tensors)
                if t not in regional_saved_tensors_list:
                    regional_saved_tensors_list.extend(input_tensors)
        if with_grad:
            # remove input tensor after recomputation
            for t in seg.input_tensors:
                if t in regional_saved_tensors_list:
                    regional_saved_tensors_list.remove(t)
                if t in saved_tensors_list:
                    saved_tensors_list.remove(t)

        for i, node in enumerate(seg.nodes):  
            execution_time += node.execution_time
            if not with_grad:
                forward_series.append(mem)  # memory before execution
            else:
                backward_series.append(mem)  # memory before execution
            output_size = sum(t.size for t in node.generated_tensors)
            mem += output_size
            if with_grad:
                # during the backward pass
                for t in node.saved_tensors:
                    if t not in regional_saved_tensors_list: 
                        regional_saved_tensors_list.append(t)
                        # print(f"Step {i} Saving tensor {t}, saved {len(saved_tensors_list)} tensors")
                
            if not with_grad:
                forward_series.append(mem)  # memory after execution
            else:
                backward_series.append(mem)  # memory after execution

            # for each previous Node, of this Node, if their inputs are not saved or is not outputed, remove them from memory
            for prev_node in node.previous_nodes:
                for t in prev_node.generated_tensors:
                    if (t not in saved_tensors_list and t not in regional_saved_tensors_list) and t not in node.output_tensors:
                        # print(f"Step {i} Removing tensor {t}")
                        mem -= t.size
                        # print(f"Segment Saving tensor {t}, saved {len(saved_tensors_list)} tensors")
        
        
        return regional_saved_tensors_list

    def execute_segment_backward(seg: Segment, regional_saved_tensors_list: List[Tensor]=[]):
        nonlocal mem
        nonlocal saved_tensors_list
        nonlocal execution_time
        # saved_tensors_list = regional_saved_tensors_list
        print(f"Executing segment {seg.name} backward")
        for i, node in enumerate(seg.nodes[::-1]):
            execution_time += node.backward_execution_time
            backward_series.append(mem)  # memory before execution
            grad_output_size = sum(t.size for t in node.grad_generated_tensors)
            mem += grad_output_size
            backward_series.append(mem)  # memory after execution
            
            # for each next Node of this Node, if their grad_inputs are not grad_outputed, remove them from memory
            for next_node in node.next_nodes:
                # print(f"Next Node of {node.id} is {next_node.id}")
                for t in next_node.grad_generated_tensors:
                    if t not in saved_tensors_list and t not in regional_saved_tensors_list and t not in node.grad_output_tensors:
                        # print(f"BWD Step {node.id} Removing tensor {t}")
                        mem -= t.size
                        
            # also, remove saved tensors of this node
            for t in node.saved_tensors:
                if t.first_save_node == node and (t in saved_tensors_list or t in regional_saved_tensors_list) and t not in persisting_tensor_list:
                    # print(f"BWD Step {node.id} Removing saved tensor {t}")
                    mem -= t.size
                    if t in saved_tensors_list:
                        saved_tensors_list.remove(t)
                    else:
                        regional_saved_tensors_list.remove(t)
                    
        for tensor in regional_saved_tensors_list:
            if tensor not in persisting_tensor_list:
                mem -= tensor.size
                if t in saved_tensors_list:
                    saved_tensors_list.remove(t)
                elif t in regional_saved_tensors_list:
                    regional_saved_tensors_list.remove(t)

    # Forward Pass Simulation
    
    for i, node in enumerate(nodes):
        if any(node in seg.nodes for seg in checkpointed_segments):
            segment_containing_node = None
            for seg in checkpointed_segments:
                if node in seg.nodes:
                    segment_containing_node = seg
                    break
            if not segment_is_executed[segment_containing_node]:
                execute_segment_forward(segment_containing_node, with_grad=False)
                segment_is_executed[segment_containing_node] = True
            else:
                pass  # already executed, skip
        else:
            execution_time += node.execution_time
            print(f"Memory usage at Node {node.id}: {mem} bytes")
            forward_series.append(mem)  # memory before execution
            output_size = sum(t.size for t in node.generated_tensors)
            mem += output_size
            for t in node.saved_tensors:
                if t not in saved_tensors_list: 
                    saved_tensors_list.append(t)
                    # print(f"Step {i} Saving tensor {t}, saved {len(saved_tensors_list)} tensors")
            forward_series.append(mem)  # memory after execution

            # for each previous Node, of this Node, if their inputs are not saved or is not outputed, remove them from memory
            for prev_node in node.previous_nodes:
                for t in prev_node.generated_tensors:
                    if t not in saved_tensors_list and t not in node.output_tensors:
                        # print(f"Step {i} Removing tensor {t}")
                        mem -= t.size
            # Drop outputs not in saved_tensors
            # saved_addrs = {t.address for t in node.saved_tensors}
            # unsaved_outputs = [t for t in node.output_tensors if t.address not in saved_addrs]
            # mem -= sum(t.size for t in unsaved_outputs)

    for key in segment_is_executed.keys():
        segment_is_executed[key] = False
    
    # Backward Pass Simulation (reverse order)
    # mem = 0
    for i, node in enumerate(bwd_nodes):
        if any(node in seg.nodes for seg in checkpointed_segments):
            segment_containing_node = None
            for seg in checkpointed_segments:
                if node in seg.nodes:
                    segment_containing_node = seg
                    break
                
            if not segment_is_executed[segment_containing_node]:
                regional_saved_tensors_list = execute_segment_forward(segment_containing_node, with_grad=True)
                execute_segment_backward(segment_containing_node, regional_saved_tensors_list)
                segment_is_executed[segment_containing_node] = True
            else:
                pass
        else:
            execution_time += node.backward_execution_time
            backward_series.append(mem)  # memory before execution
            grad_output_size = sum(t.size for t in node.grad_generated_tensors)
            mem += grad_output_size
            backward_series.append(mem)  # memory after execution
            
            # for each next Node of this Node, if their grad_inputs are not grad_outputed, remove them from memory
            for next_node in node.next_nodes:
                # print(f"Next Node of {node.id} is {next_node.id}")
                for t in next_node.grad_generated_tensors:
                    if t not in saved_tensors_list and t not in node.grad_output_tensors:
                        # print(f"BWD Step {node.id} Removing tensor {t}")
                        mem -= t.size
                        
            # also, remove saved tensors of this node
            for t in node.saved_tensors:
                if t.first_save_node == node and t in saved_tensors_list and t not in persisting_tensor_list:
                    # print(f"BWD Step {node.id} Removing saved tensor {t}")
                    mem -= t.size
                    saved_tensors_list.remove(t)
            
        # Drop inputs and saved tensors
        # drop_size = sum(t.size for t in node.input_tensors + node.saved_tensors)
        # mem -= drop_size
    print(f"Total execution time: {execution_time} us")
    return forward_series, backward_series


def simulate_memory_time_series2(nodes: List[Node], bwd_nodes: List[Node], checkpointed_segments: List[Segment], starting_mem=0) -> Tuple[List[int], List[int]]:
    forward_series = []
    backward_series = []
    execution_time = 0
    mem = starting_mem
    
    segments = []
    current_segment_nodes = []

    checkpointed_nodes = [node for seg in checkpointed_segments for node in seg.nodes]

    for i, node in enumerate(nodes):
        current_mem = 0
        for seg in segments:
            if seg in checkpointed_segments:
                current_mem += seg.input_size()
            else:
                current_mem += seg.saved_size(node)

        if node not in checkpointed_nodes:
            current_segment_nodes.append(node)
            current_segment = Segment(nodes=current_segment_nodes)

            current_mem += current_segment.saved_size(node)
        else:
            if current_segment_nodes:
                segments.append(Segment(nodes=current_segment_nodes))
                current_segment_nodes = []
            
            # node is in a checkpointed segment
            # get the segment containing node 
            segment_containing_node = None
            for seg in checkpointed_segments:
                if node in seg.nodes:
                    segment_containing_node = seg
                    break

            if segment_containing_node.nodes[-1] == node:
                segments.append(segment_containing_node)
            
            current_mem += segment_containing_node.saved_size(node) if segment_containing_node else 0
        
        segments_including_current = segments + ([Segment(nodes=current_segment_nodes)] if current_segment_nodes else [])
        saved_tensors_set = get_saved_tensors(segments_including_current, node)
        generated_size = get_generated_size(node, saved_tensors_set)
        current_mem += generated_size
        
        forward_series.append(current_mem)
        execution_time += node.execution_time

    segments.append(Segment(nodes=current_segment_nodes)) if current_segment_nodes else None
    
    
    for i, node in enumerate(bwd_nodes):
        current_mem = 0
        for seg in segments:
            if seg in checkpointed_segments:
                current_mem += seg.input_size()
            else:
                current_mem += seg.saved_size(node)
                
        segment_containing_node = None
        for seg in segments:
            if node in seg.nodes:
                segment_containing_node = seg
                break
        
        assert segment_containing_node is not None, f"Node {node.id} not found in any segment during backward pass."
        
        if node in checkpointed_nodes:
            current_mem += segment_containing_node.saved_size(node)
        
        if segment_containing_node.nodes[0] == node:
            segments.remove(segment_containing_node)

        grad_generated_size = get_grad_generated_size(node)
        current_mem += grad_generated_size
        
        backward_series.append(current_mem)
        execution_time += node.backward_execution_time
    
    return forward_series, backward_series, execution_time

def get_all_non_empty_sublists_itertools(input_list):
    sublists = []
    for i in range(1, len(input_list) + 1):  # Iterate from length 1 to full list length
        for combo in combinations(input_list, i):
            sublists.append(list(combo))
    return sublists

def is_correct_doubly_linked_graph(nodes: List[Node]) -> bool:
    """
    Ensure that previous_nodes and next_nodes are consistent.
    """
    node_set = set(nodes)
    for node in nodes:
        for prev in node.previous_nodes:
            if node not in prev.next_nodes:
                return False
        for nxt in node.next_nodes:
            if node not in nxt.previous_nodes:
                return False
    return True
        # # Remove any previous_nodes or next_nodes that are not in the original node list
        # node.previous_nodes = [prev for prev in node.previous_nodes if prev in node_set]
        # node.next_nodes = [nxt for nxt in node.next_nodes if nxt in node_set]

def is_dag(nodes: List[Node]) -> bool:
    """
    Check if the graph of nodes is a DAG using DFS.
    Each node should have `previous_nodes` pointing to parents.
    """
    visited = set()
    visiting = set()

    def dfs(node: Node) -> bool:
        if node in visiting:
            return False  # found a cycle
        if node in visited:
            return True   # already processed

        visiting.add(node)
        for prev in node.previous_nodes:
            if not dfs(prev):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(dfs(node) for node in nodes)

def is_connected(nodes: List[Node]) -> bool:
    if not nodes:
        return True
    visited = set()

    def dfs(node: Node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in node.previous_nodes + node.next_nodes:
            dfs(neighbor)

    dfs(nodes[0])
    return len(visited) == len(nodes)


def node_and_users(node: Node) -> Set[Node]:
    node_set = set([node])
    
    def node_and_users_recursive(node):
        nonlocal node_set
        for next_node in node.next_nodes:
            if next_node not in node_set:
                node_set.add(next_node)
                node_and_users_recursive(next_node)
    for next_node in node.next_nodes:
        if next_node not in node_set:
            node_set.add(next_node)
            node_and_users_recursive(next_node)

    return node_set

def generate_segments2(nodes: List[Node], budget_bytes: int) -> List[Segment]:
    if not is_dag(nodes):
        raise ValueError("The graph contains a cycle. Only DAGs are supported.")
    if not is_correct_doubly_linked_graph(nodes):
        raise ValueError("The graph's previous_nodes and next_nodes are not consistent.")
    if not is_connected(nodes):
        raise ValueError("The graph is not connected. Please provide a connected graph.")
    print("Generating segments using optimized recursive method (with tqdm).")
    segments: List[Segment] = []
    seen: Set[FrozenSet[Node]] = set()
    recursed: Set[FrozenSet[Node]] = set()

    # Progress bar tracks number of unique segments discovered
    pbar = tqdm(total=None, desc="Segments found", dynamic_ncols=True)
    # pbar2 = tqdm(total=None, desc="Current number of nodes processed", dynamic_ncols=True)
    
    def is_valid_segment(node_set: FrozenSet[Node]) -> bool:
        nonlocal nodes
        nonlocal budget_bytes
        # valid when number of nodes is smaller than total nodes / 2
        # and total size of saved tensors are smaller than the budget
        if len(node_set) > math.sqrt(len(nodes)) * 2:
            return False
        saved_tensors = set()
        for node in node_set:
            saved_tensors.update(node.saved_tensors)
        saved_size = sum(t.size for t in saved_tensors)
        if saved_size >= budget_bytes:
            return False
        return True

    def add_segment(node_set: FrozenSet[Node]):
        if node_set not in seen:
            seen.add(node_set)
            segments.append(Segment(nodes=sorted(node_set, key=lambda n: n.id)))
            pbar.update(1)  # update progress bar whenever a new unique segment is added

    def recurse(current_nodes: FrozenSet[Node], recurse_from: List[Node], depth = 0):
        if len(current_nodes) != 1 and current_nodes in recursed:
            return
        last_node = max(current_nodes, key=lambda n: n.id)
        combos = get_all_non_empty_sublists_itertools(recurse_from)
        pbar.set_postfix(seg_size=f"{len(current_nodes)}", n_combos=f"{len(combos)}", depth=f"{depth}", segments_found=f"{len(segments)}")
        
        current_node_and_users_set = node_and_users(last_node).union(current_nodes)
        for combo in combos:
            next_recurse_from = set()
            for node in combo:
                next_recurse_from.update(node.previous_nodes)
                
            # all next nodes from next_recurse_from must be in current_nodes
            # next_from_next_recurse_from = [n for node in next_recurse_from for n in node.next_nodes if n not in current_nodes]
            # if len(next_from_next_recurse_from) > 0:
            #     continue
            next_recurse_from = {node for node in next_recurse_from
                     if all(n in current_node_and_users_set for n in node.next_nodes)}

            if not next_recurse_from:
                continue

            next_current_nodes = current_nodes | next_recurse_from
            frozen_next = frozenset(next_current_nodes)

            if is_valid_segment(frozen_next):
                add_segment(frozen_next)
                recurse(frozen_next, list(next_recurse_from), depth + 1)
        recursed.add(current_nodes)

    # Outer loop progress (per starting node)
    for node in tqdm(sorted(nodes, key=lambda n: n.id, reverse=True),
                     desc="Root nodes", dynamic_ncols=True):
        single = frozenset([node])
        add_segment(single)
        recurse(single, [node])

    pbar.close()
    return segments

def filter_large_segments(segments: List[Segment], budget) -> List[Segment]:
    # filter out segments that has saved size larger than (or equal to) the budget.
    filtered_segments = []
    for segment in segments:
        if segment.saved_size(current_node=None) < budget:
            filtered_segments.append(segment)
    print(f"Filtered segments from {len(segments)} to {len(filtered_segments)} using budget {budget/(1024*1024):.2f} MB.")
    return filtered_segments

# def filter_compute_intensive_segments(segments: List[Segment]) -> List[Segment]:
#     # segments that has 3 or more compute-intensive nodes are filtered out
#     filtered_segments = []
#     for segment in segments:
#         compute_intensive_nodes = [node for node in segment.nodes if node.is_compute_intensive]
#         all_outputs_are_compute_intensive = all(node.is_compute_intensive for node in segment.output_nodes())
#         if len(compute_intensive_nodes) < 2 and not all_outputs_are_compute_intensive:
#             filtered_segments.append(segment)
#     print(f"Filtered segments from {len(segments)} to {len(filtered_segments)}.")
#     return filtered_segments

def filter_segments_simple(segments: List[Segment], nodes: List[Node]) -> List[Segment]:
    # filter out segments that has more than half the total number of nodes
    # filter out segments of two or more nodes and all nodes are compute intensive
    filtered_segments = []
    num_nodes = len(nodes)
    for segment in segments:
        is_long_segment = len(segment.nodes) > num_nodes / 2
        all_nodes_compute_intensive = all(node.is_compute_intensive for node in segment.nodes)
        if not is_long_segment and not all_nodes_compute_intensive:
            filtered_segments.append(segment)
    print(f"Filtered segments from {len(segments)} to {len(filtered_segments)} using simple criteria.")
    return filtered_segments

def print_segments(segments: List[Segment]):
    # write to file named "segments.txt"
    with open("segments.txt", "w") as f:
        for i, segment in enumerate(segments):
            f.write(f"Segment {i}: {segment}\n")

def get_saved_generated_size(segments: List[Segment], current_node: Node) -> Tuple[int, int]:
    saved_tensors = []
    generated_tensors = []
    for segment in segments:
        saved_tensors.extend(segment.saved_tensors(current_node))
    saved_tensors = list(set(saved_tensors))
        
    generated_tensors.extend(current_node.generated_tensors)
    for prev_node in current_node.previous_nodes:
        generated_tensors.extend(prev_node.generated_tensors)
    generated_tensors = list(set(generated_tensors))
    generated_tensors = [t for t in generated_tensors if t not in saved_tensors]
    saved_tensors_size = sum(t.size for t in saved_tensors)
    generated_tensors_size = sum(t.size for t in generated_tensors)
    return saved_tensors_size, generated_tensors_size

def get_saved_tensors(segments: List[Segment], current_node: Node) -> Set[Tensor]:
    nodes_set = set()
    for segment in segments:
        nodes_set = nodes_set.union(segment.nodes_set)

    tensor_set = set()
    for node in nodes_set:
        if node.id <= current_node.id:
            tensor_set = tensor_set.union(node.saved_tensors)

    return set(tensor_set)

def get_generated_tensors_fwd(nodes: List[Node]) -> Set[Tensor]:
    tensor_set = set()
    for node in nodes:
        tensor_set = tensor_set.union(node.generated_tensors)
    return tensor_set

def get_generated_tensors_bwd(nodes: List[Node]) -> Set[Tensor]:
    tensor_set = set()
    for node in nodes:
        tensor_set = tensor_set.union(node.grad_generated_tensors)
    return tensor_set

def get_to_be_used_tensors_fwd(nodes: List[Node]):
    tensor_set = set()
    for node in nodes:
        tensor_set = tensor_set.union(node.input_tensors)
    return tensor_set

def get_to_be_used_tensors_bwd(nodes: List[Node]):
    tensor_set = set()
    for node in nodes:
        tensor_set = tensor_set.union(node.grad_input_tensors)
    return tensor_set

def get_generated_size(current_node: Node, saved_tensors: Set[Tensor]) -> int:
    generated_tensors = []
    generated_tensors.extend(current_node.generated_tensors)
    for prev_node in current_node.previous_nodes:
        generated_tensors.extend(prev_node.generated_tensors)
    generated_tensors = list(set(generated_tensors))
    generated_tensors = [t for t in generated_tensors if t not in saved_tensors]
    generated_tensors_size = sum(t.size for t in generated_tensors)
    return generated_tensors_size

def get_grad_generated_size(current_node: Node) -> int:
    generated_tensors = []
    generated_tensors.extend(current_node.grad_generated_tensors)
    for next_node in current_node.next_nodes:
        generated_tensors.extend(next_node.grad_generated_tensors)
    generated_tensors = list(set(generated_tensors))
    generated_tensors = [t for t in generated_tensors if t in current_node.grad_input_tensors]
    generated_tensors_size = sum(t.size for t in generated_tensors)
    return generated_tensors_size

def get_solution_file_name(trace_file_path: str, memory_limit: int, mode: str) -> str:
    base_name = trace_file_path.split('/')[-1].split('.')[0]
    base_name = base_name.replace("operator_trace", "segment_ilp_solution")
    
    folder_name = "../cp_solution_pkl"
    # folder must already exist
    assert os.path.exists(folder_name), f"Folder {folder_name} does not exist."
    
    output_file = f"{folder_name}/{base_name}_{memory_limit}_{mode}.pkl"
    return output_file

def save_solution_to_file(nodes: List[Node], bwd_nodes: List[Node], segments: List[Segment], checkpointed_segments: List[Segment], output_file: str):
    # put all objects under one dictionary object
    save_obj = {
        "nodes": nodes,
        "bwd_nodes": bwd_nodes,
        "segments": segments,
        "checkpointed_segments": checkpointed_segments,
    }
    
    # save the dictionary object as a .pkl
    with open(output_file, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"Saved solution to {output_file}")
    
def optimizer_to_scaling_factor(optimizer: str) -> float:
    if optimizer.lower() == "adam":
        return 2.0
    elif optimizer.lower() == "sgd":
        return 1.0
    return 1.0

class CheckpointSolver:
    def __init__(self, nodes: List[Node], bwd_nodes: List[Node], memory_limit: int, starting_mem=0, weight_MB: float=0.0, optimizer: str="adam"):
        self.nodes = nodes
        self.bwd_nodes = bwd_nodes
        self.memory_limit = memory_limit
        self.starting_mem = starting_mem
        self.grad_mem = starting_mem
        # self.optimizer_state = starting_mem
        self.weight_MB = weight_MB
        self.weight_bytes = int(weight_MB * 1024 * 1024)
        self.immediate_gradients = self.weight_bytes
        self.optimizer_state = self.weight_bytes * optimizer_to_scaling_factor(optimizer)
        self.rolling_gradients = self.weight_bytes
        
        # Initial mem includes the dynamic gradients
        # Initial mem can also be used to cover for optimizer state, which scales with static weights
        static_memory_usage = self.weight_bytes + self.immediate_gradients + self.optimizer_state + self.rolling_gradients
        self.activation_limit = memory_limit - static_memory_usage - starting_mem
        
        assert(self.activation_limit > 0), "Activation limit must be greater than 0"
        
        print(f"Memory limit: {memory_limit} bytes")
        print(f"Starting memory: {starting_mem} bytes")
        print(f"Weight allocation: {self.weight_bytes} bytes")
        print(f"Activation limit: {self.activation_limit} bytes")
        
        saved_tensors_set = set()
        for node in self.nodes:
            saved_tensors_set.update(node.saved_tensors)
        total_saved_size = sum(t.size for t in saved_tensors_set)
        print(f"Total saved tensors size: {total_saved_size} bytes")
        
        find_large_nodes(self.nodes)
        find_compute_intensive_nodes(self.nodes)
        
        self.segments = generate_segments2(nodes, self.activation_limit)
        print(f"Generated {len(self.segments)} segments.")
        self.segments = filter_large_segments(self.segments, self.activation_limit)
        print(f"Filtered {len(self.segments)} segments after removing large segments.")
        # self.segments = filter_compute_intensive_segments(self.segments)
        self.segments = filter_segments_simple(self.segments, nodes)
        # print_segments(self.segments)
        
        self.problem = LpProblem("Checkpointing_Problem", LpMinimize)
        # self.solver = getSolver('PULP_CBC_CMD')
        # self.solver = getSolver('GUROBI_CMD')
        self.solver = GUROBI_CMD(msg=1, timeLimit=3600, threads=16)
        self.solved = False
        # self.execution_time = lpSum([self.segments[i].nodes[0].execution_time * self.segment_vars[i] for i in range(len(self.segments))])
        # self.problem += self.execution_time, "Total_Execution_Time"
        
        assert(memory_limit > starting_mem + self.grad_mem), "Memory limit must be greater than starting memory usage."
        assert(self.activation_limit > 0), "Activation limit must be greater than 0"
        
        print(f"Generated {len(self.segments)} segments.")
        print(f"Number of nodes: {len(nodes)}")
        
        # for seg in self.segments:
        #     print(seg)
        
        # Objective: Minimize total execution time
        self.problem += lpSum([segment.cp * segment.execution_time for segment in self.segments])
        print("Added objective function.")

        # Constraints
        # Each node must be covered by exactly one segment
        print("Adding node coverage constraints...")
        for node in tqdm(nodes):
            segments_containing_node = [segment for segment in self.segments if node in segment.nodes_set]
            assert len(segments_containing_node) > 0, f"No segment contains node {node.id}"
            self.problem += lpSum(segment.select for segment in segments_containing_node) == 1, f"Cover_Node_{node.id}"

        # Only selected segment can be checkpointed
        print("Adding checkpointing constraints...")
        for segment in self.segments:
            self.problem += (segment.cp - segment.select <= 0, f"Checkpoint_Only_If_Selected_{segment.name}")
            
        # segments that have input size larger than saved size must not be checkpointed, it might be selected
        for segment in self.segments:
            if segment.input_size() >= segment.saved_size(current_node=None):
                self.problem += segment.cp == 0, f"No_Checkpoint_If_Input_Larger_Than_Saved_{segment.name}"

        # Memory constraints of the forward pass
        # current_segments = set()
        segments_before_node = set()
        segments_before_and_containing_node = set()
        segments_before_node = set()
        saved_tensors_set = set()

        profile_exec1 = 0
        profile_exec2 = 0
        profile_exec3 = 0
        profile_exec4 = 0
        profile_exec5 = 0
        profile_exec6 = 0

        for i in tqdm(range(len(nodes))):
            node = nodes[i]
            
            fwd_executed_nodes = nodes[:i+1]
            fwd_future_nodes = nodes[i+2:]
            fwd_outputed_tensors_so_far = set()
            fwd_future_input_tensors = set()
            for fn in fwd_executed_nodes:
                fwd_outputed_tensors_so_far.update(fn.output_tensors)
            for fn in fwd_future_nodes:
                fwd_future_input_tensors.update(fn.input_tensors)
            fwd_intermediate_tensors = fwd_outputed_tensors_so_far.intersection(fwd_future_input_tensors)
            
            bwd_index = len(bwd_nodes)-1 - i
            assert node == bwd_nodes[len(bwd_nodes)-1 - i], f"Node mismatch between forward and backward passes at index {i}."
            
            bwd_executed_nodes = bwd_nodes[:bwd_index+1]
            bwd_future_nodes = bwd_nodes[bwd_index+2:]
            bwd_outputed_tensors_so_far = set()
            bwd_future_input_tensors = set()
            for fn in bwd_executed_nodes:
                bwd_outputed_tensors_so_far.update(fn.grad_output_tensors)
            for fn in bwd_future_nodes:
                bwd_future_input_tensors.update(fn.grad_input_tensors)
            bwd_intermediate_tensors = bwd_outputed_tensors_so_far.intersection(bwd_future_input_tensors)
            
            # segments_before_node, current_segments = classify_segments(current_segments, node)
            # current_segments.update([segment for segment in self.segments if node in segment.nodes_set])
            
            # segments_before_and_containing_node = segments_before_node.union(current_segments)
            curent_time = time.time()
            segments_before_node = segments_before_and_containing_node.copy()
            segments_before_and_containing_node.update([segment for segment in self.segments if node in segment.nodes_set])
            profile_exec1 = time.time() - curent_time
            
            current_time = time.time()
            current_segments = [segment for segment in segments_before_and_containing_node if node in segment.nodes_set]
            profile_exec2 = time.time() - current_time

            current_time = time.time()
            # saved_tensors_set.update([tensor for segment in segments_before_and_containing_node for tensor in segment.saved_tensors(node)])
            saved_tensors_set = get_saved_tensors(segments_before_and_containing_node, node)
            profile_exec3 = time.time() - current_time
            
            current_time = time.time()
            generated_size = get_generated_size(node, saved_tensors_set)
            profile_exec4 += time.time() - current_time

            current_time = time.time()
            grad_generated_size = get_grad_generated_size(node)
            # grad_generated_size = 0
            # generated_tensors = get_generated_tensors_bwd(nodes[i:])
            # to_be_used_tensors = get_to_be_used_tensors_bwd(nodes[:i])
            # retained_intermediate_tensors = generated_tensors.intersection(to_be_used_tensors)
            # retained_intermediate_size = sum(t.size for t in retained_intermediate_tensors)
            # assert type(retained_intermediate_size) == int, f"retained_intermediate_size is not int but {type(retained_intermediate_size)}"
            # grad_generated_size += retained_intermediate_size
            # print(f"At Node {node.name}, retained intermediate tensors for backward pass: {len(retained_intermediate_tensors)}, size: {retained_intermediate_size/1024/1024} MB")

            profile_exec5 += time.time() - current_time

            segments_before_and_containing_node_list = list(segments_before_and_containing_node)
            segments_before_node_list = list(segments_before_node)


            current_time = time.time()
            if (node.is_large_node or i in [int(len(nodes)*0.9), len(nodes)-1]):
                # self.problem += \
                #     lpSum([seg.cp * seg.input_size() * 2 + seg.z * seg.saved_size(current_node=node) for seg in segments_before_and_containing_node_list]) \
                #         + lpSum([t.size for t in node.input_tensors]) \
                #         + lpSum([t.size for t in node.output_tensors]) \
                #         + lpSum([t.size for t in node.saved_tensors]) \
                #         + generated_size <= memory_limit-starting_mem-self.grad_mem, \
                #     f"Memory_Before_Node_{node.id}"
                fwd_intermedidate_and_allocated_tensors = fwd_intermediate_tensors.union(set(node.allocated_tensors))
                self.problem += \
                    lpSum([seg.cp * seg.input_size(exclude=fwd_intermedidate_and_allocated_tensors) + seg.z * seg.saved_size(current_node=node, exclude=fwd_intermedidate_and_allocated_tensors) for seg in segments_before_and_containing_node_list]) \
                        + lpSum([t.size for t in fwd_intermedidate_and_allocated_tensors]) \
                        + lpSum([t.size for t in node.fwd_temporary_tensors()]) <= self.activation_limit, \
                    f"Memory_Before_Node_{node.id}"
                
            # self.problem += \
            #     lpSum([seg.cp * seg.input_size() + seg.z * seg.saved_size(current_node=node) for seg in segments_before_and_containing_node_list]) \
            #         + lpSum([seg.cp * (seg.saved_size(current_node=node)) for seg in current_segments]) \
            #         + grad_generated_size <= memory_limit-starting_mem-self.grad_mem, \
            #     f"Memory_Before_Node_BWD_{node.id}"
            

            # self.problem += \
            #     lpSum([seg.cp * seg.input_size() * 2 + seg.z * seg.saved_size(current_node=None) for seg in segments_before_node_list]) \
            #         + lpSum([seg.select * (seg.saved_size(current_node=node)) for seg in current_segments]) \
            #         + lpSum([t.size for t in node.grad_input_tensors]) \
            #         + lpSum([t.size for t in node.saved_tensors]) \
            #         + grad_generated_size <= memory_limit-starting_mem-self.grad_mem, \
            #     f"Memory_Before_Node_BWD_{node.id}"
            if len(node.grad_allocated_tensors) != 0: # only trainable nodes(layers) should be considered
                bwd_intermediate_and_allocated_tensors = bwd_intermediate_tensors.union(set(node.grad_allocated_tensors))
                self.problem += \
                    lpSum([seg.cp * seg.input_size() + seg.z * seg.saved_size(current_node=None) for seg in segments_before_node_list]) \
                        + lpSum([seg.select * (seg.saved_size(current_node=node)) for seg in current_segments]) \
                        + lpSum([t.size for t in bwd_intermediate_and_allocated_tensors]) \
                        + lpSum([t.size for t in node.bwd_temporary_tensors()]) <= self.activation_limit, \
                    f"Memory_Before_Node_BWD_{node.id}"

            profile_exec6 += time.time() - current_time
        print(f"Profile exec times: classify_segments {profile_exec1:.4f}s, current_segments {profile_exec2:.4f}s, get_saved_tensors {profile_exec3:.4f}s, get_generated_size {profile_exec4:.4f}s, get_grad_generated_size {profile_exec5:.4f}s, adding constraints {profile_exec6:.4f}s")
        # Profile exec times: classify_segments 0.0256s, current_segments 0.0330s, get_saved_tensors 0.2623s, get_generated_size 0.0027s, get_grad_generated_size 0.0022s, adding constraints 2258.6035s
        print("Added forward memory constraints in forward pass.")
        
    # Memory constraints of the backward pass
        # for node in bwd_nodes:
        #     segments_containing_node = [segment for segment in self.segments if node in segment.nodes]
        #     segments_before_node = [segment for segment in self.segments if segment.lvl_end < node.level]
        #     grad_generated_size = get_grad_generated_size(node)
        #     # before the operation execution
        #     self.problem += \
        #         lpSum([seg.cp * seg.input_size() + seg.z * seg.saved_size(current_node=node) for seg in segments_containing_node]) \
        #             + lpSum([seg.cp * seg.input_size() + seg.z * seg.saved_size(current_node=node) for seg in segments_before_node]) \
        #             + lpSum([seg.cp * (seg.saved_size(current_node=node)) for seg in segments_containing_node]) \
        #             + grad_generated_size <= memory_limit-starting_mem, \
        #         f"Memory_Before_BWD_Node_{node.id}"
        
        
        ## Linearization constraints
        for seg in tqdm(self.segments):
            self.problem += seg.z - 1 + seg.cp <= 0, f"Linearization0_{seg.name}"
            self.problem += seg.z - seg.select <= 0, f"Linearization1_{seg.name}"
            self.problem += seg.z + seg.cp - seg.select >= 0, f"Linearization2_{seg.name}"
        print("Added linearization constraints.")
            


    # def add_memory_constraints(self):
    #     # Add memory constraints based on the segments and memory limit
    #     for i, segment in enumerate(self.segments):
    #         segment_memory = sum(node.allocated_post_forward for node in segment.nodes if node.allocated_post_forward)
    #         self.problem += (segment_memory * self.segment_vars[i] <= self.memory_limit, f"Memory_Constraint_Segment_{i}")

    def solve(self):
        # print(f"Model has {len(self.problem.constraints)} constraints.")
        if self.solved:
            print("Model has already been solved.")
            return
        # print(f"Solving with input file: {self.input_file} and output file: {self.output_file}")
        # print(f"Model has {len(self.problem.variables())} variables.")
        # print(f"Model has {len(self.problem.constraints)} constraints.")

        print(f"Number of variables: {len(self.problem.variables())}")
        print(f"Number of constraints: {len(self.problem.constraints)}")
        start_time = time.time()
        self.problem.solve(self.solver)
        end_time = time.time()
        print(f"Solved in {end_time - start_time:.2f} seconds.")
        print(f"Status: {LpStatus[self.problem.status]}")
        
        print(f"Objective: {value(self.problem.objective)}.")

        largest_constraint = None
        largest_value = 0
        for name, constraint in self.problem.constraints.items():
            if name.startswith("Memory_Before_Node_"):
                # print(f"Constraint {name}: value {constraint.value() + (self.memory_limit-self.starting_mem)}, slack {constraint.slack}")
                if constraint.value() + (self.memory_limit-self.starting_mem) > largest_value:
                    largest_value = constraint.value() + (self.memory_limit-self.starting_mem)
                    largest_constraint = (name, constraint)

        print(f"Largest constraint: {largest_constraint[0]}: value {largest_constraint[1].value() + (self.memory_limit-self.starting_mem)}, slack {largest_constraint[1].slack}")

        print(f"MIP gap: {self.problem.solver.mipGap if hasattr(self.problem.solver, 'mipGap') else 'N/A'}")
        print(f"Total number of selected segments: {sum(1 for seg in self.segments if seg.select.varValue > 0)}")
        print(f"Total number of checkpointed segments: {sum(1 for seg in self.segments if seg.cp.varValue > 0)}")
        
        if self.problem.status != LpStatusOptimal:
            print("The model did not find an optimal solution.")
            return
        
        # Print selected segments and checkpointed segments here
        self.solved = True
        for seg in self.segments:
            if seg.select.varValue > 0:
                print(f"  Selected segment: {seg}")
        for seg in self.segments:
            if seg.cp.varValue > 0:
                print(f"  Checkpointed segment: {seg}")
                nodes = seg.nodes
                for node in nodes:
                    print(f"    Checkpoint node: {node.id}, previous nodes {[n.id for n in node.previous_nodes]}")
    
    def checkpointed_segments(self) -> List[Segment]:
        if not self.solved:
            print("Model has not been solved yet.")
            return []
        checkpointed = [seg for seg in self.segments if seg.cp.varValue > 0]
        print(f"Total checkpointed segments: {len(checkpointed)}")
        return checkpointed
    
    def selected_segments(self) -> List[Segment]:
        if not self.solved:
            print("Model has not been solved yet.")
            return []
        selected = [seg for seg in self.segments if seg.select.varValue > 0]
        print(f"Total selected segments: {len(selected)}")
        return selected

class CheckpointSolverHandler:
    def __init__(self, trace_file_path: str, budget_in_bytes: int, weight_MB: float = 0.0, optimizer: str="adam"):
        trace_file_path = trace_file_path
        nodes, bwd_nodes = parse_trace_with_links(trace_file_path)
        print_parsed_trace(nodes)
        self.nodes = nodes
        self.bwd_nodes = bwd_nodes
        self.budget_in_bytes = budget_in_bytes
        self.weight_MB = weight_MB
        self.optimizer = optimizer

    def formulate_problem(self):
        self.cp_solver = CheckpointSolver(self.nodes, self.bwd_nodes, memory_limit=self.budget_in_bytes, starting_mem=self.nodes[0].allocated_pre_forward if self.nodes and self.nodes[0].allocated_pre_forward else 0, weight_MB=self.weight_MB, optimizer=self.optimizer)
        self.checkpointed_segments: List[Segment] = []
        self.checkpointed_nodes = [] # List sorted by node_id

    def solve(self):
        self.cp_solver.solve()
        if self.cp_solver.solved:
            self.checkpointed_segments = self.cp_solver.checkpointed_segments()
            self.root_nodes = []
            self.output_nodes = []
            for seg in self.checkpointed_segments:
                self.root_nodes.extend(seg.root_nodes())
                self.output_nodes.extend(seg.output_nodes())
            self.checkpointed_nodes = sorted([node for seg in self.checkpointed_segments for node in seg.nodes], key=lambda n: n.id)
            self.root_nodes = sorted(list(set(self.root_nodes)), key=lambda n: n.id)
            self.output_nodes = sorted(list(set(self.output_nodes)), key=lambda n: n.id)
            
            # save solutions to pkl
            # solution_file_name = get_solution_file_name(trace_file_path, memory_limit, args.mode)
            # save_solution_to_file(nodes, bwd_nodes, my_solver.segments, checkpointed_segments, solution_file_name)

class Module:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []
        self.ops = []
    
    def add_child(self, child: 'Module'):
        assert child.parent is None
        child.parent = self
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        return self.parent is None
    
    def add_op(self, op: Node):
        # assert self.is_leaf(), "Must be leaf node"
        # avoid deduplicating ops in the same module
        if op not in self.ops:
            self.ops.append(op)
        
    def print_tree(self, prefix=""):
        print(f"{prefix}{self.name}")
        for child in self.children:
            child.print_tree(f"{prefix}\t")
            
    def get_all_descendants(self) -> List['Module']:
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def all_ops(self) -> List[Node]:
        ops = []
        if self.is_leaf():
            ops.extend(self.ops)
        else:
            for child in self.children:
                child_ops = child.all_ops()
                for op in child_ops:
                    assert op not in ops, f"Op {op.id} is duplicated in module {self.name}."
                ops.extend(child_ops)
        # lastly, add ops of this module if there is any
        # for op in self.ops:
        #     assert op not in ops, f"Op {op.id} is duplicated in module {self.name}."
        ops.extend(self.ops)
        return ops
    
    def saved_size(self) -> int:
        ops = self.all_ops()
        saved_tensors = set()
        for op in ops:
            saved_tensors.update(op.saved_tensors)
        return sum(t.size for t in saved_tensors)
    
    def input_op(self) -> Node:
        # first op of this node, if not, see this node's first child... and so on
        if self.children:
            return self.children[0].input_op()
        else:
            assert len(self.ops) > 0, f"Leaf module {self.name} has no ops."
            return self.ops[0]
        
    def input_size(self) -> int:
        input_op = self.input_op()
        input_tensors = input_op.input_tensors
        return sum(t.size for t in input_tensors)
    
    def get_segment(self) -> 'Segment':
        ops = self.all_ops()
        # deduplicate ops while preserving order
        seen = set()
        unique_ops = []
        for op in ops:
            if op not in seen:
                unique_ops.append(op)
                seen.add(op)
        return Segment(nodes=unique_ops, name=self.name)

def parse_module_trace(file_path: str, nodes: List[Node]) -> Module:
    root = None
    current_module = None
    
    stack = []
    in_op = False
    
    
    with open(file_path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            
            # count indentation (spaces or tabs)
            stripped = line.lstrip(" \t")
            indent = len(line) - len(stripped)
            
            # New Module
            if stripped.startswith(">>"):
                name = stripped[2:].strip()
                module = Module(name)
                
                if "--" in name:
                    # this is root module
                    assert len(stack) == 0
                    assert root is None
                    root = module
                else:
                    # this is not root module
                    assert root is not None
                    stack[-1].add_child(module)
                
                stack.append(module)
                continue
            
            # Module ends
            if stripped.startswith("<<"):
                assert len(stack) != 0, "Stack must not be 0"
                module = stack.pop()
                continue
            
            # operation
            if stripped.startswith("------>"):
                in_op = True
                continue
            
            if in_op:
                # find node in nodes that has the same name as 'stripped'
                for node in nodes:
                    if node.name == stripped:
                        stack[-1].add_op(node)
                        break
                continue
                # assert False, f"Node {stripped} is not found in nodes"
            
            if stripped.startswith("<------"):
                in_op = False
                continue
            
    assert len(stack) == 0, "Must be 0"
    return root

def reorder_bwd_nodes(bwd_nodes: List[Node], checkpointed_segments: List[Segment]) -> List[Node]:
    # go through bwd_nodes, and reorder them so that nodes in the same checkpointed segment are contiguous
    checkpointed_node_set = set()
    for seg in checkpointed_segments:
        for node in seg.nodes:
            checkpointed_node_set.add(node)
            
    bwd_nodes_queue = bwd_nodes.copy()

    reordered_bwd_nodes = []
    while len(bwd_nodes_queue) > 0:
        node = bwd_nodes_queue[0]
        if node in checkpointed_node_set:
            # node is in a checkpoined segment
            segment_containing_node = None
            for seg in checkpointed_segments:
                if node in seg.nodes:
                    segment_containing_node = seg
                    break
            assert segment_containing_node is not None, f"Node {node.id} not found in any checkpointed segment."
            reverse_nodes_in_segment = segment_containing_node.nodes[::-1]
            for seg_node in reverse_nodes_in_segment:
                if seg_node not in reordered_bwd_nodes:
                    reordered_bwd_nodes.append(seg_node)
                    bwd_nodes_queue.remove(seg_node)
        else:
            # node is not in a checkpointed segment
            reordered_bwd_nodes.append(node)
            bwd_nodes_queue.remove(node)

    assert len(reordered_bwd_nodes) == len(bwd_nodes), "Reordered backward nodes length mismatch."
    return reordered_bwd_nodes

def simulate_memory_time_series4(nodes: List[Node], bwd_nodes: List[Node], checkpointed_segments: List[Segment], starting_mem=0, weight_MB=0, optimizer="adam", do_print=True) -> Tuple[List[int], List[int]]:
    forward_series = []
    backward_series = []
    execution_time = 0
    
    weight_bytes = int(weight_MB * 1024 * 1024)
    immediate_gradients = weight_bytes
    optimizer_state = weight_bytes * optimizer_to_scaling_factor(optimizer)
    rolling_gradients = weight_bytes
    
    assert weight_bytes != 0, "Must not be 0"
    
    static_memory_usage = weight_bytes + immediate_gradients + optimizer_state + rolling_gradients + starting_mem
    # activation_limit = memory_limit - static_memory_usage - starting_mem

    # Accounting for both weights and gradients from the second iteration forwards
    # forward_series.append(starting_mem*2)
    forward_series.append(static_memory_usage)
    execution_time += 0

    # print(f"Memory usage at FWD Node None: \t\t{starting_mem*2 / (1024 * 1024):.2f} MB")   
    if do_print: print(f"Memory usage at FWD Node None: \t\t{static_memory_usage / (1024 * 1024):.2f} MB")   

    # mem = starting_mem
    
    segments = [] # segments executed so far
    current_segment_nodes = [] # current segment wrt x, checkpointed or not

    checkpointed_nodes = [node for seg in checkpointed_segments for node in seg.nodes]
    
    tensors_outputed_so_far = set()

    for i in range(len(nodes)):
        # print(f"--- Simulating FWD Node {i}/{len(nodes)} ---")
        node = nodes[i]
        executed_nodes = nodes[:i+1] # include current node
        future_nodes = nodes[i+2:]
        # recounting current_mem
        # current_mem = starting_mem*2
        current_mem = static_memory_usage
        
        tensors_outputed_so_far.update(node.output_tensors)
        future_input_tensors = set()
        for fn in future_nodes:
            future_input_tensors.update(fn.input_tensors)

        # in the executed segments, accumulate input tensors, saved tensors, and intermediate tensors that have not died yet
        input_tensors = set()
        saved_tensors = set()
        intermediate_tensors = tensors_outputed_so_far.intersection(future_input_tensors)  # tensors outputed by executed nodes that are still alive (used by future nodes)
        # if (i < len(nodes)-1): assert(len(intermediate_tensors) > 0), f"At Node {node.name}, intermediate_tensors is 0."
        for seg in segments:
            # intermediate_tensors.update(seg.fwd_intermediate_tensors(nodes[i:], current_node=None))
            if seg in checkpointed_segments:
                # current_mem += seg.input_size() # total size of input tensors
                input_tensors.update(seg.input_tensors)
                # print(f"\tSegment {seg.name} input tensors: {[t.name for t in seg.input_tensors]}")
            else:
                # current_mem += seg.saved_size(node) # total size of saved tensors upto the current node
                saved_tensors.update(seg.saved_tensors(node))
        # tensors_in_memory = input_tensors.union(saved_tensors).union(intermediate_tensors)
        # current_mem = sum(t.size for t in tensors_in_memory)

        if node not in checkpointed_nodes:
            # current regular segment is being built, not is 'segment' set yet so need to add to saved_tensors
            current_segment_nodes.append(node)
            current_segment = Segment(nodes=current_segment_nodes)

            # current_mem += current_segment.saved_size(node)
            saved_tensors.update(current_segment.saved_tensors(node))
        else:
            # We are begining a checkpointed segment, so record what happened so far as a single segment and add to 'segments' set
            if current_segment_nodes:
                segments.append(Segment(nodes=current_segment_nodes))
                current_segment_nodes = []
            
            # node is in a checkpointed segment
            # get the checkpointed segment containing node 
            segment_containing_node = None
            for seg in checkpointed_segments:
                if node in seg.nodes:
                    segment_containing_node = seg
                    break
            
            # only append to set of executed segments when the segment ends here
            if segment_containing_node.nodes[-1] == node:
                segments.append(segment_containing_node)
            
            # current_mem += segment_containing_node.saved_size(node) if segment_containing_node else 0
            if segment_containing_node:
                input_tensors.update(segment_containing_node.input_tensors)
            
            # intermediate_tensors.update(segment_containing_node.fwd_intermediate_tensors(nodes[i:], current_node=node))

        # tensors_in_memory = input_tensors.union(saved_tensors).union(intermediate_tensors)
        tensors_in_memory = input_tensors.union(saved_tensors)
        # current_mem += sum(t.size for t in tensors_in_memory) + sum(t.size for t in node.input_tensors) + sum(t.size for t in node.output_tensors)
        # current_mem += sum(t.size for t in tensors_in_memory) + sum(t.size for t in node.input_tensors) + sum(t.size for t in node.output_tensors) + sum(t.size for t in input_tensors)
        current_mem += sum(t.size for t in tensors_in_memory) + sum(t.size for t in node.allocated_tensors)

        # if node.name == "t-67":
        #     print(f"Debug at Node {node.name}:")
        #     print(f"  Input Tensors: {[t.name for t in input_tensors]}")
        #     print(f"  Saved Tensors: {[t.name for t in saved_tensors]}")
        #     print(f"  Intermediate Tensors: {[t.name for t in intermediate_tensors]}")
        #     print(f"  Total Memory: {current_mem / (1024 * 1024):.2f} MB")
        
        # segments_including_current = segments + ([Segment(nodes=current_segment_nodes)] if current_segment_nodes else [])
        # saved_tensors_set = get_saved_tensors(segments_including_current, node)
        # generated_size = get_generated_size(node, saved_tensors_set)
        # current_mem += generated_size
        
        if do_print: print(f"Memory usage at FWD Node {node.name}: \t\t{current_mem / (1024 * 1024):.2f} MB")
        forward_series.append(current_mem)
        execution_time += node.execution_time

    segments.append(Segment(nodes=current_segment_nodes)) if current_segment_nodes else None
    
    if do_print: print(f"There are {len(segments)} segments executed in forward pass.")
    
    segment_containing_node = None

    reexecuted_segments = set()
    tensors_outputed_so_far = set()
    reordered_bwd_nodes = reorder_bwd_nodes(bwd_nodes, checkpointed_segments)
    
    for i in range(len(reordered_bwd_nodes)):
        # print(f"--- Simulating BWD Node {i}/{len(bwd_nodes)} ---")
        node = reordered_bwd_nodes[i]
        executed_nodes = reordered_bwd_nodes[:i+1] # include current node
        future_nodes = reordered_bwd_nodes[i+2:]
        tensors_outputed_so_far.update(node.grad_output_tensors)
        future_input_tensors = set()
        for fn in future_nodes:
            future_input_tensors.update(fn.grad_input_tensors)
            

        # current_mem = starting_mem*2
        current_mem = static_memory_usage

        input_tensors = set()
        saved_tensors = set()
        intermediate_tensors = tensors_outputed_so_far.intersection(future_input_tensors)  # tensors outputed by executed nodes that are still alive (used by future nodes)
        # checkpointed_saved_tensors = set()
        grad_input_tensors = set()

        if segment_containing_node is not None and node not in segment_containing_node.nodes:
            segment_containing_node = None

        if segment_containing_node is None:
            for seg in segments:
                if node in seg.nodes:
                    segment_containing_node = seg
                    break
        
        assert segment_containing_node is not None, f"Node {node.id} not found in any segment during backward pass."

        
        # saved_tensors.update(segment_containing_node.saved_tensors(node))

        # if node in checkpointed_nodes:
        #     checkpointed_saved_tensors.update(segment_containing_node.saved_tensors(node))

        
        intermediate_tensors.update(segment_containing_node.bwd_intermediate_tensors(bwd_nodes[i:], current_node=node))
        # print(f"Intermediate tensors at BWD Node {node.name}: {[t.name for t in intermediate_tensors]}")

        grad_generated_tensors = segment_containing_node.grad_generated_tensors(node)

        if segment_containing_node in segments:
            segments.remove(segment_containing_node)

        for seg in segments:
            # intermediate_tensors.update(seg.bwd_intermediate_tensors(bwd_nodes[i:], current_node=None))
            if seg in checkpointed_segments:
                # current_mem += seg.input_size()
                input_tensors.update(seg.input_tensors)
            else:
                # current_mem += seg.saved_size(node)
                saved_tensors.update(seg.saved_tensors(node))
        
        
                
        # execute recomputation forward
        if node in checkpointed_nodes and segment_containing_node not in reexecuted_segments:
            # reexecute the forward pass of the checkpointed segment containing this node
            # current_mem += segment_containing_node.saved_size(node)
            # add input tensors of the segment
            # input_tensors.update(segment_containing_node.input_tensors)

            reexecuted_segments.add(segment_containing_node)
            # add an entry for each recomputed fwd in the current segment
            # recomputed_nodes = segment_containing_node.fwd_lineage_node_set(node)
            recomputed_nodes = segment_containing_node.nodes
            
            current_output_tensors = set()
            
            # Compute memory consumption of recomputed nodes
            for i in range(len(recomputed_nodes)):
                r_node = recomputed_nodes[i]
                
                executed_r_nodes = recomputed_nodes[:i+1]
                future_r_nodes = recomputed_nodes[i+2:]
                tensors_outputed_so_far_by_r_node = set()
                for fn in executed_r_nodes:
                    tensors_outputed_so_far_by_r_node.update(fn.output_tensors)
                future_input_tensors_by_r_node = set()
                for fn in future_r_nodes:
                    future_input_tensors_by_r_node.update(fn.input_tensors)
                intermediate_tensors_by_r_node = tensors_outputed_so_far_by_r_node.intersection(future_input_tensors_by_r_node)
                
                tensors_in_memory = input_tensors.union(saved_tensors).union(intermediate_tensors).union(intermediate_tensors_by_r_node).union(r_node.allocated_tensors)
                tensors_in_memory = tensors_in_memory.union(segment_containing_node.saved_tensors(r_node))
                if (i == 0): tensors_in_memory = tensors_in_memory.union(segment_containing_node.input_tensors)
                
                tmp_current_mem = current_mem + sum(t.size for t in tensors_in_memory)
                if do_print: print(f"Memory usage at FWD Node {r_node.name}: \t\t{tmp_current_mem / (1024 * 1024):.2f} MB")
                backward_series.append(tmp_current_mem)
                execution_time += r_node.execution_time
                
                # output_tensors = r_node.grad_output_tensors

                # tmp_current_mem = current_mem
                
                # current_output_tensors.update(output_tensors)
                # # current_output_tensors.intersection_update(segment_containing_node.saved_tensors(r_node))
                # assert len(segment_containing_node.saved_tensors(r_node)) >= 0, f"Segment {segment_containing_node.name} has no saved_tensors at Node {r_node.id}."
                # tensors_in_memory = input_tensors.union(saved_tensors).union(node.grad_input_tensors).union(intermediate_tensors).union(current_output_tensors).union(segment_containing_node.saved_tensors(r_node))
                # # tmp_current_mem += sum(t.size for t in tensors_in_memory) + sum(t.size for t in output_tensors) + sum(t.size for t in segment_containing_node.saved_tensors(r_node))
                # if (i == 0):
                #     tensors_in_memory = tensors_in_memory.union(segment_containing_node.input_tensors)
                # tmp_current_mem += sum(t.size for t in tensors_in_memory) + sum(t.size for t in input_tensors) + sum(t.size for t in r_node.input_tensors) + sum(t.size for t in output_tensors)

                # if do_print: print(f"Memory usage at FWD Node {r_node.name}: \t\t{tmp_current_mem / (1024 * 1024):.2f} MB")
                # backward_series.append(tmp_current_mem)
                # execution_time += r_node.execution_time
                
        # segment_containing_node = None
        # for seg in segments:
        #     if node in seg.nodes:
        #         segment_containing_node = seg
        #         break
        
        # assert segment_containing_node is not None, f"Node {node.id} not found in any segment during backward pass."
        
        # if node in checkpointed_nodes:
        #     # current_mem += segment_containing_node.saved_size(node)
        #     saved_tensors.update(segment_containing_node.saved_tensors(node))
        #     input_tensors.update(segment_containing_node.input_tensors)
        
        # intermediate_tensors.update(segment_containing_node.bwd_intermediate_tensors(bwd_nodes[i:], current_node=node))

        # if segment_containing_node.nodes[0] == node:
        #     segments.remove(segment_containing_node)
        assert len(segment_containing_node.saved_tensors(node)) >= 0, f"Segment {segment_containing_node.name} has no saved_tensors at Node {node.id}."
        # input_tensors.update(segment_containing_node.input_tensors)
        # saved_tensors.update(segment_containing_node.saved_tensors(node))
        grad_input_tensors.update(node.grad_input_tensors)

        # tensors_in_memory = input_tensors.union(saved_tensors).union(grad_input_tensors).union(intermediate_tensors)
        # tensors_in_memory = input_tensors.union(saved_tensors).union(intermediate_tensors).union(grad_input_tensors)
        tensors_in_memory = input_tensors.union(saved_tensors).union(intermediate_tensors)
        # grad_generated_tensors = node.grad_generated_tensors
        # current_mem += sum(t.size for t in tensors_in_memory)
        
        # if node in checkpointed_nodes:
            # current_mem += sum(t.size for t in segment_containing_node.saved_tensors(node).difference(input_tensors))
        # current_mem += sum(t.size for t in segment_containing_node.saved_tensors(node)) + sum(t.size for t in node.saved_tensors)
        # current_mem += sum(t.size for t in segment_containing_node.saved_tensors(node))
        
        # Double count the tensors used for this operation
        # current_mem += sum(t.size for t in node.saved_tensors) + sum(t.size for t in node.grad_output_tensors)
        # current_mem += sum(t.size for t in node.allocated_tensors)
        
        current_mem += sum(t.size for t in  tensors_in_memory.union(segment_containing_node.saved_tensors(node)).union(node.grad_allocated_tensors))

        # grad_generated_size = get_grad_generated_size(node)
        # current_mem += grad_generated_size

        if do_print: print(f"Memory usage at BWD Node {node.name}: \t\t{current_mem / (1024 * 1024):.2f} MB")
        backward_series.append(current_mem)
        execution_time += node.backward_execution_time
    
    return forward_series, backward_series, execution_time

def search_and_apply_checkpoint_wrappers(module_root: Module, nodes: List[Node], bwd_nodes: List[Node], budget: int, weight_MB=0, optimizer="adam") -> List[Segment]:
    all_modules = module_root.get_all_descendants()
    
    # filter out modules that has saved size 0
    all_modules = [m for m in all_modules if m.saved_size() > 0]
    
    # run simulation to get max memory usage
    fwd_series, bwd_series, execution_time = simulate_memory_time_series4(nodes, bwd_nodes, [], starting_mem=nodes[0].allocated_pre_forward if nodes and nodes[0].allocated_pre_forward else 0, weight_MB=weight_MB, do_print=False)
    total_series = fwd_series + bwd_series
    max_mem = max(total_series)
    print(f"Max memory usage without checkpointing: {max_mem / (1024 * 1024):.2f} MB")
    
    segments = []
    
    current_memory_usage = max_mem
    current_bwd_memory_usage = max(bwd_series)
    while True:
        if current_memory_usage <= budget:
            break
        
        if not all_modules:
            print("No more modules to checkpoint. Cannot meet memory budget.")
            break
        
        to_reduce = current_memory_usage - budget
        module_to_checkpoint = None
        
        # find the module with saved size closest to to_reduce
        current_diff_from_to_reduce = float('inf')
        for module in all_modules:
            diff_from_to_reduce = abs(module.saved_size() - to_reduce)
            if diff_from_to_reduce < current_diff_from_to_reduce:
                current_diff_from_to_reduce = diff_from_to_reduce
                module_to_checkpoint = module
        assert module_to_checkpoint is not None, "No module found to checkpoint, but memory usage is above budget."
        
        # resulting segment from module_to_checkpoint
        segment = module_to_checkpoint.get_segment()
        
        # if segment contains only 1 node, then checkpointing it won't save any memory, so skip it
        if len(segment.nodes) <= 1:
            all_modules.remove(module_to_checkpoint)
            continue
        
        print(f"Trying to checkpoint module {module_to_checkpoint.name} with saved size {module_to_checkpoint.saved_size() / (1024 * 1024):.2f} MB to reduce memory usage from {current_memory_usage / (1024 * 1024):.2f} MB to {(current_memory_usage - module_to_checkpoint.saved_size()) / (1024 * 1024):.2f} MB.")
        segments.append(segment)
        
        # get new memory usage after checkpointing module_to_checkpoint
        fwd_series, bwd_series, execution_time = simulate_memory_time_series4(nodes, bwd_nodes, segments, starting_mem=nodes[0].allocated_pre_forward if nodes and nodes[0].allocated_pre_forward else 0, weight_MB=weight_MB, do_print=False)
        total_series = fwd_series + bwd_series
        
        new_max_mem = max(total_series)
        new_bwd_max_mem = max(bwd_series)
        
        if new_max_mem <= budget:
            return segments
        
        if new_bwd_max_mem >= current_bwd_memory_usage:
            # avoid checkpointing this Module, continue to search for another module to checkpoint
            segments.pop()
            all_modules.remove(module_to_checkpoint)
            continue
        
        # new_bwd_max_mem <= current_bwd_memory_usage
        current_memory_usage = new_max_mem
        current_bwd_memory_usage = new_bwd_max_mem
        # reove checkpointed module and its decendents from all_modules to avoid checkpointing overlapping segments
        all_modules.remove(module_to_checkpoint)
        for mod in module_to_checkpoint.get_all_descendants():
            if mod in all_modules:
                all_modules.remove(mod)
                
        print(f"Checkpointed module {module_to_checkpoint.name} with saved size {module_to_checkpoint.saved_size() / (1024 * 1024):.2f} MB. New max memory usage: {current_memory_usage / (1024 * 1024):.2f} MB, New backward max memory usage: {current_bwd_memory_usage / (1024 * 1024):.2f} MB.")
                
    if current_memory_usage > budget:
        print(f"Could not meet memory budget. Final memory usage: {current_memory_usage / (1024 * 1024):.2f} MB")
    else:
        print(f"Memory budget met. Final memory usage: {current_memory_usage / (1024 * 1024):.2f} MB")
    return segments
           
class CheckpointGreedyHandler:
    def __init__(self, trace_file_path: str, module_op_trace_file_path: str, budget_in_bytes: int, weight_MB: float = 0.0, optimizer: str="adam"):
        # trace_file_path = trace_file_path
        nodes, bwd_nodes = parse_trace_with_links(trace_file_path)
        root_module = parse_module_trace(module_op_trace_file_path, nodes=nodes)
        
        # print_parsed_trace(nodes)
        self.nodes = nodes
        self.bwd_nodes = bwd_nodes
        self.root_module = root_module
        self.budget_in_bytes = budget_in_bytes
        self.weight_MB = weight_MB
        self.optimizer = optimizer

    # def formulate_problem(self):
    #     self.cp_solver = CheckpointSolver(self.nodes, self.bwd_nodes, memory_limit=self.budget_in_bytes, starting_mem=self.nodes[0].allocated_pre_forward if self.nodes and self.nodes[0].allocated_pre_forward else 0, weight_MB=self.weight_MB, optimizer=self.optimizer)
    #     self.checkpointed_segments: List[Segment] = []
    #     self.checkpointed_nodes = [] # List sorted by node_id
    
    def search_for_checkpoint_segments(self):
        self.checkpointed_segments = search_and_apply_checkpoint_wrappers(self.root_module, self.nodes, self.bwd_nodes, self.budget_in_bytes, weight_MB=self.weight_MB, optimizer=self.optimizer)
        self.root_nodes = []
        self.output_nodes = []
        for seg in self.checkpointed_segments:
            self.root_nodes.extend(seg.root_nodes())
            self.output_nodes.extend(seg.output_nodes())
        self.checkpointed_nodes = sorted([node for seg in self.checkpointed_segments for node in seg.nodes], key=lambda n: n.id)
        self.root_nodes = sorted(list(set(self.root_nodes)), key=lambda n: n.id)
        self.output_nodes = sorted(list(set(self.output_nodes)), key=lambda n: n.id)
        
class CheckpointOpListHandler:
    def __init__(self, trace_file_path: str, checkpoint_op_list_file_path: str, budget_in_bytes: int, weight_MB: float = 0.0, optimizer: str="adam"):
        # trace_file_path = trace_file_path
        nodes, bwd_nodes = parse_trace_with_links(trace_file_path)
        # checkpoint_ops = parse_checkpoint_op_list(checkpoint_op_list_file_path)
        
        # print_parsed_trace(nodes)
        self.nodes = nodes
        self.bwd_nodes = bwd_nodes
        self.budget_in_bytes = budget_in_bytes
        self.weight_MB = weight_MB
        self.optimizer = optimizer
        
        self.get_checkpointed_segments(checkpoint_op_list_file_path)
        
    def get_checkpointed_segments(self, checkpoint_op_list_file_path: str) -> List[Segment]:
        # parse the file, line by line
        # at each line, if it starts with "--------", it signal the end of the current segment, and the next line is the first op of the next segment
        # otherwise, the line is an op that belongs to the current segment
        segments = []
        current_segment_ops = []
        with open(checkpoint_op_list_file_path, "r") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.startswith("--------"):
                    if current_segment_ops:
                        segment_nodes = []
                        for op_name in current_segment_ops:
                            for node in self.nodes:
                                if node.name == op_name:
                                    segment_nodes.append(node)
                                    break
                        segments.append(Segment(nodes=segment_nodes))
                        current_segment_ops = []
                else:
                    current_segment_ops.append(line.strip())
        # add the last segment if there is any
        if current_segment_ops:
            segment_nodes = []
            for op_name in current_segment_ops:
                for node in self.nodes:
                    if node.name == op_name:
                        segment_nodes.append(node)
                        break
            segments.append(Segment(nodes=segment_nodes))
        self.checkpointed_segments = segments
        self.root_nodes = []
        self.output_nodes = []
        for seg in self.checkpointed_segments:
            self.root_nodes.extend(seg.root_nodes())
            self.output_nodes.extend(seg.output_nodes())
        self.checkpointed_nodes = sorted([node for seg in self.checkpointed_segments for node in seg.nodes], key=lambda n: n.id)
        # self.root_nodes = sorted(list(set(self.root_nodes)), key=lambda n: n.id)
        # root nodes are first node in segment
        self.root_nodes = sorted(list(set([seg.nodes[0] for seg in self.checkpointed_segments])), key=lambda n: n.id)
        self.output_nodes = sorted(list(set(self.output_nodes)), key=lambda n: n.id)
        