#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import copy
import hashlib
import itertools
from collections import defaultdict
import numpy as np
from ..process_mesh import ProcessMesh
from ..process_mesh import _flatten_nested_list
from ..completion import Completer
from ..parallelizer_v2 import Parallelizer
from ..dist_op import DistributedOperator
from ..dist_attribute import TensorDistributedAttribute
from ..dist_attribute import OperatorDistributedAttribute
from ..operators.common import find_compatible_distributed_operator_impls
from ..operators.common import is_elementwise_op
from ..utils import print_program_with_dist_attr
from .trial import Trial, TrialStatus
from .tuner import Tuner
from .tunable_space import TunableSpace
from ..cost import CostEstimator


class ParallelTuner:
    def __init__(self,
                 dist_context,
                 mode="test",
                 cluster=None,
                 num_nodes=1,
                 devices_per_node=1,
                 max_trials=None,
                 tuner_id=None,
                 seed=None,
                 logger=None,
                 loop_count=10):
        self._loop_count = loop_count
        self._estimator = None
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._mode = mode
        self._cluster = cluster
        self._num_nodes = num_nodes
        self._devices_per_node = devices_per_node
        self._space = TunableSpace()
        self._objective = "cost"
        self._direction = "min"
        # self._max_trials = max_trials
        self._max_trials = 3
        self._tuner_id = tuner_id
        self._seed = seed or np.random.randint(1, 10000)
        self._seed = 1585
        print("seed", self._seed, flush=True)
        self._seed_state = self._seed
        self._logger = logger
        self._trials = {}
        self._max_collisions = 3
        self._tried_values = set()

        self._rng = np.random.default_rng(self._seed)

        self._op_id_to_dist_attr_candidates = defaultdict(list)
        self._cached_dims_mapping_candidates = {}
        self._cached_candidates_info = defaultdict(list)

        self._special_ops = [
            "create_py_reader", "create_double_buffer_reader", "read", "while",
            "read_from_array", "write_to_array"
        ]

        self._special_tensors = [
            "lod_tensor_blocking_queue_0", "create_py_reader_0",
            "double_buffer_0"
        ]

        self._completer = Completer(self._dist_context)

        self._parallelizer = Parallelizer(self._mode, self._completer,
                                          self._dist_context)

    def _generate_combination(self,
                              elements,
                              target,
                              idx,
                              partial_candidate,
                              candidates,
                              num_candidates=None):
        if target == 0:
            candidates.append(copy.deepcopy(partial_candidate))
            return

        if target < 0 or idx == len(elements) \
            or len(candidates) > num_candidates:
            return

        # Use
        partial_candidate.append(elements[idx])
        self._generate_combination(elements, target - elements[idx], idx,
                                   partial_candidate, candidates,
                                   num_candidates)
        # Not use
        partial_candidate.pop()
        self._generate_combination(elements, target, idx + 1, partial_candidate,
                                   candidates, num_candidates)

    def _permute_combination(self,
                             combination,
                             target,
                             check,
                             partial_candidate,
                             candidates,
                             num_candidates=None,
                             skip_prob=None):
        if num_candidates is not None \
            and len(candidates) == num_candidates:
            return

        if len(partial_candidate) == len(combination):
            candidates.append(partial_candidate)
            return

        for i in range(len(combination)):
            if check[i] == 1:
                continue
            if self._rng.choice([True, False], p=[skip_prob, 1 - skip_prob]):
                continue
            if i > 0 and combination[i] == combination[i - 1] \
                and check[i -1] == 0:
                continue
            check[i] = 1
            self._permute_combination(combination, target, check,
                                      partial_candidate + [combination[i]],
                                      candidates, num_candidates, skip_prob)
            check[i] = 0

    def _partition_number(self, target):
        log2_target = int(math.log2(target))
        elements = [pow(2, i) for i in range(log2_target)]
        if pow(2, log2_target) == target:
            elements.append(target)
        seed_candidates = []
        num_seed_candidates = 1000
        partial_results = []
        self._generate_combination(elements, target, 0, partial_results,
                                   seed_candidates, num_seed_candidates)

        candidates = []
        for seed_candidate in seed_candidates:
            cur_candidates = []
            num_cur_candidates = 16
            seed_candidate.sort()
            check = [0 for i in range(len(seed_candidate))]
            if target <= 8:
                skip_prob = 0.0
            else:
                skip_prob = (len(seed_candidate) / target)
            self._permute_combination(seed_candidate, target, check, [],
                                      cur_candidates, num_cur_candidates,
                                      skip_prob)
            candidates.extend(cur_candidates)
        return candidates

    def _partition_devices(self, num_nodes, devices_per_node):
        inter_node_partitions = self._partition_number(num_nodes)
        intra_node_partitions = self._partition_number(devices_per_node)
        # print("inter and intra",
        #       inter_node_partitions,
        #       intra_node_partitions,
        #       flush=True)
        # for inter in inter_node_partitions:
        #     for intra in intra_node_partitions:
        #         process_mesh_list = self._generate_process_mesh_list(
        #             inter, intra, self._num_nodes, self._devices_per_node)
        #         print(inter, intra, process_mesh_list, flush=True)
        return inter_node_partitions, intra_node_partitions

    def _generate_process_mesh_list(self, inter_node_partition,
                                    intra_node_partition):
        process_mesh_list = []
        start_row = 0
        start_col = 0
        for m in inter_node_partition:
            start_col = 0
            for n in intra_node_partition:
                process_mesh = []
                for p in range(m):
                    start = (start_row + p) * self._devices_per_node + start_col
                    tmp = []
                    for q in range(n):
                        tmp.append(start + q)
                    process_mesh.append(tmp)
                # if len(process_mesh) == 1:
                #     process_mesh_list.append(copy.deepcopy(process_mesh[0]))
                # else:
                #     process_mesh_list.append(copy.deepcopy(process_mesh))
                process_mesh_list.append(copy.deepcopy(process_mesh))
                start_col += n
            start_row += m
        return process_mesh_list

    def _generate_dims_mapping_candidates_helper(self, dims_mapping, dims_list,
                                                 start, visited, candidates):
        if start == len(dims_mapping) or all(visited):
            candidates.append(copy.deepcopy(dims_mapping))
            return

        for idx, dim in enumerate(dims_list):
            if visited[idx] == False:
                dims_mapping[start] = dim
                visited[idx] = True
                self._generate_dims_mapping_candidates_helper(
                    dims_mapping, dims_list, start + 1, visited, candidates)
                visited[idx] = False
        dims_mapping[start] = -1
        self._generate_dims_mapping_candidates_helper(
            dims_mapping, dims_list, start + 1, visited, candidates)

    def _generate_dims_mapping_candidates(self, dims_mapping_len,
                                          process_mesh_len):
        assert dims_mapping_len >= 1 and process_mesh_len >= 1
        key = (dims_mapping_len, process_mesh_len)
        if key in self._cached_dims_mapping_candidates:
            return self._cached_dims_mapping_candidates[key]
        candidates = []
        dims_mapping = [-1 for i in range(dims_mapping_len)]
        dims_list = [i for i in range(process_mesh_len)]
        visited = [False for i in range(process_mesh_len)]
        self._generate_dims_mapping_candidates_helper(dims_mapping, dims_list,
                                                      0, visited, candidates)
        self._cached_dims_mapping_candidates[key] = candidates
        return candidates

    def _generate_dist_attr_candidates(self, op_id, dist_op):
        # For now, only allow the process meshes have two dimensions
        process_mesh_len = 2
        serial_op = dist_op.serial_op
        op_dist_attr = dist_op.dist_attr
        if serial_op.type in self._special_ops:
            return [copy.deepcopy(op_dist_attr)]
        # print(
        #     "~~~~~~~~~~~~~~~~~~~~~~~",
        #     serial_op.type,
        #     serial_op.desc.id(),
        #     serial_op.desc.original_id(),
        #     flush=True)
        key = []
        key.append(serial_op.type)
        for input_name in serial_op.input_names:
            key.append(input_name)
            for input_arg_name in serial_op.input(input_name):
                key.append(
                    len(op_dist_attr.get_input_dims_mapping(input_arg_name)))
        for output_name in serial_op.output_names:
            key.append(output_name)
            for output_arg_name in serial_op.output(output_name):
                key.append(
                    len(op_dist_attr.get_output_dims_mapping(output_arg_name)))
        key = tuple(key)

        if key in self._cached_candidates_info:
            cached_dist_attr_candidates = []
            cached_input_arg_names = self._cached_candidates_info[key][0]
            cached_output_arg_names = self._cached_candidates_info[key][1]
            for cached_dist_attr in self._cached_candidates_info[key][2]:
                new_op_dist_attr = copy.deepcopy(dist_op.dist_attr)
                i = 0
                for input_name in serial_op.input_names:
                    for input_arg_name in serial_op.input(input_name):
                        cached_dims_mapping = cached_dist_attr.get_input_dims_mapping(
                            cached_input_arg_names[i])
                        new_op_dist_attr.set_input_dims_mapping(
                            input_arg_name, cached_dims_mapping)
                        i += 1
                i = 0
                for output_name in serial_op.output_names:
                    for output_arg_name in serial_op.output(output_name):
                        cached_dims_mapping = cached_dist_attr.get_output_dims_mapping(
                            cached_output_arg_names[i])
                        new_op_dist_attr.set_output_dims_mapping(
                            output_arg_name, cached_dims_mapping)
                        i += 1
                cached_dist_attr_candidates.append(new_op_dist_attr)
            return cached_dist_attr_candidates

        cached_candidates_info = []
        input_arg_names = []
        for input_name in serial_op.input_names:
            for input_arg_name in serial_op.input(input_name):
                input_arg_names.append(input_arg_name)
        self._cached_candidates_info[key].append(input_arg_names)
        # cached_candidates_info.append(input_arg_names)
        output_arg_names = []
        for output_name in serial_op.output_names:
            for output_arg_name in serial_op.output(output_name):
                output_arg_names.append(output_arg_name)
        self._cached_candidates_info[key].append(output_arg_names)
        # cached_candidates_info.append(output_arg_names)

        new_op_dist_attr = copy.deepcopy(dist_op.dist_attr)
        # Find valid dims_mapping candidates for inputs
        input_names = []
        dims_mapping_generated = []
        inputs_dist_attrs = op_dist_attr.inputs_dist_attrs
        for tensor_name, tensor_dist_attr in inputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            input_names.append(tensor_name)
            if dims_mapping_len < 1:
                dims_mapping_generated.append(
                    [copy.deepcopy(original_dims_mapping)])
            else:
                dims_mapping_generated.append(
                    self._generate_dims_mapping_candidates(dims_mapping_len,
                                                           process_mesh_len))
        input_dims_mapping_candidates = []
        for dims_mapping_list in itertools.product(*dims_mapping_generated):
            dims_mapping_list = list(dims_mapping_list)
            assert len(dims_mapping_list) == len(input_names)
            for i, dims_mapping in enumerate(dims_mapping_list):
                new_op_dist_attr.set_input_dims_mapping(input_names[i],
                                                        dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(
                new_dist_op, fwd=True)
            if dist_op_impls is not None:
                input_dims_mapping_candidates.append(dims_mapping_list)

        # Find valid dims_mapping candidates for outputs
        output_names = []
        dims_mapping_generated = []
        outputs_dist_attrs = op_dist_attr.outputs_dist_attrs
        for tensor_name, tensor_dist_attr in outputs_dist_attrs.items():
            original_dims_mapping = tensor_dist_attr.dims_mapping
            dims_mapping_len = len(original_dims_mapping)
            output_names.append(tensor_name)
            if dims_mapping_len < 1:
                dims_mapping_generated.append(
                    [copy.deepcopy(original_dims_mapping)])
            else:
                dims_mapping_generated.append(
                    self._generate_dims_mapping_candidates(dims_mapping_len,
                                                           process_mesh_len))
        output_dims_mapping_candidates = []
        for dims_mapping_list in itertools.product(*dims_mapping_generated):
            dims_mapping_list = list(dims_mapping_list)
            assert len(dims_mapping_list) == len(output_names)
            for i, dims_mapping in enumerate(dims_mapping_list):
                new_op_dist_attr.set_output_dims_mapping(output_names[i],
                                                         dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(
                new_dist_op, fwd=False)
            if dist_op_impls is not None:
                output_dims_mapping_candidates.append(dims_mapping_list)

        if not input_dims_mapping_candidates and output_dims_mapping_candidates:
            inout_dims_mapping_generated = [[[[-2]]],
                                            output_dims_mapping_candidates]
        elif input_dims_mapping_candidates and not output_dims_mapping_candidates:
            inout_dims_mapping_generated = [
                input_dims_mapping_candidates, [[[-2]]]
            ]
        elif not input_dims_mapping_candidates and not output_dims_mapping_candidates:
            inout_dims_mapping_generated = [[[[-2]]], [[[-2]]]]
        else:
            inout_dims_mapping_generated = [
                input_dims_mapping_candidates, output_dims_mapping_candidates
            ]
        # Find valid dims_mapping generated for both inputs and outputs
        cached_dist_attr_candidates = []
        for inout_dims_mapping_list in itertools.product(
                *inout_dims_mapping_generated):
            assert len(inout_dims_mapping_list) == 2
            if input_dims_mapping_candidates:
                assert len(inout_dims_mapping_list[0]) == len(input_names)
            if output_dims_mapping_candidates:
                assert len(inout_dims_mapping_list[1]) == len(output_names)
            # set the dims_mappings for inputs
            for i, dims_mapping in enumerate(inout_dims_mapping_list[0]):
                if dims_mapping != [-2]:
                    new_op_dist_attr.set_input_dims_mapping(input_names[i],
                                                            dims_mapping)
            # set the dims_mappings for outputs
            for i, dims_mapping in enumerate(inout_dims_mapping_list[1]):
                if dims_mapping != [-2]:
                    new_op_dist_attr.set_output_dims_mapping(output_names[i],
                                                             dims_mapping)
            new_dist_op = DistributedOperator(dist_op.serial_op,
                                              new_op_dist_attr)
            dist_op_impls = find_compatible_distributed_operator_impls(
                new_dist_op, partial=False)
            if dist_op_impls is None:
                continue
            for dist_op_impl in dist_op_impls:
                new_op_dist_attr.impl_type = dist_op_impl.type
                new_op_dist_attr.impl_idx = dist_op_impl.idx
                cached_dist_attr_candidates.append(
                    copy.deepcopy(new_op_dist_attr))
        self._cached_candidates_info[key].append(cached_dist_attr_candidates)
        return self._cached_candidates_info[key][2]
        # cached_candidates_info.append(cached_dist_attr_candidates)
        # return cached_candidates_info[2]

    def construct_space(self):
        inter_node_partitions, intra_node_partitions = self._partition_devices(
            self._num_nodes, self._devices_per_node)
        self._space.choice(
            "inter_node_partitions",
            inter_node_partitions,
            default=inter_node_partitions[0])
        self._space.choice(
            "intra_node_partitions",
            intra_node_partitions,
            default=intra_node_partitions[0])

        dist_ops = self._dist_context._dist_ops_for_program
        for op_id, dist_op in dist_ops.items():
            op_dist_attr_candidates = self._generate_dist_attr_candidates(
                op_id, dist_op)
            self._space.choice(
                str(op_id),
                op_dist_attr_candidates,
                default=op_dist_attr_candidates[0])

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = "".join(str(k) + "=" + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

    def _random_values(self):
        space = TunableSpace()
        collisions = 0
        while True:
            for v in self._space.variables.values():
                space._register(v)
                space.values[v.name] = v.random(self._seed_state)
                self._seed_state += 1
            values = space.values
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_values:
                collisions += 1
                if collisions > self._max_collisions:
                    return None
                continue
            self._tried_values.add(values_hash)
            break
        return values

    def populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": TrialStatus.STOPPED, "values": None}
        return {"status": TrialStatus.RUNNING, "values": values}

    def create_trial(self):
        trial_id = "{{:0{}d}}".format(len(str(self._max_trials)))
        trial_id = trial_id.format(len(self._trials))

        if self._max_trials and len(self._trials) >= self._max_trials:
            status = TrialStatus.STOPPED
            values = None
        else:
            results = self.populate_space()
            status = results["status"]
            values = results["values"]

        space = TunableSpace()
        space.variables = self._space.variables
        space.values = values
        trial = Trial(tunable_space=space, trial_id=trial_id, status=status)
        self._trials[trial.id] = trial

        return trial

    def _apply_pipeline_partition(self, process_mesh_list):
        op_id_to_process_mesh = {}
        total_ops = len(self._dist_context._dist_ops_for_program)
        total_stages = len(process_mesh_list)
        ops_per_stages = total_ops // total_stages
        if ops_per_stages == 0:
            return None
        pipeline_starts = []
        start = 0
        pipeline_starts.append(0)
        for _ in process_mesh_list:
            start += ops_per_stages
            pipeline_starts.append(start)
        pipeline_starts[-1] = total_ops
        start = 1
        sorted_op_ids = sorted(self._dist_context._dist_ops_for_program.keys())
        for idx, op_id in enumerate(sorted_op_ids):
            if idx < pipeline_starts[start]:
                op_id_to_process_mesh[op_id] = process_mesh_list[start - 1]
            else:
                start += 1
                op_id_to_process_mesh[op_id] = process_mesh_list[start - 1]
        # print(
        #     "pipeline partition",
        #     total_ops,
        #     total_stages,
        #     ops_per_stages,
        #     pipeline_starts,
        #     op_id_to_process_mesh,
        #     flush=True)
        return op_id_to_process_mesh

    def _amend_dist_attr(self):
        # Reshape the process mesh of [1, x] to [x] or [x, 1] to [x],
        # and amend the corresponding dims_mapping
        for dist_op in self._dist_context._dist_ops_for_program.values():
            dist_attr = dist_op.dist_attr
            process_mesh = dist_attr.process_mesh
            if process_mesh is None:
                continue
            assert process_mesh.ndim == 2
            dim_of_one = None
            dim_of_other = None
            if process_mesh.topology[0] == 1:
                dim_of_one = 0
                dim_of_other = 1
            elif process_mesh.topology[1] == 1:
                dim_of_one = 1
                dim_of_other = 0
            if dim_of_one is None:
                continue

            dist_attr.process_mesh = ProcessMesh(process_mesh.processes)
            self._dist_context.add_process_mesh(dist_attr.process_mesh)
            for arg_name in dist_attr.inputs_dist_attrs.keys():
                new_dims_mapping = []
                dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                for dim_mapping in dims_mapping:
                    if dim_mapping == dim_of_one:
                        new_dims_mapping.append(-1)
                    elif dim_mapping == dim_of_other:
                        new_dims_mapping.append(0)
                    else:
                        new_dims_mapping.append(dim_mapping)
                dist_attr.set_input_dims_mapping(arg_name, new_dims_mapping)
            for arg_name in dist_attr.outputs_dist_attrs.keys():
                new_dims_mapping = []
                dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                for dim_mapping in dims_mapping:
                    if dim_mapping == dim_of_one:
                        new_dims_mapping.append(-1)
                    elif dim_mapping == dim_of_other:
                        new_dims_mapping.append(0)
                    else:
                        new_dims_mapping.append(dim_mapping)
                dist_attr.set_output_dims_mapping(arg_name, new_dims_mapping)

            dist_op_impls = find_compatible_distributed_operator_impls(
                dist_op, partial=False)
            if dist_op_impls is not None:
                # Select the first compatible dist op impl
                dist_op.dist_attr.impl_type = dist_op_impls[0].type
                dist_op.dist_attr.impl_idx = dist_op_impls[0].idx
            else:
                # Use the default dist op impl
                for arg_name in dist_attr.inputs_dist_attrs.keys():
                    dims_mapping = dist_attr.get_input_dims_mapping(arg_name)
                    for i, _ in enumerate(dims_mapping):
                        dims_mapping[i] = -1
                for arg_name in dist_attr.outputs_dist_attrs.keys():
                    dims_mapping = dist_attr.get_output_dims_mapping(arg_name)
                    for i, _ in enumerate(dims_mapping):
                        dims_mapping[i] = -1
                dist_op.dist_attr.impl_type = "default"
                dist_op.dist_attr.impl_idx = 0

    def eval_trial(self, trial):
        self._dist_context._reset()
        # print_program_with_dist_attr(self._dist_context.serial_main_program, self._dist_context)
        results = None
        inter_node_partition = trial.space.values["inter_node_partitions"]
        intra_node_partition = trial.space.values["intra_node_partitions"]
        process_mesh_list = self._generate_process_mesh_list(
            inter_node_partition, intra_node_partition)
        # print("process_mesh_list",
        #       inter_node_partition,
        #       intra_node_partition,
        #       process_mesh_list,
        #       flush=True)
        op_id_to_process_mesh = self._apply_pipeline_partition(
            process_mesh_list)
        if op_id_to_process_mesh is None:
            print("Operators are less than pipeline stages", flush=True)
            return results

        op_id_to_dist_attr = {}
        for name, value in trial.space.values.items():
            if name != "inter_node_partitions" \
                and name !="intra_node_partitions":
                op_id_to_dist_attr[int(name)] = value

        # print("len assert", len(op_id_to_process_mesh), len(op_id_to_dist_attr), flush=True)
        assert len(op_id_to_process_mesh) == len(op_id_to_dist_attr)

        skip_dist_ops = {}
        for op_id, process_mesh in op_id_to_process_mesh.items():
            dist_op = self._dist_context._dist_ops_for_program[op_id]
            # if not is_elementwise_op(dist_op.serial_op.type):
            dist_op.dist_attr = copy.deepcopy(op_id_to_dist_attr[op_id])
            assert dist_op.dist_attr.impl_type == op_id_to_dist_attr[
                op_id].impl_type
            assert dist_op.dist_attr.impl_idx == op_id_to_dist_attr[
                op_id].impl_idx
            # skip_dist_ops[op_id] = True
            dist_op.dist_attr.process_mesh = process_mesh
        # print_program_with_dist_attr(self._dist_context.serial_main_program, self._dist_context)
        self._amend_dist_attr()

        self._completer.complete_forward_annotation()
        self._dist_context.block_state.parse_forward_blocks(
            self._dist_context.serial_main_program)
        # print_program_with_dist_attr(self._dist_context.serial_main_program, self._dist_context)
        self._parallelizer.parallel_all()

        # print("after reset dist context",
        #       self._dist_context._dist_ops_for_program.keys(),
        #       self._dist_context._dist_tensors_for_program.keys(), flush=True)
        # for block in self._dist_context.serial_main_program.blocks:
        #     for tensor in block.vars.values():
        #         print("tensor id",
        #               tensor.desc.id(),
        #               "tensor original_id",
        #               tensor.desc.original_id(),
        #               flush=True)
        #     for op in block.ops:
        #         print("op id",
        #               op.desc.id(),
        #               "op original_id",
        #               op.desc.original_id(),
        #               flush=True)
        # print_program_with_dist_attr(completed_main_program, self._dist_context)

        # TODO: eval the partiton strategy by calling the cost model
        return results

    def update_trial(self, trial_id, metrics, step=0):
        trial = self._trials[trial_id]
        for metric_name, metric_value in metrics.items():
            trial.metrics.update(metric_name, metric_value, step=step)
        return trial.status

    def estimate_trail(self):
        assert self._cluster is not None
        if self._estimator is None:
            self._estimator = CostEstimator(
                self._dist_context.serial_main_program,
                self._cluster,
                loop_count=self._loop_count)
        global_cost = self._estimator.estimate(self._dist_context)
        return global_cost.time

    def tune(self):
        self.times = 0
        self.construct_space()
        while True:
            print("times: ", self.times)
            print("tune 1", flush=True)
            # print_program_with_dist_attr(self._dist_context.serial_program, self._dist_context)
            trial = self.create_trial()
            print("tune 2", flush=True)
            if trial.status == TrialStatus.STOPPED:
                break
            print("tune 3", flush=True)
            results = self.eval_trial(trial)
            print("tune 4 serial program", flush=True)
            print_program_with_dist_attr(self._dist_context.serial_main_program,
                                         self._dist_context)
            time = self.estimate_trail()
            print("exec time: ", time)
            print("----------------------------")
            self.times += 1
            # self.update_trial(trial.id, results)
            # print("5")
