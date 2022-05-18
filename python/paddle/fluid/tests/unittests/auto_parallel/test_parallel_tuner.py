# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os
import json

import paddle
import numpy as np
import paddle.nn as nn
import paddle.utils as utils
import paddle.static as static
import paddle.nn.functional as F

from paddle.distributed import fleet
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.utils import make_data_unshard
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr
from paddle.distributed.auto_parallel.tuner.parallel_tuner import ParallelTuner

import sys
sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from test_cluster import cluster_json

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512
_g_process_mesh = [[0, 1], [2, 3]]


def get_random_inputs_and_labels(input_shape, label_shape):
    input = np.random.random(size=input_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('float32')
    return input, label


def batch_generator_creator():
    def __reader__():
        for _ in range(batch_size):
            batch_input, batch_label = get_random_inputs_and_labels(
                [batch_size, sequence_len, hidden_size],
                [batch_size, sequence_len, 1])
            yield batch_input, batch_label

    return __reader__


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        param_initializer = nn.initializer.Normal(
            mean=0.0, std=initializer_range)

        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.linear0 = nn.Linear(
            d_model,
            dim_feedforward,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None)
        self.linear1 = nn.Linear(
            dim_feedforward,
            d_model,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None)

    def forward(self, input):
        out = self.norm(input)
        auto.shard_tensor(
            self.linear0.weight,
            dist_attr={
                "process_mesh": _g_process_mesh[0],
                "dims_mapping": [-1, 0]
            })
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        auto.shard_tensor(
            self.linear1.weight,
            dist_attr={
                "process_mesh": _g_process_mesh[1],
                "dims_mapping": [0, -1]
            })
        out = self.linear1(out)

        return out


def loop_cond(i, loop_len, input_array):
    return i < loop_len


def loop_body(i, loop_len, input_array):
    pre_input = paddle.tensor.array_read(array=input_array, i=i)
    mlp_while0 = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02)

    mlp_while1 = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02)

    output = mlp_while0(pre_input)
    cur_pred = mlp_while1(output)

    # 更新循环条件
    i = paddle.increment(x=i, value=1)
    paddle.tensor.array_write(cur_pred, array=input_array, i=i)
    return i, loop_len, input_array


def get_program_v1():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    with static.program_guard(train_program, start_program):

        # 循环计数器
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        # 循环次数
        loop_len = paddle.full(shape=[1], fill_value=epoch_num, dtype='int64')

        # input
        input = static.data(
            name="input",
            shape=[batch_size, sequence_len, hidden_size],
            dtype='float32')
        label = static.data(
            name="label", shape=[batch_size, sequence_len, 1], dtype='float32')
        data_holder = [input, label]
        # dataloader
        dataloader = paddle.io.DataLoader.from_generator(
            feed_list=data_holder, capacity=4 * batch_size, iterable=False)
        dataloader.set_batch_generator(
            batch_generator_creator(), places=paddle.static.cuda_places())
        # data dist_attr
        auto.shard_tensor(
            input,
            dist_attr={
                "process_mesh": _g_process_mesh[0],
                "dims_mapping": [0, -1, -1]
            })
        auto.shard_tensor(
            label,
            dist_attr={
                "process_mesh": _g_process_mesh[0],
                "dims_mapping": [0, -1, -1]
            })

        mlp_start = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_start(input)

        input_array = paddle.tensor.array_write(pred, i)
        i, loop_len, input_array = static.nn.while_loop(
            cond=loop_cond,
            body=loop_body,
            loop_vars=[i, loop_len, input_array])
        end_pred = paddle.tensor.array_read(array=input_array, i=i)

        mlp_end = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_end(end_pred)

        error_cost = paddle.nn.functional.square_error_cost(pred, label)
        # error_cost = paddle.nn.functional.square_error_cost(end_pred, label)
        loss = paddle.mean(error_cost)

        optimizer = paddle.optimizer.Adam(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None)

        feed_vars = {"inputs": [input], "labels": [label]}
        fetch_vars = {"loss": [loss]}

    return train_program, start_program, dataloader, loss, optimizer, feed_vars, fetch_vars


def get_program():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)

    train_program = static.Program()
    start_program = static.Program()
    with static.program_guard(train_program, start_program):

        # input
        input = static.data(
            name="input",
            shape=[batch_size, sequence_len, hidden_size],
            dtype='float32')
        label = static.data(
            name="label", shape=[batch_size, sequence_len, 1], dtype='float32')
        data_holder = [input, label]
        # dataloader
        dataloader = paddle.io.DataLoader.from_generator(
            feed_list=data_holder, capacity=4 * batch_size, iterable=False)
        dataloader.set_batch_generator(
            batch_generator_creator(), places=paddle.static.cuda_places())
        # data dist_attr
        auto.shard_tensor(
            input,
            dist_attr={
                "process_mesh": _g_process_mesh[0],
                "dims_mapping": [0, -1, -1]
            })
        auto.shard_tensor(
            label,
            dist_attr={
                "process_mesh": _g_process_mesh[0],
                "dims_mapping": [0, -1, -1]
            })

        mlp_start = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_start(input)

        mlp_mid = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_mid(pred)

        mlp_end = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            dropout_ratio=0.1,
            initializer_range=0.02)
        pred = mlp_end(pred)

        error_cost = paddle.nn.functional.square_error_cost(pred, label)
        loss = paddle.mean(error_cost)

        optimizer = paddle.optimizer.Adam(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None)

        feed_vars = {"inputs": [input], "labels": [label]}
        fetch_vars = {"loss": [loss]}

    return train_program, start_program, dataloader, loss, optimizer, feed_vars, fetch_vars


def get_program_v3():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)
    place = paddle.set_device("gpu")
    gpus = [0, 1]
    batch_size = 8
    sequence_len = 512
    vocab_size = 1000

    train_program = static.Program()
    start_program = static.Program()
    modeling.init_global()
    with static.program_guard(train_program, start_program):
        tokens = paddle.static.data(
            name="tokens", shape=[batch_size, sequence_len], dtype='int64')
        position_ids = paddle.static.data(
            name="position_ids",
            shape=[batch_size, sequence_len],
            dtype='int64')
        attention_mask = paddle.static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32')
        labels = paddle.static.data(
            name="labels", shape=[batch_size, sequence_len], dtype='int64')
        loss_mask = paddle.static.data(
            name="loss_mask", shape=[batch_size, sequence_len], dtype='float32')
        data_holder = [tokens, position_ids, attention_mask, labels, loss_mask]

        gpt = GPTModel(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=256,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=1024,
            type_vocab_size=1,
            initializer_range=0.02,
            pad_token_id=0,
            eos_token_id=7,
            bos_token_id=0,
            eol_token_id=3)

        model = GPTForPretraining(
            gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02)
        preds = model(tokens, position_ids, attention_mask)
        criterion = GPTPretrainingCriterion()
        loss = criterion(preds, labels, loss_mask)

        optimizer = paddle.fluid.optimizer.AdamOptimizer(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None)

        feed_vars = {
            "inputs": [tokens, position_ids, attention_mask, loss_mask],
            "labels": [labels]
        }
        fetch_vars = {"loss": [loss]}

    return train_program, start_program, None, loss, optimizer, feed_vars, fetch_vars


class TestParallelTuner(unittest.TestCase):
    def setUp(self):
        train_program, start_program, dataloader, loss, optimizer, feed_vars, fetch_vars = get_program_v3(
        )
        dist_context = DistributedContext(train_program, start_program,
                                          optimizer, loss, feed_vars,
                                          fetch_vars)
        dist_context.initialize()
        dist_context.block_state.parse_forward_blocks(
            dist_context.serial_main_program)

        self.num_nodes = 1
        self.device_per_nodes = 8
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        self.parallel_tuner = ParallelTuner(
            dist_context,
            num_nodes=self.num_nodes,
            devices_per_node=self.device_per_nodes,
            cluster=cluster,
            loop_count=10,
            mode="test")

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)

# def test_generate_process_mesh_candidates(self):
#     process_mesh_candidates = self.parallel_tuner._partition_devices(1, 8)
#     print(process_mesh_candidates)

# def test_generate_dims_mapping_candidates(self):
#     dims_mapping_candidates = self.parallel_tuner._generate_dims_mapping_candidates(
#         1, 1)
#     print(dims_mapping_candidates, "\n\n")
#     dims_mapping_candidates = self.parallel_tuner._generate_dims_mapping_candidates(
#         2, 1)
#     print(dims_mapping_candidates, "\n\n")
#     dims_mapping_candidates = self.parallel_tuner._generate_dims_mapping_candidates(
#         3, 2)
#     print(dims_mapping_candidates, "\n\n")
#     dims_mapping_candidates = self.parallel_tuner._generate_dims_mapping_candidates(
#         3, 2)
#     print(dims_mapping_candidates, "\n\n")
#     dims_mapping_candidates = self.parallel_tuner._generate_dims_mapping_candidates(
#         4, 3)
#     print(dims_mapping_candidates, "\n\n")

# def test_construct_space(self):
#     self.parallel_tuner.construct_space()

# def test_create_trial(self):
#     self.parallel_tuner.construct_space()
#     self.parallel_tuner.create_trial()

# def test_eval_trial(self):
#     self.parallel_tuner.construct_space()
#     trail = self.parallel_tuner.create_trial()
#     self.parallel_tuner.eval_trial(trail.id)

    def test_tune(self):

        self.parallel_tuner.tune()

if __name__ == "__main__":
    unittest.main()
