# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import json
import shutil

from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.cost import CommContext

cluster_json = """
{ 
    "alpha_latency": {"inter": {"ring": "NVL", "tree": "PHB"},
                      "intra": {"ring": "NET", "tree": "NET"},
                      "base": {"ring": 8.4, "tree": 0},
                      "switch": 10.0},
    "machines": [
        {
            "hostname": "yq01-sys-hic-v100-box-a225-0266",
            "addr": "10.127.9.147",
            "port": "60009",
            "devices": [
                {
                    "global_id": 0,
                    "local_id": 0,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 1,
                    "local_id": 1,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 2,
                    "local_id": 2,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 3,
                    "local_id": 3,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 4,
                    "local_id": 4,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 5,
                    "local_id": 5,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 6,
                    "local_id": 6,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 7,
                    "local_id": 7,
                    "type": "GPU",
                    "model": " Tesla V100-SXM2-32GB",
                    "memory": "32",
                    "sp_gflops": "15700",
                    "dp_gflops": "7800"
                },
                {
                    "global_id": 8,
                    "local_id": 0,
                    "type": "CPU",
                    "arch": "x86_64",
                    "vendor": "GenuineIntel",
                    "model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GH",
                    "memory": "502",
                    "sp_gflops": "150",
                    "dp_gflops": "75"
                },
                {
                    "global_id": 9,
                    "local_id": 0,
                    "type": "NIC",
                    "width": 12.5,
                    "ip": "10.127.9.147"
                }
            ],
            "links": [
                {
                    "source_global_id": 0,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 0,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 1,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 7,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 2,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 4,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 5,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 6,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 3,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 0,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 4,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 1,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 5,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 2,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 3,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 7,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 6,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 0,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 1,
                    "type": "NVB",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 2,
                    "type": "NVB",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 3,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 4,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 5,
                    "type": "NVL",
                    "bandwidth": 22.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 6,
                    "type": "NVL",
                    "bandwidth": 44.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 7,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 8,
                    "target_global_id": 9,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 0,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 1,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 2,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 3,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 4,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 5,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 6,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 7,
                    "type": "PHB",
                    "bandwidth": 12.0
                },
                {
                    "source_global_id": 9,
                    "target_global_id": 8,
                    "type": "PHB",
                    "bandwidth": 12.0
                }
            ]
        }
    ]
}
"""


class TestCommOpCost(unittest.TestCase):
    def test_cluster_alpha_latency(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        comm_context = CommContext(cluster)
        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)


if __name__ == "__main__":
    unittest.main()
