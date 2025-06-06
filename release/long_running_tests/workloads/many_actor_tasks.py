# This workload tests submitting many actor methods.
import time

import numpy as np

import ray
from ray.cluster_utils import Cluster
from ray._private.test_utils import safe_write_to_results_json


def update_progress(result):
    result["last_update"] = time.time()
    safe_write_to_results_json(result)


object_store_memory = 10**8
num_nodes = 10

message = (
    "Make sure there is enough memory on this machine to run this "
    "workload. We divide the system memory by 2 to provide a buffer."
)
assert (
    num_nodes * object_store_memory < ray._common.utils.get_system_memory() / 2
), message

# Simulate a cluster on one machine.

cluster = Cluster()
for i in range(num_nodes):
    cluster.add_node(
        redis_port=6379 if i == 0 else None,
        num_cpus=5,
        num_gpus=0,
        resources={str(i): 2},
        object_store_memory=object_store_memory,
        dashboard_host="0.0.0.0",
    )
ray.init(address=cluster.address)

# Run the workload.


@ray.remote
class Actor(object):
    def __init__(self):
        self.value = 0

    def method(self):
        self.value += 1
        return np.zeros(1024, dtype=np.uint8)


actors = [
    Actor._remote([], {}, num_cpus=0.1, resources={str(i % num_nodes): 0.1})
    for i in range(num_nodes * 5)
]

iteration = 0
start_time = time.time()
previous_time = start_time
while True:
    for _ in range(100):
        previous_ids = [a.method.remote() for a in actors]

    ray.get(previous_ids)

    new_time = time.time()
    print(
        "Iteration {}:\n"
        "  - Iteration time: {}.\n"
        "  - Absolute time: {}.\n"
        "  - Total elapsed time: {}.".format(
            iteration, new_time - previous_time, new_time, new_time - start_time
        )
    )
    update_progress(
        {
            "iteration": iteration,
            "iteration_time": new_time - previous_time,
            "absolute_time": new_time,
            "elapsed_time": new_time - start_time,
        }
    )
    previous_time = new_time
    iteration += 1
