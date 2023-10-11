# We need a way to simulate an all_reduce operation. And eventually, to
# implement it to communicate over different machines

# What we will want, is to simulate that all nodes send their results to a common
# pool of results, and get the final result when all worker chunks are commited.

# This simulates a blocking all_reduce.

# ref. https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce

import multiprocessing

import torch


class ReduceController:
    """Naive implementation of a simulated distributed all_reduce SUM

    Each all reduce is scoped to an operation id. Each worker must call
    this controller from a different thread. The final result is only
    returned when all workers have provided their tensor chunk.

    operation ids cannot be re-used. no effort was done to ensure that
    state is deleted for the synchronization points.
    """

    def __init__(self, number_of_workers: int = 1):
        self.number_of_workers = number_of_workers
        self.state = dict()
        self.cv = multiprocessing.Condition()

    def all_reduce(self, op_id: str, tensor: torch.Tensor):
        with self.cv:
            if op_id not in self.state:
                self.state[op_id] = (tensor, 1)
            else:
                state_tensor, count = self.state[op_id]
                self.state[op_id] = (
                    torch.add(state_tensor, tensor), count + 1)

            if self.state[op_id][1] < self.number_of_workers:
                self.cv.wait()
            else:
                self.cv.notify_all()
            return self.state[op_id][0]

    def lock(self):
        return self.cv.acquire()

    def unlock(self):
        return self.cv.release()
