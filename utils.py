from multiprocessing.managers import BaseManager
from multiprocessing import Lock
from termcolor import colored
from distributed_state import ReduceController

lock = Lock()

PROCESS_COLOR_MAPPING = {
    0: "red",
    1: "yellow",
    2: "blue",
    3: "magenta",
    4: "cyan",
    6: "white"
}


def process_print(worker_index: int, msg: str):
    with lock:
        print(colored(f"worker_{worker_index} -> ",
                      PROCESS_COLOR_MAPPING[worker_index]) + msg)


def estimate_model_size(model):
    mem_params = sum([param.nelement()*param.element_size()
                     for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size()
                   for buf in model.buffers()])
    return mem_params + mem_bufs


def setup_distributed_state(number_of_workers: int = 1):
    BaseManager.register('ReduceController', ReduceController)
    manager = BaseManager()
    manager.start()
    return manager.ReduceController(
        number_of_workers=number_of_workers)
