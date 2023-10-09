from contextlib import nullcontext
import torch
import tiktoken
from distributed_model import DGPT
from distributed_state import ReduceController
import threading
from multiprocessing import Process
from multiprocessing.managers import BaseManager
seed = 1337


torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

enc = tiktoken.get_encoding("gpt2")
def encode(s): return enc.encode(s, allowed_special={"<|endoftext|>"})
def decode(l): return enc.decode(l)


def run_node(reduce_controller: ReduceController,
             worker_index: int,
             number_of_workers: int,
             model_name: str,
             x: torch.Tensor,
             max_new_tokens: int,
             temperature: float,
             top_k: int):

    print(f"running worker_{worker_index}")

    model = DGPT.from_pretrained(
        model_name, reduce_controller, worker_index, number_of_workers)

    # for later use in torch.autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float32)

    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens,
                               temperature=temperature, top_k=top_k)
            print("\n ->" + decode(y[0].tolist()))


model_name = 'gpt2'
# or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
start = "Tom Cruise is"
max_new_tokens = 100  # number of tokens generated in each sample
# 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
temperature = 1
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
number_of_workers = 4

# Used to simulate a distributed all_reduce operation
# reduce_controller = ReduceController(number_of_workers=number_of_workers)
if __name__ == '__main__':
    BaseManager.register('ReduceController', ReduceController)
    manager = BaseManager()
    manager.start()
    reduce_controller = manager.ReduceController(
        number_of_workers=number_of_workers)

    # Prepare input for the LLM
    start_ids = encode(start)
    x: torch.Tensor = (torch.tensor(
        start_ids, dtype=torch.long, device=device)[None, ...])

    # No, we simulate two distributed nodes
    workers = []

    for worker_index in range(number_of_workers):
        workers.append(Process(target=run_node, args=[
                    reduce_controller, worker_index, number_of_workers, model_name, x, max_new_tokens, temperature, top_k]))

    for w in workers:
        w.start()

    for w in workers:
        w.join()
