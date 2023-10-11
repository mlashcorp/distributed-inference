from contextlib import nullcontext
from multiprocessing import Process

import tiktoken
import torch

from distributed_model import DGPT
from distributed_state import ReduceController
from utils import process_print, setup_distributed_state

# Used to reproduce the same results with Torch
seed = 1337


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

enc = tiktoken.get_encoding("gpt2")
def encode(s): return enc.encode(s, allowed_special={"<|endoftext|>"})
def decode(l): return enc.decode(l)


def run_node(reduce_controller: ReduceController,
             worker_index: int,
             number_of_workers: int,
             model_name: str,
             device: str,
             x: torch.Tensor,
             max_new_tokens: int,
             temperature: float,
             top_k: int):

    process_print(worker_index, "starting worker")

    model = DGPT.from_pretrained(
        model_name, reduce_controller, worker_index, number_of_workers)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float32)

    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens,
                               temperature=temperature, top_k=top_k)
            process_print(
                worker_index, f"generated text: " + decode(y[0].tolist()))


def run_distributed_inference(start: str = "\n", max_new_tokens: int = 100, temperature: float = 1,
                              top_k: int = 200, device: str = "cpu", number_of_workers: int = 1):

    # Prepare input for the LLM
    start_ids = encode(start)
    x: torch.Tensor = (torch.tensor(
        start_ids, dtype=torch.long, device=device)[None, ...])

    workers = []

    # Get the ReduceController instance to simulate distributed state
    reduce_controller = setup_distributed_state(number_of_workers)

    for worker_index in range(number_of_workers):
        workers.append(Process(target=run_node, args=[
                       reduce_controller, worker_index, number_of_workers,
                       "gpt2", device, x, max_new_tokens, temperature, top_k]))

    # Launch processes
    for w in workers:
        w.start()

    # Wait for processes to finish
    for w in workers:
        w.join()
