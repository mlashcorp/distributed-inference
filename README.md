# distributed-inference

## Overview
This repository contains an implementation of the [Megatron-LM paper](https://arxiv.org/abs/1909.08053) based on [Andrej Karpathy's Nano GPT codebase](https://github.com/karpathy/nanoGPT).

The purpose of this exercise was to explore distributed inference so the code was simplified where possible for readability (e.g., removing anything specifically related to training, implementing attention without torch's flash attention).

I’ve opted to use Andrej's GPT-2 implementation because it's super small and easy to understand. Also, he has a great video walking through all the code on YouTube. This code simulates machine-to-machine communication by leveraging Python's BaseManager process to hold the distributed state between processes and simulate a blocking, distributed, all reduce operation.

There are a bunch of files in here, let's go over them:

```
.
├── README.md
├── run.py                       <- CLI application wrapper
├── distributed_inference.py     <- Called by the CLI. Launches the processes and waits for the results.
├── distributed_model.py         <- This is the only file that really matters. GPT-2 model in pytorch, using tensor parallelism
├── distributed_state.py         <- Used for inter-process state sharing. Simulated distributed all reduce
├── download_model.py            <- Use this to download the GPT-2 124M model from HF
├── models
│   └── <MODEL WILL BE DOWNLOADED HERE>
├── requirements.txt
└── utils.py    
```

## Setup 

First, setup the required dependencies.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the GPT-2 model locally with the following command

```
python download_models.py
```

## Run

```
python run.py
Prompt: <Insert start of prompt here>
```

Usage options
```
Usage: run.py [OPTIONS]

Options:
  -mt, --max_tokens INTEGER  How many new tokens to generate.
  -t, --temperature INTEGER  The temperature of the sampling.
  -tk, --top_k INTEGER       The top_k of the sampling.
  -d, --device TEXT          The device to run on.
  -w, --workers [1|2|3|4]    The number of workers to simulate.
  -s, --start TEXT           The start of the prompt to use for generation.
  --help                     Show this message and exit.
```

## References

1. https://github.com/karpathy/nanoGPT
2. https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4139s&ab_channel=AndrejKarpathy
3. https://arxiv.org/abs/1909.08053
4. https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce
5. https://huggingface.co/docs/safetensors/index#load-tensors
