import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# Create models folder in current working directory, if it does not exist
Path("models").mkdir(exist_ok=True)

# Force download of the GPT-2 124M model in safetensors format
hf_hub_download(repo_id="gpt2", filename="model.safetensors",
                local_dir="models", force_download=True)

os.rename("models/model.safetensors", "models/gpt2.safetensors")
