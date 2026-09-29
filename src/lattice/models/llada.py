"""The released LLaDA-8B-Base and the GPT-2-large scorer, loaded from local directories.

Both load in bfloat16 from the directories setup/prepare_llada.sh writes; nothing is downloaded
here. LLaDA's modelling code comes with its weights (trust_remote_code).
"""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

import torch


def load_llada(path, device):
    """(model, tokenizer, mask_id): LLaDA-8B-Base in eval mode and its mask token."""
    import transformers
    config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True,
                                                     local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                                           local_files_only=True)
    model = transformers.AutoModel.from_pretrained(path, config=config, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16,
                                                   local_files_only=True).to(device).eval()
    return model, tokenizer, int(config.mask_token_id)


def load_scorer(path, device):
    """(model, tokenizer, context): the causal scorer in eval mode and its context length."""
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                              local_files_only=True).to(device).eval()
    context = int(getattr(model.config, "max_position_embeddings", None) or model.config.n_positions)
    return model, tokenizer, context


def logits(model, x):
    """The model's logits for the token ids x, whichever output type it returns."""
    out = model(x)
    return out.logits if hasattr(out, "logits") else out[0]


def environment(model_path, device):
    """What a record depends on beyond its settings: the weights (the sha256 of the model's
    config.json), the GPU model (generator draws and bfloat16 arithmetic differ between models of
    card, so a run must resume on the same model) and the library versions."""
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    return dict(model_config_sha256=hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest(),
                device=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu",
                torch=torch.__version__, transformers=version("transformers"))


def sync(device):
    """Wait for the device, so that a timer read after it counts the work queued before it."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
