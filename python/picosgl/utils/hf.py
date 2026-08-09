import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    return AutoConfig.from_pretrained(model_path)


def _load_config_json(model_path: str) -> dict:
    """Read config.json either from a local dir or via huggingface_hub."""
    import json
    import os

    if os.path.isdir(model_path):
        path = os.path.join(model_path, "config.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No config.json in {model_path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(hf_hub_download(model_path, "config.json").read_text())


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    try:
        config = _load_hf_config(model_path)
        return type(config)(**config.to_dict())
    except Exception:
        # Fallback for architectures not yet registered in this transformers version
        # (e.g. Qwen3.5). Builds a lightweight attribute object mirroring the HF config;
        # picosgl only reads attributes off it, so this is sufficient.
        from types import SimpleNamespace

        logger = __import__("logging").getLogger("picosgl.hf")
        logger.warning("AutoConfig failed for %s, falling back to raw config.json", model_path)
        # Mirror HF PretrainedConfig: dicts become attribute-accessible namespaces, except
        # rope_parameters / rope_scaling which picosgl reads as dicts.
        _DICT_KEYS = ("rope_parameters", "rope_scaling")

        def _ns(d: dict) -> Any:
            return SimpleNamespace(
                **{
                    k: (_ns(v) if isinstance(v, dict) and k not in _DICT_KEYS else v)
                    for k, v in d.items()
                }
            )

        return _ns(_load_config_json(model_path))


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )