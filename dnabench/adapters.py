import os
import sys

import torch

from .env import mean_pool

HYENA_PAD_ID = 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_dir(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _freeze(model, device):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)


class Adapter:
    uses_attention_mask = True

    def __init__(self, name, cfg):
        self.name = name
        self.tag = cfg["tag"]
        self.hf_id = cfg["hf_id"]
        self.model_dir = cfg.get("model_dir", "")
        self.local_cache = cfg.get("local_cache", "")
        self.trust_remote_code = cfg.get("trust_remote_code", False)

    def build_tokenizer(self, max_length):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=self.trust_remote_code
        )

    def build_model(self, num_layers, device):
        raise NotImplementedError

    def pool(self, model, batch, device):
        raise NotImplementedError


class Dnabert2Adapter(Adapter):
    def build_model(self, num_layers, device):
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
        config.use_flash_attn = False
        config.num_hidden_layers = num_layers
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 3
        model = AutoModel.from_pretrained(
            self.hf_id, config=config, trust_remote_code=True
        )
        return _freeze(model, device)

    def pool(self, model, batch, device):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        return mean_pool(hidden, mask).cpu().numpy()


class GenaAdapter(Adapter):
    def _add_cache_to_path(self):
        import glob

        path = os.path.expanduser(self.local_cache)
        if any(ch in path for ch in "*?["):
            matches = sorted(glob.glob(path))
            if not matches:
                raise FileNotFoundError(f"GENA local cache not found: {path}")
            path = matches[-1]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"GENA local cache not found: {path}")
        if path not in sys.path:
            sys.path.insert(0, path)

    def build_tokenizer(self, max_length):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id)
        tok.pad_token = tok.unk_token
        return tok

    def build_model(self, num_layers, device):
        self._add_cache_to_path()
        from modeling_bert import BertModel
        from transformers import BertConfig

        config = BertConfig.from_pretrained(self.hf_id)
        model = BertModel.from_pretrained(self.hf_id, config=config)
        model.encoder.layer = model.encoder.layer[:num_layers]
        return _freeze(model, device)

    def pool(self, model, batch, device):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        return mean_pool(out.last_hidden_state, mask).cpu().numpy()


class GroverAdapter(Adapter):
    def build_model(self, num_layers, device):
        from transformers import AutoConfig, AutoModelForMaskedLM

        config = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0
        model = AutoModelForMaskedLM.from_pretrained(
            self.hf_id,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        model.bert.encoder.layer = model.bert.encoder.layer[:num_layers]
        return _freeze(model, device)

    def pool(self, model, batch, device):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return mean_pool(out.hidden_states[-1], mask).cpu().numpy()


class Ntv2Adapter(Adapter):
    def build_model(self, num_layers, device):
        from transformers import AutoConfig, AutoModelForMaskedLM

        config = AutoConfig.from_pretrained(self.hf_id, trust_remote_code=True)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0
        model = AutoModelForMaskedLM.from_pretrained(
            self.hf_id,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        model.esm.encoder.layer = model.esm.encoder.layer[:num_layers]
        return _freeze(model, device)

    def pool(self, model, batch, device):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return mean_pool(out.hidden_states[-1], mask).cpu().numpy()


class HyenaAdapter(Adapter):
    uses_attention_mask = False

    def _source(self):
        return _resolve_dir(self.model_dir) or self.hf_id

    def build_tokenizer(self, max_length):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self._source(), trust_remote_code=True)
        tok.model_max_length = max_length
        return tok

    def build_model(self, num_layers, device):
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(self._source(), trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self._source(), config=config, trust_remote_code=True
        )
        model.backbone.layers = model.backbone.layers[:num_layers]
        return _freeze(model, device)

    def pool(self, model, batch, device):
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            hidden = model(input_ids=ids, output_hidden_states=False, return_dict=False)
        mask = (ids != HYENA_PAD_ID).float()
        return mean_pool(hidden, mask).cpu().numpy()


ADAPTERS = {
    "dnabert2": Dnabert2Adapter,
    "gena": GenaAdapter,
    "grover": GroverAdapter,
    "ntv2": Ntv2Adapter,
    "hyena": HyenaAdapter,
}


def build_adapter(name, cfg):
    key = cfg["adapter"]
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter: {key}")
    return ADAPTERS[key](name, cfg)
