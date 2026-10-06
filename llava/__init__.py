import transformers

# peft >= 0.14 imports these at import time; the pinned transformers commit predates them and they are not used here
for _name in ("EncoderDecoderCache", "HybridCache"):
    if not hasattr(transformers, _name):
        setattr(transformers, _name, transformers.DynamicCache)

from .model import LlavaLlamaForCausalLM
