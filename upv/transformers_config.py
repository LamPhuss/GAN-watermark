# ============================================================
# transformers_config.py  (ported from MarkLLM — standalone)
# Description: Minimal config wrapper for HuggingFace model + tokenizer
# ============================================================


class TransformersConfig:
    """
    Minimal configuration wrapper for a HuggingFace model and tokenizer.
    Ported from MarkLLM/utils/transformers_config.py — no external deps.
    """

    def __init__(self, model, tokenizer, vocab_size=None, device="cuda", **kwargs):
        """
        Args:
            model:      HuggingFace AutoModelForCausalLM instance
            tokenizer:  HuggingFace AutoTokenizer instance
            vocab_size: Optional override; defaults to len(tokenizer)
            device:     "cuda" or "cpu"
            **kwargs:   Extra generation kwargs (max_new_tokens, do_sample, etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        # All extra kwargs become generation kwargs (max_new_tokens, do_sample, …)
        self.gen_kwargs = dict(kwargs)
