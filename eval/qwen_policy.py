"""Qwen policy: observation JSON (+ last 2 actions) -> action JSON, greedy decoding + strict validation."""
from __future__ import annotations

import torch

from train.prompt import parse_action, prompt_text


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenPolicy:
    name = "qwen"

    def __init__(self, model, tokenizer, max_new_tokens=48, name="qwen"):
        self.model = model
        self.tok = tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_new_tokens = max_new_tokens
        self.name = name
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    @classmethod
    def load(cls, model_path, adapter_path=None, max_new_tokens=48, name="qwen"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = pick_device()
        tok = AutoTokenizer.from_pretrained(adapter_path or model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(dev)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return cls(model, tok, max_new_tokens, name)

    def reset(self, seed, obs):
        return {}

    @torch.no_grad()
    def act_batch(self, obs_list, hist_list):
        prompts = [prompt_text(self.tok, o, h) for o, h in zip(obs_list, hist_list)]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  eos_token_id=[self.im_end, self.tok.eos_token_id],
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [(parse_action(t), t.strip()) for t in texts]

    def act(self, obs, history):
        return self.act_batch([obs], [history])[0]
