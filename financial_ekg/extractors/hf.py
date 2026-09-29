from __future__ import annotations

import gc
import sys
from typing import Any, Dict, Optional

from financial_ekg.utils.json_repair import (
    extract_first_json_object,
    normalize_llm_output_text,
    print_parse_failure_visualization,
)


class HFExtractor:
    """Local Hugging Face JSON extractor."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 4096,
        disable_thinking: bool = False,
    ) -> None:
        """Load a Hugging Face causal or image-text model for JSON extraction.

        Args:
            model_name: Hugging Face model identifier or local model path.
            device: Target device string such as `cuda`, `cuda:0`, or `cpu`.
            load_in_4bit: Whether to request bitsandbytes 4-bit quantization.
            max_new_tokens: Default generation budget for each extraction call.
            disable_thinking: Whether to ask Qwen-style chat templates to skip
                reasoning blocks and emit final JSON directly.

        Returns:
            None.
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import AutoModelForImageTextToText
        except Exception:  # pragma: no cover - depends on transformers version
            AutoModelForImageTextToText = None

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.disable_thinking = disable_thinking
        self.json_assistant_prefill = ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto" if device.startswith("cuda") else None,
        }
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                print("WARNING: CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
                device = "cpu"
                kwargs["device_map"] = None
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                print(f"WARNING: Could not enable 4-bit quantization: {e}", file=sys.stderr)

        model_cls = AutoModelForCausalLM
        architecture_names = [str(name).lower() for name in getattr(config, "architectures", []) or []]
        if (
            AutoModelForImageTextToText is not None
            and (
                getattr(config, "model_type", "") == "qwen3_5"
                or any("conditionalgeneration" in name for name in architecture_names)
            )
        ):
            model_cls = AutoModelForImageTextToText

        try:
            self.model = model_cls.from_pretrained(model_name, **{k: v for k, v in kwargs.items() if v is not None})
        except Exception as exc:
            if model_cls is AutoModelForCausalLM or AutoModelForImageTextToText is None:
                raise
            print(
                f"WARNING: AutoModelForImageTextToText failed for {model_name} ({exc}). "
                "Falling back to AutoModelForCausalLM.",
                file=sys.stderr,
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **{k: v for k, v in kwargs.items() if v is not None})
        if not device.startswith("cuda"):
            self.model.to(device)
        self.device = device
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompt: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate deterministic JSON from a prompt and parse it leniently.

        Args:
            prompt: Fully rendered extraction prompt.
            max_new_tokens: Optional per-call token budget. When omitted or
                non-positive, the extractor default is used.

        Returns:
            Parsed extraction dictionary. The result always includes generation
            metadata and includes `_parse_error=True` when JSON recovery fails.
        """
        # We restore these cache clears! They are vital for sequential long-context processing.
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise financial event extraction engine for event knowledge graphs. "
                    "Extract only article-supported financial events for all companies discussed in the article. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            template_kwargs = {"enable_thinking": False} if self.disable_thinking else {}
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        else:
            input_text = "\n\n".join([m["role"].upper() + ": " + m["content"] for m in messages]) + "\nASSISTANT:"
        if self.json_assistant_prefill:
            input_text += self.json_assistant_prefill

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=False)
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        requested_tokens = max_new_tokens if max_new_tokens is not None and max_new_tokens > 0 else self.max_new_tokens
        fallback_candidates = [requested_tokens, 3072, 2048, 1024, 768, 512, 256]

        generation_budgets = []
        for budget in fallback_candidates:
            if 0 < budget <= requested_tokens and budget not in generation_budgets:
                generation_budgets.append(budget)

        last_error: Optional[Exception] = None
        output_ids = None
        used_budget = 0
        oom_retry_count = 0
        for budget in generation_budgets:
            try:
                with self.torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                used_budget = budget
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                oom_retry_count += 1
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                continue
        if output_ids is None:
            return {
                "events": [],
                "_raw_output": "",
                "_parse_error": True,
                "_error": str(last_error) if last_error else "generation failed",
                "_requested_max_new_tokens": requested_tokens,
                "_used_max_new_tokens": 0,
                "_generated_token_count": 0,
                "_output_may_be_truncated": False,
                "_generation_budget_reduced": bool(oom_retry_count),
                "_oom_retry_count": oom_retry_count,
            }
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated_token_count = int(gen_ids.shape[-1])
        output_may_be_truncated = bool(used_budget and generated_token_count >= used_budget)

        # Decode WITHOUT skipping special tokens so the </think> tag is preserved
        decoded_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        decoded_start = normalize_llm_output_text(decoded_text).lstrip()
        if self.json_assistant_prefill and decoded_start.startswith(('"', "}")):
            output_text = self.json_assistant_prefill + decoded_text
        else:
            output_text = decoded_text

        # Parse the JSON
        parsed = extract_first_json_object(output_text)

        if parsed is None:
            print_parse_failure_visualization(output_text)

        del inputs, output_ids, gen_ids
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        generation_meta = {
            "_requested_max_new_tokens": requested_tokens,
            "_used_max_new_tokens": used_budget,
            "_generated_token_count": generated_token_count,
            "_output_may_be_truncated": output_may_be_truncated,
            "_generation_budget_reduced": used_budget != requested_tokens,
            "_oom_retry_count": oom_retry_count,
        }
        if parsed is None:
            return {"events": [], "_raw_output": output_text, "_parse_error": True, **generation_meta}
        parsed["_raw_output"] = output_text
        parsed["_parse_error"] = False
        parsed.update(generation_meta)
        return parsed
