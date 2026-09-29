from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Optional

from financial_ekg.utils.json_repair import extract_first_json_object, print_parse_failure_visualization


class LlamaCppExtractor:
    """Local llama.cpp JSON extractor for GGUF models.

    This backend is useful on clusters where recent Transformers/vLLM CUDA
    stacks do not match the available driver. It expects a local `.gguf` model
    file and keeps the same output contract as `HFExtractor`.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 4096,
        disable_thinking: bool = False,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        n_threads: int = 0,
        verbose: bool = False,
    ) -> None:
        """Load a GGUF model with llama.cpp.

        Args:
            model_path: Local path to the `.gguf` model file.
            max_new_tokens: Default generation budget.
            disable_thinking: Whether to ask reasoning-capable models to return
                final JSON directly.
            n_ctx: llama.cpp context window.
            n_gpu_layers: Number of layers to offload to GPU. `-1` means all.
            n_batch: Prompt-processing batch size.
            n_threads: CPU thread count. `0` lets llama.cpp choose.
            verbose: Whether llama.cpp should print detailed logs.

        Returns:
            None.
        """
        from llama_cpp import Llama

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"llama.cpp model file not found: {model_path}. "
                "Download a GGUF file first and pass it with --llama_model_path."
            )

        kwargs: Dict[str, Any] = {
            "model_path": str(model_file),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "verbose": verbose,
        }
        if n_threads and n_threads > 0:
            kwargs["n_threads"] = n_threads

        self.model = Llama(**kwargs)
        self.max_new_tokens = max_new_tokens
        self.disable_thinking = disable_thinking

    def _format_prompt(self, prompt: str) -> str:
        """Format an extraction prompt with the Qwen chat template.

        Args:
            prompt: User extraction prompt.

        Returns:
            Full chat-formatted prompt string for llama.cpp completion.
        """
        system = (
            "You are a precise financial event extraction engine for event knowledge graphs. "
            "Extract only article-supported financial events for all companies discussed in the article. "
            "Return only valid JSON."
        )
        if self.disable_thinking:
            system += " Do not include a thinking block. Return the final JSON only."
        assistant_prefix = "<think>\n\n</think>\n\n" if self.disable_thinking else "<think>\n"
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )

    def __call__(self, prompt: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate and parse a JSON extraction with llama.cpp.

        Args:
            prompt: Extraction prompt.
            max_new_tokens: Optional generation budget.

        Returns:
            Parsed extraction object with generation metadata.
        """
        gc.collect()
        requested_tokens = max_new_tokens if max_new_tokens is not None and max_new_tokens > 0 else self.max_new_tokens
        input_text = self._format_prompt(prompt)

        try:
            response = self.model(
                input_text,
                max_tokens=requested_tokens,
                temperature=0.0,
                top_p=1.0,
                echo=False,
                stop=["<|im_end|>"],
            )
            choice = response.get("choices", [{}])[0]
            output_text = str(choice.get("text", ""))
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            generated_token_count = int(usage.get("completion_tokens", 0) or 0)
            if generated_token_count <= 0:
                generated_token_count = len(self.model.tokenize(output_text.encode("utf-8"), add_bos=False))
            finish_reason = str(choice.get("finish_reason", ""))
            output_may_be_truncated = finish_reason == "length" or generated_token_count >= requested_tokens
        except Exception as exc:
            return {
                "events": [],
                "_raw_output": "",
                "_parse_error": True,
                "_error": str(exc),
                "_requested_max_new_tokens": requested_tokens,
                "_used_max_new_tokens": requested_tokens,
                "_generated_token_count": 0,
                "_output_may_be_truncated": False,
                "_generation_budget_reduced": False,
                "_oom_retry_count": 0,
            }

        parsed = extract_first_json_object(output_text)
        if parsed is None:
            print_parse_failure_visualization(output_text)

        generation_meta = {
            "_requested_max_new_tokens": requested_tokens,
            "_used_max_new_tokens": requested_tokens,
            "_generated_token_count": generated_token_count,
            "_output_may_be_truncated": output_may_be_truncated,
            "_generation_budget_reduced": False,
            "_oom_retry_count": 0,
            "_finish_reason": finish_reason,
        }
        if parsed is None:
            return {"events": [], "_raw_output": output_text, "_parse_error": True, **generation_meta}
        parsed["_raw_output"] = output_text
        parsed["_parse_error"] = False
        parsed.update(generation_meta)
        return parsed
