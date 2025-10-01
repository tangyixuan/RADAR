from __future__ import annotations

from typing import List, Tuple


def _get_tokenizer_name(tokenizer) -> str:
    try:
        return tokenizer.__class__.__name__.lower()
    except AttributeError:
        return ""


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> Tuple[str, bool]:
    """Return a prompt string for chat models and flag whether a chat template was used."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    prefer_chat_template = True
    tokenizer_name = _get_tokenizer_name(tokenizer)
    if tokenizer_name:
        prefer_chat_template = any(name in tokenizer_name for name in ("qwen", "llama")) or bool(
            getattr(tokenizer, "chat_template", None)
        )

    if callable(apply_chat) and prefer_chat_template:
        try:
            text = apply_chat(messages, tokenize=False, add_generation_prompt=True)
            return text, True
        except TypeError:
            # Older versions may not accept tokenize argument
            text = apply_chat(messages, add_generation_prompt=True)
            return text, True
        except Exception:
            # Fallback to manual template below
            pass

    manual_prompt = f"<|begin_of_text|><|system|>\n{system_prompt}\n<|user|>\n{user_prompt}<|assistant|>\n"
    return manual_prompt, False


def build_chat_prompts(tokenizer, system_prompts: List[str], user_prompts: List[str]) -> Tuple[List[str], List[bool]]:
    """Vectorised helper that mirrors build_chat_prompt for multiple prompts."""
    prompts: List[str] = []
    flags: List[bool] = []
    for sys_prompt, usr_prompt in zip(system_prompts, user_prompts):
        text, used_template = build_chat_prompt(tokenizer, sys_prompt, usr_prompt)
        prompts.append(text)
        flags.append(used_template)
    return prompts, flags


def extract_assistant_response(raw_text: str, used_chat_template: bool) -> str:
    """Extract assistant response from raw model output."""
    if not isinstance(raw_text, str):
        return raw_text

    if used_chat_template:
        markers = [
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|assistant|>",
            "assistant\n",
            "assistant:",
        ]
        for marker in markers:
            if marker in raw_text:
                return raw_text.split(marker)[-1].strip()
        return raw_text.strip()

    if "<|assistant|>" in raw_text:
        return raw_text.split("<|assistant|>")[-1].strip()

    return raw_text.strip()
