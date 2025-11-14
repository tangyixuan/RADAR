import math
import re
import torch
from typing import Dict, List, Optional, Tuple

from model.loader import load_model
from prompts.templates_people import (
    get_system_prompt,
    politician_opening_prompt,
    politician_rebuttal_prompt,
    politician_closing_prompt,
    scientist_opening_prompt,
    scientist_rebuttal_prompt,
    scientist_closing_prompt,
    judge_prompt,
    judge_prompt_1r,
    judge_prompt_2r
)
from agents.chat_template_utils import (
    build_chat_prompt,
    extract_assistant_response,
    inference_generate,
)

# Global model info - will be set by main.py
model_info = None

# Choices for different decision types
CONTINUATION_CHOICES = ("STOP", "CONTINUE")
MCQ_CHOICES = ("TRUE", "FALSE", "HALF-TRUE")
MCQ_VERDICT_MAP = {
    "TRUE": "TRUE",
    "FALSE": "FALSE",
    "HALF-TRUE": "HALF-TRUE",
}

_ROUNDS = ["opening", "rebuttal", "closing"]


def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info


def run_model(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    collect_scores: bool = False,
):
    """Run model inference based on model type"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")

    if len(model_info) == 2:
        first, second = model_info

        if isinstance(second, str):
            # GPT model: (client, model_name)
            client, model_name = model_info
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            if collect_scores:
                # Use completion API with logprobs for probability estimation
                try:
                    response = client.completions.create(
                        model=model_name,
                        prompt=f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                        max_tokens=max_tokens,
                        temperature=0.7,
                        logprobs=5,
                        top_logprobs=5
                    )
                    content = response.choices[0].text.strip()
                    logprobs = response.choices[0].logprobs
                    return content, logprobs
                except Exception as e:
                    print(f"Warning: Could not get logprobs from GPT model: {e}")
                    # Fallback to regular chat completion
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip(), None
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()

        # Local model: (tokenizer, model)
        tokenizer, model = model_info
        text, used_chat_template = build_chat_prompt(tokenizer, system_prompt, user_prompt)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if collect_scores:
            outputs = inference_generate(
                model,
                inputs,
                return_dict_in_generate=True,
                output_scores=True,
                **generate_kwargs,
            )
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            extracted = extract_assistant_response(response, used_chat_template)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs.sequences[0][input_length:].tolist()
            generation_info = {
                "scores": [score.detach().cpu() for score in outputs.scores],
                "generated_tokens": generated_tokens,
                "tokenizer": tokenizer,
            }
            return extracted, generation_info

        outputs = inference_generate(
            model,
            inputs,
            **generate_kwargs,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return extract_assistant_response(response, used_chat_template)

    raise ValueError("Invalid model_info format")


# === Utility Functions ===

def _logprobs_to_probabilities(logprob_map: Dict[str, float]) -> Dict[str, float]:
    if not logprob_map:
        return {}
    max_logprob = max(logprob_map.values())
    exp_map = {
        choice: math.exp(logprob - max_logprob) if math.isfinite(logprob) else 0.0
        for choice, logprob in logprob_map.items()
    }
    total = sum(exp_map.values())
    return {choice: (value / total if total > 0 else 0.0) for choice, value in exp_map.items()}


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    max_len = min(len(first), len(second))
    idx = 0
    while idx < max_len and first[idx] == second[idx]:
        idx += 1
    return idx


def _extract_verdict_label(response_text: str) -> str:
    """Extract the verdict label from the model response text."""
    verdict = "UNKNOWN"
    patterns = [
        r"\[VERDICT\]:\s*(TRUE|FALSE|HALF-TRUE)",
        r"VERDICT:\s*(TRUE|FALSE|HALF-TRUE)",
        r"\b(TRUE|FALSE|HALF-TRUE)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            break
    return verdict


def _extract_reason_section(response_text: str) -> str:
    """Extract the reason section without the verdict line."""
    reason_pattern = r'((?:\[)?REASON(?:\])?:.*?)(?=\n\s*(?:\[)?VERDICT(?:\])?:)'
    match = re.search(reason_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        reason_section = match.group(1).strip()
    else:
        # Fallback: remove any verdict lines and keep the rest
        lines = [
            line for line in response_text.strip().splitlines()
            if not re.match(r'\s*(?:\[)?VERDICT(?:\])?:', line, re.IGNORECASE)
        ]
        reason_section = "\n".join(lines).strip()

    if not reason_section:
        return ""

    reason_section = reason_section.rstrip()
    if reason_section and not reason_section.endswith("\n"):
        reason_section += "\n"
    return reason_section


# === Continuation Scoring (from FJ version) ===

def _compute_choice_logprobs_local(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    assistant_context: str,
    choices: tuple[str, ...],
) -> tuple[Dict[str, float], Dict[str, str]]:
    prompt_text, _ = build_chat_prompt(tokenizer, system_prompt, user_prompt)
    base_text = prompt_text + assistant_context

    base_encoding = tokenizer([base_text], return_tensors="pt")
    base_ids = base_encoding["input_ids"][0].tolist()

    choice_logprobs: Dict[str, float] = {}
    choice_first_tokens: Dict[str, str] = {}

    model_inputs = {k: v.to(model.device) for k, v in base_encoding.items()}
    model.eval()
    with torch.no_grad():
        base_outputs = model(**model_inputs)
        base_logits = base_outputs.logits[0, -1, :]
        base_logprobs = torch.log_softmax(base_logits, dim=-1)

        for choice in choices:
            choice_text = base_text + choice
            choice_encoding = tokenizer([choice_text], return_tensors="pt")
            choice_ids = choice_encoding["input_ids"][0].tolist()

            prefix_len = _common_prefix_length(base_ids, choice_ids)
            if prefix_len >= len(choice_ids):
                choice_logprobs[choice] = float("-inf")
                choice_first_tokens[choice] = ""
                continue

            first_token_id = choice_ids[prefix_len]
            token_logprob = float(base_logprobs[first_token_id].item())
            choice_logprobs[choice] = token_logprob
            choice_first_tokens[choice] = tokenizer.decode([first_token_id])

    return choice_logprobs, choice_first_tokens


def _compute_choice_logprobs_gpt_mcq(client, model_name: str, system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Compute next-token log-probabilities for MCQ choices using GPT-style models."""
    base_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: {assistant_context}"
    base_length = len(base_prompt)

    choice_logprobs: Dict[str, float] = {}
    for choice in MCQ_CHOICES:
        prompt_with_choice = base_prompt + choice
        response = client.completions.create(
            model=model_name,
            prompt=prompt_with_choice,
            max_tokens=0,
            temperature=0,
            logprobs=1,
            echo=True
        )
        logprob_data = response.choices[0].logprobs
        if logprob_data is None or logprob_data.tokens is None:
            raise RuntimeError("Logprob data unavailable for GPT MCQ scoring.")

        tokens = logprob_data.tokens
        token_logprobs = logprob_data.token_logprobs
        text_offsets = logprob_data.text_offset

        recorded = False
        for delta in (0, 1, 2):
            first_idx = None
            for idx, offset in enumerate(text_offsets):
                if offset is not None and offset >= base_length - delta:
                    first_idx = idx
                    break

            if first_idx is None:
                continue

            sum_logprob = 0.0
            token_count = 0
            for idx in range(first_idx, len(tokens)):
                offset = text_offsets[idx]
                token_logprob = token_logprobs[idx]
                if offset is None or token_logprob is None:
                    continue
                
                # Stop when we reach content beyond the choice
                choice_end_offset = base_length + len(choice)
                if offset >= choice_end_offset:
                    break
                    
                sum_logprob += token_logprob
                token_count += 1

            if token_count > 0:
                choice_logprobs[choice] = sum_logprob
            else:
                choice_logprobs[choice] = float("-inf")
            recorded = True
            break

        if not recorded:
            choice_logprobs[choice] = float("-inf")

    return choice_logprobs


def compute_continuation_logprobs(
    system_prompt: str,
    user_prompt: str,
    assistant_context: str,
) -> tuple[Dict[str, float], Dict[str, str]]:
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")

    if len(model_info) != 2:
        raise ValueError("Invalid model_info format")

    first, second = model_info
    if isinstance(second, str):
        client, model_name = model_info
        return _compute_choice_logprobs_gpt(
            client,
            model_name,
            system_prompt,
            user_prompt,
            assistant_context,
            CONTINUATION_CHOICES,
        )

    tokenizer, model = model_info
    return _compute_choice_logprobs_local(
        tokenizer,
        model,
        system_prompt,
        user_prompt,
        assistant_context,
        CONTINUATION_CHOICES,
    )


def _compute_choice_logprobs_from_generation(
    generation_info: Dict[str, object],
    choices: tuple[str, ...]
) -> tuple[Dict[str, float], Dict[str, str]]:
    tokenizer = generation_info.get("tokenizer")
    scores = generation_info.get("scores")
    generated_tokens = generation_info.get("generated_tokens")

    if tokenizer is None or scores is None or generated_tokens is None:
        raise ValueError("Incomplete generation info for choice scoring.")

    if not scores:
        raise ValueError("Empty generation scores for choice scoring.")

    # Get token IDs for choices
    choice_token_ids = {}
    for choice in choices:
        token_ids = tokenizer.encode(f" {choice}", add_special_tokens=False)
        if token_ids:
            choice_token_ids[choice] = token_ids[0]
        else:
            raise ValueError(f"Failed to obtain token ids for choice '{choice}'.")

    decision_index = None
    for idx, token_id in enumerate(generated_tokens):
        if token_id in choice_token_ids.values():
            decision_index = idx
            break

    if decision_index is None or decision_index >= len(scores):
        raise ValueError("Could not locate decision token in generated sequence.")

    score_tensor = scores[decision_index]
    if isinstance(score_tensor, torch.Tensor) and score_tensor.ndim == 2:
        score_tensor = score_tensor[0]
    log_probs = torch.log_softmax(score_tensor, dim=-1)

    choice_logprobs = {}
    choice_first_tokens = {}
    for choice, token_id in choice_token_ids.items():
        choice_logprobs[choice] = float(log_probs[token_id].item())
        choice_first_tokens[choice] = tokenizer.decode([token_id])

    return choice_logprobs, choice_first_tokens


# === Verdict Scoring (from Round version) ===

def _find_last_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return the starting index of the last occurrence of needle within haystack."""
    if not haystack or not needle or len(needle) > len(haystack):
        return None
    for idx in range(len(haystack) - len(needle), -1, -1):
        if haystack[idx: idx + len(needle)] == needle:
            return idx
    return None


def _label_token_variants(tokenizer, label: str) -> List[List[int]]:
    """Return plausible token sequences for a label with different leading characters."""
    prefixes = [" ", "\n", "\n ", ""]
    variants: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for prefix in prefixes:
        tokens = tokenizer.encode(prefix + label, add_special_tokens=False)
        if tokens:
            token_tuple = tuple(tokens)
            if token_tuple not in seen:
                seen.add(token_tuple)
                variants.append(tokens)
    return variants


def _compute_verdict_probability_info_from_generation(
    response_text: str,
    generation_info: Dict[str, object],
    choices: Tuple[str, ...] = MCQ_CHOICES,
) -> Dict[str, object]:
    """Compute per-choice logprobs using generation scores captured during decoding."""
    tokenizer = generation_info.get("tokenizer")
    scores = generation_info.get("scores")
    generated_tokens = generation_info.get("generated_tokens")

    if tokenizer is None or scores is None or generated_tokens is None:
        raise ValueError("Incomplete generation info for verdict probability computation.")

    if not isinstance(generated_tokens, (list, tuple)):
        raise ValueError("generated_tokens must be a list or tuple of token ids.")

    score_list: List[torch.Tensor] = []
    for score in scores:
        if isinstance(score, torch.Tensor):
            score_list.append(score.detach().cpu())
        else:
            score_list.append(torch.tensor(score))

    if not score_list:
        raise ValueError("Empty score list for verdict probability computation.")

    generated_token_ids = list(generated_tokens)

    label_positions: Dict[str, Tuple[int, List[int]]] = {}
    for label in choices:
        variants = _label_token_variants(tokenizer, label)
        for variant in variants:
            position = _find_last_subsequence(generated_token_ids, variant)
            if position is None:
                continue
            stored = label_positions.get(label)
            if stored is None or position > stored[0]:
                label_positions[label] = (position, variant)

    if not label_positions:
        raise ValueError("Failed to locate any verdict label tokens in generated sequence.")

    extracted_verdict = _extract_verdict_label(response_text)
    target_label = MCQ_VERDICT_MAP.get(extracted_verdict, extracted_verdict)
    if target_label not in label_positions:
        target_label = max(label_positions.items(), key=lambda item: item[1][0])[0]

    decision_index, matched_sequence = label_positions[target_label]
    if decision_index >= len(score_list):
        raise ValueError("Decision token index exceeds available score tensors.")

    score_tensor = score_list[decision_index]
    if score_tensor.ndim == 2:
        score_tensor = score_tensor[0]
    log_probs = torch.log_softmax(score_tensor, dim=-1)

    choice_logprobs: Dict[str, float] = {}
    choice_first_tokens: Dict[str, str] = {}
    first_token_ids: Dict[str, int] = {}
    for label in choices:
        variants = _label_token_variants(tokenizer, label)
        if not variants:
            raise ValueError(f"Failed to obtain token ids for verdict label '{label}'.")
        first_token_id = variants[0][0]
        first_token_ids[label] = first_token_id
        choice_logprobs[label] = float(log_probs[first_token_id].item())
        choice_first_tokens[label] = tokenizer.decode([first_token_id])

    choice_probabilities = _logprobs_to_probabilities(choice_logprobs)
    predicted_choice = (
        max(choice_probabilities, key=choice_probabilities.get)
        if choice_probabilities
        else None
    )

    verdict_logprobs = {
        MCQ_VERDICT_MAP[label]: logprob for label, logprob in choice_logprobs.items()
    }
    verdict_probabilities = {
        MCQ_VERDICT_MAP[label]: prob for label, prob in choice_probabilities.items()
    }

    return {
        "choice_logprobs": choice_logprobs,
        "choice_probabilities": choice_probabilities,
        "choice_first_tokens": choice_first_tokens,
        "first_token_ids": first_token_ids,
        "predicted_verdict": MCQ_VERDICT_MAP.get(predicted_choice, predicted_choice),
        "verdict_logprobs": verdict_logprobs,
        "verdict_probabilities": verdict_probabilities,
    }


# === Continuation Decision Functions ===

def _compute_continuation_probability_info_with_generation(
    user_prompt: str,
    assistant_context: str = "DECISION: ",
    generation_info: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if generation_info is not None:
        try:
            choice_logprobs, choice_first_tokens = _compute_choice_logprobs_from_generation(
                generation_info, CONTINUATION_CHOICES
            )
            choice_probabilities = _logprobs_to_probabilities(choice_logprobs)
            predicted_decision = (
                max(choice_probabilities, key=choice_probabilities.get)
                if choice_probabilities
                else None
            )
            return {
                "choice_logprobs": choice_logprobs,
                "choice_probabilities": choice_probabilities,
                "predicted_decision": predicted_decision,
                "choice_first_tokens": choice_first_tokens,
            }
        except Exception as exc:
            print(f"Warning: generation-based continuation scoring failed: {exc}")

    try:
        choice_logprobs, choice_first_tokens = compute_continuation_logprobs(
            get_system_prompt("judge"),
            user_prompt,
            assistant_context,
        )
    except Exception as exc:
        print(f"Warning: continuation probability estimation failed: {exc}")
        return {
            "choice_logprobs": {},
            "choice_probabilities": {},
            "predicted_decision": None,
            "choice_first_tokens": {},
        }

    choice_probabilities = _logprobs_to_probabilities(choice_logprobs)
    predicted_decision = (
        max(choice_probabilities, key=choice_probabilities.get)
        if choice_probabilities
        else None
    )
    return {
        "choice_logprobs": choice_logprobs,
        "choice_probabilities": choice_probabilities,
        "predicted_decision": predicted_decision,
        "choice_first_tokens": choice_first_tokens,
    }


def _format_round_summary(transcripts, executed_rounds):
    if not executed_rounds:
        return "No debate rounds have been completed yet."

    sections = []
    for round_name in executed_rounds:
        round_data = transcripts.get(round_name, {})
        pol = round_data.get("politician", "").strip() or "[No statement]"
        sci = round_data.get("scientist", "").strip() or "[No statement]"
        sections.append(
            f"[{round_name.title()} - Politician]\n{pol}\n\n"
            f"[{round_name.title()} - Scientist]\n{sci}"
        )
    return "\n\n".join(sections)


def _continuation_prompt_before_round(claim, evidence, upcoming_round, transcripts, executed_rounds):
    if not executed_rounds:
        summary = (
            "No debate rounds have been completed yet. You must rely solely on the evidence provided. "
            "If the evidence already makes the claim clearly TRUE, FALSE, or HALF-TRUE, you should issue a final verdict now."
        )
    else:
        summary = _format_round_summary(transcripts, executed_rounds)
    return f"""
You are acting as an interim adjudicator during a two-agent fact-checking debate.

Claim: {claim}

Evidence:
{evidence}

Rounds completed so far:
{summary}

You are about to listen to the {upcoming_round.title()} round. Decide whether you already have enough information to reach a final verdict now, or whether the debate should continue and you should listen to this upcoming round.

Respond strictly with:
DECISION: CONTINUE or STOP
REASON: <one or two sentences>
"""


def _should_continue_before_round(claim, evidence, upcoming_round, transcripts, executed_rounds):
    prompt = _continuation_prompt_before_round(claim, evidence, upcoming_round, transcripts, executed_rounds)
    response, generation_info = run_model(
        get_system_prompt("judge"),
        prompt,
        max_tokens=200,
        collect_scores=True,
    )
    probability_info = _compute_continuation_probability_info_with_generation(
        prompt,
        generation_info=generation_info,
    )

    decision_line = ""
    for line in response.splitlines():
        upper = line.upper()
        if "DECISION" in upper:
            decision_line = upper
            break

    if "STOP" in decision_line and "CONTINUE" not in decision_line:
        return False, response, probability_info
    if "CONTINUE" in decision_line and "STOP" not in decision_line:
        return True, response, probability_info

    upper_response = response.upper()
    if "STOP" in upper_response and "CONTINUE" not in upper_response:
        return False, response, probability_info

    return True, response, probability_info


# === Verdict Computation Functions ===

def _compute_verdict_probability_info(
    user_prompt: str,
    assistant_context: str,
    generation_info: Optional[Dict[str, object]] = None,
    response_text: Optional[str] = None,
) -> Dict[str, object]:
    """Compute verdict probabilities, preferring generation scores when available."""
    if generation_info is not None and response_text is not None:
        try:
            info = _compute_verdict_probability_info_from_generation(response_text, generation_info)
            if info.get("choice_logprobs"):
                return info
        except Exception as exc:
            print(f"Warning: generation-based verdict probability computation failed: {exc}")

    # Fallback to recomputation (for GPT models)
    system_prompt = get_system_prompt("judge")
    try:
        if model_info and len(model_info) == 2 and isinstance(model_info[1], str):
            # GPT model - use MCQ-specific function
            choice_logprobs = _compute_choice_logprobs_gpt_mcq(
                model_info[0], model_info[1], system_prompt, user_prompt, assistant_context
            )
            choice_first_tokens = {}  # Not needed for MCQ case
        else:
            raise RuntimeError("Local model verdict scoring requires generation scores.")
    except Exception as exc:
        print(f"Warning: verdict probability computation failed: {exc}")
        return {
            "predicted_verdict": "UNKNOWN",
            "verdict_probabilities": {},
            "verdict_logprobs": {},
            "choice_logprobs": {},
            "choice_probabilities": {},
        }

    verdict_logprobs = {
        MCQ_VERDICT_MAP[choice]: logprob for choice, logprob in choice_logprobs.items()
    }
    verdict_probabilities = _logprobs_to_probabilities(verdict_logprobs)
    predicted_verdict = max(verdict_probabilities, key=verdict_probabilities.get)

    return {
        "predicted_verdict": predicted_verdict,
        "verdict_probabilities": verdict_probabilities,
        "verdict_logprobs": verdict_logprobs,
        "choice_logprobs": choice_logprobs,
        "choice_probabilities": _logprobs_to_probabilities(choice_logprobs),
        "choice_first_tokens": choice_first_tokens if 'choice_first_tokens' in locals() else {},
    }


# === Agent Functions ===

def opening_politician(claim, evidence):
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)


def rebuttal_politician(claim, evidence, opponent_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)


def closing_politician(claim, evidence):
    prompt = politician_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)


def opening_scientist(claim, evidence):
    prompt = scientist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)


def rebuttal_scientist(claim, evidence, opponent_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)


def closing_scientist(claim, evidence):
    prompt = scientist_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)


# === Judge Functions ===

def judge_round_1(claim, evidence, pol_open, sci_open):
    """Judge after first round (opening statements)"""
    prompt = judge_prompt_1r(claim, evidence, pol_open, sci_open)
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, collect_scores=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
    else:
        response = result
        probability_data = None

    generation_info: Optional[Dict[str, object]] = None
    if probability_data is not None:
        if isinstance(probability_data, dict):
            generation_info = probability_data
        elif hasattr(probability_data, "__iter__") and not isinstance(probability_data, str):
            try:
                # Try to parse as generation info
                generation_info = probability_data
            except:
                generation_info = None

    reason_section = _extract_reason_section(response)
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "

    probability_info = _compute_verdict_probability_info(
        user_prompt=prompt,
        assistant_context=assistant_context,
        generation_info=generation_info,
        response_text=response,
    )

    verdict = _extract_verdict_label(response)
    if verdict == "UNKNOWN" and probability_info.get("predicted_verdict"):
        verdict = probability_info["predicted_verdict"]

    mapped_verdict = MCQ_VERDICT_MAP.get(verdict, verdict)
    probability = probability_info.get("verdict_probabilities", {}).get(mapped_verdict)

    return {
        "verdict_text": response,
        "response": response,
        "verdict": verdict,
        "probability": probability,
        "probability_info": probability_info,
        "reason_section": reason_section.strip(),
    }


def judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut):
    """Judge after second round (opening + rebuttal statements)"""
    prompt = judge_prompt_2r(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, collect_scores=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
    else:
        response = result
        probability_data = None

    generation_info: Optional[Dict[str, object]] = None
    if probability_data is not None:
        if isinstance(probability_data, dict):
            generation_info = probability_data
        elif hasattr(probability_data, "__iter__") and not isinstance(probability_data, str):
            try:
                generation_info = probability_data
            except:
                generation_info = None

    reason_section = _extract_reason_section(response)
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "

    probability_info = _compute_verdict_probability_info(
        user_prompt=prompt,
        assistant_context=assistant_context,
        generation_info=generation_info,
        response_text=response,
    )

    verdict = _extract_verdict_label(response)
    if verdict == "UNKNOWN" and probability_info.get("predicted_verdict"):
        verdict = probability_info["predicted_verdict"]

    mapped_verdict = MCQ_VERDICT_MAP.get(verdict, verdict)
    probability = probability_info.get("verdict_probabilities", {}).get(mapped_verdict)

    return {
        "verdict_text": response,
        "response": response,
        "verdict": verdict,
        "probability": probability,
        "probability_info": probability_info,
        "reason_section": reason_section.strip(),
    }


def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    prompt = judge_prompt(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, collect_scores=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
    else:
        response = result
        probability_data = None

    generation_info: Optional[Dict[str, object]] = None
    if probability_data is not None:
        if isinstance(probability_data, dict):
            generation_info = probability_data
        elif hasattr(probability_data, "__iter__") and not isinstance(probability_data, str):
            try:
                generation_info = probability_data
            except:
                generation_info = None

    reason_section = _extract_reason_section(response)
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "

    probability_info = _compute_verdict_probability_info(
        user_prompt=prompt,
        assistant_context=assistant_context,
        generation_info=generation_info,
        response_text=response,
    )

    verdict = _extract_verdict_label(response)
    if verdict == "UNKNOWN" and probability_info.get("predicted_verdict"):
        verdict = probability_info["predicted_verdict"]

    mapped_verdict = MCQ_VERDICT_MAP.get(verdict, verdict)
    probability = probability_info.get("verdict_probabilities", {}).get(mapped_verdict)

    return {
        "verdict_text": response,
        "response": response,
        "verdict": verdict,
        "probability": probability,
        "probability_info": probability_info,
        "reason_section": reason_section.strip(),
    }


# === MCQ Judge Functions ===

def judge_round_1_mcq(claim, evidence, pol_open, sci_open):
    """Judge after first round with MCQ probability analysis"""
    base_result = judge_round_1(claim, evidence, pol_open, sci_open)
    
    # Extract essential probability information, removing redundancy
    probability_info = base_result.get("probability_info", {})
    verdict_probabilities = probability_info.get("verdict_probabilities", {})
    
    # Return streamlined result with only essential fields
    return {
        "verdict": base_result.get("verdict"),
        "response": base_result.get("response"),
        "probability": base_result.get("probability"),
        "verdict_probabilities": verdict_probabilities
    }


def judge_round_2_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut):
    """Judge after second round with MCQ probability analysis"""
    base_result = judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    
    # Extract essential probability information, removing redundancy
    probability_info = base_result.get("probability_info", {})
    verdict_probabilities = probability_info.get("verdict_probabilities", {})
    
    # Return streamlined result with only essential fields
    return {
        "verdict": base_result.get("verdict"),
        "response": base_result.get("response"),
        "probability": base_result.get("probability"),
        "verdict_probabilities": verdict_probabilities
    }


def judge_final_verdict_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    """Final judge with MCQ probability analysis"""
    base_result = judge_final_verdict(
        claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close
    )
    
    # Extract essential probability information, removing redundancy
    probability_info = base_result.get("probability_info", {})
    verdict_probabilities = probability_info.get("verdict_probabilities", {})
    
    # Return streamlined result with only essential fields
    return {
        "verdict": base_result.get("verdict"),
        "response": base_result.get("response"),
        "probability": base_result.get("probability"),
        "verdict_probabilities": verdict_probabilities
    }


# === Main Hybrid Debate Function ===
def run_multi_agent_people_round_mcq(claim, evidence):
    """Run the debate and compute verdict token probabilities at each judge (Round-only mode)"""
    print("\n=== Running Multi-Agent People Debate with Round Judges (MCQ probability view) ===")

    print("Round 1: Opening statements...")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)

    print("Round 1 Judge evaluation (probabilities)...")
    round_1_judge = judge_round_1_mcq(claim, evidence, pol_open, sci_open)

    print("Round 2: Rebuttal statements...")
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)

    print("Round 2 Judge evaluation (probabilities)...")
    round_2_judge = judge_round_2_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)

    print("Round 3: Closing statements...")
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)

    print("Final Judge evaluation (probabilities)...")
    final_judge = judge_final_verdict_mcq(
        claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close
    )

    return {
        "politician_opening": pol_open,
        "scientist_opening": sci_open,
        "round_1_judge": round_1_judge,
        "politician_rebuttal": pol_rebut,
        "scientist_rebuttal": sci_rebut,
        "round_2_judge": round_2_judge,
        "politician_closing": pol_close,
        "scientist_closing": sci_close,
        "final_judge": final_judge
    }


def run_multi_agent_people_hybrid_mcq(claim, evidence, tau_s=-0.6, tau_v=0.75):
    """
    Run hybrid debate with MCQ probability analysis and adaptive early stopping.
    
    This function combines:
    1. Forward judging: Decide before each round whether to continue
    2. Adaptive early stopping: Use dual thresholds (stop margin + confidence)
    3. Round judging: Evaluate after each completed round with MCQ probabilities
    4. Final judging: Final verdict with probabilities
    
    Args:
        claim: The claim to be debated
        evidence: Evidence related to the claim
        tau_s: Stop margin threshold for early termination (default -0.6)
        tau_v: Veracity confidence threshold for early termination (default 0.75)
    
    Returns:
        Dictionary containing comprehensive results with MCQ probability analysis and early stopping metadata
    """
    print("\n" + "="*80)
    print("RUNNING HYBRID MULTI-AGENT DEBATE (ADAPTIVE EARLY STOPPING MODE)")
    print("Forward Judging + Adaptive Early Stopping + Round Judging + Probability Analysis")
    print(f"Stop margin threshold (tau_s): {tau_s}")
    print(f"Veracity confidence threshold (tau_v): {tau_v}")
    print("="*80)
    
    # Initialize data structures
    transcripts = {name: {"politician": "", "scientist": ""} for name in _ROUNDS}
    executed_rounds = []
    continuation_decisions = []
    round_judges = {}
    
    pol_open = sci_open = ""
    pol_rebut = sci_rebut = ""
    pol_close = sci_close = ""

    # Process each round with both forward and round judging
    for round_name in _ROUNDS:
        print(f"\n{'='*50}")
        print(f"PROCESSING {round_name.upper()} ROUND")
        print(f"{'='*50}")
        
        # FORWARD JUDGING WITH ADAPTIVE EARLY STOPPING: Decide whether to continue BEFORE the round
        print(f"Forward Judge with Adaptive Early Stopping: Evaluating whether to proceed with {round_name} round...")
        continue_debate, decision_text, probability_info, terminated_early = adaptive_early_stopping_decision(
            claim, evidence, round_name, transcripts, executed_rounds, tau_s, tau_v, round_judges
        )
        
        continuation_decision = {
            "round": round_name,
            "decision": "continue" if continue_debate else "stop",
            "rationale": decision_text,
            "choice_logprobs": probability_info.get("choice_logprobs"),
            "choice_probabilities": probability_info.get("choice_probabilities"),
            "choice_first_tokens": probability_info.get("choice_first_tokens"),
            "stop_margin": probability_info.get("stop_margin"),
            "confidence": probability_info.get("confidence"),
            "tau_s": probability_info.get("tau_s"),
            "tau_v": probability_info.get("tau_v"),
            "terminated_early": terminated_early,
            "standard_continue_decision": probability_info.get("standard_continue_decision")
        }
        continuation_decisions.append(continuation_decision)
        
        print(f"   → Decision: {'CONTINUE' if continue_debate else 'STOP'}")
        if terminated_early:
            print(f"   → Early termination triggered (s={probability_info.get('stop_margin', 0):.4f}, c={probability_info.get('confidence', 0):.4f})")
        
        # If decided to stop, break the loop
        if not continue_debate:
            print(f"   → Stopping debate before {round_name} round")
            break

        # EXECUTE THE ROUND
        print(f"Executing {round_name} round...")
        
        if round_name == "opening":
            pol_open = opening_politician(claim, evidence)
            sci_open = opening_scientist(claim, evidence)
            transcripts["opening"] = {"politician": pol_open, "scientist": sci_open}
            executed_rounds.append("opening")
            
            # ROUND JUDGING: Evaluate AFTER the round with MCQ probabilities
            print(f"Round Judge: Evaluating after {round_name} round (MCQ probabilities)...")
            round_1_judge = judge_round_1_mcq(claim, evidence, pol_open, sci_open)
            round_judges["round_1"] = round_1_judge
            print(f"   → Round 1 verdict: {round_1_judge['verdict']}")

        elif round_name == "rebuttal":
            pol_rebut = rebuttal_politician(claim, evidence, sci_open)
            sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
            transcripts["rebuttal"] = {"politician": pol_rebut, "scientist": sci_rebut}
            executed_rounds.append("rebuttal")
            
            # ROUND JUDGING: Evaluate AFTER the round with MCQ probabilities
            print(f"Round Judge: Evaluating after {round_name} round (MCQ probabilities)...")
            round_2_judge = judge_round_2_mcq(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
            round_judges["round_2"] = round_2_judge
            print(f"   → Round 2 verdict: {round_2_judge['verdict']}")

        elif round_name == "closing":
            pol_close = closing_politician(claim, evidence)
            sci_close = closing_scientist(claim, evidence)
            transcripts["closing"] = {"politician": pol_close, "scientist": sci_close}
            executed_rounds.append("closing")

    # FINAL JUDGING with MCQ probabilities
    print(f"\n{'='*50}")
    print("FINAL VERDICT GENERATION (MCQ PROBABILITIES)")
    print(f"{'='*50}")
    
    final_verdict = judge_final_verdict_mcq(
        claim, evidence,
        transcripts["opening"]["politician"],
        transcripts["opening"]["scientist"],
        transcripts["rebuttal"]["politician"],
        transcripts["rebuttal"]["scientist"],
        transcripts["closing"]["politician"],
        transcripts["closing"]["scientist"],
    )
    
    print(f"Final verdict: {final_verdict['verdict']}")
    
    # SUMMARY ANALYSIS WITH ADAPTIVE EARLY STOPPING
    print(f"\n{'='*60}")
    print("HYBRID MCQ ANALYSIS WITH ADAPTIVE EARLY STOPPING SUMMARY")
    print(f"{'='*60}")
    
    print(f"Thresholds used - Stop margin (tau_s): {tau_s}, Veracity confidence (tau_v): {tau_v}")
    print(f"Rounds executed by forward judge: {executed_rounds}")
    print(f"Forward judge decisions with adaptive early stopping:")
    
    early_terminations = 0
    for decision in continuation_decisions:
        decision_str = f"  - {decision['round']}: {decision['decision'].upper()}"
        if decision.get('terminated_early', False):
            decision_str += f" (EARLY STOP: s={decision.get('stop_margin', 0):.4f}, c={decision.get('confidence', 0):.4f})"
            early_terminations += 1
        else:
            decision_str += f" (s={decision.get('stop_margin', 0):.4f}, c={decision.get('confidence', 0):.4f})"
        print(decision_str)
    
    print(f"Total early terminations triggered: {early_terminations}")
    
    if round_judges:
        print(f"Round-by-round verdicts with probabilities:")
        for round_key, judge_result in round_judges.items():
            verdict_probs = judge_result.get("verdict_probabilities", {})
            prob_str = ", ".join([f"{k}: {v:.3f}" for k, v in verdict_probs.items()])
            print(f"  - {round_key}: {judge_result['verdict']} ({prob_str})")
    
    final_verdict_probs = final_verdict.get("verdict_probabilities", {})
    final_prob_str = ", ".join([f"{k}: {v:.3f}" for k, v in final_verdict_probs.items()])
    print(f"Final verdict: {final_verdict['verdict']} ({final_prob_str})")
    
    return {
        "mode": "hybrid_mcq_adaptive",
        "thresholds": {"tau_s": tau_s, "tau_v": tau_v},
        "transcripts": transcripts,
        "executed_rounds": executed_rounds,
        "continuation_decisions": continuation_decisions,
        "round_judges": round_judges,
        "final_verdict": final_verdict,
        "early_termination_count": early_terminations
    }


def compute_choice_logprobs_mcq(system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Dispatch MCQ choice log-probability computation based on loaded model."""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")

    if len(model_info) != 2:
        raise ValueError("Invalid model_info format")

    first, second = model_info
    if isinstance(second, str):
        client, model_name = model_info
        return _compute_choice_logprobs_gpt_mcq(client, model_name, system_prompt, user_prompt, assistant_context)

    raise RuntimeError(
        "Local model MCQ scoring requires generation_info scores; direct recomputation path has been removed."
    )


# === Convenience Functions for Adaptive Early Stopping ===

def run_multi_agent_people_hybrid_adaptive(claim, evidence, tau_s=-0.6, tau_v=0.75):
    """
    Convenience function to run hybrid debate with adaptive early stopping.
    
    This is an alias for run_multi_agent_people_hybrid_mcq with explicit adaptive early stopping.
    
    Args:
        claim: The claim to be debated
        evidence: Evidence related to the claim  
        tau_s: Stop margin threshold (default -0.6)
        tau_v: Veracity confidence threshold (default 0.75)
        
    Returns:
        Dictionary containing comprehensive results with adaptive early stopping analysis
    """
    return run_multi_agent_people_hybrid_mcq(claim, evidence, tau_s, tau_v)


def analyze_early_stopping_performance(results: Dict) -> Dict[str, object]:
    """
    Analyze the performance of the adaptive early stopping mechanism.
    
    Args:
        results: Results dictionary from run_multi_agent_people_hybrid_mcq
        
    Returns:
        Dictionary containing early stopping performance metrics
    """
    continuation_decisions = results.get("continuation_decisions", [])
    early_termination_count = results.get("early_termination_count", 0)
    total_decisions = len(continuation_decisions)
    
    # Count decisions by type
    stop_decisions = sum(1 for d in continuation_decisions if d["decision"] == "stop")
    continue_decisions = total_decisions - stop_decisions
    
    # Count early terminations
    early_stop_decisions = sum(1 for d in continuation_decisions if d.get("terminated_early", False))
    standard_stop_decisions = stop_decisions - early_stop_decisions
    
    # Calculate stopping statistics  
    early_stop_rate = early_stop_decisions / total_decisions if total_decisions > 0 else 0
    total_stop_rate = stop_decisions / total_decisions if total_decisions > 0 else 0
    
    # Extract threshold information
    thresholds = results.get("thresholds", {})
    tau_s = thresholds.get("tau_s", 0.5)
    tau_v = thresholds.get("tau_v", 0.7)
    
    # Analyze margin and confidence distributions
    stop_margins = []
    confidences = []
    
    for decision in continuation_decisions:
        if decision.get("stop_margin") is not None:
            stop_margins.append(decision["stop_margin"])
        if decision.get("confidence") is not None:
            confidences.append(decision["confidence"])
    
    analysis = {
        "total_decisions": total_decisions,
        "stop_decisions": stop_decisions,
        "continue_decisions": continue_decisions,
        "early_stop_decisions": early_stop_decisions,
        "standard_stop_decisions": standard_stop_decisions,
        "early_stop_rate": early_stop_rate,
        "total_stop_rate": total_stop_rate,
        "thresholds_used": {"tau_s": tau_s, "tau_v": tau_v},
        "stop_margin_stats": {
            "values": stop_margins,
            "min": min(stop_margins) if stop_margins else None,
            "max": max(stop_margins) if stop_margins else None,
            "avg": sum(stop_margins) / len(stop_margins) if stop_margins else None
        },
        "confidence_stats": {
            "values": confidences,
            "min": min(confidences) if confidences else None,
            "max": max(confidences) if confidences else None,
            "avg": sum(confidences) / len(confidences) if confidences else None
        }
    }
    
    return analysis


def recommend_thresholds(historical_results: List[Dict], target_early_stop_rate: float = 0.3) -> Tuple[float, float]:
    """
    Recommend optimal thresholds based on historical performance.
    
    Args:
        historical_results: List of results from multiple debate runs
        target_early_stop_rate: Target early stopping rate (default 0.3)
        
    Returns:
        Tuple of recommended (tau_s, tau_v) values
    """
    all_margins = []
    all_confidences = []
    all_early_stops = []
    
    # Collect all decision data
    for result in historical_results:
        for decision in result.get("continuation_decisions", []):
            if decision.get("stop_margin") is not None:
                all_margins.append(decision["stop_margin"])
                all_confidences.append(decision.get("confidence", 0))
                all_early_stops.append(decision.get("terminated_early", False))
    
    if not all_margins:
        return -0.6, 0.75  # Default thresholds
    
    # Sort by stop margin
    sorted_data = sorted(zip(all_margins, all_confidences, all_early_stops), key=lambda x: x[0])
    
    # Find threshold that achieves target early stop rate
    target_count = int(len(sorted_data) * target_early_stop_rate)
    
    if target_count < len(sorted_data):
        recommended_tau_s = sorted_data[target_count][0]
    else:
        recommended_tau_s = sorted_data[-1][0]
    
    # For tau_v, use median confidence of early stopped cases
    early_stop_confidences = [conf for margin, conf, early in sorted_data if early]
    if early_stop_confidences:
        early_stop_confidences.sort()
        median_idx = len(early_stop_confidences) // 2
        recommended_tau_v = early_stop_confidences[median_idx]
    else:
        recommended_tau_v = 0.75  # Default
    
    return recommended_tau_s, recommended_tau_v


def adaptive_early_stopping_decision(claim, evidence, upcoming_round, transcripts, executed_rounds, tau_s=-0.6, tau_v=0.75, round_judges=None):
    """
    Adaptive early stopping decision using dual thresholds (RADAR approach).
    
    Args:
        claim: The claim being debated
        evidence: Evidence provided 
        upcoming_round: The round about to be executed
        transcripts: Current debate transcripts
        executed_rounds: List of rounds already executed
        tau_s: Stop margin threshold (default -0.6)
        tau_v: Veracity confidence threshold (default 0.75)
        round_judges: Dictionary containing round judge results for confidence calculation
    
    Returns:
        Tuple of (continue_debate, decision_text, probability_info, terminated_early)
    """
    # Opening round must always be executed
    if upcoming_round == "opening":
        return True, "Opening round must always be executed.", {
            "stop_margin": 0.0,
            "confidence": 0.0,
            "tau_s": tau_s,
            "tau_v": tau_v,
            "early_stop_condition": False,
            "standard_continue_decision": True,
            "p_stop": 0.0,
            "p_continue": 1.0
        }, False
    
    # Get the basic continuation decision and probabilities
    continue_debate, decision_text, probability_info = _should_continue_before_round(
        claim, evidence, upcoming_round, transcripts, executed_rounds
    )
    
    # Extract continuation probabilities
    choice_probabilities = probability_info.get("choice_probabilities", {})
    p_stop = choice_probabilities.get("STOP", 0.0)
    p_continue = choice_probabilities.get("CONTINUE", 1.0)
    
    # Calculate stop-continue difference: stop_continue_diff = p(STOP) - p(CONTINUE)
    stop_continue_diff = p_stop - p_continue
    
    # Get veracity confidence from the latest round judge result
    confidence = 0.0
    if round_judges and executed_rounds:
        # Use the probability from the most recent round judge
        if "round_2" in round_judges:
            # Use round_2 judge if available
            confidence = round_judges["round_2"].get("probability", 0.0)
        elif "round_1" in round_judges:
            # Use round_1 judge if round_2 not available
            confidence = round_judges["round_1"].get("probability", 0.0)
    
    # Apply dual-threshold logic
    # Early stopping occurs when BOTH conditions are met:
    # 1. stop_continue_diff >= tau_s (stop margin threshold)
    # 2. confidence >= tau_v (confidence threshold from round judge)
    early_stop_condition = (stop_continue_diff >= tau_s) and (confidence >= tau_v)
    
    # Determine final decision
    if early_stop_condition:
        # Override continue decision with early stop
        final_continue_debate = False
        terminated_early = True
        decision_text += f"\n[EARLY STOP TRIGGERED: s={stop_continue_diff:.4f} >= {tau_s}, c={confidence:.4f} >= {tau_v}]"
    else:
        # Use standard continuation decision
        final_continue_debate = continue_debate
        terminated_early = False
        if not continue_debate:
            decision_text += f"\n[STANDARD STOP: s={stop_continue_diff:.4f}, c={confidence:.4f}]"
    
    # Enhanced probability info with dual-threshold metrics
    enhanced_probability_info = {
        **probability_info,
        "stop_margin": stop_continue_diff,
        "confidence": confidence,
        "tau_s": tau_s,
        "tau_v": tau_v,
        "early_stop_condition": early_stop_condition,
        "standard_continue_decision": continue_debate,
        "p_stop": p_stop,
        "p_continue": p_continue
    }
    
    return final_continue_debate, decision_text, enhanced_probability_info, terminated_early
