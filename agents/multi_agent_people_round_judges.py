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
import torch
import math
import re
from typing import Dict, List, Optional, Tuple

# Global model info - will be set by main.py
model_info = None


MCQ_CHOICES = ("TRUE", "FALSE", "HALF-TRUE")
MCQ_VERDICT_MAP = {
    "TRUE": "TRUE",
    "FALSE": "FALSE",
    "HALF-TRUE": "HALF-TRUE",
}

def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info

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
    # Check if this is GPT model generation_info
    if generation_info.get("model_type") == "gpt":
        return _compute_verdict_probability_info_from_gpt_generation(generation_info)
    
    # Otherwise, handle local model generation_info
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
        "extracted_verdict": extracted_verdict,
        "decision_index": decision_index,
        "matched_sequence": matched_sequence,
        "matched_label": target_label,
    }


def _compute_verdict_probability_info_from_gpt_generation(generation_info: Dict[str, object]) -> Dict[str, object]:
    """Extract TRUE/FALSE/HALF-TRUE logprobs from GPT chat completion response"""
    logprobs_data = generation_info.get("logprobs")
    
    if logprobs_data is None:
        raise ValueError("No logprobs data available in GPT response - logprobs_data is None")
    
    if not hasattr(logprobs_data, "content"):
        raise ValueError(f"No logprobs data available in GPT response - missing 'content' attribute. Type: {type(logprobs_data)}")
    
    choice_logprobs: Dict[str, float] = {}
    choice_first_tokens: Dict[str, str] = {}
    
    # Debug: only print verdict-related tokens
    # (We'll print details when we find TRUE/FALSE/HALF tokens below)
    
    # Look for verdict tokens: TRUE, FALSE, HALF
    # The pattern is usually: [VERDICT]: TRUE/FALSE/HALF-TRUE
    verdict_token_idx = -1
    
    # Search for tokens that contain TRUE, FALSE, or HALF
    for idx, token_logprob_obj in enumerate(logprobs_data.content):
        token = token_logprob_obj.token.strip().upper()
        
        # Check if this token is one of our verdict tokens
        if token in ('TRUE', 'FALSE', 'HALF'):
            verdict_token_idx = idx
            print(f"\n=== DEBUG: Found verdict token at position {idx} ===")
            print(f"Token: '{token_logprob_obj.token}' (logprob: {token_logprob_obj.logprob})")
            
            # Record the generated token's logprob
            if token == 'TRUE':
                # Check if it's HALF-TRUE by looking at previous token
                if idx > 0 and 'HALF' in logprobs_data.content[idx - 1].token.upper():
                    choice_logprobs["HALF-TRUE"] = token_logprob_obj.logprob
                    choice_first_tokens["HALF-TRUE"] = token_logprob_obj.token
                    print(f"    -> Generated verdict: 'HALF-TRUE'")
                else:
                    choice_logprobs["TRUE"] = token_logprob_obj.logprob
                    choice_first_tokens["TRUE"] = token_logprob_obj.token
                    print(f"    -> Generated verdict: 'TRUE'")
            elif token == 'FALSE':
                choice_logprobs["FALSE"] = token_logprob_obj.logprob
                choice_first_tokens["FALSE"] = token_logprob_obj.token
                print(f"    -> Generated verdict: 'FALSE'")
            elif token == 'HALF':
                # This is likely HALF-TRUE, check next token
                if idx + 1 < len(logprobs_data.content):
                    next_token = logprobs_data.content[idx + 1].token.strip().upper()
                    if 'TRUE' in next_token:
                        # Combine logprobs (approximate)
                        combined_logprob = token_logprob_obj.logprob + logprobs_data.content[idx + 1].logprob
                        choice_logprobs["HALF-TRUE"] = combined_logprob
                        choice_first_tokens["HALF-TRUE"] = token_logprob_obj.token
                        print(f"    -> Generated 'HALF-TRUE' (combined HALF + TRUE tokens)")
            
            # Extract alternative choices from top_logprobs
            if hasattr(token_logprob_obj, "top_logprobs") and token_logprob_obj.top_logprobs:
                print(f"\nTop alternatives for verdict token ({len(token_logprob_obj.top_logprobs)} total):")
                for top_token_obj in token_logprob_obj.top_logprobs:
                    print(f"  '{top_token_obj.token}' (logprob: {top_token_obj.logprob})")
                    top_token = top_token_obj.token.strip().upper()
                    
                    if top_token == 'TRUE' and "TRUE" not in choice_logprobs:
                        choice_logprobs["TRUE"] = top_token_obj.logprob
                        choice_first_tokens["TRUE"] = top_token_obj.token
                        print(f"    -> Found 'TRUE' alternative")
                    
                    elif top_token == 'FALSE' and "FALSE" not in choice_logprobs:
                        choice_logprobs["FALSE"] = top_token_obj.logprob
                        choice_first_tokens["FALSE"] = top_token_obj.token
                        print(f"    -> Found 'FALSE' alternative")
                    
                    elif top_token == 'HALF' and "HALF-TRUE" not in choice_logprobs:
                        # Approximate HALF-TRUE logprob
                        choice_logprobs["HALF-TRUE"] = top_token_obj.logprob
                        choice_first_tokens["HALF-TRUE"] = top_token_obj.token
                        print(f"    -> Found 'HALF' alternative (treating as HALF-TRUE)")
            
            print("=== END DEBUG ===\n")
            break
    
    if verdict_token_idx == -1:
        print("DEBUG: WARNING - Could not find verdict token (TRUE/FALSE/HALF) in response")
    
    # Summary of collected verdicts
    print(f"\nCollected verdict logprobs: {list(choice_logprobs.keys())}")
    for choice in MCQ_CHOICES:
        if choice in choice_logprobs and choice_logprobs[choice] != float("-inf"):
            print(f"  {choice}: {choice_logprobs[choice]:.4f}")
    
    # For any missing choices, set to -inf
    for choice in MCQ_CHOICES:
        if choice not in choice_logprobs:
            choice_logprobs[choice] = float("-inf")
            choice_first_tokens[choice] = ""
    
    # Convert to probabilities
    choice_probabilities = _logprobs_to_probabilities(choice_logprobs)
    predicted_verdict = max(choice_probabilities, key=choice_probabilities.get) if choice_probabilities else None
    
    verdict_logprobs = {
        MCQ_VERDICT_MAP[choice]: logprob for choice, logprob in choice_logprobs.items()
    }
    verdict_probabilities = {
        MCQ_VERDICT_MAP[choice]: prob for choice, prob in choice_probabilities.items()
    }
    
    print(f"\nFinal verdict probabilities:")
    for verdict, prob in verdict_probabilities.items():
        print(f"  {verdict}: {prob:.4f}")
    print()
    
    return {
        "choice_logprobs": choice_logprobs,
        "choice_probabilities": choice_probabilities,
        "choice_first_tokens": choice_first_tokens,
        "predicted_verdict": MCQ_VERDICT_MAP.get(predicted_verdict, predicted_verdict),
        "verdict_logprobs": verdict_logprobs,
        "verdict_probabilities": verdict_probabilities,
    }


def extract_verdict_and_probability(response_text, logprobs=None, scores=None, tokenizer=None, generated_tokens=None):
    """Extract verdict and probability from judge response"""
    verdict = _extract_verdict_label(response_text)
    print(f"Extracted verdict: {verdict} from text ending: ...{response_text[-100:]}")  # Debug print

    probability = None
    print(
        "Starting probability extraction - logprobs: "
        f"{logprobs is not None}, scores: {scores is not None}, "
        f"tokenizer: {tokenizer is not None}, tokens: {generated_tokens is not None}"
    )

    probability_info: Optional[Dict[str, object]] = None

    if scores is not None and tokenizer is not None and generated_tokens is not None:
        try:
            generation_info = {
                "scores": list(scores) if isinstance(scores, (list, tuple)) else scores,
                "tokenizer": tokenizer,
                "generated_tokens": generated_tokens,
            }
            probability_info = _compute_verdict_probability_info_from_generation(
                response_text,
                generation_info,
            )
            mapped_verdict = MCQ_VERDICT_MAP.get(verdict, verdict)
            probability = probability_info["verdict_probabilities"].get(mapped_verdict)
        except Exception as exc:
            print(f"Warning: generation-based verdict probability extraction failed: {exc}")

    elif logprobs is not None:
        try:
            verdict_tokens = ["TRUE", "FALSE", "HALF-TRUE"]
            for token_logprob in logprobs:
                if token_logprob.token in verdict_tokens:
                    probability = math.exp(token_logprob.logprob)
                    print(
                        f"Found verdict token '{token_logprob.token}' with probability: {probability}"
                    )
                    break
        except Exception as exc:
            print(f"Warning: Could not extract probability from logprobs: {exc}")

    if probability is None and probability_info is None and scores is not None:
        try:
            if len(scores) > 0:
                last_scores = scores[-1]
                if isinstance(last_scores, torch.Tensor) and last_scores.ndim == 2:
                    last_scores = last_scores[0]
                probabilities = torch.softmax(last_scores, dim=-1)
                probability = float(torch.max(probabilities).item())
                print(f"Fallback: Using max probability of last token: {probability}")
        except Exception as exc:
            print(f"Warning: Could not extract probability from scores (fallback): {exc}")

    return verdict, probability

def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300, get_probabilities: bool = False):
    """Run model inference based on model type"""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")
    
    if len(model_info) == 2:
        first, second = model_info
        
        # Determine the model type by inspecting the second element
        if isinstance(second, str):
            # GPT model: (client, model_name)
            client, model_name = model_info
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]  
            
            if get_probabilities:
                # Use chat completion API with logprobs for probability estimation
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    logprobs=True,
                    top_logprobs=20,  # Get top 20 token logprobs to capture TRUE/FALSE/HALF-TRUE tokens
                )
                content = response.choices[0].message.content.strip()
                
                # Extract logprobs information for GPT models
                generation_info = {
                    "logprobs": response.choices[0].logprobs,
                    "model_type": "gpt",
                }
                return content, generation_info
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
        
        else:
            # Local model: (tokenizer, model)
            tokenizer, model = model_info

            text, used_chat_template = build_chat_prompt(tokenizer, system_prompt, user_prompt)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            if get_probabilities:
                generate_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                outputs = inference_generate(
                    model,
                    inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generate_kwargs,
                )
                response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                extracted_response = extract_assistant_response(response, used_chat_template)
                
                # Extract only the newly generated tokens (excluding input tokens)
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[0][input_length:].tolist()
                
                print(f"Generated response length: {len(extracted_response)}")  # Debug print
                print(f"Scores available: {outputs.scores is not None if hasattr(outputs, 'scores') else 'No scores'}")
                if hasattr(outputs, 'scores') and outputs.scores is not None:
                    print(f"Scores type: {type(outputs.scores)}, length: {len(outputs.scores)}")
                    print(f"Generated tokens: {generated_tokens}")
                
                score_list = [score.detach().cpu() for score in outputs.scores] if outputs.scores else []
                generation_info = {
                    "scores": score_list,
                    "tokenizer": tokenizer,
                    "generated_tokens": generated_tokens,
                }
                return extracted_response, generation_info
            else:
                generate_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                outputs = inference_generate(
                    model,
                    inputs,
                    **generate_kwargs,
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return extract_assistant_response(response, used_chat_template)
    
    else:
        raise ValueError("Invalid model_info format")


def _extract_reason_section(response_text: str) -> str:
    """Extract the reason section without the verdict line."""
    reason_pattern = r'((?:\[)?REASON(?:\])?:.*?)(?=\n\s*(?:\[)?VERDICT(?:\])?:)'  # capture until verdict marker (with/without brackets)
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


def _logprobs_to_probabilities(logprob_map: Dict[str, float]) -> Dict[str, float]:
    """Convert choice log-probabilities into a probability distribution."""
    max_logprob = max(logprob_map.values())
    exp_map = {choice: math.exp(logprob - max_logprob) for choice, logprob in logprob_map.items()}
    total = sum(exp_map.values())
    return {choice: (value / total if total > 0 else 0.0) for choice, value in exp_map.items()}


def _compute_choice_logprobs_gpt(client, model_name: str, system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Compute next-token log-probabilities for choices using GPT-style models."""
    base_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: {assistant_context}"
    base_length = len(base_prompt)

    choice_logprobs: Dict[str, float] = {}
    for choice in MCQ_CHOICES:
        prompt_with_choice = base_prompt + choice
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_with_choice}
            ],
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
                if offset < base_length - delta:
                    continue
                sum_logprob += token_logprob
                token_count += 1

            if token_count > 0:
                choice_logprobs[choice] = sum_logprob / token_count
                recorded = True
                break

        if not recorded:
            choice_logprobs[choice] = float("-inf")

    return choice_logprobs


def compute_choice_logprobs(system_prompt: str, user_prompt: str, assistant_context: str) -> Dict[str, float]:
    """Dispatch next-token log-probability computation based on loaded model."""
    if model_info is None:
        raise ValueError("Model not loaded. Please call set_model_info() first.")

    if len(model_info) != 2:
        raise ValueError("Invalid model_info format")

    first, second = model_info
    if isinstance(second, str):
        client, model_name = model_info
        return _compute_choice_logprobs_gpt(client, model_name, system_prompt, user_prompt, assistant_context)

    raise RuntimeError(
        "Local model MCQ scoring requires generation_info scores; direct recomputation path has been removed."
    )


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

    system_prompt = get_system_prompt("judge")
    try:
        choice_logprobs = compute_choice_logprobs(system_prompt, user_prompt, assistant_context)
    except RuntimeError as exc:
        raise RuntimeError(
            "Generation scores missing for local model MCQ scoring; rerun with get_probabilities=True to capture generation_info."
        ) from exc

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
    }

# === Politician Agent ===
def opening_politician(claim, evidence):
    prompt = politician_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

def rebuttal_politician(claim, evidence, opponent_argument):
    prompt = politician_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("politician"), prompt)

def closing_politician(claim, evidence):
    prompt = politician_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("politician"), prompt)

# === Scientist Agent ===
def opening_scientist(claim, evidence):
    prompt = scientist_opening_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

def rebuttal_scientist(claim, evidence, opponent_argument):
    prompt = scientist_rebuttal_prompt(claim, evidence, opponent_argument)
    return run_model(get_system_prompt("scientist"), prompt)

def closing_scientist(claim, evidence):
    prompt = scientist_closing_prompt(claim, evidence)
    return run_model(get_system_prompt("scientist"), prompt)

# === Round Judge Agents ===
def judge_round_1(claim, evidence, pol_open, sci_open):
    """Judge after first round (opening statements)"""
    prompt = judge_prompt_1r(claim, evidence, pol_open, sci_open)
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, get_probabilities=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
        print(f"Round 1 - Got probability data type: {type(probability_data)}")  # Debug print
    else:
        response = result
        probability_data = None

    generation_info: Optional[Dict[str, object]] = None
    is_gpt_model = False

    if probability_data is not None:
        if isinstance(probability_data, dict):
            generation_info = probability_data
            # Check if it's GPT model
            if generation_info.get("model_type") == "gpt":
                is_gpt_model = True
        else:
            # Legacy: assume it's logprobs from OpenAI if not a dict
            is_gpt_model = True

    reason_section = _extract_reason_section(response)
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "

    # Compute probability info
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
    if probability is None and probability_info.get("predicted_verdict"):
        probability = probability_info["verdict_probabilities"].get(
            probability_info["predicted_verdict"]
        )

    base_result = {
        "verdict_text": response,
        "response": response,
        "verdict": verdict,
        "probability": probability,
        "probability_info": probability_info,
        "reason_section": reason_section.strip(),
        "generation_info": None,
    }

    return base_result

def judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut):
    """Judge after second round (opening + rebuttal statements)"""
    prompt = judge_prompt_2r(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, get_probabilities=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
        print(f"Round 2 - Got probability data type: {type(probability_data)}")  # Debug print
    else:
        response = result
        probability_data = None

    generation_info: Optional[Dict[str, object]] = None
    is_gpt_model = False

    if probability_data is not None:
        if isinstance(probability_data, dict):
            generation_info = probability_data
            # Check if it's GPT model
            if generation_info.get("model_type") == "gpt":
                is_gpt_model = True
        else:
            # Legacy: assume it's logprobs from OpenAI if not a dict
            is_gpt_model = True

    reason_section = _extract_reason_section(response)
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "

    # Compute probability info
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
    if probability is None and probability_info.get("predicted_verdict"):
        probability = probability_info["verdict_probabilities"].get(
            probability_info["predicted_verdict"]
        )

    base_result = {
        "verdict_text": response,
        "response": response,
        "verdict": verdict,
        "probability": probability,
        "probability_info": probability_info,
        "reason_section": reason_section.strip(),
        "generation_info": None,
    }

    return base_result

# === Final Judge Agent ===
def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    prompt = judge_prompt(
        claim, evidence,
        pol_open, sci_open,
        pol_rebut, sci_rebut,
        pol_close, sci_close
    )
    result = run_model(get_system_prompt("judge"), prompt, max_tokens=400, get_probabilities=True)
    
    if isinstance(result, tuple):
        response, probability_data = result
        print(f"Final - Got probability data type: {type(probability_data)}")  # Debug print
    else:
        response = result
        probability_data = None

    generation_info: Optional[Dict[str, object]] = None
    is_gpt_model = False

    if probability_data is not None:
        if isinstance(probability_data, dict):
            generation_info = probability_data
            # Check if it's GPT model
            if generation_info.get("model_type") == "gpt":
                is_gpt_model = True
        else:
            # Legacy: assume it's logprobs from OpenAI if not a dict
            is_gpt_model = True

    reason_section = _extract_reason_section(response)
    reason_context = reason_section.rstrip("\n ")
    assistant_context = f"{reason_context}\n[VERDICT]: " if reason_context else "[VERDICT]: "

    # Compute probability info
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
    if probability is None and probability_info.get("predicted_verdict"):
        probability = probability_info["verdict_probabilities"].get(
            probability_info["predicted_verdict"]
        )

    base_result = {
        "verdict_text": response,
        "response": response,
        "verdict": verdict,
        "probability": probability,
        "probability_info": probability_info,
        "reason_section": reason_section.strip(),
        "generation_info": None,
    }

    return base_result

# === Main Debate Function ===
def run_multi_agent_people_round_judges(claim, evidence):
    """Run the complete multi-agent people debate with round judges"""
    print("\n=== Running Multi-Agent People Debate with Round Judges ===")
    
    # Round 1: Opening statements
    print("Round 1: Opening statements...")
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    
    # Round 1 Judge
    print("Round 1 Judge evaluation...")
    round_1_judge = judge_round_1(claim, evidence, pol_open, sci_open)
    
    # Round 2: Rebuttal statements
    print("Round 2: Rebuttal statements...")
    pol_rebut = rebuttal_politician(claim, evidence, sci_open)
    sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
    
    # Round 2 Judge
    print("Round 2 Judge evaluation...")
    round_2_judge = judge_round_2(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut)
    
    # Round 3: Closing statements
    print("Round 3: Closing statements...")
    pol_close = closing_politician(claim, evidence)
    sci_close = closing_scientist(claim, evidence)
    
    # Final Judge
    print("Final Judge evaluation...")
    final_judge = judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close)
    
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
