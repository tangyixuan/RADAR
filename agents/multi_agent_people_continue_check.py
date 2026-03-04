import math
from typing import Dict

import torch

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
)
from agents.chat_template_utils import (
    build_chat_prompt,
    extract_assistant_response,
    inference_generate,
)

# Global model info - will be set by main.py
model_info = None

CONTINUATION_CHOICES = ("STOP", "CONTINUE")


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
            
            # For GPT models, we can use logprobs in chat completions
            if collect_scores:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    logprobs=True,
                    top_logprobs=20,  # Get top 20 token logprobs to capture STOP/CONTINUE tokens
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
                    temperature=0.7,
                )
                content = response.choices[0].message.content.strip()
                return content

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


# === Continuation scoring helpers ===

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


def _compute_choice_logprobs_gpt(
    client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    assistant_context: str,
    choices: tuple[str, ...],
) -> tuple[Dict[str, float], Dict[str, str]]:
    base_prompt = (
        f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: {assistant_context}"
    )
    base_length = len(base_prompt)

    choice_logprobs: Dict[str, float] = {}
    choice_first_tokens: Dict[str, str] = {}
    for choice in choices:
        prompt_with_choice = base_prompt + choice
        response = client.completions.create(
            model=model_name,
            prompt=prompt_with_choice,
            max_tokens=0,
            temperature=0,
            logprobs=1,
            echo=True,
        )
        logprob_data = response.choices[0].logprobs
        if logprob_data is None or logprob_data.tokens is None:
            raise RuntimeError("Logprob data unavailable for continuation scoring.")

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

            first_token_logprob = token_logprobs[first_idx]
            first_token_text = tokens[first_idx]
            if first_token_logprob is not None:
                choice_logprobs[choice] = first_token_logprob
            else:
                choice_logprobs[choice] = float("-inf")
            choice_first_tokens[choice] = first_token_text
            recorded = True
            break

        if not recorded:
            choice_logprobs[choice] = float("-inf")
            choice_first_tokens[choice] = ""

    return choice_logprobs, choice_first_tokens


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


def _compute_choice_logprobs_from_generation(generation_info: Dict[str, object]) -> tuple[Dict[str, float], Dict[str, str]]:
    # Check if this is GPT model generation_info
    if generation_info.get("model_type") == "gpt":
        return _compute_choice_logprobs_from_gpt_generation(generation_info)
    
    # Otherwise, handle local model generation_info
    tokenizer = generation_info.get("tokenizer")
    scores = generation_info.get("scores")
    generated_tokens = generation_info.get("generated_tokens")

    if tokenizer is None or scores is None or generated_tokens is None:
        raise ValueError("Incomplete generation info for continuation scoring.")

    if not scores:
        raise ValueError("Empty generation scores for continuation scoring.")

    stop_token_ids = tokenizer.encode(" STOP", add_special_tokens=False)
    continue_token_ids = tokenizer.encode(" CONTINUE", add_special_tokens=False)
    if not stop_token_ids or not continue_token_ids:
        raise ValueError("Failed to obtain STOP/CONTINUE token ids from tokenizer.")

    stop_token_id = stop_token_ids[0]
    continue_token_id = continue_token_ids[0]

    decision_index = None
    for idx, token_id in enumerate(generated_tokens):
        if token_id in (stop_token_id, continue_token_id):
            decision_index = idx
            break

    if decision_index is None or decision_index >= len(scores):
        raise ValueError("Could not locate decision token in generated sequence.")

    score_tensor = scores[decision_index][0]
    log_probs = torch.log_softmax(score_tensor, dim=-1)

    choice_logprobs = {
        "STOP": float(log_probs[stop_token_id].item()),
        "CONTINUE": float(log_probs[continue_token_id].item()),
    }
    choice_first_tokens = {
        "STOP": tokenizer.decode([stop_token_id]),
        "CONTINUE": tokenizer.decode([continue_token_id]),
    }

    print("choice_logprobs:", choice_logprobs)
    print("choice_first_tokens:", choice_first_tokens)

    return choice_logprobs, choice_first_tokens


def _compute_choice_logprobs_from_gpt_generation(generation_info: Dict[str, object]) -> tuple[Dict[str, float], Dict[str, str]]:
    """Extract STOP/CONTINUE logprobs from GPT chat completion response"""
    logprobs_data = generation_info.get("logprobs")
    
    if logprobs_data is None or not hasattr(logprobs_data, "content"):
        raise ValueError("No logprobs data available in GPT response")
    
    choice_logprobs: Dict[str, float] = {}
    choice_first_tokens: Dict[str, str] = {}
    
    # Debug: print all tokens to see what we're getting
    print("\n=== DEBUG: GPT Logprobs Tokens ===")
    for idx, token_logprob_obj in enumerate(logprobs_data.content[:20]):  # Show first 20 tokens
        print(f"Token {idx}: '{token_logprob_obj.token}' (logprob: {token_logprob_obj.logprob})")
        if hasattr(token_logprob_obj, "top_logprobs") and token_logprob_obj.top_logprobs:
            print(f"  Top alternatives: {[t.token for t in token_logprob_obj.top_logprobs[:5]]}")
    print("=== END DEBUG ===\n")
    
    # Find the first token that is ' STOP' or ' CONT' (usually Token 3 after 'DEC' + 'ISION' + ':')
    # Then extract both ' STOP' and ' CONT' logprobs from that token's top_logprobs
    
    decision_token_idx = -1
    
    # Search for the token containing ' STOP' or ' CONT'
    for idx, token_logprob_obj in enumerate(logprobs_data.content[:10]):
        token = token_logprob_obj.token
        if token in (' STOP', ' CONT'):
            decision_token_idx = idx
            print(f"DEBUG: Found decision token at position {idx}: '{token}'")
            
            # Record the generated token's logprob
            if token == ' STOP':
                choice_logprobs["STOP"] = token_logprob_obj.logprob
                choice_first_tokens["STOP"] = token
                print(f"DEBUG: Generated ' STOP' with logprob {token_logprob_obj.logprob}")
            else:  # ' CONT'
                choice_logprobs["CONTINUE"] = token_logprob_obj.logprob
                choice_first_tokens["CONTINUE"] = token
                print(f"DEBUG: Generated ' CONT' with logprob {token_logprob_obj.logprob}")
            
            # Extract the other choice from top_logprobs
            if hasattr(token_logprob_obj, "top_logprobs") and token_logprob_obj.top_logprobs:
                print(f"DEBUG: Checking top_logprobs for the alternative:")
                for top_token_obj in token_logprob_obj.top_logprobs:
                    # Look for ' STOP' if we haven't found it yet
                    if top_token_obj.token == ' STOP' and "STOP" not in choice_logprobs:
                        choice_logprobs["STOP"] = top_token_obj.logprob
                        choice_first_tokens["STOP"] = top_token_obj.token
                        print(f"DEBUG: Found ' STOP' in top_logprobs with logprob {top_token_obj.logprob}")
                    
                    # Look for ' CONT' if we haven't found it yet
                    elif top_token_obj.token == ' CONT' and "CONTINUE" not in choice_logprobs:
                        choice_logprobs["CONTINUE"] = top_token_obj.logprob
                        choice_first_tokens["CONTINUE"] = top_token_obj.token
                        print(f"DEBUG: Found ' CONT' in top_logprobs with logprob {top_token_obj.logprob}")
            
            break
    
    if decision_token_idx == -1:
        print("DEBUG: ERROR - Could not find ' STOP' or ' CONT' token in first 10 tokens")
    
    print(f"DEBUG: Done. STOP found: {'STOP' in choice_logprobs}, CONTINUE found: {'CONTINUE' in choice_logprobs}")
    
    # For any missing choices, set to -inf
    if "STOP" not in choice_logprobs:
        choice_logprobs["STOP"] = float("-inf")
        choice_first_tokens["STOP"] = ""
    if "CONTINUE" not in choice_logprobs:
        choice_logprobs["CONTINUE"] = float("-inf")
        choice_first_tokens["CONTINUE"] = ""
    
    print(f"\n=== DEBUG: Final choice_logprobs: {choice_logprobs} ===\n")
        
    return choice_logprobs, choice_first_tokens


def _compute_continuation_probability_info_with_generation(
    user_prompt: str,
    assistant_context: str = "DECISION: ",
    generation_info: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if generation_info is not None:
        try:
            choice_logprobs, choice_first_tokens = _compute_choice_logprobs_from_generation(generation_info)
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


# === Judge Agent ===

def judge_final_verdict(claim, evidence, pol_open, sci_open, pol_rebut, sci_rebut, pol_close, sci_close):
    prompt = judge_prompt(
        claim,
        evidence,
        pol_open,
        sci_open,
        pol_rebut,
        sci_rebut,
        pol_close,
        sci_close,
    )
    return run_model(get_system_prompt("judge"), prompt, max_tokens=400)


# === Interim Decision Agent (Before Rounds) ===

_ROUNDS = ["opening", "rebuttal", "closing"]


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


def run_multi_agent_people_continue_check(claim, evidence):
    """Run multi-agent debate with a continuation judge before each round."""

    transcripts = {name: {"politician": "", "scientist": ""} for name in _ROUNDS}
    executed_rounds = []
    continuation_decisions = []

    pol_open = sci_open = ""
    pol_rebut = sci_rebut = ""
    pol_close = sci_close = ""

    for round_name in _ROUNDS:
        continue_debate, decision_text, probability_info = _should_continue_before_round(
            claim, evidence, round_name, transcripts, executed_rounds
        )
        continuation_decisions.append(
            {
                "round": round_name,
                "decision": "continue" if continue_debate else "stop",
                "rationale": decision_text,
                "choice_logprobs": probability_info.get("choice_logprobs"),
                "choice_probabilities": probability_info.get("choice_probabilities"),
                "choice_first_tokens": probability_info.get("choice_first_tokens"),
            }
        )

        # if not continue_debate:
        #     break

        if round_name == "opening":
            pol_open = opening_politician(claim, evidence)
            sci_open = opening_scientist(claim, evidence)
            transcripts["opening"] = {"politician": pol_open, "scientist": sci_open}
            executed_rounds.append("opening")

        elif round_name == "rebuttal":
            pol_rebut = rebuttal_politician(claim, evidence, sci_open)
            sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
            transcripts["rebuttal"] = {"politician": pol_rebut, "scientist": sci_rebut}
            executed_rounds.append("rebuttal")

        elif round_name == "closing":
            pol_close = closing_politician(claim, evidence)
            sci_close = closing_scientist(claim, evidence)
            transcripts["closing"] = {"politician": pol_close, "scientist": sci_close}
            executed_rounds.append("closing")

    final_verdict = judge_final_verdict(
        claim,
        evidence,
        transcripts["opening"]["politician"],
        transcripts["opening"]["scientist"],
        transcripts["rebuttal"]["politician"],
        transcripts["rebuttal"]["scientist"],
        transcripts["closing"]["politician"],
        transcripts["closing"]["scientist"],
    )

    return {
        "transcripts": transcripts,
        "continuation_decisions": continuation_decisions,
        "final_verdict": final_verdict,
    }
