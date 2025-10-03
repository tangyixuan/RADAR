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

        # 通过检查第二个元素来区分模型类型
        if isinstance(second, str):
            # GPT model: (client, model_name)
            client, model_name = model_info
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            if collect_scores:
                print("Warning: collect_scores not supported for GPT models; falling back to recomputation.")
                return content, None
            return content

        else:
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
    return choice_logprobs, choice_first_tokens


def _compute_continuation_probability_info(
    user_prompt: str,
    assistant_context: str = "DECISION: ",
) -> Dict[str, object]:
    return _compute_continuation_probability_info_with_generation(
        user_prompt=user_prompt,
        assistant_context=assistant_context,
        generation_info=None,
    )


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


# === Interim Decision Agent (Between Rounds) ===

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


def _continuation_prompt_after_round(claim, evidence, round_name, transcripts, executed_rounds):
    summary = _format_round_summary(transcripts, executed_rounds)
    return f"""
You are acting as an interim adjudicator during a two-agent fact-checking debate.

Claim: {claim}

Evidence:
{evidence}

Rounds completed so far (most recent round: {round_name}):
{summary}

Decide whether the debate should continue to the next round or whether a final verdict can already be reached based on the arguments so far.

Respond strictly with:
DECISION: CONTINUE or STOP
REASON: <one or two sentences>
"""


def _should_continue_after_round(claim, evidence, round_name, transcripts, executed_rounds):
    prompt = _continuation_prompt_after_round(claim, evidence, round_name, transcripts, executed_rounds)
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


def run_multi_agent_people_bj(claim, evidence):
    """Run multi-agent debate with a continuation judge between rounds."""

    transcripts = {name: {"politician": "", "scientist": ""} for name in _ROUNDS}
    executed_rounds = []
    continuation_decisions = []

    # Opening round
    pol_open = opening_politician(claim, evidence)
    sci_open = opening_scientist(claim, evidence)
    transcripts["opening"] = {"politician": pol_open, "scientist": sci_open}
    executed_rounds.append("opening")

    continue_debate, decision_text, probability_info = _should_continue_after_round(
        claim, evidence, "opening", transcripts, executed_rounds
    )
    continuation_decisions.append(
        {
            "round": "opening",
            "decision": "continue" if continue_debate else "stop",
            "rationale": decision_text,
            "choice_logprobs": probability_info.get("choice_logprobs"),
            "choice_probabilities": probability_info.get("choice_probabilities"),
            "choice_first_tokens": probability_info.get("choice_first_tokens"),
        }
    )

    pol_rebut = sci_rebut = ""
    pol_close = sci_close = ""

    if continue_debate:
        # Rebuttal round
        pol_rebut = rebuttal_politician(claim, evidence, sci_open)
        sci_rebut = rebuttal_scientist(claim, evidence, pol_open)
        transcripts["rebuttal"] = {"politician": pol_rebut, "scientist": sci_rebut}
        executed_rounds.append("rebuttal")

        continue_debate, decision_text, probability_info = _should_continue_after_round(
            claim, evidence, "rebuttal", transcripts, executed_rounds
        )
        continuation_decisions.append(
            {
                "round": "rebuttal",
                "decision": "continue" if continue_debate else "stop",
                "rationale": decision_text,
                "choice_logprobs": probability_info.get("choice_logprobs"),
                "choice_probabilities": probability_info.get("choice_probabilities"),
                "choice_first_tokens": probability_info.get("choice_first_tokens"),
            }
        )

    if continue_debate:
        # Closing round
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
