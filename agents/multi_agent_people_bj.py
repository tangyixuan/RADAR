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
from agents.chat_template_utils import build_chat_prompt, extract_assistant_response

# Global model info - will be set by main.py
model_info = None


def set_model_info(info):
    """Set the global model info"""
    global model_info
    model_info = info


def run_model(system_prompt: str, user_prompt: str, max_tokens: int = 300):
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
            return response.choices[0].message.content.strip()

        else:
            # Local model: (tokenizer, model)
            tokenizer, model = model_info

            text, used_chat_template = build_chat_prompt(tokenizer, system_prompt, user_prompt)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return extract_assistant_response(response, used_chat_template)

    raise ValueError("Invalid model_info format")


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
    response = run_model(get_system_prompt("judge"), prompt, max_tokens=200)

    decision_line = ""
    for line in response.splitlines():
        upper = line.upper()
        if "DECISION" in upper:
            decision_line = upper
            break

    if "STOP" in decision_line and "CONTINUE" not in decision_line:
        return False, response
    if "CONTINUE" in decision_line and "STOP" not in decision_line:
        return True, response

    upper_response = response.upper()
    if "STOP" in upper_response and "CONTINUE" not in upper_response:
        return False, response

    return True, response


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

    continue_debate, decision_text = _should_continue_after_round(
        claim, evidence, "opening", transcripts, executed_rounds
    )
    continuation_decisions.append(
        {
            "round": "opening",
            "decision": "continue" if continue_debate else "stop",
            "rationale": decision_text,
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

        continue_debate, decision_text = _should_continue_after_round(
            claim, evidence, "rebuttal", transcripts, executed_rounds
        )
        continuation_decisions.append(
            {
                "round": "rebuttal",
                "decision": "continue" if continue_debate else "stop",
                "rationale": decision_text,
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
