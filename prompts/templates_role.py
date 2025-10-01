# === SYSTEM PROMPTS ===

def get_system_prompt(role: str = "fact_checker") -> str:
    if role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "debater":
        return "You are a critical thinker participating in a factual debate."
    elif role == "judge":
        return "You are a neutral judge who evaluates factual debates."
    else:
        return "You are a helpful assistant."

# === INTENT INFERENCE ===

def user_prompt_intent_inference(claim):
    return f"""A claim may be literally accurate but still misleading due to the message it implies.

Your task is to infer the intended message or implied conclusion of the following claim.

Claim: "{claim}"

What is the intended conclusion of this claim?

Respond with one clear and concise sentence."""

# === ROLE INFERENCE ===


def user_prompt_role_inference(intent):
    return f"""You are analyzing a social or political claim.

The following is the **intended conclusion** of a claim:

"{intent}"

Your task is to infer **two clearly opposing real-world social, political, or cultural roles** that would likely support or oppose this conclusion in an actual public debate.

DO NOT use generic labels like "Pro", "Con", "Supporter", or "Opponent".

Instead, name **specific groups or identity categories**, such as:
- Political parties (e.g., "Republicans", "Democratic Socialists")
- Professions or interest groups (e.g., "Oil industry executives", "Climate activists")
- Demographic groups (e.g., "Young urban voters", "Evangelical Christians")
- Organizations (e.g., "ACLU", "NRA")

Respond **strictly** using this format:

SUPPORTING_ROLE: <name of the group likely to support>
OPPOSING_ROLE: <name of the group likely to oppose>

Example:
SUPPORTING_ROLE: Environmental activists  
OPPOSING_ROLE: Fossil fuel industry lobbyists
"""
# === PRO AGENT ===

def user_prompt_opening_pro(claim, evidence, role):
    return f"""You are a {role} who supports the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE."""

def user_prompt_rebuttal_pro(claim, evidence, con_argument, role):
    return f"""You are a {role} who supports the following claim in a factual debate. Your opponent has argued against the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{con_argument}

Write your rebuttal, explaining why the opponent is wrong and defending the claim."""

def user_prompt_closing_pro(claim, evidence, role):
    return f"""You are a {role} who supports the claim below. Provide your final summary.

Claim: {claim}

Evidence:
{evidence}

Provide a closing statement reinforcing why the claim is TRUE."""

# === CON AGENT ===

def user_prompt_opening_con(claim, evidence, role):
    return f"""You are a {role} who opposes the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Begin your argument by explaining why the claim is FALSE or misleading, referencing specific points in the evidence."""

def user_prompt_rebuttal_con(claim, evidence, pro_argument, role):
    return f"""You are a {role} who opposes the following claim in a factual debate. Your opponent has argued in favor of the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{pro_argument}

Write your rebuttal, explaining why the opponent is incorrect and the claim is still FALSE or HALF-TRUE."""

def user_prompt_closing_con(claim, evidence, role):
    return f"""You are a {role} who opposes the claim below. Provide your final summary.
    
Claim: {claim}

Evidence:
{evidence}

Provide a closing statement reinforcing why the claim is FALSE or HALF-TRUE."""

# === JUDGE AGENT ===

def user_prompt_judge_full(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
    return f"""You are a neutral judge evaluating a factual debate.

Claim: {claim}

Evidence:
{evidence}

--- Opening Statements ---
Pro Agent:
{pro_open}

Con Agent:
{con_open}

--- Rebuttals ---
Pro Agent:
{pro_rebut}

Con Agent:
{con_rebut}

--- Closing Statements ---
Pro Agent:
{pro_close}

Con Agent:
{con_close}

Based on the arguments and evidence, decide whether the claim is TRUE, FALSE, or HALF-TRUE.

Answer format:
[REASON]: <your justification>
[VERDICT]: TRUE / FALSE / HALF-TRUE  
"""