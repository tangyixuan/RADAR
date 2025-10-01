# === System Prompt ===
def get_system_prompt(role: str = "fact_checker") -> str:
    if role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "debater":
        return "You are a critical thinker participating in a factual debate."
    elif role == "judge":
        return "You are a neutral judge who evaluates factual debates."
    else:
        return "You are a helpful assistant."

def system_prompt_fact_checker():
    return get_system_prompt("fact_checker")

# === INTENT INFERENCE ===

def user_prompt_intent_inference(claim):
    return f"""A claim may be literally accurate but still misleading due to the message it implies.

Your task is to infer the intended message or implied conclusion of the following claim.

Claim: "{claim}"

What is the intended conclusion of this claim?

Respond with one clear and concise sentence."""

# === Single-Agent Fact-Checking ===
def user_prompt_single_agent(claim: str, evidence: str) -> str:
    """
    User prompt template for single-agent fact verification.
    Fills in the claim and retrieved evidence.
    """
    return f"""
Given a claim and some retrieved evidence, determine whether the claim is TRUE, FALSE, or HALF-TRUE.

Claim: {claim}

Retrieved Evidence:
{evidence}

Answer format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE  
"""

# === Multi-Agent Fact-Checking ===
# === Opening Round ===
def user_prompt_opening_pro(claim, evidence):
    return f"""You support the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE."""

def user_prompt_opening_con(claim, evidence):
    return f"""You oppose the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Begin your argument by explaining why the claim is FALSE or misleading, referencing specific points in the evidence."""


# === Rebuttal Round ===
def user_prompt_rebuttal_pro(claim, evidence, con_argument):
    return f"""You are the supporting agent in a debate about the claim below. Your opponent has made an argument against the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{con_argument}

Write your rebuttal, explaining why the opponent is wrong and defending the claim."""

def user_prompt_rebuttal_con(claim, evidence, pro_argument):
    return f"""You are the opposing agent in a debate about the claim below. Your opponent has made an argument supporting the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{pro_argument}

Write your rebuttal, explaining why the opponent is incorrect and the claim is still FALSE or HALF-TRUE."""


# === Closing Round ===
def user_prompt_closing_pro(claim, evidence):
    return f"""You are the supporting agent in a debate. Summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Provide a closing statement reinforcing why the claim is TRUE."""

def user_prompt_closing_con(claim, evidence):
    return f"""You are the opposing agent in a debate. Summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Provide a closing statement reinforcing why the claim is FALSE or HALF-TRUE."""


# === Judge Prompt ===
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

# === Intent Enhanced Reformulation ===
def user_prompt_reformulate_pro(claim, intent):
    return f"""You support the following claim and want to restate it in a way that clearly reflects your agreement with its message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to rephrase the claim using a positive, supporting tone that affirms the claim’s main idea.

Only output the reformulated claim directly, without extra explanations.

Reformulated (Pro) Claim:"""

def user_prompt_reformulate_con(claim, intent):
    return f"""You oppose the following claim and want to restate it in a way that clearly reflects your disagreement with its message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to rephrase the claim using a critical, opposing tone that challenges or rejects its main idea.

Only output the reformulated claim directly, without extra explanations.

Reformulated (Con) Claim:"""
