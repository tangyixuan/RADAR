# === 4-Agent Debate System Prompts ===
def get_system_prompt(role: str = "fact_checker") -> str:
    if role == "pro1":
        return "You are a factual expert supporting the claim. Focus on verifying factual accuracy and identifying reliable evidence."
    elif role == "pro2":
        return "You are a reasoning expert supporting the claim. Focus on building logical arguments and analyzing reasoning chains."
    elif role == "con1":
        return "You are a source critic opposing the claim. Focus on identifying unreliable sources and conflicting evidence."
    elif role == "con2":
        return "You are a reasoning critic opposing the claim. Focus on exposing logical flaws and reasoning errors."
    elif role == "judge":
        return "You are a neutral judge evaluating the debate. Assess all arguments objectively and make final judgment."
    elif role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "debater":
        return "You are a critical thinker participating in a factual debate."
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

# === 4-Agent Debate System ===
# === Opening Round ===
def user_prompt_opening_pro1(claim, evidence):
    return f"""As a factual expert, support this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: factual accuracy, reliable evidence, specific facts supporting the claim."""

def user_prompt_opening_pro2(claim, evidence):
    return f"""As a reasoning expert, support this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: logical arguments, reasoning chains, claim rationality."""

def user_prompt_opening_con1(claim, evidence):
    return f"""As a source critic, oppose this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: unreliable sources, conflicting evidence, credibility issues."""

def user_prompt_opening_con2(claim, evidence):
    return f"""As a reasoning critic, oppose this claim. Present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Focus on: logical flaws, reasoning errors, argument weaknesses."""

# === Legacy Prompts ===
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
def user_prompt_rebuttal_pro1(claim, evidence, con1_argument, con2_argument):
    return f"""As a factual expert, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Con-1 (source critic) argument:
{con1_argument}

Con-2 (reasoning critic) argument:
{con2_argument}

Focus on: defending factual accuracy, reliable evidence, source credibility."""

def user_prompt_rebuttal_pro2(claim, evidence, con1_argument, con2_argument):
    return f"""As a reasoning expert, rebut the opposing arguments.

Claim: {claim}

Evidence:
{evidence}

Con-1 (source critic) argument:
{con1_argument}

Con-2 (reasoning critic) argument:
{con2_argument}

Focus on: defending logical arguments, reasoning chains, logical validity."""

def user_prompt_rebuttal_con1(claim, evidence, pro1_argument, pro2_argument):
    return f"""As a source critic, rebut the supporting arguments.

Claim: {claim}

Evidence:
{evidence}

Pro-1 (factual expert) argument:
{pro1_argument}

Pro-2 (reasoning expert) argument:
{pro2_argument}

Focus on: questioning source reliability, identifying conflicts, credibility issues."""

def user_prompt_rebuttal_con2(claim, evidence, pro1_argument, pro2_argument):
    return f"""As a reasoning critic, rebut the supporting arguments.

Claim: {claim}

Evidence:
{evidence}

Pro-1 (factual expert) argument:
{pro1_argument}

Pro-2 (reasoning expert) argument:
{pro2_argument}

Focus on: exposing logical flaws, reasoning errors, argument weaknesses."""

# === Legacy Prompts ===
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
def user_prompt_closing_pro1(claim, evidence):
    return f"""As a factual expert, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: factual accuracy, key evidence, why claim is factually correct."""

def user_prompt_closing_pro2(claim, evidence):
    return f"""As a reasoning expert, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: logical validity, reasoning chains, why claim is logically sound."""

def user_prompt_closing_con1(claim, evidence):
    return f"""As a source critic, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: source unreliability, conflicting evidence, why claim is not credible."""

def user_prompt_closing_con2(claim, evidence):
    return f"""As a reasoning critic, summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Summarize: logical flaws, reasoning errors, why claim is logically unsound."""

# === Legacy Prompts ===
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
def user_prompt_judge_4_agents(claim, evidence, pro1_open, pro2_open, con1_open, con2_open, 
                              pro1_rebut, pro2_rebut, con1_rebut, con2_rebut,
                              pro1_close, pro2_close, con1_close, con2_close):
    return f"""You are a neutral judge evaluating a 4-agent debate.

Claim: {claim}

Evidence:
{evidence}

=== Opening Statements ===
Pro-1 (Factual Expert):
{pro1_open}

Pro-2 (Reasoning Expert):
{pro2_open}

Con-1 (Source Critic):
{con1_open}

Con-2 (Reasoning Critic):
{con2_open}

=== Rebuttals ===
Pro-1 (Factual Expert):
{pro1_rebut}

Pro-2 (Reasoning Expert):
{pro2_rebut}

Con-1 (Source Critic):
{con1_rebut}

Con-2 (Reasoning Critic):
{con2_rebut}

=== Closing Statements ===
Pro-1 (Factual Expert):
{pro1_close}

Pro-2 (Reasoning Expert):
{pro2_close}

Con-1 (Source Critic):
{con1_close}

Con-2 (Reasoning Critic):
{con2_close}

Based on all arguments and evidence, decide whether the claim is TRUE, FALSE, or HALF-TRUE.

Answer format:
[REASON]: <your justification>
[VERDICT]: TRUE / FALSE / HALF-TRUE  
"""

# === Legacy Judge Prompt ===
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
