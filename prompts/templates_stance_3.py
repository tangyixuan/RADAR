# === System Prompt ===
def get_system_prompt(role: str = "fact_checker") -> str:
    if role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "pro":
        return """You are a supporting agent in a factual debate. You represent the position that supports the claim.
When debating, frame arguments in terms of why the claim is TRUE and supported by evidence. Keep responses concise and focused."""
    elif role == "con":
        return """You are an opposing agent in a factual debate. You represent the position that opposes the claim.
When debating, frame arguments in terms of why the claim is FALSE or HALF-TRUE. Keep responses concise and focused."""
    elif role == "flexible":
        return """You are a flexible analyst in a factual debate. You represent an independent perspective that can take any position based on evidence.

Key characteristics:
- Analyze evidence from multiple angles
- Consider different perspectives and interpretations
- Take a position based on the strongest evidence
- Emphasize nuanced analysis and balanced reasoning
- Focus on the most compelling arguments regardless of stance

When debating, choose your position based on the evidence and present arguments from that perspective. Keep responses concise and focused."""
    elif role == "judge":
        return """You are a neutral judge who evaluates factual debates."""
    else:
        return "You are a helpful assistant."

def system_prompt_fact_checker():
    return get_system_prompt("fact_checker")

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

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support the claim as TRUE."""

def user_prompt_opening_con(claim, evidence):
    return f"""You oppose the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:{evidence}

Begin your argument by explaining why the claim is FALSE or misleading, referencing specific points in the evidence."""

def user_prompt_opening_flexible(claim, evidence, pro_argument, con_argument):
    return f"""You are a flexible analyst evaluating the following claim. Based on the evidence and the arguments from both sides, choose your position and present your opening argument.

Claim: {claim}

Evidence:
{evidence}

Proponent's argument:
{pro_argument}

Opponent's argument:
{con_argument}

Analyze the evidence and both arguments from a fresh, different perspective. Consider what angles or interpretations the other two agents might have missed. Then choose whether to support or oppose the claim, but present your opening argument from a unique viewpoint that differs from both the proponent and opponent."""

# === Rebuttal Round ===
def user_prompt_rebuttal_pro(claim, evidence, con_argument):
    return f"""You are the supporting agent in a debate about the claim below. Your opponent has made an argument against the claim.

Claim: {claim}

Evidence:{evidence}

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

def user_prompt_rebuttal_flexible(claim, evidence, pro_argument, con_argument):
    return f"""You are a flexible analyst in a debate about the claim below. Both opponents have made their arguments.

Claim: {claim}

Evidence: {evidence}

Proponent's argument: {pro_argument}

Opponent's argument: {con_argument}

Write your rebuttal from a unique perspective that differs from both the proponent and opponent. Address the weaknesses in their arguments while bringing in alternative viewpoints or interpretations they may have missed."""

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

def user_prompt_closing_flexible(claim, evidence, pro_argument, con_argument):
    return f"""You are a flexible analyst in a debate. Summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Proponent's argument:
{pro_argument}

Opponent's argument:
{con_argument}

Provide a closing statement that reinforces your position while highlighting the unique perspective and different angles you brought to this debate that the other agents may have overlooked."""

# === Judge Prompt ===
# def judge_prompt(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
#     return f"""
# You are an ordinary American citizen evaluating a political debate.

# Claim: {claim}

# Evidence: {evidence}

# --- Opening Statements ---
# Proponent's opening: {pro_open}
# Opponent's opening: {con_open}

# --- Rebuttals ---
# Proponent's rebuttal: {pro_rebut}
# Opponent's rebuttal: {con_rebut}

# --- Closing Statements ---
# Proponent's closing: {pro_close}
# Opponent's closing: {con_close}

# Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

# You MUST respond in exactly this format:
# [REASON]: <your explanation>
# [VERDICT]: TRUE / FALSE / HALF-TRUE

# Make the verdict in the last line.
# """

def judge_prompt_three_agents(claim, evidence, flexible_open, pro_open, con_open, flexible_rebut, pro_rebut, con_rebut, flexible_close, pro_close, con_close):
    return f"""
You are an ordinary American citizen evaluating a political debate with three participants.

Claim: {claim}

Evidence: {evidence}

--- Opening Statements ---
Proponent's opening: {pro_open}
Opponent's opening: {con_open}
Flexible analyst's opening: {flexible_open}

--- Rebuttals ---
Proponent's rebuttal: {pro_rebut}
Opponent's rebuttal: {con_rebut}
Flexible analyst's rebuttal: {flexible_rebut}

--- Closing Statements ---
Proponent's closing: {pro_close}
Opponent's closing: {con_close}
Flexible analyst's closing: {flexible_close}

Based on the evidence and arguments from all three participants, decide if the claim is TRUE, FALSE, or HALF-TRUE.

You MUST respond in exactly this format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE

Make the verdict in the last line.
"""

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

def user_prompt_judge_three_agents_full(claim, evidence, flexible_open, pro_open, con_open, flexible_rebut, pro_rebut, con_rebut, flexible_close, pro_close, con_close):
    return f"""You are a neutral judge evaluating a factual debate with three participants.

Claim: {claim}

Evidence:
{evidence}

--- Opening Statements ---

Pro Agent:
{pro_open}

Con Agent:
{con_open}

Flexible Analyst:
{flexible_open}

--- Rebuttals ---
Pro Agent:
{pro_rebut}

Con Agent:
{con_rebut}

Flexible Analyst:
{flexible_rebut}

--- Closing Statements ---
Pro Agent:
{pro_close}

Con Agent:
{con_close}

Flexible Analyst:
{flexible_close}

Based on the arguments and evidence from all three participants, decide whether the claim is TRUE, FALSE, or HALF-TRUE.

Answer format:
[REASON]: <your justification>
[VERDICT]: TRUE / FALSE / HALF-TRUE  
"""

def user_prompt_intent_inference(claim):
    return f"""A claim may be literally accurate but still misleading due to the message it implies.

Your task is to infer the intended message or implied conclusion of the following claim.

Claim: "{claim}"

What is the intended conclusion of this claim?

Respond with one clear and concise sentence."""

def user_prompt_reformulate_pro(claim, intent):
    return f"""You support the following claim and aim to reinforce its implied message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to reformulate the claim in a way that makes the implied conclusion more explicit and persuasive.

Only output the reformulated claim directly, without any introductory phrases or explanations.

Reformulated (Pro) Claim:"""

def user_prompt_reformulate_con(claim, intent):
    return f"""You oppose the following claim and aim to highlight any issues with its implied message.

Claim: "{claim}"

Inferred Intent: "{intent}"

Your task is to reformulate the claim in a way that emphasizes its potential problems, uncertainties, or misleading assumptions.

Only output the reformulated claim directly, without any introductory phrases or explanations.

Reformulated (Con) Claim:"""