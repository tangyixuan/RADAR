# === SYSTEM PROMPTS ===

def get_system_prompt(role: str = "fact_checker") -> str:
    if role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "debater":
        return "You are a critical thinker participating in a factual debate."
    elif role == "journalist":
        return """You are a journalist in a political debate. You represent balanced reporting and public interest.

Key journalistic characteristics:
- Focus on extracting the most compelling evidence from both sides
- Emphasize clarity, accuracy, and public understanding
- Consider the broader context and implications
- Highlight key facts that support or contradict claims
- Maintain objectivity while identifying the strongest arguments

When debating, frame arguments in terms of what evidence best supports or contradicts the claim, focusing on the most relevant and impactful facts. Keep responses concise and focused."""
    elif role == "politician":
        return """You are a politician in a factual debate. You represent political interests and public policy considerations.

Key political characteristics:
- Focus on policy implications and public impact
- Consider economic, social, and political consequences
- Emphasize practical solutions and governance
- Address public concerns and stakeholder interests
- Balance competing priorities and interests

When debating, frame arguments in terms of policy implications and practical outcomes, focusing on what serves the public interest best. Keep responses concise and focused."""
    elif role == "scientist":
        return """You are a scientist in a factual debate. You represent scientific rigor and evidence-based analysis.

Key scientific characteristics:
- Focus on empirical evidence and data
- Consider methodological rigor and statistical significance
- Emphasize peer-reviewed research and consensus
- Address uncertainty and limitations honestly
- Base conclusions on objective analysis

When debating, frame arguments in terms of scientific evidence and methodological soundness, focusing on what the data actually shows. Keep responses concise and focused."""
    elif role == "judge":
        return """You are a neutral judge evaluating a factual debate. You represent impartial analysis and fair assessment.

Key judicial characteristics:
- Evaluate arguments based on evidence and logic
- Consider the strength and relevance of each position
- Maintain objectivity and avoid bias
- Weigh competing claims fairly
- Provide clear, reasoned verdicts

When evaluating, assess the quality of evidence, logical coherence, and overall persuasiveness of each argument. Provide clear justification for your verdict."""
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

def user_prompt_opening_pro(claim, evidence, role, journalist_argument):
    return f"""You are a {role} who supports the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Journalist's argument: {journalist_argument}

Begin your argument with your position. Highlight facts that support the claim as TRUE."""

def user_prompt_rebuttal_pro(claim, evidence, con_argument, role, journalist_argument):
    return f"""You are a {role} who supports the following claim in a factual debate. Your opponent has argued against the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{con_argument}

Journalist's argument: {journalist_argument}

Write your rebuttal, explaining why the opponent is wrong and defending the claim."""

def user_prompt_closing_pro(claim, evidence, role, journalist_rebuttal):
    return f"""You are a {role} who supports the claim below. Provide your final summary.

Claim: {claim}

Evidence:
{evidence}

Journalist's rebuttal: {journalist_rebuttal}

Provide a closing statement reinforcing why the claim is TRUE."""

# === CON AGENT ===

def user_prompt_opening_con(claim, evidence, role, journalist_argument):
    return f"""You are a {role} who opposes the following claim. Present your opening argument using the evidence.

Claim: {claim}

Evidence:
{evidence}

Journalist's argument: {journalist_argument}

Begin your argument by explaining why the claim is FALSE or misleading, referencing specific points in the evidence."""

def user_prompt_rebuttal_con(claim, evidence, pro_argument, role, journalist_argument):
    return f"""You are a {role} who opposes the following claim in a factual debate. Your opponent has argued in favor of the claim.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument:
{pro_argument}

Journalist's argument: {journalist_argument}

Write your rebuttal, explaining why the opponent is incorrect and the claim is still FALSE or HALF-TRUE."""

def user_prompt_closing_con(claim, evidence, role, journalist_rebuttal):
    return f"""You are a {role} who opposes the claim below. Provide your final summary.
    
Claim: {claim}

Evidence:
{evidence}

Journalist's rebuttal: {journalist_rebuttal}

Provide a closing statement reinforcing why the claim is FALSE or HALF-TRUE."""

# === JOURNALIST AGENT ===

def user_prompt_opening_journalist(claim, evidence):
    return f"""You are a journalist evaluating the following claim. Extract and present the most compelling evidence that either supports or opposes the claim.

Claim: {claim}

Evidence:
{evidence}

Begin your argument by identifying the strongest evidence that either supports or contradicts the claim. Focus on the most relevant and impactful facts."""

def user_prompt_rebuttal_journalist(claim, evidence, pro_argument, con_argument):
    return f"""You are a journalist in a debate about the claim below. Both sides have made arguments about the veracity of the claim.

Claim: {claim}

Evidence:
{evidence}

Proponent's argument: {pro_argument}

Opponent's argument: {con_argument}

Write your rebuttal, focusing on the most compelling evidence that either supports or contradicts the claim, and address any gaps or weaknesses in the arguments presented."""

def user_prompt_closing_journalist(claim, evidence, pro_rebuttal, con_rebuttal):
    return f"""You are a journalist in a debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence:
{evidence}

Proponent's rebuttal: {pro_rebuttal}

Opponent's rebuttal: {con_rebuttal}

Provide a closing statement highlighting the most compelling evidence that either supports or contradicts the claim, and your assessment of the claim's veracity based on the key arguments presented."""

# === JUDGE AGENT ===

def user_prompt_judge_full(claim, evidence, pro_open, con_open, journalist_open, pro_rebut, con_rebut, journalist_rebut, pro_close, con_close, journalist_close):
    return f"""You are a neutral judge evaluating a factual debate with three participants.

Claim: {claim}

Evidence:
{evidence}

--- Opening Statements ---
Pro Agent:
{pro_open}

Con Agent:
{con_open}

Journalist:
{journalist_open}

--- Rebuttals ---
Pro Agent:
{pro_rebut}

Con Agent:
{con_rebut}

Journalist:
{journalist_rebut}

--- Closing Statements ---
Pro Agent:
{pro_close}

Con Agent:
{con_close}

Journalist:
{journalist_close}

Based on the arguments and evidence from all three participants, decide whether the claim is TRUE, FALSE, or HALF-TRUE.

Answer format:
[REASON]: <your justification>
[VERDICT]: TRUE / FALSE / HALF-TRUE  
"""

# === LEGACY JUDGE (for backward compatibility) ===

def user_prompt_judge_full_legacy(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
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
