# === System Prompt ===
def get_system_prompt(role: str = "fact_checker") -> str:
    if role == "fact_checker":
        return "You are a precise and critical fact checker."
    elif role == "politician":
        return """You are a politician in a political debate. You represent political interests and policy positions.

Key political characteristics:
- Focus on public opinion and voter concerns
- Emphasize policy benefits and political messaging
- Consider electoral implications and public perception
- Frame arguments in terms of public interest and national priorities
- Balance facts with political strategy and messaging

When debating, frame arguments in terms of public benefit, policy effectiveness, and political priorities. Keep responses concise and focused."""
    elif role == "scientist":
        return """You are a scientist/fact-checker in a political debate. You represent scientific evidence and factual accuracy.

Key scientific characteristics:
- Rely strictly on empirical evidence and data
- Question claims that lack scientific support
- Emphasize methodology and research quality
- Focus on accuracy over political messaging
- Consider uncertainty and limitations in evidence

When debating, frame arguments in terms of scientific evidence, data analysis, and factual accuracy. Keep responses concise and focused."""
    elif role == "judge":
        return """You are an ordinary American citizen serving as a judge in a political debate.

Your role is to evaluate the evidence and arguments from both parties and make a clear decision.

When making your decision, consider:
- Is this claim supported by credible evidence?
- Does this align with common sense and American values?
- What would be the practical impact on families and communities?

Do not deviate from this format. Do not ask for more information. Make your decision based on what is provided."""
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
def politician_opening_prompt(claim, evidence):
    return f"""Evaluate the following claim. Based on your stance as a politician, either support or oppose the claim. 
Present your opening argument using the evidence given.

Claim: {claim}

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support your position.
"""

def scientist_opening_prompt(claim, evidence):
    return f"""Evaluate the following claim. Based on your stance as a scientist, either support or oppose the claim. 
Present your opening argument using the evidence given.

Claim: {claim}

Evidence: {evidence}

Begin your argument with your position. Highlight facts that support your position.
"""

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
def politician_rebuttal_prompt(claim, evidence, con_argument):
    return f"""You are a politician in a debate about the claim below. Your opponent, the scientist, has made an argument about the veracity of the claim.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {con_argument}

Write your rebuttal, explaining why you disagree with the opponent and defending your position.
"""

def scientist_rebuttal_prompt(claim, evidence, con_argument):
    return f"""You are a scientist in a debate about the claim below. Your opponent, the politician, has made an argument about the veracity of the claim.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {con_argument}

Write your rebuttal, explaining why you disagree with the opponent and defending your position.
"""

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


# === Cross-Examination Round ===
def politician_cross_examination_prompt(claim, evidence, opponent_argument):
    return f"""You are the politician in a political debate. Conduct a cross-examination of your opponent's argument.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {opponent_argument}

Ask 2-3 specific, challenging questions about your opponent's argument. Focus on:
- Weaknesses in their evidence or reasoning
- Potential biases or assumptions
- Alternative interpretations of the facts
- Political implications they may have overlooked

Format your response as direct questions to your opponent."""

def scientist_cross_examination_prompt(claim, evidence, opponent_argument):
    return f"""You are the scientist in a political debate. Conduct a cross-examination of your opponent's argument.

Claim: {claim}

Evidence: {evidence}

Opponent's argument: {opponent_argument}

Ask 2-3 specific, challenging questions about your opponent's argument. Focus on:
- Methodological flaws in their reasoning
- Lack of scientific evidence for their claims
- Statistical or factual inaccuracies
- Logical fallacies or unsupported assumptions

Format your response as direct questions to your opponent."""

def user_prompt_cross_examination_pro(claim, evidence, con_argument):
    return f"""You are the supporting agent in a debate. Conduct a cross-examination of your opponent's argument.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument: {con_argument}

Ask 2-3 specific, challenging questions about your opponent's argument. Focus on weaknesses in their reasoning or evidence.

Format your response as direct questions to your opponent."""

def user_prompt_cross_examination_con(claim, evidence, pro_argument):
    return f"""You are the opposing agent in a debate. Conduct a cross-examination of your opponent's argument.

Claim: {claim}

Evidence:
{evidence}

Opponent's argument: {pro_argument}

Ask 2-3 specific, challenging questions about your opponent's argument. Focus on weaknesses in their reasoning or evidence.

Format your response as direct questions to your opponent."""

# === Closing Round ===
def politician_closing_prompt(claim, evidence):
    return f"""You are the politician in a political debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence: {evidence}

Provide a closing statement reinforcing your position about the claim.
"""

def scientist_closing_prompt(claim, evidence):
    return f"""You are the scientist in a political debate about the claim below. Summarize your final position.

Claim: {claim}

Evidence: {evidence}

Provide a closing statement reinforcing your position about the claim.
"""

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
def judge_prompt(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_close, con_close):
    return f"""
You are an ordinary American citizen evaluating a political debate.

Claim: {claim}

Evidence: {evidence}

--- Opening Statements ---
Politician's opening: {pro_open}
Scientist's opening: {con_open}

--- Rebuttals ---
Politician's rebuttal: {pro_rebut}
Scientist's rebuttal: {con_rebut}

--- Closing Statements ---
Politician's closing: {pro_close}
Scientist's closing: {con_close}

Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

You MUST respond in exactly this format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE

Make the verdict in the last line.
"""

def judge_prompt_1r(claim, evidence, pro_open, con_open):
    return f"""
You are an ordinary American citizen evaluating a political debate.

Claim: {claim}

Evidence: {evidence}

--- Opening Statements ---
Politician's opening: {pro_open}
Scientist's opening: {con_open}

Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

You MUST respond in exactly this format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE

Make the verdict in the last line.
"""

def judge_prompt_2r(claim, evidence, pro_open, con_open, pro_rebut, con_rebut):
    return f"""
You are an ordinary American citizen evaluating a political debate.

Claim: {claim}

Evidence: {evidence}

--- Opening Statements ---
Politician's opening: {pro_open}
Scientist's opening: {con_open}

--- Rebuttals ---
Politician's rebuttal: {pro_rebut}
Scientist's rebuttal: {con_rebut}

Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

You MUST respond in exactly this format:
[REASON]: <your explanation>
[VERDICT]: TRUE / FALSE / HALF-TRUE

Make the verdict in the last line.
"""

def judge_prompt_4r(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_cross, con_cross, pro_close, con_close):
    return f"""
You are an ordinary American citizen evaluating a political debate.

Claim: {claim}

Evidence: {evidence}

--- Opening Statements ---
Politician's opening: {pro_open}
Scientist's opening: {con_open}

--- Rebuttals ---
Politician's rebuttal: {pro_rebut}
Scientist's rebuttal: {con_rebut}

--- Cross-Examination ---
Politician's cross-examination: {pro_cross}
Scientist's cross-examination: {con_cross}

--- Closing Statements ---
Politician's closing: {pro_close}
Scientist's closing: {con_close}

Based on the evidence and arguments, decide if the claim is TRUE, FALSE, or HALF-TRUE.

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

def user_prompt_judge_full_4r(claim, evidence, pro_open, con_open, pro_rebut, con_rebut, pro_cross, con_cross, pro_close, con_close):
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

--- Cross-Examination ---
Pro Agent:
{pro_cross}

Con Agent:
{con_cross}

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